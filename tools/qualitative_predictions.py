import argparse
import json
import os
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from helper import CKPT_BEST_FNAME, CKPT_PATH, load_model
from scalant.config import load_config
from scalant.datasets import build_dataset
from scalant.models import QueryPredictor


BACKGROUND = "__background__"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Print qualitative top-k action anticipation predictions."
    )
    parser.add_argument("--cfg", default="configs/ek100/default.yaml")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help=(
            "Checkpoint directory name under checkpoints/, checkpoint directory path, "
            "or full checkpoint_best.pth path."
        ),
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument(
        "--feat-dir",
        default="epickitchens100/features/rgb_kinetics_bninception",
    )
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--indices", type=int, nargs="*", default=None)
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--include-background", action="store_true")
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    parser.add_argument(
        "--opts",
        nargs=argparse.REMAINDER,
        default=None,
        help="Additional config overrides, using the same KEY VALUE format as main.py.",
    )
    return parser.parse_args()


def resolve_checkpoint_path(checkpoint_arg):
    checkpoint = Path(checkpoint_arg)
    if checkpoint.is_file():
        return checkpoint

    if checkpoint.is_dir():
        return checkpoint / CKPT_BEST_FNAME

    candidate = Path(CKPT_PATH) / checkpoint_arg / CKPT_BEST_FNAME
    return candidate


def invert_indexed_names(name_to_idx):
    names = {0: BACKGROUND}
    for name, idx in name_to_idx.items():
        names[idx + 1] = name
    return names


def class_name(names, class_idx):
    if class_idx is None:
        return None
    return names.get(int(class_idx), f"id={int(class_idx)}")


def seconds_to_timestamp(seconds):
    seconds = max(float(seconds), 0.0)
    centiseconds = int(round(seconds * 100))
    hours, remainder = divmod(centiseconds, 3600 * 100)
    minutes, remainder = divmod(remainder, 60 * 100)
    secs, centis = divmod(remainder, 100)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{centis:02d}"


def positive_label_ids(target_tensor):
    if target_tensor is None:
        return []
    labels = target_tensor[-1].detach().cpu()
    indices = torch.nonzero(labels > 0, as_tuple=False).flatten().tolist()
    if not indices:
        indices = [int(labels.argmax(dim=-1).item())]
    return [int(index) for index in indices]


def label_list(target_tensor, names):
    return [
        {
            "id": class_idx,
            "name": class_name(names, class_idx),
        }
        for class_idx in positive_label_ids(target_tensor)
    ]


def label_names(target_tensor, names):
    labels = label_list(target_tensor, names)
    if len(labels) == 1:
        return labels[0]["name"]
    return [label["name"] for label in labels]


def topk_predictions(logits, names, k, include_background, label_key):
    scores = logits.detach().float()
    if not include_background and scores.numel() > 0:
        scores = scores.clone()
        scores[0] = float("-inf")

    probs = torch.softmax(scores, dim=-1)
    values, indices = probs.topk(min(k, probs.numel()))
    return [
        {
            "rank": rank,
            "id": int(class_idx),
            label_key: class_name(names, class_idx),
            "probability": float(prob),
        }
        for rank, (prob, class_idx) in enumerate(
            zip(values.cpu(), indices.cpu()), start=1
        )
    ]


def get_sample_summary(dataset, index, item_cpu, action_names):
    if not hasattr(dataset, "df"):
        return {
            "sample_index": index,
            "ground_truth_action": label_names(item_cpu.get("future_act"), action_names),
        }

    row = dataset.df.loc[index]
    video_id = row["video_id"] if "video_id" in row else None
    participant_id = row["participant_id"] if "participant_id" in row else None

    observed_start = max(float(row["start"]), 0.0)
    observed_end = max(float(row["end"]), 0.0)
    action_start = float(row["orig_start"])
    action_end = float(row["orig_end"])

    return {
        "sample_index": index,
        "uid": int(row["uid"]) if "uid" in row else None,
        "video_id": video_id,
        "raw_video_hint": (
            f"{participant_id}/{video_id}.MP4"
            if participant_id is not None and video_id is not None
            else None
        ),
        "observed_timestamp": {
            "start": seconds_to_timestamp(observed_start),
            "end": seconds_to_timestamp(observed_end),
            "duration_sec": round(observed_end - observed_start, 2),
            "padded_at_start": bool(float(row["start"]) < 0),
        },
        "ground_truth_next_action_timestamp": {
            "start": row["start_timestamp"]
            if "start_timestamp" in row
            else seconds_to_timestamp(action_start),
            "end": row["stop_timestamp"]
            if "stop_timestamp" in row
            else seconds_to_timestamp(action_end),
        },
        "ground_truth_action": label_names(item_cpu.get("future_act"), action_names),
        "annotation_narration": row["narration"] if "narration" in row else None,
    }


def sample_indices(dataset_size, args):
    if args.indices:
        indices = args.indices
    elif args.random:
        rng = random.Random(args.seed)
        indices = rng.sample(range(dataset_size), min(args.num_samples, dataset_size))
    else:
        stop = args.start_index + args.num_samples * args.stride
        indices = list(range(args.start_index, stop, args.stride))

    return [idx for idx in indices if 0 <= idx < dataset_size]


def move_item_to_device(item, device):
    return {
        key: value.unsqueeze(0).to(device=device, non_blocking=True)
        if hasattr(value, "unsqueeze")
        else value
        for key, value in item.items()
    }


@torch.inference_mode()
def main():
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    device = torch.device(args.device)

    cfg = load_config(args.cfg, args.opts)
    cfg.DATA.DATA_ROOT_PATH = args.data_root
    cfg.DATA.FEAT_DIR = args.feat_dir
    cfg.TRAIN.ENABLE = False
    cfg.TEST.ENABLE = True

    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    dataset = build_dataset(cfg, args.split)
    model = QueryPredictor(cfg, num_classes=dataset.num_classes, dataset=dataset)
    load_model(model, str(checkpoint_path))
    model.to(device=device)
    model.eval()

    action_names = invert_indexed_names(dataset.action_classes)
    verb_names = invert_indexed_names(dataset.verb_classes)
    noun_names = invert_indexed_names(dataset.noun_classes)

    rows = []
    indices = sample_indices(len(dataset), args)
    if not indices:
        raise ValueError("No valid sample indices selected.")

    for index in indices:
        item_cpu = dataset[index]
        item = move_item_to_device(item_cpu, device)
        pred = model(item["past_feats"])

        result = {
            **get_sample_summary(dataset, index, item_cpu, action_names),
            "top5_actions": topk_predictions(
                pred.future_actions[0, -1],
                action_names,
                args.topk,
                args.include_background,
                "action",
            ),
            "top5_verbs": topk_predictions(
                pred.future_verbs[0, -1],
                verb_names,
                args.topk,
                args.include_background,
                "verb",
            )
            if pred.future_verbs is not None
            else [],
            "top5_nouns": topk_predictions(
                pred.future_nouns[0, -1],
                noun_names,
                args.topk,
                args.include_background,
                "noun",
            )
            if pred.future_nouns is not None
            else [],
        }
        rows.append(result)

    print(json.dumps(rows, indent=2))

    if args.output:
        output_path = Path(args.output)
        if output_path.parent:
            os.makedirs(output_path.parent, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
