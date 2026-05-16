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
from scalant.datasets import build_action_id_to_verb_noun_maps, build_action_similarity_matrix, build_dataset, diverse_action_rerank, topk_action_ids
from scalant.models import QueryPredictor


BACKGROUND = "__background__"


def parse_args():
    parser = argparse.ArgumentParser(description="Print qualitative top-k action anticipation predictions.")
    parser.add_argument("--cfg", default="configs/ek100/default.yaml")
    parser.add_argument("--checkpoint", required=True, help=("Checkpoint directory name under checkpoints/, checkpoint directory path, " "or full checkpoint_best.pth path."))
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--feat-dir", default="epickitchens100/features/rgb_kinetics_bninception")
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
    parser.add_argument("--diverse-rerank", action="store_true", help="Output a diverse predicted action set using MODEL.DIVERSE_SET settings.")
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    parser.add_argument("--opts", nargs=argparse.REMAINDER, default=None, help="Additional config overrides, using the same KEY VALUE format as main.py.")
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
    return [{"id": class_idx, "name": class_name(names, class_idx)} for class_idx in positive_label_ids(target_tensor)]


def action_predictions_from_ids(logits, action_ids, names, include_background, label_key):
    if action_ids.numel() == 0:
        return []

    scores = logits.detach().float()
    if not include_background and scores.numel() > 0:
        scores = scores.clone()
        scores[0] = float("-inf")

    probs = torch.softmax(scores, dim=-1)
    return [{"rank": rank, "id": int(class_idx), label_key: class_name(names, class_idx), "probability": float(probs[int(class_idx)].cpu())} for rank, class_idx in enumerate(action_ids.detach().cpu().tolist(), start=1)]


def topk_predictions(logits, names, k, include_background, label_key):
    action_ids = topk_action_ids(logits, k, include_background=include_background)
    return action_predictions_from_ids(logits, action_ids, names, include_background, label_key)


def primary_future_logits(future_logits, diverse_set_enabled):
    if diverse_set_enabled and future_logits.size(1) > 1:
        return future_logits[0, 0]
    return future_logits[0, -1]


def query_slot_action_ids(future_logits, include_background):
    scores = future_logits.detach().float()
    if not include_background and scores.size(-1) > 0:
        scores = scores.clone()
        scores[..., 0] = float("-inf")
    return scores.argmax(dim=-1)


def lookup_id(lookup, class_idx, unknown_id=-1):
    if lookup is None:
        return None
    class_idx = int(class_idx)
    if class_idx < 0 or class_idx >= lookup.numel():
        return None
    value = int(lookup[class_idx].item())
    return None if value == unknown_id else value


def action_set_predictions(logits, action_ids, action_names, verb_names, noun_names, action_to_verb_id, action_to_noun_id, include_background):
    if logits.ndim == 2:
        rows = []
        for rank, (slot_logits, action_id) in enumerate(zip(logits, action_ids), start=1):
            row = action_predictions_from_ids(slot_logits, action_id.view(1), action_names, include_background, "action")[0]
            row["rank"] = rank
            rows.append(row)
    else:
        rows = action_predictions_from_ids(logits, action_ids, action_names, include_background, "action")

    for row in rows:
        verb_id = lookup_id(action_to_verb_id, row["id"])
        noun_id = lookup_id(action_to_noun_id, row["id"])
        row["verb"] = {"id": verb_id, "name": class_name(verb_names, verb_id)} if verb_id is not None else None
        row["noun"] = {"id": noun_id, "name": class_name(noun_names, noun_id)} if noun_id is not None else None
    return rows


def observed_label_ids(target_tensor, ignore_id=0):
    if target_tensor is None:
        return set()

    labels = target_tensor.detach().cpu()
    if labels.ndim >= 2 and labels.shape[-1] > 1:
        ids = torch.nonzero((labels > 0).flatten(0, -2).any(dim=0), as_tuple=False).flatten().tolist()
    else:
        ids = labels.long().flatten().unique().tolist()
    return {int(class_idx) for class_idx in ids if int(class_idx) != ignore_id}


def count_matching_pairs(values):
    count = 0
    for left_idx in range(len(values)):
        if values[left_idx] is None:
            continue
        for right_idx in range(left_idx + 1, len(values)):
            if values[left_idx] == values[right_idx]:
                count += 1
    return count


def action_set_sample_metrics(action_ids, action_to_verb_id, action_to_noun_id, past_nouns, ignore_id=0):
    ids = [int(class_idx) for class_idx in action_ids.detach().cpu().tolist()]
    verb_ids = [lookup_id(action_to_verb_id, class_idx) for class_idx in ids]
    noun_ids = [lookup_id(action_to_noun_id, class_idx) for class_idx in ids]
    observed_nouns = observed_label_ids(past_nouns, ignore_id=ignore_id)
    plausible_nouns = [noun_id for noun_id in noun_ids if noun_id is not None and noun_id != ignore_id]

    return {"exact_duplicate": len(ids) - len(set(ids)), "verb_duplicate": count_matching_pairs(verb_ids), "noun_duplicate": count_matching_pairs(noun_ids), "object_match": (100.0 * sum(noun_id in observed_nouns for noun_id in plausible_nouns) / len(plausible_nouns) if plausible_nouns else None)}


def get_sample_summary(dataset, index, item_cpu, action_names):
    if not hasattr(dataset, "df"):
        return {"sample": {"index": index}, "target": {"actions": label_list(item_cpu.get("future_act"), action_names)}}

    row = dataset.df.loc[index]
    video_id = row["video_id"] if "video_id" in row else None

    observed_start = max(float(row["start"]), 0.0)
    observed_end = max(float(row["end"]), 0.0)
    action_start = float(row["orig_start"])
    action_end = float(row["orig_end"])

    return {
        "sample": {
            "index": index,
            "uid": int(row["uid"]) if "uid" in row else None,
            "video_id": video_id,
        },
        "observed": {
            "start": seconds_to_timestamp(observed_start),
            "end": seconds_to_timestamp(observed_end),
            "duration_sec": round(observed_end - observed_start, 2),
            "padded_at_start": bool(float(row["start"]) < 0),
        },
        "target": {
            "timestamp": {
                "start": row["start_timestamp"]
                if "start_timestamp" in row
                else seconds_to_timestamp(action_start),
                "end": row["stop_timestamp"]
                if "stop_timestamp" in row
                else seconds_to_timestamp(action_end),
            },
            "actions": label_list(item_cpu.get("future_act"), action_names),
            "narration": row["narration"] if "narration" in row else None,
        },
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
    return {key: value.unsqueeze(0).to(device=device, non_blocking=True) if hasattr(value, "unsqueeze") else value for key, value in item.items()}


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
    diverse_cfg = cfg.MODEL.DIVERSE_SET
    diverse_set_model = bool(diverse_cfg.MULTI_QUERY)
    diverse_enabled = bool(args.diverse_rerank or diverse_cfg.ENABLE or diverse_set_model)
    diverse_set_size = int(diverse_cfg.SET_SIZE)
    diversity_weight = float(diverse_cfg.DIVERSITY_WEIGHT)
    action_to_verb_id, action_to_noun_id, action_similarity = None, None, None
    if diverse_enabled:
        action_to_verb_id, action_to_noun_id = build_action_id_to_verb_noun_maps(dataset=dataset, device=device)
        action_similarity = build_action_similarity_matrix(dataset=dataset, device=device, dtype=torch.float) if not diverse_set_model and diversity_weight != 0 else None

    rows = []
    indices = sample_indices(len(dataset), args)
    if not indices:
        raise ValueError("No valid sample indices selected.")

    for index in indices:
        item_cpu = dataset[index]
        item = move_item_to_device(item_cpu, device)
        pred = model(item["past_feats"])
        action_logits = primary_future_logits(pred.future_actions, diverse_set_model)

        result = get_sample_summary(dataset, index, item_cpu, action_names)
        result["predictions"] = {"top_actions": topk_predictions(action_logits, action_names, args.topk, args.include_background, "action")}
        if pred.future_verbs is not None:
            result["predictions"]["top_verbs"] = topk_predictions(primary_future_logits(pred.future_verbs, diverse_set_model), verb_names, args.topk, args.include_background,  "verb")
        if pred.future_nouns is not None:
            result["predictions"]["top_nouns"] = topk_predictions(primary_future_logits(pred.future_nouns, diverse_set_model), noun_names, args.topk, args.include_background, "noun")
        if diverse_enabled:
            if diverse_set_model and pred.future_actions.size(1) > 1:
                diverse_action_ids = query_slot_action_ids(pred.future_actions[0], args.include_background)
                action_set_logits = pred.future_actions[0]
            else:
                diverse_action_ids = diverse_action_rerank(action_logits, diverse_set_size, action_similarity=action_similarity, diversity_weight=diversity_weight, include_background=args.include_background,)
                action_set_logits = action_logits
            result["predictions"]["action_set"] = action_set_predictions(action_set_logits, diverse_action_ids, action_names, verb_names, noun_names, action_to_verb_id, action_to_noun_id, args.include_background)
            result["set_metrics"] = action_set_sample_metrics(diverse_action_ids, action_to_verb_id, action_to_noun_id, item_cpu.get("past_noun"))
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
