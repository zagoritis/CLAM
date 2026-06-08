"""
Regression check for scalant/datasets/transition_prior.py (run: python tools/check_transition_prior.py).
Loaded by file path to skip scalant.datasets.__init__ (heavy deps absent on the dev box).
"""

import csv
import importlib.util
import os.path as osp
from collections import defaultdict
import torch

HERE = osp.dirname(osp.abspath(__file__))
ROOT = osp.dirname(HERE)
ANNO = osp.join(ROOT, "annotations", "ek100_rulstm")

spec = importlib.util.spec_from_file_location("transition_prior", osp.join(ROOT, "scalant", "datasets", "transition_prior.py"))
tp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tp)


def val_pairs():
    by_vid = defaultdict(list)
    with open(osp.join(ANNO, "validation.csv"), newline="") as f:
        for r in csv.reader(f):
            if not r or r[0] == "id":
                continue
            by_vid[r[1]].append((int(r[2]), int(r[6])))
    
    pairs = []
    for segs in by_vid.values():
        segs.sort()
        for i in range(1, len(segs)):
            pairs.append((segs[i - 1][1], segs[i][1]))  # raw prev, raw next
    return pairs


def max_raw_action():
    m = 0
    for name in ("training.csv", "validation.csv"):
        with open(osp.join(ANNO, name), newline="") as f:
            for r in csv.reader(f):
                if not r or r[0] == "id":
                    continue
                m = max(m, int(r[6]))
    return m


def main():
    raw_max = max_raw_action()
    A_raw = raw_max + 1
    print(f"raw action id max = {raw_max}  (A_raw={A_raw})")

    # 1. raw-space recall@5 should reproduce ~34.9
    logp_raw = tp.build_action_transition_logprior(A_raw, anno_path=ANNO + osp.sep, label_offset=0, background_id=None, pop_smoothing=0.01)
    pairs = val_pairs()
    prev = torch.tensor([p for p, _ in pairs])
    nxt = torch.tensor([n for _, n in pairs])
    top5 = logp_raw[prev].topk(5, dim=-1).indices  # [N, 5]
    recall5 = (top5 == nxt[:, None]).any(dim=-1).float().mean().item() * 100
    top3 = logp_raw[prev].topk(3, dim=-1).indices
    recall3 = (top3 == nxt[:, None]).any(dim=-1).float().mean().item() * 100
    print(f"raw-space val recall@3 = {recall3:.2f}   recall@5 = {recall5:.2f}   (expect ~29.2 / ~34.9)")
    assert 33.0 <= recall5 <= 37.0, f"recall@5 {recall5:.2f} off expected ~34.9 -- build logic changed?"

    # 2. model-space invariants (label_offset=1, background_id=0)
    A = raw_max + 2  # shifted space includes background at 0
    prior = tp.ActionTransitionPrior.from_annotations(A, anno_path=ANNO + osp.sep, label_offset=1, background_id=0, pop_smoothing=0.1)
    assert prior.log_prob.shape == (A, A), prior.log_prob.shape
    assert (prior.log_prob[:, 0] < -20).all(), "background column must be strongly suppressed (~log eps)"
    row_mass = prior.prob.sum(dim=-1)
    assert torch.allclose(row_mass, torch.ones_like(row_mass), atol=1e-3), f"rows not normalized: {row_mass.min()}..{row_mass.max()}"
    assert float(prior.prob[:, 0].max()) < 1e-6, "background must carry negligible successor prob"

    # fuse: shapes [B,A] and [B,K,A]; top_modes excludes bg + GT
    B, K = 4, 5
    logits = torch.randn(B, K, A)
    prev_ids = torch.tensor([10, 250, 1000, 3000])
    fused = prior.fuse(logits, prev_ids=prev_ids, weight=1.0)
    assert fused.shape == (B, K, A)
    gt = torch.tensor([11, 251, 1001, 3001])
    modes = prior.top_modes(prev_ids, K - 1, exclude_ids=gt, exclude_background=True)
    assert modes.shape == (B, K - 1)
    assert (modes != 0).all(), "modes must exclude background"
    assert (modes != gt[:, None]).all(), "modes must exclude GT"
    # soft (past-distribution) fusion path
    past = torch.softmax(torch.randn(B, A), dim=-1)
    fused_soft = prior.fuse(torch.randn(B, A), past_dist=past, weight=0.5)
    assert fused_soft.shape == (B, A) and torch.isfinite(fused_soft).all()

    print("model-space invariants OK (shape, bg column, normalization, fuse, top_modes, soft path)")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
