"""Mirror-math smoke test for the transition wiring in criterion.py.

The criterion itself can't be imported on the dev box (mamba/nestconfig/dataset
deps), so this reproduces the exact tensor ops of the two new code paths in
isolation and checks their invariants:

  Step A  - _set_metric_dict fusion: fusing log P(next|prev) into head logits
            pulls a true-but-low-ranked successor into the top-5.
  Step B  - _diverse_coverage_loss with COVERAGE_SOURCE='transition': the
            Hungarian assignment + CE over transition modes is finite and
            differentiable, gradients reach slots 1..K-1 but NOT the slot-0
            anchor, and the targets are the prior's transition successors.

Run:  python tools/check_transition_criterion.py
"""
import importlib.util
import math
import os.path as osp
from itertools import permutations

import torch

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
spec = importlib.util.spec_from_file_location("transition_prior", osp.join(ROOT, "scalant", "datasets", "transition_prior.py"))
tp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tp)


def make_prior(A=10, bg=0):
    """Hand-built prior: prev=3 -> successor 7 dominant (0.6), 5 secondary (0.3)."""
    prob = torch.full((A, A), 1e-6)
    prob[:, bg] = 0.0
    prob[3, 7] = 0.6
    prob[3, 5] = 0.3
    prob = prob / prob.sum(-1, keepdim=True).clamp_min(1e-12)
    return tp.ActionTransitionPrior(prob.clamp_min(1e-12).log(), background_id=bg)


def test_step_a_fusion():
    A, bg = 10, 0
    prior = make_prior(A, bg)
    # Head ranks distractors {1,2,4,6,8} top (+5); the true next action is 7, the
    # dominant transition successor of prev=3, but the head buries it at 0.
    head = torch.zeros(1, A)
    head[0, [1, 2, 4, 6, 8]] = 5.0
    gt, prev = 7, torch.tensor([3])

    base_top5 = head[0].clone()
    base_top5[bg] = float("-inf")
    before = set(base_top5.topk(5).indices.tolist())

    fused = prior.fuse(head.float(), prev_ids=prev, weight=10.0)[0]
    fused[bg] = float("-inf")
    after = set(fused.topk(5).indices.tolist())

    assert gt not in before, f"setup wrong: GT already in head top-5 {before}"
    assert gt in after, f"fusion failed to surface GT successor; top-5={after}"
    assert fused.argmax().item() == gt, "GT should dominate after fusing a strong successor prior"
    print(f"Step A: recall@5 0 -> 1 after fusion (top-5 {sorted(before)} -> {sorted(after)})  OK")


def coverage_loss_transition(future_logits, target_future, prev_ids, prior, ignore_index=0, diversity_temp=1.0):
    """Verbatim reproduction of criterion._diverse_coverage_loss (transition branch)."""
    B, K, A = future_logits.shape
    device = future_logits.device
    bg = ignore_index
    has_bg = 0 <= bg < A

    t = target_future[:, -1].float()
    if has_bg:
        t = t.clone(); t[:, bg] = float("-inf")
    gt = t.argmax(dim=-1)

    modes = prior.to(device).top_modes(prev_ids.to(device), K - 1, exclude_ids=gt, exclude_background=has_bg)

    nw = future_logits[:, 1:].float()
    if has_bg:
        nw = nw.clone(); nw[:, :, bg] = float("-inf")
    logp = torch.log_softmax(nw, dim=-1)

    modes_exp = modes.unsqueeze(1).expand(-1, K - 1, -1)
    cost = -logp.gather(2, modes_exp)
    perms = torch.tensor(list(permutations(range(K - 1))), dtype=torch.long, device=device)
    slot_index = torch.arange(K - 1, device=device).unsqueeze(0).expand(perms.size(0), -1)
    perm_cost = cost.detach()[:, slot_index, perms].sum(dim=-1)
    best = perm_cost.argmin(dim=-1)
    assigned = modes.gather(1, perms[best])
    chosen_logp = logp.gather(2, assigned.unsqueeze(-1)).squeeze(-1)
    return (-chosen_logp).mean(), modes, assigned, gt


def test_step_b_coverage():
    A, bg = 10, 0
    B, K = 6, 5
    prior = make_prior(A, bg)
    logits = torch.randn(B, K, A, requires_grad=True)
    target = torch.zeros(B, 1, A); target[:, 0, 4] = 1.0   # GT action = 4
    prev = torch.tensor([3, 3, 3, 3, 3, 3])

    loss, modes, assigned, gt = coverage_loss_transition(logits, target, prev, prior, ignore_index=bg)
    assert torch.isfinite(loss) and loss.item() > 0, f"loss not finite/positive: {loss}"

    # transition modes for prev=3 must be its top successors {7,5,...}, excl GT/bg
    assert (modes != bg).all() and (modes != gt[:, None]).all(), "modes must exclude bg and GT"
    assert modes[0, 0].item() == 7 and modes[0, 1].item() == 5, f"top transition modes wrong: {modes[0].tolist()}"

    loss.backward()
    g = logits.grad
    assert torch.isfinite(g).all(), "non-finite gradient"
    assert g[:, 0].abs().sum().item() == 0.0, "slot 0 (GT anchor) must receive NO coverage gradient"
    assert g[:, 1:].abs().sum().item() > 0.0, "slots 1..K-1 must receive coverage gradient"
    print(f"Step B: transition-coverage loss={loss.item():.3f}, modes[0]={modes[0].tolist()}, "
          f"grad slot0={g[:,0].abs().sum():.1e} slots1+={g[:,1:].abs().sum():.2f}  OK")


if __name__ == "__main__":
    test_step_a_fusion()
    test_step_b_coverage()
    print("ALL CRITERION-MATH CHECKS PASSED")
