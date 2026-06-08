import os.path as osp
import csv
from collections import defaultdict

import torch
from torch import Tensor


def _load_segments(path: str) -> dict[str, list[tuple[int, int]]]:
    """
    Group (start_frame, raw_action_id) by video from an EK100 rulstm csv.
    Columns are headerless: id, video, start_f, end_f, verb, noun, action.
    """

    by_vid: dict[str, list[tuple[int, int]]] = defaultdict(list)
    with open(path, newline="") as f:
        for r in csv.reader(f):
            if not r or r[0] == "id":
                continue
            by_vid[r[1]].append((int(r[2]), int(r[6])))
    return by_vid


def build_action_transition_logprior(num_actions: int, anno_path: str = "annotations/ek100_rulstm/", csv_name: str = "training.csv", label_offset: int = 1, background_id: int | None = 0, pop_smoothing: float = 0.1, eps: float = 1e-12) -> Tensor:
    """
    Return a dense [num_actions, num_actions] log P(next | prev) table, rows indexed by
    previous action, columns by next action (both in 'label_offset'-shifted space). Each
    row is a proper distribution over foreground next-actions, Dirichlet-smoothed toward
    global successor popularity ('pop_smoothing'), so previous actions never seen as
    predecessors fall back to popularity rather than a degenerate row. The 'background_id'
    column is zeroed (background is never a valid successor target).

    'label_offset' matches the dataset's +1 background shift: use 1 with background_id=0 to
    live in model-logit space (model-index 0 = background); use 0 with background_id=None to
    stay in raw csv action-id space.
    """

    path = osp.join(anno_path, csv_name)
    by_vid = _load_segments(path)
    counts = torch.zeros(num_actions, num_actions, dtype=torch.float64)

    for segs in by_vid.values():
        segs.sort()  # temporal order within a video
        for i in range(1, len(segs)):
            p = segs[i - 1][1] + label_offset
            n = segs[i][1] + label_offset
            if 0 <= p < num_actions and 0 <= n < num_actions:
                counts[p, n] += 1.0

    if background_id is not None and 0 <= background_id < num_actions:
        counts[:, background_id] = 0.0  # background is never a successor target

    # Global successor popularity (foreground) -> Dirichlet prior for smoothing
    # and as the fallback distribution for unseen previous-actions.
    pop = counts.sum(dim=0)  # [A]
    pop = pop / pop.sum().clamp_min(eps)
    counts = counts + pop_smoothing * pop.unsqueeze(0)  # broadcast over rows
    row_sum = counts.sum(dim=-1, keepdim=True).clamp_min(eps)
    prob = counts / row_sum
    return torch.log(prob.clamp_min(eps)).float()


class ActionTransitionPrior:
    """
    Thin holder around a [A, A] log P(next | prev) table with fusion helpers.
    Kept device-agnostic: '.to(device)' once, then the fuse/top_modes helpers
    operate on the cached tensors. 'prob' is materialized lazily (only the soft,
    past-distribution-weighted fusion path needs it).
    """

    def __init__(self, log_prob: Tensor, background_id: int | None = 0):
        if log_prob.ndim != 2 or log_prob.size(0) != log_prob.size(1):
            raise ValueError(f"log_prob must be square [A, A]; got {tuple(log_prob.shape)}.")
        self.log_prob = log_prob
        self.background_id = background_id
        self._prob: Tensor | None = None

    @classmethod
    def from_annotations(cls, num_actions: int, background_id: int | None = 0, **kwargs) -> "ActionTransitionPrior":
        log_prob = build_action_transition_logprior(num_actions, background_id=background_id, **kwargs)
        return cls(log_prob, background_id=background_id)

    @property
    def num_actions(self) -> int:
        return self.log_prob.size(0)

    @property
    def prob(self) -> Tensor:
        if self._prob is None or self._prob.device != self.log_prob.device:
            self._prob = self.log_prob.exp()
        return self._prob

    def to(self, device=None, dtype=None) -> "ActionTransitionPrior":
        moved = self.log_prob.to(device=device, dtype=dtype)
        if moved is not self.log_prob:  # tensor.to is a no-op (same object) when already there
            self.log_prob = moved
            self._prob = None  # invalidate cache; re-materialized lazily on the new device
        return self

    def _logrow(self, prev_ids: Tensor | None, past_dist: Tensor | None, eps: float = 1e-12) -> Tensor:
        """
        Return per-sample [B, A] log-prior rows.
            - 'prev_ids' [B] long: hard lookup of the prev action's successor row.
            - 'past_dist' [B, A] prob: soft mixture sum_a past(a) * P(next | a), then
            logged, robust when the previous action is uncertain (low past_top1).
        """

        if (prev_ids is None) == (past_dist is None):
            raise ValueError("Provide exactly one of prev_ids or past_dist.")
        if prev_ids is not None:
            return self.log_prob[prev_ids.clamp(min=0).long()]
        mix = past_dist.float() @ self.prob  # [B, A]
        return mix.clamp_min(eps).log()

    def fuse(self, logits: Tensor, prev_ids: Tensor | None = None, past_dist: Tensor | None = None, weight: float = 1.0) -> Tensor:
        """
        Add 'weight * log P(next | prev)' to action logits.
        'logits' may be [B, A] (single head) or [B, K, A] (per slot); the prior
        row is broadcast across any middle (slot) dimensions.
        """

        logrow = self._logrow(prev_ids, past_dist)  # [B, A]
        while logrow.dim() < logits.dim():
            logrow = logrow.unsqueeze(1)
        return logits + float(weight) * logrow

    def top_modes(self, prev_ids: Tensor, k: int, exclude_ids: Tensor | None = None, exclude_background: bool = True) -> Tensor:
        """
        Top-k successor action ids of each prev action (excluding GT/background).
        Returns [B, k] long. Used to build external coverage targets for the
        non-anchor slots in the fixed-role scheme.
        """

        rows = self.log_prob[prev_ids.clamp(min=0).long()].clone()  # [B, A]
        B, A = rows.shape
        if exclude_background and self.background_id is not None and 0 <= self.background_id < A:
            rows[:, self.background_id] = float("-inf")
        if exclude_ids is not None:
            rows[torch.arange(B, device=rows.device), exclude_ids.clamp(min=0).long()] = float("-inf")
        return rows.topk(min(k, A), dim=-1).indices
