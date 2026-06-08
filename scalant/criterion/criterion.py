import logging
from collections.abc import Mapping

import torch
from torch import Tensor

from scalant.config import Config
from scalant.utils import accuracy
from scalant.utils.ouput_target_structure import Prediction, Target
from scalant.criterion.build import Criterion_REGISTRY
from scalant.datasets import (EpicKitchens, action2verbnoun, action_set_metrics, build_action_id_to_verb_noun_maps, build_action_similarity_matrix, diverse_action_rerank, topk_action_ids, verbnoun2action)
from scalant.datasets.transition_prior import ActionTransitionPrior
from scalant.criterion.loss import *

logger = logging.getLogger(__name__)


@Criterion_REGISTRY.register()
class Criterion_LSTR:
    def __init__(self, cfg: Config, dataset: EpicKitchens):
        ignore_index = cfg.MODEL.IGNORE_INDEX

        if isinstance(dataset, EpicKitchens):
            self.action_cls = MultipCrossEntropyEqualizedLoss(ignore_index=ignore_index)
        else:
            self.action_cls = MultipCrossEntropyLoss(ignore_index=ignore_index)
        self.verb_noun_cls = MultipCrossEntropyLoss(ignore_index=ignore_index)
        self.ignore_index = ignore_index
        self.dataset = dataset
        self.cfg = cfg
        self.set_metric_k = int(cfg.MODEL.DIVERSE_SET.SET_SIZE)
        self.set_metric_background_id = ignore_index if ignore_index >= 0 else None
        self.multi_query = bool(cfg.MODEL.DIVERSE_SET.MULTI_QUERY)
        self.set_metrics_enabled = False

        # Gate for the set-based ('hit-anywhere') future-action loss. Requires
        # both the diverse-set training regime (ENABLE) and the K-slot
        # architecture (MULTI_QUERY); the loss is meaningless with K=1.
        self.set_loss_enabled = bool(cfg.MODEL.DIVERSE_SET.ENABLE) and self.multi_query
        self.hit_weight = float(cfg.MODEL.DIVERSE_SET.HIT_WEIGHT)
        if self.set_loss_enabled and self.hit_weight == 0.0:
            logger.warning("DIVERSE_SET.ENABLE=True with HIT_WEIGHT=0 would leave the future-action head unsupervised; defaulting effective weight to 1.0. Set HIT_WEIGHT explicitly to scale the set loss.")
            self.hit_weight = 1.0
        if self.multi_query and not self.set_loss_enabled:
            logger.warning("MULTI_QUERY=True but DIVERSE_SET.ENABLE=False: only slot 0 receives action-CE gradient; slots 1..K-1 stay unsupervised. Set DIVERSE_SET.ENABLE=True to activate the hit-anywhere set loss before training a multi-query checkpoint.")

        # Wrap the SAME per-row CE used in single-query mode so the
        # multi-query ablation changes only the aggregation, not the
        # per-class weighting (EPIC equalization is preserved).
        self.set_hit_loss = MultipSetHitLoss(base_loss=self.action_cls, epsilon=float(cfg.MODEL.DIVERSE_SET.HIT_EPSILON))

        # Step 8: fixed-role coverage loss. Active only when the set loss is
        # active AND DIVERSITY_WEIGHT > 0. DIVERSITY_WEIGHT scales the coverage
        # CE; DIVERSITY_TEMP is the temperature of slot 0's detached distribution
        # used to rank its runner-up modes; DIVERSITY_WARMUP_EPOCHS linearly
        # ramps the coverage weight from 0. 'current_epoch' is set by the train
        # loop each epoch (defaults to past-warmup for eval). The slot-permutation
        # table for the assignment is cached lazily on first use.
        self.diversity_weight = float(cfg.MODEL.DIVERSE_SET.DIVERSITY_WEIGHT)
        self.diversity_temp = float(cfg.MODEL.DIVERSE_SET.DIVERSITY_TEMP)
        if self.diversity_temp <= 0:
            raise ValueError(f"DIVERSE_SET.DIVERSITY_TEMP must be > 0; got {self.diversity_temp}.")
        self.diversity_warmup_epochs = int(cfg.MODEL.DIVERSE_SET.DIVERSITY_WARMUP_EPOCHS)
        self.current_epoch = 0
        self._perm_cache = None

        try:
            self.action_to_verb_id, self.action_to_noun_id = build_action_id_to_verb_noun_maps(dataset=dataset, background_id=self.set_metric_background_id)
            self.action_similarity = build_action_similarity_matrix(dataset=dataset, background_id=self.set_metric_background_id)
            self.set_metrics_enabled = True
        except (AttributeError, KeyError, ValueError):
            self.action_to_verb_id, self.action_to_noun_id, self.action_similarity = None, None, None

        # First-order action-transition prior P(next | prev). Built once, lazily,
        # only when needed: TRANSITION_WEIGHT > 0 (Step A inference fusion) or
        # COVERAGE_SOURCE == "transition" (Step B training targets). It is external,
        # data-driven structure that lets the diverse set exceed the single-head
        # top-K ceiling (see scalant/datasets/transition_prior.py).
        self.transition_weight = float(cfg.MODEL.DIVERSE_SET.TRANSITION_WEIGHT)
        self.transition_fuse_source = str(cfg.MODEL.DIVERSE_SET.TRANSITION_FUSE_SOURCE)
        self.coverage_source = str(cfg.MODEL.DIVERSE_SET.COVERAGE_SOURCE)
        self.transition_prior = None
        if self.transition_weight > 0.0 or self.coverage_source == "transition":
            try:
                num_actions = dataset.num_classes
                num_actions = int(num_actions["action"] if isinstance(num_actions, Mapping) else num_actions)
                self.transition_prior = ActionTransitionPrior.from_annotations(num_actions, background_id=self.set_metric_background_id if self.set_metric_background_id is not None else 0, anno_path=str(cfg.MODEL.DIVERSE_SET.TRANSITION_ANNO), label_offset=1)
                logger.info(f"Built action-transition prior [{num_actions}x{num_actions}] "f"(transition_weight={self.transition_weight}, fuse_source={self.transition_fuse_source}, coverage_source={self.coverage_source}).")
            except (FileNotFoundError, AttributeError, KeyError, ValueError) as e:
                logger.warning(f"Could not build action-transition prior ({e!r}); transition fusion/coverage stays OFF.")
                self.transition_prior = None
        if self.coverage_source == "transition" and self.transition_prior is None:
            logger.warning("COVERAGE_SOURCE='transition' but the prior failed to build; coverage falls back to slot-0 self-distillation ('self').")

    def __call__(self, pred: Prediction, target: Target, is_training=True) -> (Tensor, dict):
        # Always materialize the slot-0 view so accuracy/top-1 stay defined the
        # same way whether or not the set loss is active.
        future_action_pred, future_action_target = self._future_loss_pair(pred.future_actions, target.future_actions)
        notice_index = [i for i in range(target.past_actions.shape[-1]) if i != self.ignore_index]
        past_cls = self.action_cls(pred.past_actions, target.past_actions)

        # Future-action supervision has three regimes:
        #   - fixed-role (Step 8): slot 0 is anchored to the GT with plain
        #     equalized CE and slots 1..K-1 are pushed onto slot 0's own
        #     runner-up modes by the coverage loss. Active when the set loss is
        #     on AND DIVERSITY_WEIGHT > 0.
        #   - hit-anywhere (Step 7): epsilon-relaxed WTA over all K slots. Active
        #     when the set loss is on AND DIVERSITY_WEIGHT == 0 (the ablation).
        #   - single-query: plain CE on the lone slot.
        use_set_loss = self.set_loss_enabled and pred.future_actions.size(1) > 1
        fixed_role = use_set_loss and self.diversity_weight > 0.0
        action_winners, valid_mask = None, None
        if fixed_role:
            # Slot 0 is the plausibility anchor: the SAME equalized CE used in
            # single-query mode, applied to slot 0 vs the GT next action. This
            # keeps slot 0 a strong, input-dependent predictor (mt5r/top1) and
            # gives the coverage loss a grounded ranking to expand from.
            future_action_loss = self.action_cls(future_action_pred, future_action_target)
            future_loss_key = "future_cls_loss"
            future_loss_value = future_action_loss.item()
        elif use_set_loss:
            future_set_hit, action_winners, valid_mask = self.set_hit_loss.aggregate(pred.future_actions, target.future_actions)
            future_action_loss = self.hit_weight * future_set_hit
            future_loss_key = "future_set_hit_loss"
            future_loss_value = future_set_hit.item()
        else:
            future_action_loss = self.action_cls(future_action_pred, future_action_target)
            future_loss_key = "future_cls_loss"
            future_loss_value = future_action_loss.item()

        loss = past_cls + future_action_loss

        # Step 8: coverage loss. Matches slots 1..K-1 to slot 0's detached
        # top-(K-1) non-GT modes so the K slots span a diverse, plausible set.
        # The weight is linearly warmed up so slot 0 first becomes a meaningful
        # ranker before the other slots are asked to mirror its modes.
        future_coverage_value = None
        if fixed_role:
            cov_prev_ids = self._prev_action_ids(target) if (self.coverage_source == "transition" and self.transition_prior is not None) else None
            future_coverage = self._diverse_coverage_loss(pred.future_actions, target.future_actions, prev_ids=cov_prev_ids)
            cov_weight = self.diversity_weight * (self._diversity_warmup() if is_training else 1.0)
            loss = loss + cov_weight * future_coverage
            future_coverage_value = float(future_coverage.item())

        # Compute metrics
        (past_top1,), past_counts = accuracy(pred.past_actions[..., notice_index], target.past_actions[..., notice_index])
        (future_top1,), future_counts = accuracy(future_action_pred[..., notice_index], future_action_target[..., notice_index])

        # Mean top 5
        mt5r_dict = {"logits": self._primary_future_logits(pred.future_actions)[:, notice_index], "labels": target.future_actions[:, -1, notice_index].argmax(dim=-1)}
        loss_dict = {"past_cls_loss": past_cls.item(), future_loss_key: future_loss_value, "past_top1": [None, past_top1, past_counts], "future_top1": [None, future_top1, future_counts], "mt5r": ["MeanTopKRecallMeter", mt5r_dict, None]}
        if future_coverage_value is not None:
            loss_dict["future_coverage_loss"] = future_coverage_value

        if pred.past_verbs is not None:
            past_verb = self.verb_noun_cls(pred.past_verbs, target.past_verbs)
            past_noun = self.verb_noun_cls(pred.past_nouns, target.past_nouns)
            if use_set_loss and not fixed_role:
                # Tied WTA: supervise verb/noun on the slot the action head
                # picked as the winner, so each slot stays a coherent action
                # hypothesis instead of slot 0 being anchored as the default.
                future_verb = self._winner_aux_loss(pred.future_verbs, target.future_verbs, action_winners, valid_mask)
                future_noun = self._winner_aux_loss(pred.future_nouns, target.future_nouns, action_winners, valid_mask)
            else:
                # Single-query and fixed-role: verb/noun follow slot 0, the GT anchor.
                future_verb_pred, future_verb_target = self._future_loss_pair(pred.future_verbs, target.future_verbs)
                future_noun_pred, future_noun_target = self._future_loss_pair(pred.future_nouns, target.future_nouns)
                future_verb = self.verb_noun_cls(future_verb_pred, future_verb_target)
                future_noun = self.verb_noun_cls(future_noun_pred, future_noun_target)

            loss += past_verb + past_noun + future_verb + future_noun
            loss_dict.update({"past_verb_loss": past_verb.item(), "past_noun_loss": past_noun.item(), "future_verb_loss": future_verb.item(), "future_noun_loss": future_noun.item()})
            verb_notice_index = [i for i in range(target.future_verbs.shape[-1]) if i != self.ignore_index]
            noun_notice_index = [i for i in range(target.future_nouns.shape[-1]) if i != self.ignore_index]
            # Mean top 5
            verb_mt5r_dict = {"logits": self._primary_future_logits(pred.future_verbs)[:, verb_notice_index], "labels": target.future_verbs[:, -1, verb_notice_index].argmax(dim=-1)}
            noun_mt5r_dict = {"logits": self._primary_future_logits(pred.future_nouns)[:, noun_notice_index], "labels": target.future_nouns[:, -1, noun_notice_index].argmax(dim=-1)}
            loss_dict.update({"verb_mt5r_cls": ["MeanTopKRecallMeter", verb_mt5r_dict, None], "noun_mt5r_cls": ["MeanTopKRecallMeter", noun_mt5r_dict, None]})


        future_verbs, future_nouns = action2verbnoun(pred.future_actions, self.dataset.class_mappings)
        verb_notice_index = [i for i in range(target.future_verbs.shape[-1]) if i != self.ignore_index]
        noun_notice_index = [i for i in range(target.future_nouns.shape[-1]) if i != self.ignore_index]
        # Mean top 5
        verb_mt5r_dict = {"logits": self._primary_future_logits(future_verbs)[:, verb_notice_index], "labels": target.future_verbs[:, -1, verb_notice_index].argmax(dim=-1)}
        noun_mt5r_dict = {"logits": self._primary_future_logits(future_nouns)[:, noun_notice_index], "labels": target.future_nouns[:, -1, noun_notice_index].argmax(dim=-1)}
        loss_dict.update({"verb_mt5r": ["MeanTopKRecallMeter", verb_mt5r_dict, None], "noun_mt5r": ["MeanTopKRecallMeter", noun_mt5r_dict, None]})

        if not is_training and self.set_metrics_enabled:
            loss_dict.update(self._set_metric_dict(pred, target))

        return loss, loss_dict

    def _set_metric_dict(self, pred: Prediction, target: Target) -> dict:
        future_logits = self._primary_future_logits(pred.future_actions)
        slot_logits = pred.future_actions
        device = future_logits.device

        # Step A (probe): fuse the transition prior into the action logits feeding
        # BOTH the single-head top-K set and the per-slot diverse set. Adding the
        # same prior to every slot biases them toward the shared top successors, so
        # the multi-query diversity drops here, that is expected, and is exactly
        # why Step B instead grounds each slot in a DISTINCT transition successor at
        # train time. The headline number for this probe is topk_set_recall@5: does
        # the single head, reranked by transitions, clear its own ~22.4 ceiling?
        if self.transition_prior is not None and self.transition_weight > 0.0:
            self.transition_prior.to(device)
            fuse_kw = self._transition_fuse_kwargs(pred, target, device)
            future_logits = self.transition_prior.fuse(future_logits.float(), weight=self.transition_weight, **fuse_kw)
            slot_logits = self.transition_prior.fuse(slot_logits.float(), weight=self.transition_weight, **fuse_kw)

        action_to_verb_id = self.action_to_verb_id.to(device=device)
        action_to_noun_id = self.action_to_noun_id.to(device=device)
        action_similarity = self.action_similarity.to(device=device)
        topk_sets = torch.stack([topk_action_ids(logits, self.set_metric_k, include_background=False, background_id=self.set_metric_background_id) for logits in future_logits])

        metric_dict = {}
        topk_metrics = action_set_metrics(topk_sets, target.future_actions[:, -1], action_to_verb_id, action_to_noun_id, action_similarity=action_similarity, past_nouns=target.past_nouns, ignore_index=self.ignore_index if self.ignore_index >= 0 else None)
        metric_dict.update(self._format_set_metrics("topk", topk_metrics))

        if self.cfg.MODEL.DIVERSE_SET.ENABLE or self.multi_query:
            if self.multi_query and slot_logits.size(1) > 1:
                diverse_sets = self._query_slot_action_ids(slot_logits)
            else:
                diverse_sets = torch.stack([diverse_action_rerank(logits, self.set_metric_k, action_similarity=action_similarity, diversity_weight=float(self.cfg.MODEL.DIVERSE_SET.DIVERSITY_WEIGHT), include_background=False, background_id=self.set_metric_background_id) for logits in future_logits])
            diverse_metrics = action_set_metrics(diverse_sets, target.future_actions[:, -1], action_to_verb_id, action_to_noun_id, action_similarity=action_similarity, past_nouns=target.past_nouns, ignore_index=self.ignore_index if self.ignore_index >= 0 else None)
            metric_dict.update(self._format_set_metrics("diverse", diverse_metrics))

        return metric_dict

    def _format_set_metrics(self, prefix: str, metrics: dict) -> dict:
        return {f"{prefix}_{metric_name}@{self.set_metric_k}": [None, value, count] for metric_name, (value, count) in metrics.items()}

    def _future_loss_pair(self, pred_tensor: Tensor, target_tensor: Tensor) -> tuple[Tensor, Tensor]:
        # Multi-query slots are alternatives along dim 1, not temporal steps.
        # Until a set-based ("hit-anywhere") loss is wired in, supervise slot 0
        # only with the GT next action (target's last step).
        if self.multi_query and pred_tensor is not None and target_tensor is not None:
            if target_tensor.ndim < 3:
                raise ValueError(f"Expected target tensor with shape [B, T, C]; got {tuple(target_tensor.shape)}.")
            if pred_tensor.size(0) != target_tensor.size(0):
                raise ValueError(f"Batch size mismatch between pred {tuple(pred_tensor.shape)} and target {tuple(target_tensor.shape)}.")
            return pred_tensor[:, :1], target_tensor[:, -1:]
        return pred_tensor, target_tensor

    def _diversity_warmup(self) -> float:
        """
        Linear 0->1 ramp of the coverage weight over the first
        DIVERSITY_WARMUP_EPOCHS epochs, so slot 0 (the anchor) becomes a
        meaningful ranker before slots 1..K-1 are asked to mirror its modes.
        Returns 1.0 when warmup is disabled or already complete.
        """

        w = int(getattr(self, "diversity_warmup_epochs", 0))
        if w <= 0:
            return 1.0
        e = int(getattr(self, "current_epoch", w))
        return min(1.0, max(0.0, e / w))

    def _assignment_perms(self, n: int, device) -> Tensor:
        """Cached [n!, n] table of all slot->target permutations (n is small, K-1)."""
        if self._perm_cache is None or self._perm_cache[0] != n:
            from itertools import permutations
            perms = torch.tensor(list(permutations(range(n))), dtype=torch.long)
            self._perm_cache = (n, perms)
        return self._perm_cache[1].to(device=device)

    def _diverse_coverage_loss(self, future_logits: Tensor, target_future: Tensor, prev_ids: Tensor | None = None) -> Tensor:
        """
        Coverage loss for the fixed-role scheme.

        Slot 0 is the GT anchor (supervised separately by the equalized CE). This
        term spreads slots 1..K-1 across K-1 target modes, then:
          1. pick K-1 distinct target action ids (the coverage modes, see below);
          2. match slots 1..K-1 to those K-1 modes one-to-one, minimizing total
             assignment cost (each slot covers the mode it already prefers);
          3. cross-entropy pushing each slot toward its assigned mode.

        The coverage modes come from one of two sources (COVERAGE_SOURCE):
        - 'self' (Step 8(5)): slot 0's OWN detached top-(K-1) non-GT modes, ranked
          by softmax(z_0 / tau). Grounded slot 0 is pinned to the real GT, so the
          runner modes track a real, input-dependent predictor and cannot drift to
          an input-independent fixed set. BUT the diverse set is then by construction
          an approximation of slot 0's top-K, so it cannot exceed the single-head
          ~23% set-recall ceiling.
        - 'transition' (Step B): the top-(K-1) transition successors of the observed
          previous action, P(next | prev), excluding the GT and background. These are
          EXTERNAL, data-driven conditional modes, often actions the single head
          ranks below its top-K (transition recall@5 ~34.9 vs head ~22.4), so the
          slots can cover ground the single head's top-K misses and the diverse set
          can exceed the ceiling. Still collapse-safe: targets are fixed dataset
          statistics, not a GT-excluding self-referential signal.

        The CE toward concrete target actions keeps a non-vanishing gradient at
        collapse: a slot sitting elsewhere assigns low probability to its assigned
        mode and is pulled toward it. Returns a graph-preserving zero scalar when K < 2.
        """

        B, K, A = future_logits.shape
        if K < 2:
            return future_logits.sum() * 0.0

        device = future_logits.device
        bg = self.ignore_index
        has_bg = 0 <= bg < A

        # GT action = dominant foreground class of the (possibly mixed) target.
        t = target_future[:, -1].float()
        if has_bg:
            t = t.clone()
            t[:, bg] = float("-inf")
        gt = t.argmax(dim=-1)                                # [B]

        # Coverage modes for slots 1..K-1: external transition successors of the
        # observed action when COVERAGE_SOURCE=='transition' (can exceed the
        # single-head ceiling), else slot 0's own detached top-(K-1) modes.
        use_transition = (self.coverage_source == "transition" and self.transition_prior is not None and prev_ids is not None)
        if use_transition:
            modes = self.transition_prior.to(device).top_modes(prev_ids.to(device), K - 1, exclude_ids=gt, exclude_background=has_bg)  # [B, K-1]
        else:
            ref0 = future_logits[:, 0].detach().float() / self.diversity_temp  # [B, A]
            if has_bg:
                ref0[:, bg] = float("-inf")
            ref0[torch.arange(B, device=device), gt] = float("-inf")
            modes = ref0.topk(K - 1, dim=-1).indices            # [B, K-1] distinct ids

        # Trained slots 1..K-1, foreground log-probs (background removed).
        nw = future_logits[:, 1:].float()                   # [B, K-1, A]
        if has_bg:
            nw = nw.clone()
            nw[:, :, bg] = float("-inf")
        
        logp = torch.log_softmax(nw, dim=-1)                # [B, K-1, A]
        # Cost C[b, i, m] = -logp(slot_{i+1} -> mode_m). Match to minimize total
        # cost over all (K-1)! permutations (K-1 is small).
        modes_exp = modes.unsqueeze(1).expand(-1, K - 1, -1)                 # [B, K-1, K-1]
        cost = -logp.gather(2, modes_exp)                                    # [B, K-1(slot), K-1(mode)]
        perms = self._assignment_perms(K - 1, device)                       # [P, K-1]
        slot_index = torch.arange(K - 1, device=device).unsqueeze(0).expand(perms.size(0), -1)  # [P, K-1]
        perm_cost = cost.detach()[:, slot_index, perms].sum(dim=-1)         # [B, P]
        best = perm_cost.argmin(dim=-1)                                     # [B]
        assigned = modes.gather(1, perms[best])                            # [B, K-1] mode per slot
        chosen_logp = logp.gather(2, assigned.unsqueeze(-1)).squeeze(-1)    # [B, K-1]
        return (-chosen_logp).mean()

    def _winner_aux_loss(self, pred_full: Tensor, target_full: Tensor, winners: Tensor, valid_mask: Tensor) -> Tensor:
        """
        Compute a verb/noun future CE on the slot the action head won.

        pred_full:   [B, K, C]
        target_full: [B, T, C]
        winners:     [num_valid] long, slot index per valid sample.
        valid_mask:  [B] bool.
        """

        if pred_full is None or target_full is None or winners.numel() == 0:
            return pred_full.sum() * 0.0
        valid_pred = pred_full[valid_mask]  # [num_valid, K, C]
        C = valid_pred.size(-1)
        winner_idx = winners.view(-1, 1, 1).expand(-1, 1, C)
        winner_pred = valid_pred.gather(1, winner_idx).squeeze(1)  # [num_valid, C]
        winner_target = target_full[valid_mask, -1]  # [num_valid, C]
        return self.verb_noun_cls(winner_pred, winner_target)

    def _primary_future_logits(self, future_logits: Tensor) -> Tensor:
        if self.multi_query and future_logits.size(1) > 1:
            return future_logits[:, 0]
        return future_logits[:, -1]

    def _query_slot_action_ids(self, future_logits: Tensor) -> Tensor:
        scores = future_logits.detach().float()
        if self.set_metric_background_id is not None and 0 <= self.set_metric_background_id < scores.size(-1):
            scores = scores.clone()
            scores[..., self.set_metric_background_id] = float("-inf")
        return scores.argmax(dim=-1)

    def _prev_action_ids(self, target: Target) -> Tensor:
        """
        GT last-observed action id per sample (model index space), [B] long.

        target.past_actions is [B, T, A] (one-hot / mixup-mixed over the observed
        window); the last step is the most recent observed action, i.e. the
        'previous action' that conditions the transition prior.
        """

        pa = target.past_actions
        last = pa[:, -1] if pa.ndim == 3 else pa
        return last.argmax(dim=-1)

    def _past_action_dist(self, pred: Prediction) -> Tensor:
        """
        Softmax of the model's own past-action head over the last observed step,
        [B, A], background-masked and renormalized. Used for the soft, end-to-end
        ('past_head') transition-fusion path when the previous action is uncertain.
        """

        pa = pred.past_actions
        last = pa[:, -1] if pa.ndim == 3 else pa
        dist = torch.softmax(last.detach().float(), dim=-1)
        bg = self.set_metric_background_id
        if bg is not None and 0 <= bg < dist.size(-1):
            dist = dist.clone()
            dist[:, bg] = 0.0
            dist = dist / dist.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return dist

    def _transition_fuse_kwargs(self, pred: Prediction, target: Target, device) -> dict:
        """
        Pick the previous-action signal for fusion: hard GT-prev row, or the
        soft past-head mixture, per TRANSITION_FUSE_SOURCE.
        """

        if self.transition_fuse_source == "past_head":
            return {"past_dist": self._past_action_dist(pred).to(device)}
        return {"prev_ids": self._prev_action_ids(target).to(device)}
