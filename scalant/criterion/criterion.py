import logging

import torch
from torch import Tensor

from scalant.config import Config
from scalant.utils import accuracy
from scalant.utils.ouput_target_structure import Prediction, Target
from scalant.criterion.build import Criterion_REGISTRY
from scalant.datasets import (EpicKitchens, action2verbnoun, action_set_metrics, build_action_id_to_verb_noun_maps, build_action_similarity_matrix, diverse_action_rerank, topk_action_ids, verbnoun2action)
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

        try:
            self.action_to_verb_id, self.action_to_noun_id = build_action_id_to_verb_noun_maps(dataset=dataset, background_id=self.set_metric_background_id)
            self.action_similarity = build_action_similarity_matrix(dataset=dataset, background_id=self.set_metric_background_id)
            self.set_metrics_enabled = True
        except (AttributeError, KeyError, ValueError):
            self.action_to_verb_id, self.action_to_noun_id, self.action_similarity = None, None, None

    def __call__(self, pred: Prediction, target: Target, is_training=True) -> (Tensor, dict):
        # Always materialize the slot-0 view so accuracy/top-1 stay defined the
        # same way whether or not the set loss is active.
        future_action_pred, future_action_target = self._future_loss_pair(pred.future_actions, target.future_actions)
        notice_index = [i for i in range(target.past_actions.shape[-1]) if i != self.ignore_index]
        past_cls = self.action_cls(pred.past_actions, target.past_actions)

        # Step 7: replace future-action CE with the 'hit-anywhere' set loss when
        # the diverse-set training regime is active and we have K>1 slots.
        use_set_loss = self.set_loss_enabled and pred.future_actions.size(1) > 1
        action_winners, valid_mask = None, None
        if use_set_loss:
            future_set_hit, action_winners, valid_mask = self.set_hit_loss.aggregate(pred.future_actions, target.future_actions)
            future_action_loss = self.hit_weight * future_set_hit
            future_loss_key = "future_set_hit_loss"
            future_loss_value = future_set_hit.item()
        else:
            future_action_loss = self.action_cls(future_action_pred, future_action_target)
            future_loss_key = "future_cls_loss"
            future_loss_value = future_action_loss.item()

        loss = past_cls + future_action_loss

        # Compute metrics
        (past_top1,), past_counts = accuracy(pred.past_actions[..., notice_index], target.past_actions[..., notice_index])
        (future_top1,), future_counts = accuracy(future_action_pred[..., notice_index], future_action_target[..., notice_index])

        # Mean top 5
        mt5r_dict = {"logits": self._primary_future_logits(pred.future_actions)[:, notice_index], "labels": target.future_actions[:, -1, notice_index].argmax(dim=-1)}
        loss_dict = {"past_cls_loss": past_cls.item(), future_loss_key: future_loss_value, "past_top1": [None, past_top1, past_counts], "future_top1": [None, future_top1, future_counts], "mt5r": ["MeanTopKRecallMeter", mt5r_dict, None]}

        if pred.past_verbs is not None:
            past_verb = self.verb_noun_cls(pred.past_verbs, target.past_verbs)
            past_noun = self.verb_noun_cls(pred.past_nouns, target.past_nouns)
            if use_set_loss:
                # Tied WTA: supervise verb/noun on the slot the action head
                # picked as the winner, so each slot stays a coherent action
                # hypothesis instead of slot 0 being anchored as the default.
                future_verb = self._winner_aux_loss(pred.future_verbs, target.future_verbs, action_winners, valid_mask)
                future_noun = self._winner_aux_loss(pred.future_nouns, target.future_nouns, action_winners, valid_mask)
            else:
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
        device = future_logits.device
        action_to_verb_id = self.action_to_verb_id.to(device=device)
        action_to_noun_id = self.action_to_noun_id.to(device=device)
        action_similarity = self.action_similarity.to(device=device)
        topk_sets = torch.stack([topk_action_ids(logits, self.set_metric_k, include_background=False, background_id=self.set_metric_background_id) for logits in future_logits])

        metric_dict = {}
        topk_metrics = action_set_metrics(topk_sets, target.future_actions[:, -1], action_to_verb_id, action_to_noun_id, action_similarity=action_similarity, past_nouns=target.past_nouns, ignore_index=self.ignore_index if self.ignore_index >= 0 else None)
        metric_dict.update(self._format_set_metrics("topk", topk_metrics))

        if self.cfg.MODEL.DIVERSE_SET.ENABLE or self.multi_query:
            if self.multi_query and pred.future_actions.size(1) > 1:
                diverse_sets = self._query_slot_action_ids(pred.future_actions)
            else:
                diverse_sets = torch.stack([
                    diverse_action_rerank(logits, self.set_metric_k, action_similarity=action_similarity, diversity_weight=float(self.cfg.MODEL.DIVERSE_SET.DIVERSITY_WEIGHT), include_background=False, background_id=self.set_metric_background_id) for logits in future_logits])
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

    def _winner_aux_loss(self, pred_full: Tensor, target_full: Tensor, winners: Tensor, valid_mask: Tensor) -> Tensor:
        """Compute a verb/noun future CE on the slot the action head won.

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
