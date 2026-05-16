import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os.path as osp


class MultipCrossEntropyLoss(nn.Module):

    def __init__(self, reduction='mean', ignore_index=-100):
        super(MultipCrossEntropyLoss, self).__init__()

        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, input, target):
        input = input.reshape(-1, input.size(-1))
        target = target.reshape(-1, target.size(-1))

        logsoftmax = nn.LogSoftmax(dim=1).to(input.device)

        if self.ignore_index >= 0:
            notice_index = [
                i for i in range(target.shape[-1]) if i != self.ignore_index]
            output = torch.sum(-target[:, notice_index] * logsoftmax(input[:, notice_index]), dim=1)

            if self.reduction == 'mean':
                return torch.mean(output[target[:, self.ignore_index] != 1])
            elif self.reduction == 'sum':
                return torch.sum(output[target[:, self.ignore_index] != 1])
            else:
                return output[target[:, self.ignore_index] != 1]
        else:
            output = torch.sum(-target * logsoftmax(input), dim=1)

            if self.reduction == 'mean':
                return torch.mean(output)
            elif self.reduction == 'sum':
                return torch.sum(output)
            else:
                return output


class MultipCrossEntropyEqualizedLoss(nn.Module):

    def __init__(self, gamma=0.95, lambda_=1.76e-3, reduction='mean', ignore_index=-100, anno_path='annotations/ek100_rulstm/', freq_info=None):
        super(MultipCrossEntropyEqualizedLoss, self).__init__()

        if freq_info is None:
            # get label distribution
            segment_list = pd.read_csv(osp.join(anno_path, 'training.csv'), names=['id', 'video', 'start_f', 'end_f', 'verb', 'noun', 'action'], skipinitialspace=True)
            freq_info = np.zeros((max(segment_list['action']) + 1,))
            assert ignore_index == 0
            for segment in segment_list.iterrows():
                freq_info[segment[1]['action']] += 1.
        freq_info = freq_info / freq_info.sum()
        self.freq_info = torch.FloatTensor(freq_info)

        self.gamma = gamma
        self.lambda_ = lambda_
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, input, target):
        input = input.reshape(-1, input.size(-1))
        target = target.reshape(-1, target.size(-1))

        logsoftmax = nn.LogSoftmax(dim=1).to(input.device)

        bg_target = target[:, self.ignore_index]
        notice_index = [i for i in range(target.shape[-1]) if i != self.ignore_index]
        input = input[:, notice_index]
        target = target[:, notice_index]

        weight = input.new_zeros(len(notice_index))
        weight[self.freq_info < self.lambda_] = 1.
        weight = weight.view(1, -1).repeat(input.shape[0], 1)

        eql_w = 1 - (torch.rand_like(target) < self.gamma) * weight * (1 - target)
        input = torch.log(eql_w + 1e-8) + input

        output = torch.sum(-target * logsoftmax(input), dim=1)
        if (bg_target != 1).sum().item() == 0:
            return torch.mean(torch.zeros_like(output))
        if self.reduction == 'mean':
            return torch.mean(output[bg_target != 1])
        elif self.reduction == 'sum':
            return torch.sum(output[bg_target != 1])
        else:
            return output[bg_target != 1]


class MultipSetHitLoss(nn.Module):
    """Set-based 'hit-anywhere' loss for multi-query future action prediction.

    Wraps a per-row CE loss (the same one used in single-query mode) and
    aggregates its K per-slot scores with epsilon-relaxed Winner-Takes-All
    (Multiple-Choice Learning, Lee et al. 2016):

        loss_b = (1 - epsilon) * min_k CE(slot_k, y_b) + epsilon * mean_k CE(...)

    The min term sends gradient mainly to the slot that best matches the GT,
    so different slots can specialize on different samples. The mean term is
    optional: epsilon > 0 protects against slot starvation but contaminates
    the "hit anywhere" interpretation. Default epsilon=0 = pure WTA.

    By delegating per-row scoring to the supplied `base_loss`, any class
    weighting in the base loss (e.g. EPIC class-frequency equalization in
    MultipCrossEntropyEqualizedLoss) carries through unchanged. The
    single-query -> set-loss ablation therefore changes ONLY the aggregation,
    not the per-class weighting.
    """

    def __init__(self, base_loss: nn.Module, reduction='mean', epsilon=0.0):
        super().__init__()
        if not hasattr(base_loss, 'reduction') or not hasattr(base_loss, 'ignore_index'):
            raise ValueError("base_loss must expose 'reduction' and 'ignore_index' attributes (e.g. MultipCrossEntropyLoss / MultipCrossEntropyEqualizedLoss).")
        self.base_loss = base_loss
        self.reduction = reduction
        self.epsilon = float(epsilon)

    def forward(self, input, target):
        loss, _, _ = self.aggregate(input, target)
        return loss

    def aggregate(self, input, target):
        """Returns (loss, winners, valid_mask).

        - loss: scalar, the WTA-aggregated set loss.
        - winners: long tensor of shape [num_valid]. winners[i] = slot index
          that minimized the base CE for the i-th valid (non-ignore) sample,
          in the order in which valid samples appear along the batch.
        - valid_mask: bool tensor of shape [B]. True = sample is foreground
          (not ignore-class); False = filtered out.

        The winners are useful for tying auxiliary heads (verb/noun) to the
        same slot the action head picked, so per-slot supervision stays
        coherent and slot-0 is not anchored as a "default winner".
        """
        if input.ndim != 3:
            raise ValueError(f"MultipSetHitLoss expects input shape [B, K, A]; got {tuple(input.shape)}.")
        if target.ndim != 3:
            raise ValueError(f"MultipSetHitLoss expects target shape [B, T, A]; got {tuple(target.shape)}.")

        B, K, A = input.shape
        target_step = target[:, -1:]  # [B, 1, A]
        target_expanded = target_step.expand(-1, K, -1).contiguous()  # [B, K, A]

        ignore_index = int(getattr(self.base_loss, 'ignore_index', -1))
        if 0 <= ignore_index < A:
            valid_mask = target_step[:, 0, ignore_index] != 1  # [B]
        else:
            valid_mask = torch.ones(B, dtype=torch.bool, device=target.device)
        num_valid = int(valid_mask.sum().item())

        if num_valid == 0:
            # Zero scalar that still tracks the graph so optimizer.step() is a no-op.
            zero_loss = input.sum() * 0.0
            empty_winners = torch.zeros(0, dtype=torch.long, device=input.device)
            return zero_loss, empty_winners, valid_mask

        # Score every (sample, slot) with the SAME per-row CE used in the
        # single-query baseline. The base loss handles ignore-index masking
        # and any class equalization internally.
        saved_reduction = self.base_loss.reduction
        self.base_loss.reduction = 'none'
        try:
            per_row = self.base_loss(input, target_expanded)
        finally:
            self.base_loss.reduction = saved_reduction

        expected = num_valid * K
        if per_row.numel() != expected:
            raise RuntimeError(f"Expected base_loss to return {expected} per-row values (num_valid * K), got {per_row.numel()}. Confirm base_loss filters by ignore_index consistently.")

        per_slot_ce = per_row.view(num_valid, K)
        min_ce, winners = per_slot_ce.min(dim=-1)  # winners: [num_valid] long

        if self.epsilon > 0.0:
            mean_ce = per_slot_ce.mean(dim=-1)
            loss = (1.0 - self.epsilon) * min_ce + self.epsilon * mean_ce
        else:
            loss = min_ce

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss, winners.detach(), valid_mask
