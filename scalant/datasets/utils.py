import json
from collections.abc import Mapping

import torch
from torch import Tensor


def load_json(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    return data


def dump_json(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f)


def action2verbnoun(res_action: Tensor, class_mappings: dict[tuple[str, str], Tensor]) -> tuple[Tensor, Tensor]:
    res_verb = torch.matmul(res_action, class_mappings[('verb', 'action')].to(res_action.device))
    res_noun = torch.matmul(res_action, class_mappings[('noun', 'action')].to(res_action.device))
    return res_verb, res_noun


def _get_dataset_mapping(dataset, attr_name):
    return getattr(dataset, attr_name, None) if dataset is not None else None


def _get_dataset_num_actions(dataset):
    if dataset is None or not hasattr(dataset, "num_classes"):
        return None

    num_classes = dataset.num_classes
    if isinstance(num_classes, Mapping):
        return num_classes.get("action")
    return num_classes


def _get_action_class_mapping(class_mappings: Mapping, component: str) -> Tensor:
    key = (component, "action")
    if key not in class_mappings:
        raise KeyError(f"class_mappings must contain {key!r}.")
    return class_mappings[key]


def _component_ids_from_class_mapping(component_in_action: Tensor, background_id: int | None = 0, unknown_id: int = -1, device=None) -> Tensor:
    if component_in_action.ndim != 2:
        raise ValueError("Action class mapping tensors must have shape (num_actions, num_components).")

    if device is not None:
        component_in_action = component_in_action.to(device=device)

    action_to_component = torch.full((component_in_action.shape[0],), unknown_id, dtype=torch.long, device=component_in_action.device)
    has_mapping = component_in_action != 0
    mapped_actions = has_mapping.any(dim=-1)
    if mapped_actions.any():
        action_to_component[mapped_actions] = component_in_action[mapped_actions].argmax(dim=-1).long()

    if background_id is not None and 0 <= background_id < action_to_component.numel() and not mapped_actions[background_id]:
        action_to_component[background_id] = background_id
    return action_to_component


def _num_actions_from_sources(dataset=None, class_mappings: Mapping | None = None, verb_noun_to_action: Mapping[tuple[int, int], int] | None = None, num_actions: int | None = None, background_id: int | None = 0) -> int:
    if num_actions is not None:
        resolved_num_actions = int(num_actions)
    elif class_mappings is not None and ("verb", "action") in class_mappings:
        resolved_num_actions = int(class_mappings[("verb", "action")].shape[0])
    else:
        dataset_num_actions = _get_dataset_num_actions(dataset)
        if dataset_num_actions is not None:
            resolved_num_actions = int(dataset_num_actions)
        elif verb_noun_to_action:
            resolved_num_actions = max(int(action_id) for action_id in verb_noun_to_action.values()) + 1
        else:
            resolved_num_actions = 0

    if background_id is not None:
        resolved_num_actions = max(resolved_num_actions, background_id + 1)
    return resolved_num_actions


def build_action_id_to_verb_noun_maps(dataset=None, class_mappings: Mapping[tuple[str, str], Tensor] | None = None, verb_noun_to_action: Mapping[tuple[int, int], int] | None = None, num_actions: int | None = None, background_id: int | None = 0, unknown_id: int = -1, device=None) -> tuple[Tensor, Tensor]:
    """
    Build action_id -> verb_id and action_id -> noun_id lookup tensors.

    The helper accepts a dataset with EPIC-style metadata, or the raw mappings
    directly. `class_mappings` is preferred when available because it already
    stores action-to-component relationships.

    `background_id=0` matches the current EPIC dataset path, where labels are
    shifted and class 0 is background. For datasets without a background class,
    pass `background_id=None`. If a raw mapping explicitly assigns action 0 to a
    real verb/noun pair, that explicit mapping is preserved.
    """
    if class_mappings is None:
        class_mappings = _get_dataset_mapping(dataset, "class_mappings")
    if verb_noun_to_action is None:
        verb_noun_to_action = _get_dataset_mapping(dataset, "verb_noun_to_action")

    if class_mappings is not None:
        action_to_verb = _component_ids_from_class_mapping(_get_action_class_mapping(class_mappings, "verb"), background_id=background_id, unknown_id=unknown_id, device=device)
        action_to_noun = _component_ids_from_class_mapping( _get_action_class_mapping(class_mappings, "noun"), background_id=background_id, unknown_id=unknown_id, device=action_to_verb.device)
        return action_to_verb, action_to_noun

    if verb_noun_to_action is None:
        raise ValueError("Provide dataset.class_mappings, dataset.verb_noun_to_action, or mappings directly.")

    num_actions = _num_actions_from_sources(dataset=dataset, verb_noun_to_action=verb_noun_to_action, num_actions=num_actions, background_id=background_id)
    tensor_kwargs = {"dtype": torch.long}
    if device is not None:
        tensor_kwargs["device"] = device

    action_to_verb = torch.full((num_actions,), unknown_id, **tensor_kwargs)
    action_to_noun = torch.full((num_actions,), unknown_id, **tensor_kwargs)

    for (verb_id, noun_id), action_id in verb_noun_to_action.items():
        action_id = int(action_id)
        if 0 <= action_id < num_actions:
            action_to_verb[action_id] = int(verb_id)
            action_to_noun[action_id] = int(noun_id)

    has_background_mapping = (background_id is not None and any(int(action_id) == background_id for action_id in verb_noun_to_action.values()))
    if background_id is not None and 0 <= background_id < num_actions and not has_background_mapping:
        action_to_verb[background_id] = background_id
        action_to_noun[background_id] = background_id

    return action_to_verb, action_to_noun


def build_action_id_to_verb_id_map(*args, **kwargs) -> Tensor:
    action_to_verb, _ = build_action_id_to_verb_noun_maps(*args, **kwargs)
    return action_to_verb


def build_action_id_to_noun_id_map(*args, **kwargs) -> Tensor:
    _, action_to_noun = build_action_id_to_verb_noun_maps(*args, **kwargs)
    return action_to_noun


def action_id_to_verb_id(action_id: int, action_to_verb_id: Tensor | None = None, background_id: int | None = 0, unknown_id: int = -1, **kwargs) -> int:
    if action_to_verb_id is None:
        action_to_verb_id = build_action_id_to_verb_id_map(background_id=background_id, unknown_id=unknown_id, **kwargs)
    action_id = int(action_id)
    if action_id < 0 or action_id >= action_to_verb_id.numel():
        return unknown_id
    return int(action_to_verb_id[action_id].item())


def action_id_to_noun_id(action_id: int, action_to_noun_id: Tensor | None = None, background_id: int | None = 0, unknown_id: int = -1, **kwargs) -> int:
    if action_to_noun_id is None:
        action_to_noun_id = build_action_id_to_noun_id_map(background_id=background_id, unknown_id=unknown_id, **kwargs)
    action_id = int(action_id)
    if action_id < 0 or action_id >= action_to_noun_id.numel():
        return unknown_id
    return int(action_to_noun_id[action_id].item())


def build_action_similarity_matrix(dataset=None, class_mappings: Mapping[tuple[str, str], Tensor] | None = None, verb_noun_to_action: Mapping[tuple[int, int], int] | None = None, num_actions: int | None = None, same_action_similarity: float = 1.0, same_component_similarity: float = 0.5, unrelated_similarity: float = 0.0, background_id: int | None = 0, unknown_id: int = -1, device=None, dtype=torch.float) -> Tensor:
    """
    Build a dense action similarity matrix from action/verb/noun metadata.

    Diagonal entries receive `same_action_similarity`, action pairs sharing the
    same verb or noun receive `same_component_similarity`, and all other pairs
    receive `unrelated_similarity`. `background_id` is passed through to
    `build_action_id_to_verb_noun_maps`; use `background_id=None` for datasets
    that do not reserve an action background class.
    """
    action_to_verb, action_to_noun = build_action_id_to_verb_noun_maps(dataset=dataset, class_mappings=class_mappings, verb_noun_to_action=verb_noun_to_action, num_actions=num_actions, background_id=background_id, unknown_id=unknown_id, device=device)

    num_actions = action_to_verb.numel()
    similarity = torch.full((num_actions, num_actions), unrelated_similarity, dtype=dtype, device=action_to_verb.device)

    valid_verbs = action_to_verb != unknown_id
    valid_nouns = action_to_noun != unknown_id
    same_verb = (action_to_verb[:, None] == action_to_verb[None, :]) & valid_verbs[:, None] & valid_verbs[None, :]
    same_noun = (action_to_noun[:, None] == action_to_noun[None, :]) & valid_nouns[:, None] & valid_nouns[None, :]
    similarity[same_verb | same_noun] = same_component_similarity

    diagonal = torch.eye(num_actions, dtype=torch.bool, device=similarity.device)
    similarity[diagonal] = same_action_similarity
    return similarity


def _candidate_action_ids(num_actions: int, device, include_background: bool, background_id: int | None = 0) -> Tensor:
    candidate_ids = torch.arange(num_actions, device=device, dtype=torch.long)
    if not include_background and background_id is not None and 0 <= background_id < num_actions:
        candidate_ids = candidate_ids[candidate_ids != background_id]
    return candidate_ids


def topk_action_ids(action_logits: Tensor, k: int, include_background: bool = False, background_id: int | None = 0) -> Tensor:
    """
    Return normal top-k action ids from a single action-logit vector.
    """
    if action_logits.ndim != 1:
        raise ValueError("topk_action_ids expects a single action-logit vector with shape (num_actions,).")

    candidate_ids = _candidate_action_ids(action_logits.numel(), action_logits.device, include_background, background_id)
    k = min(int(k), candidate_ids.numel())
    if k <= 0:
        return candidate_ids[:0]

    candidate_scores = action_logits.detach().float()[candidate_ids]
    topk_indices = candidate_scores.topk(k).indices
    return candidate_ids[topk_indices]


def diverse_action_rerank(action_logits: Tensor, k: int, action_similarity: Tensor | None = None, dataset=None, class_mappings: Mapping[tuple[str, str], Tensor] | None = None, verb_noun_to_action: Mapping[tuple[int, int], int] | None = None, diversity_weight: float = 1.0, include_background: bool = False, background_id: int | None = 0,) -> Tensor:
    """
    Greedily select diverse action ids from a single action-logit vector.

    At each step, candidates are scored as:
        model_score - diversity_weight * max_similarity_to_selected

    When `action_similarity` is not supplied, it is built from the metadata
    helper using the provided dataset or mappings.
    """
    if action_logits.ndim != 1:
        raise ValueError("diverse_action_rerank expects a single action-logit vector with shape (num_actions,).")

    k = int(k)
    if k <= 0:
        return torch.empty((0,), dtype=torch.long, device=action_logits.device)
    if float(diversity_weight) == 0:
        return topk_action_ids(action_logits, k, include_background=include_background, background_id=background_id)

    scores = action_logits.detach().float()
    num_actions = scores.numel()
    remaining_ids = _candidate_action_ids(num_actions, scores.device, include_background, background_id)
    if remaining_ids.numel() == 0:
        return remaining_ids

    if action_similarity is None:
        action_similarity = build_action_similarity_matrix(dataset=dataset, class_mappings=class_mappings, verb_noun_to_action=verb_noun_to_action, num_actions=num_actions, background_id=background_id, device=scores.device, dtype=scores.dtype)
    else:
        action_similarity = action_similarity.to(device=scores.device, dtype=scores.dtype)

    expected_shape = (num_actions, num_actions)
    if tuple(action_similarity.shape) != expected_shape:
        raise ValueError(f"action_similarity must have shape {expected_shape}, got {tuple(action_similarity.shape)}.")

    selected = []
    for _ in range(min(k, remaining_ids.numel())):
        candidate_scores = scores[remaining_ids]
        if selected:
            selected_ids = torch.stack(selected).long()
            diversity_penalty = action_similarity[remaining_ids][:, selected_ids].max(dim=-1).values
            candidate_scores = candidate_scores - float(diversity_weight) * diversity_penalty

        best_pos = candidate_scores.argmax()
        selected.append(remaining_ids[best_pos])

        keep = torch.ones(remaining_ids.numel(), dtype=torch.bool, device=remaining_ids.device)
        keep[best_pos] = False
        remaining_ids = remaining_ids[keep]

    return torch.stack(selected).long()


def _lookup_action_ids(action_ids: Tensor, lookup: Tensor, unknown_id: int = -1) -> Tensor:
    lookup = lookup.to(device=action_ids.device)
    values = torch.full_like(action_ids, unknown_id)
    valid = (action_ids >= 0) & (action_ids < lookup.numel())
    if valid.any():
        values[valid] = lookup[action_ids[valid]]
    return values


def _target_action_ids(target_actions: Tensor) -> Tensor:
    if target_actions.ndim >= 2 and target_actions.shape[-1] > 1:
        return target_actions[..., :].argmax(dim=-1).flatten(0, -1)
    return target_actions.long().flatten()


def _observed_class_mask(labels: Tensor, num_classes: int, ignore_index: int | None = 0) -> Tensor:
    labels = labels.detach()
    if labels.ndim >= 2 and labels.shape[-1] == num_classes:
        observed = labels > 0
        if observed.ndim > 2:
            observed = observed.flatten(1, -2).any(dim=1)
    else:
        label_ids = labels.long()
        observed = torch.zeros((label_ids.shape[0], num_classes), dtype=torch.bool, device=labels.device)
        valid = (label_ids >= 0) & (label_ids < num_classes)
        if ignore_index is not None:
            valid &= label_ids != ignore_index
        if valid.any():
            batch_ids = torch.arange(label_ids.shape[0], device=labels.device).view(-1, *([1] * (label_ids.ndim - 1))).expand_as(label_ids)
            observed[batch_ids[valid], label_ids[valid]] = True

    if ignore_index is not None and 0 <= ignore_index < num_classes:
        observed[:, ignore_index] = False
    return observed


@torch.inference_mode()
def action_set_metrics(predicted_action_sets: Tensor, target_actions: Tensor, action_to_verb_id: Tensor, action_to_noun_id: Tensor, action_similarity: Tensor | None = None, past_nouns: Tensor | None = None, ignore_index: int | None = 0, unknown_id: int = -1) -> dict[str, tuple[float, int]]:
    """
    Compute set metrics for predicted next-action sets.

    Returns metric values as percentages except `set_similarity`.
    Each value is paired with the number of valid ground-truth samples.
    """
    if predicted_action_sets.ndim != 2:
        raise ValueError("predicted_action_sets must have shape (batch, set_size).")
    if predicted_action_sets.shape[0] == 0:
        return {}

    predicted_action_sets = predicted_action_sets.long()
    target_ids = _target_action_ids(target_actions)
    if target_ids.numel() != predicted_action_sets.shape[0]:
        target_ids = target_ids.reshape(predicted_action_sets.shape[0], -1)[:, -1]

    valid_samples = torch.ones_like(target_ids, dtype=torch.bool)
    if ignore_index is not None:
        valid_samples &= target_ids != ignore_index
    if not valid_samples.any():
        return {}

    predicted_action_sets = predicted_action_sets[valid_samples]
    target_ids = target_ids[valid_samples]
    batch_size, set_size = predicted_action_sets.shape

    set_recall = (predicted_action_sets == target_ids[:, None]).any(dim=-1).float().mean() * 100.0

    exact_duplicate_rates = []
    for action_set in predicted_action_sets:
        exact_duplicate_rates.append(0.0 if set_size == 0 else 1.0 - (action_set.unique().numel() / set_size))
    exact_duplicate_rate = torch.tensor(exact_duplicate_rates, device=predicted_action_sets.device).mean() * 100.0

    metrics = {"set_recall": (float(set_recall.item()), batch_size), "exact_duplicate": (float(exact_duplicate_rate.item()), batch_size)}

    if set_size < 2:
        metrics.update({"verb_duplicate": (0.0, batch_size), "noun_duplicate": (0.0, batch_size), "set_similarity": (0.0, batch_size)})
    else:
        pair_i, pair_j = torch.triu_indices(set_size, set_size, offset=1, device=predicted_action_sets.device)
        pair_count = pair_i.numel()

        action_to_verb_id = action_to_verb_id.to(device=predicted_action_sets.device)
        action_to_noun_id = action_to_noun_id.to(device=predicted_action_sets.device)
        set_verbs = _lookup_action_ids(predicted_action_sets, action_to_verb_id, unknown_id=unknown_id)
        set_nouns = _lookup_action_ids(predicted_action_sets, action_to_noun_id, unknown_id=unknown_id)

        verb_pair_valid = (set_verbs[:, pair_i] != unknown_id) & (set_verbs[:, pair_j] != unknown_id)
        noun_pair_valid = (set_nouns[:, pair_i] != unknown_id) & (set_nouns[:, pair_j] != unknown_id)
        same_verb = (set_verbs[:, pair_i] == set_verbs[:, pair_j]) & verb_pair_valid
        same_noun = (set_nouns[:, pair_i] == set_nouns[:, pair_j]) & noun_pair_valid

        metrics.update({"verb_duplicate": (float((same_verb.float().sum(dim=-1) / pair_count).mean().item() * 100.0), batch_size), "noun_duplicate": (float((same_noun.float().sum(dim=-1) / pair_count).mean().item() * 100.0), batch_size)})

        if action_similarity is not None:
            action_similarity = action_similarity.to(device=predicted_action_sets.device)
            pair_similarity = action_similarity[predicted_action_sets[:, pair_i], predicted_action_sets[:, pair_j]]
            metrics["set_similarity"] = (float(pair_similarity.mean().item()), batch_size)
        else:
            metrics["set_similarity"] = (0.0, batch_size)

    if past_nouns is not None:
        set_nouns = _lookup_action_ids(predicted_action_sets, action_to_noun_id, unknown_id=unknown_id)
        num_noun_classes = past_nouns.shape[-1] if past_nouns.ndim >= 2 else int(action_to_noun_id.max().item()) + 1
        observed_nouns = _observed_class_mask(past_nouns[valid_samples], num_noun_classes, ignore_index=ignore_index)
        valid_nouns = (set_nouns != unknown_id) & (set_nouns >= 0) & (set_nouns < num_noun_classes)
        if ignore_index is not None:
            valid_nouns &= set_nouns != ignore_index

        plausible = torch.zeros_like(valid_nouns, dtype=torch.bool)
        if valid_nouns.any():
            plausible[valid_nouns] = observed_nouns.gather(1, set_nouns.clamp(min=0, max=num_noun_classes - 1))[valid_nouns]
        per_sample_valid = valid_nouns.sum(dim=-1).clamp(min=1)
        object_match = (plausible.float().sum(dim=-1) / per_sample_valid).mean() * 100.0
        metrics["object_match"] = (float(object_match.item()), batch_size)

    return metrics


def verbnoun2action(res_verb: Tensor, res_noun: Tensor, verb_noun_to_action: dict[tuple[int, int], int]) -> Tensor:
    verb_ids, noun_ids = zip(*verb_noun_to_action.keys())

    # Convert to tensors
    verb_ids = torch.tensor(verb_ids, device=res_verb.device)
    noun_ids = torch.tensor(noun_ids, device=res_noun.device)

    # Index into the verb and noun probabilities
    verb_action_probs = res_verb[..., verb_ids]
    noun_action_probs = res_noun[..., noun_ids]

    # Calculate action probabilities
    res_action = verb_action_probs * noun_action_probs  # element-wise multiplication

    return res_action
