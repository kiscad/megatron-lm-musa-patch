
import math
from typing import Optional

import torch

from megatron.core import parallel_state
import megatron.core.transformer.moe.moe_utils
get_capacity = megatron.core.transformer.moe.moe_utils.get_capacity
device_limited_topk = getattr(megatron.core.transformer.moe.moe_utils, "device_limited_topk", None)
group_limited_topk = megatron.core.transformer.moe.moe_utils.group_limited_topk


def node_limited_topk(
    scores: torch.Tensor,
    topk: int,
    num_tokens: int,
    num_experts: int,
    moe_router_topk_limited_devices: int,
    num_node_group: int=None,
):
    """Perform top-k routing on a subset of expert parallel ranks.

    Selects N ranks for each token, then conducts top-k selection among experts on these node.
    See DeepSeek-V3 technical report for details.

    Args:
        scores (torch.Tensor): Softmax scores from the router.
        topk (int): The number of experts to select for each token.
        num_tokens (int): The number of tokens.
        num_experts (int): The number of experts.
        moe_router_topk_limited_devices (int): Number of expert parallel ranks to consider for
            each token during routing. None means no device limitation.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Probs and indices tensor.
    """

    # Organize the experts into groups
    if num_node_group is None:
        ep_size = (
            parallel_state.get_expert_model_parallel_world_size()
        )  # num_node_group equals to expert parallel size/8
        assert ep_size % 8 == 0, f"ep_size should be multiple of 8, but get {ep_size}"
        num_node_group = ep_size // 8
    node_k = topk // moe_router_topk_limited_devices #each token select node according to the sum of the highest K/M affinity scores
    group_scores = (
                scores.view(num_tokens, num_node_group, -1).topk(node_k, dim=-1)[0].sum(dim = -1)
            )  # [n, n_group]
    group_idx = torch.topk(
                group_scores, k=moe_router_topk_limited_devices, dim=-1, sorted=False
            )[
                1
            ]  # [n, moe_router_topk_limited_devices]
    group_mask = torch.zeros_like(group_scores)  # [n, n_group]
    group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_tokens, num_node_group, num_experts // num_node_group)
        .reshape(num_tokens, -1)
    )  # [n, e]
    masked_scores = scores.masked_fill(~score_mask.bool(), 0.0)  # [n, e]
    _, top_indices = torch.topk(masked_scores, k=topk, dim=-1)
    return top_indices


def sequence_load_balancing_loss_func(
    probs: torch.Tensor,
    routing_map: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    batch_size: int,
    seq_length: int,
    topk: int,
    moe_aux_loss_coeff: float,
    moe_device_level_aux_loss_coeff: float=None,
    moe_comm_aux_loss_coeff: float=None,
    moe_router_topk_limited_devices: float=None,
    moe_complementary_seq_aux_loss: bool=False,
    sequence_partition_group=None,
):
    """
    Calculate the auxiliary loss in sequence-level by computing the loss for each individual sample.
    Refer to the DeepSeek-V2 huggingface repo
    (https://huggingface.co/deepseek-ai/DeepSeek-V2) for details.
    """
    num_sub_sequence = 1

    # If the sequence is partitioned by certain parallelism strategies like Sequence Parallelism
    # or Context Parallelism, compute the gradient of the auxiliary loss with respect to the full
    # sequence.
    if sequence_partition_group is not None:
        # We can keep `aggregated_probs_per_expert` local since we don't need the gradient for
        # `tokens_per_expert`, saving one allreduce operation for `aggregated_probs_per_expert`.
        num_sub_sequence = torch.distributed.get_world_size(sequence_partition_group)
        torch.distributed.all_reduce(tokens_per_expert, group=sequence_partition_group)

    assert num_sub_sequence == 1, "Do not support sequence aux loss in sequence partition case"

    num_experts = probs.shape[1]

    probs_for_aux_loss = probs.view(seq_length, batch_size, -1)
    cost_coeff = routing_map.view(seq_length, batch_size, -1).sum(dim=0).float()
    cost_coeff.div_(seq_length * topk / num_experts)
    if moe_complementary_seq_aux_loss:
        assert (
            (moe_device_level_aux_loss_coeff is None) and 
            (moe_comm_aux_loss_coeff is None)
            ), "moe_complementary_seq_aux_loss only used in deepseekV3, which means no other aux loss used"
        probs_for_aux_loss = probs.view(seq_length, batch_size, -1)
        sum_value = probs_for_aux_loss.sum(dim=-1, keepdim=True)
        probs_for_aux_loss = probs_for_aux_loss / (sum_value + 1e-20)
    seq_aux_loss = (cost_coeff * probs_for_aux_loss.mean(dim=0)).sum(dim=1).mean()
    seq_aux_loss *= moe_aux_loss_coeff

    if moe_device_level_aux_loss_coeff:
        num_group = (
        parallel_state.get_expert_model_parallel_world_size()
        )  # num_group equals to expert parallel size
        device_aux_loss = (cost_coeff.view(batch_size, num_group, -1).mean(dim=2) * 
                           probs_for_aux_loss.mean(dim=0).view(batch_size, num_group, -1).sum(dim=2)).sum(dim=1).mean()
        device_aux_loss *= moe_device_level_aux_loss_coeff
        seq_aux_loss += device_aux_loss
    if moe_comm_aux_loss_coeff:
        num_group = (
        parallel_state.get_expert_model_parallel_world_size()
        )  # num_group equals to expert parallel size
        cost_coeff = routing_map.view(seq_length, batch_size, num_group, -1).any(dim=3).sum(dim=0).float()
        cost_coeff.div_(seq_length *  moe_router_topk_limited_devices/ num_group)
        comm_aux_loss = (cost_coeff * 
                           probs_for_aux_loss.mean(dim=0).view(batch_size, num_group, -1).sum(dim=2)).sum(dim=1).mean()
        comm_aux_loss *= moe_comm_aux_loss_coeff
        seq_aux_loss += comm_aux_loss
        
    return seq_aux_loss

def topk_softmax_with_capacity(
    logits: torch.Tensor,
    topk: int,
    capacity_factor: Optional[float] = None,
    pad_to_capacity: bool = False,
    drop_policy: str = "probs",
    use_pre_softmax: bool = False,
    moe_router_topk_limited_devices: int = None,
    moe_router_topk_scaling_factor: float = None,
    device_level_capacity: Optional[bool] = False,
    use_sigmoid: bool = False,
    norm_topk_prob: bool = False,
    num_node_group: int = None,
    e_score_correction_bias: torch.Tensor = None,
    deterministic_mode: bool = False,
    num_groups: Optional[int] = None,
    group_topk: Optional[int] = None,
    scaling_factor: Optional[float] = None,
    score_function: str = "softmax",
    expert_bias: Optional[torch.Tensor] = None,
):
    """Apply top-k routing with compatibility for old and current Megatron args."""

    assert logits.dim() == 2, f"Expected 2D logits [num_tokens, num_experts], got {logits.dim()}."
    num_tokens, num_experts = logits.shape

    if scaling_factor is None:
        scaling_factor = moe_router_topk_scaling_factor
    if expert_bias is None:
        expert_bias = e_score_correction_bias
    if use_sigmoid:
        score_function = "sigmoid"

    def compute_topk(scores: torch.Tensor):
        if group_topk:
            return group_limited_topk(
                scores=scores,
                topk=topk,
                num_tokens=num_tokens,
                num_experts=num_experts,
                num_groups=num_groups,
                group_topk=group_topk,
            )
        if moe_router_topk_limited_devices:
            if num_node_group:
                top_indices = node_limited_topk(
                    scores,
                    topk,
                    num_tokens,
                    num_experts,
                    moe_router_topk_limited_devices,
                    num_node_group,
                )
                return scores.gather(1, top_indices), top_indices
            if device_limited_topk is not None:
                return device_limited_topk(
                    scores, topk, num_tokens, num_experts, moe_router_topk_limited_devices
                )
            return group_limited_topk(
                scores=scores,
                topk=topk,
                num_tokens=num_tokens,
                num_experts=num_experts,
                num_groups=parallel_state.get_expert_model_parallel_world_size(),
                group_topk=moe_router_topk_limited_devices,
            )
        return torch.topk(scores, k=topk, dim=1)

    if score_function == "softmax":
        if use_pre_softmax:
            scores = torch.softmax(logits, dim=-1, dtype=torch.float32).type_as(logits)
            probs, top_indices = compute_topk(scores)
        else:
            scores, top_indices = compute_topk(logits)
            probs = torch.softmax(scores, dim=-1, dtype=torch.float32).type_as(logits)
    elif score_function == "sigmoid":
        scores = torch.sigmoid(logits.float()).type_as(logits)
        if expert_bias is not None:
            _, top_indices = compute_topk(scores + expert_bias)
            probs = scores.gather(1, top_indices)
        else:
            probs, top_indices = compute_topk(scores)
        if norm_topk_prob or topk > 1:
            probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-20)
    else:
        raise ValueError(f"Invalid score_function: {score_function}")

    if scaling_factor:
        probs = probs * scaling_factor

    topk_masked_gates = torch.zeros_like(logits).scatter(1, top_indices, probs)
    topk_map = torch.zeros_like(logits).int().scatter(1, top_indices, 1).bool()
    tokens_per_expert = topk_map.sum(dim=0)

    if capacity_factor is None:
        return topk_masked_gates, topk_map, tokens_per_expert
    elif device_level_capacity:
        assert drop_policy == "probs", f"only support 'probs' for device_level capacity, but get {drop_policy}"
        num_group = parallel_state.get_expert_model_parallel_world_size()
        device_expert_capacity = get_capacity(
            num_tokens=num_tokens * topk, num_experts=num_experts, capacity_factor=capacity_factor
        ) * num_experts // num_group
        topk_masked_group_gates = topk_masked_gates.view(num_tokens, num_group, -1)
        topk_masked_group_gates = topk_masked_group_gates.permute(0, 2, 1).reshape(-1, num_group)
        _, capacity_indices = torch.topk(
            topk_masked_group_gates, k=device_expert_capacity, dim=0, sorted=False
        )
        capacity_mask = torch.zeros(
            [num_tokens * num_experts // num_group, num_group], device=logits.device
        ).scatter(0, capacity_indices, 1).bool()
        capacity_mask = capacity_mask.view(num_tokens, num_experts // num_group, num_group)
        capacity_mask = capacity_mask.permute(0, 2, 1).reshape(num_tokens, -1)
        if pad_to_capacity:
            final_map = capacity_mask
            final_probs = topk_masked_gates * final_map
        else:
            final_map = torch.logical_and(topk_map, capacity_mask)
            final_probs = topk_masked_gates * final_map
        return final_probs, final_map, tokens_per_expert
    else:
        expert_capacity = get_capacity(
            num_tokens=num_tokens * topk, num_experts=num_experts, capacity_factor=capacity_factor
        )
        if drop_policy == "probs":
            _, capacity_indices = torch.topk(
                topk_masked_gates, k=expert_capacity, dim=0, sorted=False
            )
            capacity_mask = torch.zeros_like(logits).scatter(0, capacity_indices, 1).bool()
        elif drop_policy == "position":
            _, capacity_indices = torch.topk(topk_map.int(), k=expert_capacity, dim=0, sorted=False)
            capacity_mask = torch.zeros_like(logits).scatter(0, capacity_indices, 1).bool()
        else:
            raise ValueError(f"Invalid drop_policy: {drop_policy}")

        if pad_to_capacity:
            final_map = capacity_mask
            final_probs = topk_masked_gates * final_map
        else:
            final_map = torch.logical_and(topk_map, capacity_mask)
            final_probs = topk_masked_gates * final_map
        return final_probs, final_map, tokens_per_expert


def reduce_aux_losses_tracker_across_ranks(track_names: Optional[list] = None):
    """Reduce MoE aux-loss trackers without hanging on dense-only pipeline stages.

    Current Megatron reduces tracker values across the pipeline-parallel group, but
    a pipeline stage without MoE layers may have an empty local tracker. All ranks
    in a pipeline group must still enter the same collectives, so create zero
    placeholders for missing tracker names before reducing.
    """
    moe_utils = megatron.core.transformer.moe.moe_utils
    tracker = moe_utils.get_moe_layer_wise_logging_tracker()
    pp_group = parallel_state.get_pipeline_model_parallel_group()

    if track_names is None:
        local_meta = {name: tuple(entry["values"].shape) for name, entry in tracker.items()}
        pp_world_size = torch.distributed.get_world_size(pp_group)
        gathered_meta = [None for _ in range(pp_world_size)]
        torch.distributed.all_gather_object(gathered_meta, local_meta, group=pp_group)

        merged_meta = {}
        for meta in gathered_meta:
            if not meta:
                continue
            merged_meta.update(meta)
        track_names = list(merged_meta.keys())
    else:
        track_names = list(track_names)
        merged_meta = {name: tuple(tracker[name]["values"].shape) for name in track_names if name in tracker}

    for name in track_names:
        if name not in tracker:
            shape = merged_meta.get(name)
            if shape is None:
                continue
            tracker[name] = {
                "values": torch.zeros(shape, device=torch.cuda.current_device()),
                "reduce_group": None,
                "avg_group": None,
            }

        values = tracker[name]["values"]
        torch.distributed.all_reduce(values, group=pp_group)
        if tracker[name].get("reduce_group") is not None:
            torch.distributed.all_reduce(values, group=tracker[name].get("reduce_group"))
        if tracker[name].get("avg_group") is not None:
            torch.distributed.all_reduce(
                values, group=tracker[name]["avg_group"], op=torch.distributed.ReduceOp.AVG
            )


megatron.core.transformer.moe.moe_utils.reduce_aux_losses_tracker_across_ranks = reduce_aux_losses_tracker_across_ranks
megatron.core.transformer.moe.moe_utils.sequence_load_balancing_loss_func = sequence_load_balancing_loss_func
megatron.core.transformer.moe.moe_utils.topk_softmax_with_capacity = topk_softmax_with_capacity