# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from functools import partial

import torch

from megatron.core.transformer.moe.router import TopKRouter

from .moe_utils import (
    sequence_load_balancing_loss_func,
    topk_softmax_with_capacity,
)


def init(self, config, model_comm_pgs=None):
    """Initialize the router with current Megatron MoE fields."""

    super(TopKRouter, self).__init__(config=config, model_comm_pgs=model_comm_pgs)
    self.topk = self.config.moe_router_topk
    self.routing_type = self.config.moe_router_load_balancing_type
    self.score_function = getattr(self.config, "moe_router_score_function", None)
    if self.score_function is None:
        self.score_function = "sigmoid" if getattr(self.config, "moe_router_use_sigmoid", False) else "softmax"
    self.input_jitter = None

    self.enable_expert_bias = getattr(self.config, "moe_router_enable_expert_bias", False)
    if self.enable_expert_bias:
        self.register_buffer(
            "local_tokens_per_expert",
            torch.zeros(self.config.num_moe_experts, dtype=torch.float32, device=torch.cuda.current_device()),
            persistent=False,
        )
        self.register_buffer(
            "expert_bias",
            torch.zeros(self.config.num_moe_experts, dtype=torch.float32, device=torch.cuda.current_device()),
        )
    else:
        self.local_tokens_per_expert = None
        self.expert_bias = None


def seq_aux_loss_load_balancing(self, logits: torch.Tensor, bsz: int, seq_length: int):
    """Apply sequence-auxiliary loss-based load balancing to router logits."""

    probs, routing_map, _ = topk_softmax_with_capacity(
        logits,
        self.topk,
        capacity_factor=self.config.moe_expert_capacity_factor,
        pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
        drop_policy=self.config.moe_token_drop_policy,
        use_pre_softmax=self.config.moe_router_pre_softmax,
        moe_router_topk_limited_devices=getattr(self.config, "moe_router_topk_limited_devices", None),
        moe_router_topk_scaling_factor=self.config.moe_router_topk_scaling_factor,
        device_level_capacity=getattr(self.config, "moe_device_level_capacity", False),
        num_node_group=getattr(self.config, "moe_router_num_node_group", None),
        deterministic_mode=self.config.deterministic_mode,
        num_groups=getattr(self.config, "moe_router_num_groups", None),
        group_topk=getattr(self.config, "moe_router_group_topk", None),
        scaling_factor=self.config.moe_router_topk_scaling_factor,
        score_function=self.score_function,
        expert_bias=self.expert_bias,
    )

    if self.training and torch.is_grad_enabled():
        scores, loss_routing_map = self.compute_routing_scores_for_aux_loss(logits)
        aux_loss_func = partial(
            sequence_load_balancing_loss_func,
            probs=scores,
            routing_map=loss_routing_map,
            tokens_per_expert=loss_routing_map.sum(dim=0),
            batch_size=bsz,
            seq_length=seq_length,
            topk=self.topk,
            moe_router_topk_limited_devices=getattr(self.config, "moe_router_topk_limited_devices", None),
            moe_device_level_aux_loss_coeff=getattr(self.config, "moe_device_level_aux_loss_coeff", None),
            moe_comm_aux_loss_coeff=getattr(self.config, "moe_comm_aux_loss_coeff", None),
            moe_complementary_seq_aux_loss=getattr(self.config, "moe_complementary_seq_aux_loss", False),
        )
        probs = self.apply_load_balancing_loss(
            activation=probs, load_balancing_loss_func=aux_loss_func
        )

    return probs, routing_map


import megatron.core.transformer.moe.router

megatron.core.transformer.moe.router.TopKRouter.__init__ = init
megatron.core.transformer.moe.router.TopKRouter.seq_aux_loss_load_balancing = seq_aux_loss_load_balancing
