"""
================================== MoE Router相关算法 ====================================
moe_router_norm_before_softmax: 
    默认关闭
    开启: MOE_ROUTER_NORM_BEFORE_SOFTMAX=1
    可设置缩放系数 (默认为1): i.e., MOE_ROUTER_NORM_SCALE=2
=========================================================================================
"""


import os
import torch
import torch.nn.functional as F
from megatron.core.transformer.moe.router import TopKRouter


moe_router_norm_before_softmax = int(os.getenv('MOE_ROUTER_NORM_BEFORE_SOFTMAX', 0)) == 1
moe_router_norm_scale = float(os.getenv('MOE_ROUTER_NORM_SCALE', 1))


def router_forward_with_normalization(self, input: torch.Tensor):
    """
    Forward pass of the router.

    Args:
        input (torch.Tensor): Input tensor.
    """
    self._maintain_float32_expert_bias()

    # Apply input jitter
    input = self.apply_input_jitter(input)
    logits = self.gating(input)

    # ---- Add normalization before softmax ----
    if moe_router_norm_before_softmax:
        logits = F.layer_norm(
            logits, 
            normalized_shape=(logits.size(-1),), 
            weight=None, bias=None)
        logits.mul_(moe_router_norm_scale)
    # ------------------------------------------

    if self.config.moe_router_force_load_balancing:
        # Apply force load balancing with random logits for benchmark
        logits = apply_random_logits(logits)

    scores, routing_map = self.routing(logits)
    return scores, routing_map


from transformer_engine.musa.pytorch.utils import replace_attr, add_attr
if moe_router_norm_before_softmax and moe_router_norm_scale != 0.:
    from megatron.core.transformer.moe.router import TopKRouter
    replace_attr(TopKRouter, "forward", router_forward_with_normalization)

