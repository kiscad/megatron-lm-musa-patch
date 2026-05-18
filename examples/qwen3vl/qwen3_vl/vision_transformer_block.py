# Copyright (c) 2025, BAAI. All rights reserved.
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from contextlib import nullcontext
from dataclasses import dataclass
from typing import List, Optional, Union

import torch
from torch import Tensor, nn

from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core import parallel_state, tensor_parallel
from megatron.core.enums import Fp8Recipe
# from megatron.core.fp4_utils import get_fp4_context
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.utils import (
    WrappedTensor,
    deprecate_inference_params,
    get_pg_rank,
    make_viewless_tensor,
)
from megatron.core.models.vision.multimodal_projector import MultimodalProjector
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
try:
    import transformer_engine.pytorch as te  # pylint: disable=unused-import

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

try:
    import apex  # pylint: disable=unused-import

    HAVE_APEX = True
except ImportError:
    HAVE_APEX = False

get_cpu_offload_context = None
te_checkpoint = None
try:
    from transformer_engine.pytorch.cpu_offload import get_fine_grained_offload_handler
except ImportError:
    get_fine_grained_offload_handler = None

# if HAVE_TE:
try:
    from megatron.core.extensions.transformer_engine import (
        TENorm,
        get_cpu_offload_context,
        te_checkpoint,
    )

    # LayerNormImpl = TENorm
    raise ImportError("not supported")

except ImportError:
    HAVE_TE = False
    try:
        from megatron.core.fusions.fused_layer_norm import FusedLayerNorm

        # LayerNormImpl = FusedLayerNorm
        # print("LayerNormImpl",LayerNormImpl.__module__)
    except ImportError:
        from megatron.core.transformer.torch_norm import WrappedTorchNorm

        LayerNormImpl = WrappedTorchNorm
    from megatron.core.transformer.torch_norm import WrappedTorchNorm
    LayerNormImpl = WrappedTorchNorm


from megatron.core.transformer.enums import LayerType
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import (
    BaseTransformerLayer,
    get_transformer_layer_offset,
)

class VisionTransformerBlock(TransformerBlock):
    """
    The difference between this class and the base TransformerBlock is that:
     1. It adds deepstack merger and norm layers
     2. It returns hidden_states and deepstack features in the forward function
    """
    def __init__(self, config, spec,
                 post_layer_norm = True, pre_process = True, post_process = True,
                 pg_collection = None, vp_stage = None,model_comm_pgs = None,
                 projection_config = None,  # Note: DeepStack
                 projection_layer_spec = None,  # Note: DeepStack
                 projection_type = 'mlp',  # Note: DeepStack):
                ):
        super().__init__(config, spec, post_layer_norm, pre_process, post_process, model_comm_pgs, vp_stage)
        self.submodules = _get_block_submodules(config, spec, vp_stage)
        if self.final_layernorm != None:
            # NOTE(lizhiyu): replace final layernorm with TENorm if using TE
            self.final_layernorm = None
            self.final_layernorm = torch.nn.LayerNorm(normalized_shape=self.config.hidden_size, eps=self.config.layernorm_epsilon,)
            # self.final_layernorm = build_module(
            #         self.submodules.layer_norm,
            #         config=self.config,
            #         hidden_size=self.config.hidden_size,
            #         eps=self.config.layernorm_epsilon,
            #     )
        # NOTE: DeepStack
        
        self.deepstack_visual_indexes = self.config.deepstack_visual_indexes
        self.deepstack_merger_list = nn.ModuleList(
            [
                MultimodalProjector(
                    projection_config,
                    projection_layer_spec,
                    projection_type,
                    projection_config.ffn_hidden_size
                )
                for _ in range(len(self.config.deepstack_visual_indexes))
            ]
        )
        self.deepstack_norm_list = nn.ModuleList(
            [
                build_module(
                    self.submodules.layer_norm,
                    config=self.config,
                    # NOTE: check here for DeepStack
                    hidden_size=projection_config.ffn_hidden_size,
                    eps=self.config.layernorm_epsilon,
                )
                # NOTE(lizhiyu): replace with torch LayerNorm
                # torch.nn.LayerNorm(normalized_shape = projection_config.ffn_hidden_size, eps = self.config.layernorm_epsilon,)
                for _ in range(len(self.config.deepstack_visual_indexes))
            ]
        )

    def _checkpointed_forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        context: Tensor,
        context_mask: Tensor,
        rotary_pos_emb: Tensor,
        attention_bias: Tensor,
        packed_seq_params: PackedSeqParams,
        use_inner_quantization_context: bool,
        deepstack_feature_lists: List[int],  # Note: DeepStack
    ):
        """Forward method with activation checkpointing."""

        def custom(start: int, end: int):
            def custom_forward(
                hidden_states, attention_mask, context, context_mask, rotary_pos_emb
            ):
                for index in range(start, end):
                    layer = self._get_layer(index)

                    # Get appropriate inner quantization context
                    if use_inner_quantization_context:
                        if self.config.fp8:
                            inner_quantization_context = get_fp8_context(
                                self.config, layer.layer_number - 1
                            )
                        # TODO: check if fp4 is supported in this case
                        elif self.config.fp4:
                            inner_quantization_context = get_fp4_context(
                                self.config, layer.layer_number - 1
                            )
                        else:
                            inner_quantization_context = nullcontext()
                    else:
                        inner_quantization_context = nullcontext()

                    with inner_quantization_context:
                        hidden_states, context = layer(
                            hidden_states=hidden_states,
                            attention_mask=attention_mask,
                            context=context,
                            context_mask=context_mask,
                            rotary_pos_emb=rotary_pos_emb,
                            attention_bias=attention_bias,
                            inference_context=None,
                            packed_seq_params=packed_seq_params,
                        )
                return hidden_states, context

            return custom_forward

        def checkpoint_handler(forward_func):
            """Determines whether to use the `te_checkpoint` or `tensor_parallel.checkpoint`"""
            # TODO: check if fp4 is supported in this case
            if self.config.fp8 or self.config.fp4:
                return te_checkpoint(
                    forward_func,
                    self.config.distribute_saved_activations,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    self.pg_collection.tp,
                    hidden_states,
                    attention_mask,
                    context,
                    context_mask,
                    rotary_pos_emb,
                )
            else:
                return tensor_parallel.checkpoint(
                    forward_func,
                    self.config.distribute_saved_activations,
                    hidden_states,
                    attention_mask,
                    context,
                    context_mask,
                    rotary_pos_emb,
                )

        if self.config.recompute_method == 'uniform':
            # Uniformly divide the total number of Transformer layers and checkpoint
            # the input activation of each divided chunk.
            # A method to further reduce memory usage reducing checkpoints.
            layer_idx = 0
            while layer_idx < self.num_layers_per_pipeline_rank:
                hidden_states, context = checkpoint_handler(
                    custom(layer_idx, layer_idx + self.config.recompute_num_layers)
                )

                # Deepstack
                # stack visual embeddings at specified layers
                if layer_idx in self.deepstack_visual_indexes:
                    idx = self.deepstack_visual_indexes.index(layer_idx)
                    merger = self.deepstack_merger_list[idx]
                    deepstack_feature = merger(
                        self.deepstack_norm_list[idx](
                            hidden_states.view(-1, merger.config.ffn_hidden_size)
                        )
                    )
                    deepstack_feature_lists.append(deepstack_feature)

                if (
                    torch.is_grad_enabled()
                    and self.config.cpu_offloading
                    and self.group_prefetch_offload_commit_async is not None
                ):
                    hidden_states = self.group_prefetch_offload_commit_async(hidden_states)

                layer_idx += self.config.recompute_num_layers

        elif self.config.recompute_method == 'block':
            # Checkpoint the input activation of only a set number of individual
            # Transformer layers and skip the rest.
            # A method fully use the device memory removing redundant re-computation.
            recompute_skip_num_layers = 0
            for layer_idx in range(self.num_layers_per_pipeline_rank):
                # Skip recomputation when input grad computation is not needed.
                # Need to have at least one input tensor with gradient computation
                # for re-enterant autograd engine.
                # TODO: check if fp4 is supported in this case
                if (self.config.fp8 or self.config.fp4) and not hidden_states.requires_grad:
                    recompute_skip_num_layers += 1
                if (
                    layer_idx >= recompute_skip_num_layers
                    and layer_idx < self.config.recompute_num_layers + recompute_skip_num_layers
                ):
                    hidden_states, context = checkpoint_handler(custom(layer_idx, layer_idx + 1))
                else:
                    hidden_states, context = custom(layer_idx, layer_idx + 1)(
                        hidden_states, attention_mask, context, context_mask, rotary_pos_emb
                    )
        else:
            raise ValueError("Invalid activation recompute method.")

        return hidden_states

    """Transformer class."""
    def forward(
        self,
        hidden_states: Union[Tensor, WrappedTensor],
        attention_mask: Optional[Tensor],
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
    ):
        """
        Perform the forward pass through the transformer block.

        This method handles the core computation of the transformer, including
        self-attention, optional cross-attention, and feed-forward operations.

        Args:
            hidden_states (Union[Tensor, WrappedTensor]): Input tensor of shape [s, b, h]
                where s is the sequence length, b is the batch size, and h is the hidden size.
                Can be passed as a WrappedTensor during inference to avoid an obsolete
                reference in the calling function.
            attention_mask (Tensor): Boolean tensor of shape [1, 1, s, s] for masking
                self-attention.
            context (Tensor, optional): Context tensor for cross-attention.
            context_mask (Tensor, optional): Mask for cross-attention context
            rotary_pos_emb (Tensor, optional): Rotary positional embeddings.
            attention_bias (Tensor): Bias tensor for Q * K.T of shape in shape broadcastable
                to [b, num_head, sq, skv], e.g. [1, 1, sq, skv].
                Used as an alternative to apply attention mask for TE cuDNN attention.
            inference_context (BaseInferenceContext, optional): Parameters for inference-time
                optimizations.
            packed_seq_params (PackedSeqParams, optional): Parameters for packed sequence
                processing.

        Returns:
            Union[Tensor, Tuple[Tensor, Tensor]]: The output hidden states tensor of shape
            [s, b, h], and optionally the updated context tensor if cross-attention is used.
        """
        deepstack_feature_lists = []  # Note: DeepStack, reset feature list
        inference_context = deprecate_inference_params(inference_context, inference_params)

        # Delete the obsolete reference to the initial input tensor if necessary
        if isinstance(hidden_states, WrappedTensor):
            hidden_states = hidden_states.unwrap()

        if not self.pre_process:
            # See set_input_tensor()
            hidden_states = self.input_tensor

        # Viewless tensor.
        # - We only need to create a viewless tensor in the case of micro batch
        #   size (mbs) == 1, since in this case, 'hidden_states.transpose()'
        #   above creates a view tensor, and '.contiguous()' is a pass-through.
        #   For mbs >= 2, '.contiguous()' creates a new tensor, eliminating
        #   the need to make it viewless.
        #
        #   However, we don't explicitly check mbs == 1 here because
        #   make_viewless_tensor() has negligible overhead when its input
        #   is already viewless.
        #
        # - For the 'else' case above, calling make_viewless_tensor() here is
        #   likely redundant, since p2p_communication.py (likely originator)
        #   already creates viewless tensors. That said, make_viewless_tensor()
        #   is called here to be future-proof and corner-case-proof.
        hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)

        if self.config.sequence_parallel:
            rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()

        # If fp8_recipe is delayed, wrap the entire pass with get_fp8_context(),
        # otherwise do nothing extra at the outer level
        # if we are using other fp8 recipes, then the context manager enter&exit are free
        # we can wrap fp8_context within the for loop over layers, so that we can fine-grained
        # control which layer will be fp8 or bf16
        # For FP4: NVFP4BlockScaling doesn't have delayed scaling, always uses inner context
        if self.config.fp8:
            use_outer_quantization_context = self.config.fp8_recipe == Fp8Recipe.delayed
            use_inner_quantization_context = self.config.fp8_recipe != Fp8Recipe.delayed
            outer_quantization_context = (
                get_fp8_context(self.config) if use_outer_quantization_context else nullcontext()
            )
        # elif self.config.fp4:
        #     use_outer_quantization_context = False
        #     use_inner_quantization_context = True
        #     outer_quantization_context = nullcontext()
        else:
            # No quantization
            use_outer_quantization_context = False
            use_inner_quantization_context = False
            outer_quantization_context = nullcontext()

        if get_fine_grained_offload_handler is not None:
            fine_grained_offload_handler = get_fine_grained_offload_handler()
            fine_grained_offload_handler.num_layers = len(self.layers)
            fine_grained_offload_handler.pp_size = self.config.pipeline_model_parallel_size
            fine_grained_offload_handler.is_pipeline_last_stage = parallel_state.is_pipeline_last_stage()

        with rng_context, outer_quantization_context:
            # Forward pass.
            if self.config.recompute_granularity == 'full' and self.training:
                assert self.config.recompute_method == 'uniform' and self.config.recompute_num_layers == 1, \
                    f"Only uniform recompute with recompute_num_layers=1 is supported for full recompute in Qwen3-VL."
                hidden_states = self._checkpointed_forward(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    context=context,
                    context_mask=context_mask,
                    rotary_pos_emb=rotary_pos_emb,
                    attention_bias=attention_bias,
                    packed_seq_params=packed_seq_params,
                    use_inner_quantization_context=use_inner_quantization_context,
                    deepstack_feature_lists = deepstack_feature_lists,  # Note: DeepStack
                )
            else:
                for l_no, layer in enumerate(self.layers):
                    # Get appropriate inner quantization context
                    if use_inner_quantization_context:
                        if self.config.fp8:
                            inner_quantization_context = get_fp8_context(
                                self.config, layer.layer_number - 1
                            )
                        elif self.config.fp4:
                            inner_quantization_context = get_fp4_context(
                                self.config, layer.layer_number - 1
                            )
                        else:
                            inner_quantization_context = nullcontext()
                    else:
                        inner_quantization_context = nullcontext()

                    with self.offload_context, inner_quantization_context:
                        hidden_states, context = layer(
                            hidden_states=hidden_states,
                            attention_mask=attention_mask,
                            context=context,
                            context_mask=context_mask,
                            rotary_pos_emb=rotary_pos_emb,
                            rotary_pos_cos=rotary_pos_cos,
                            rotary_pos_sin=rotary_pos_sin,
                            attention_bias=attention_bias,
                            inference_context=inference_context,
                            packed_seq_params=packed_seq_params,
                            sequence_len_offset=sequence_len_offset,
                        )

                    # Deepstack
                    # stack visual embeddings at specified layers
                    if l_no in self.deepstack_visual_indexes:
                        idx = self.deepstack_visual_indexes.index(l_no)
                        merger = self.deepstack_merger_list[idx]
                        deepstack_feature = merger(
                            self.deepstack_norm_list[idx](
                                hidden_states.view(-1, merger.config.ffn_hidden_size)
                            )
                        )
                        deepstack_feature_lists.append(deepstack_feature)

                    if (
                        torch.is_grad_enabled()
                        and self.config.cpu_offloading
                        and self.group_prefetch_offload_commit_async is not None
                    ):
                        hidden_states = self.group_prefetch_offload_commit_async(hidden_states)

        # Final layer norm.
        if self.final_layernorm is not None:
            hidden_states = self.final_layernorm(hidden_states)
            # TENorm produces a "viewed" tensor. This will result in schedule.py's
            # deallocate_output_tensor() throwing an error, so a viewless tensor is
            # created to prevent this.
            hidden_states = make_viewless_tensor(
                inp=hidden_states, requires_grad=True, keep_graph=True
            )

        # If this TransformerBlock is empty, input and output hidden states will be the same node
        # on the computational graph and will lead to unexpected errors in pipeline schedules.
        if not self.pre_process and len(self.layers) == 0 and not self.final_layernorm:
            hidden_states = hidden_states.clone()

        return hidden_states, deepstack_feature_lists  # Note: DeepStack

@dataclass
class TransformerBlockSubmodules:
    """
    Dataclass for specifying the submodules of a transformer block.

    This class defines the structure for configuring the layers and normalization
    within a transformer block, allowing for flexible and customizable architecture designs.

    Args:
        layer_specs (List[ModuleSpec], optional): A list of module specifications for
            the layers within the transformer block. Each specification typically
            defines a complete transformer layer (e.g., self-attention, feed-forward network).
        layer_norm (Optional[Union[ModuleSpec, torch.nn.Module]], optional): Specification
            or instance of the layer normalization to be applied.
    """

    layer_specs: List[ModuleSpec] = None
    layer_norm: Optional[Union[ModuleSpec, torch.nn.Module]] = None

def _get_block_submodules(
    config: TransformerConfig,
    spec: Union[TransformerBlockSubmodules, ModuleSpec],
    vp_stage: Optional[int] = None,
) -> TransformerBlockSubmodules:
    """
    Retrieve or construct TransformerBlockSubmodules based on the provided specification.

    Args:
        config (TransformerConfig): Configuration object for the transformer model.
        spec (Union[TransformerBlockSubmodules, ModuleSpec]): Specification for the
            transformer block submodules. Can be either a TransformerBlockSubmodules
            instance or a ModuleSpec.
        vp_stage (Optional[int]): Virtual pipeline stage number.

    Returns:
        TransformerBlockSubmodules: The submodules for the transformer block.
    """

    # Transformer block submodules.
    if isinstance(spec, TransformerBlockSubmodules):
        return spec

    # ModuleSpec here is generally assumed to be for a transformer layer that
    # is implemented in `transformer_layer.py` or if it subclasses
    # `BaseTransformerLayer` from the `transformer_layer.py` file.
    elif isinstance(spec, ModuleSpec):
        if issubclass(spec.module, TransformerBlock):
            return spec.submodules
        elif issubclass(spec.module, BaseTransformerLayer):
            num_layers = get_num_layers_to_build(config, vp_stage)
            return TransformerBlockSubmodules(
                layer_specs=[spec] * num_layers, layer_norm=LayerNormImpl
            )
        else:
            raise Exception(f"specialize for {spec.module.__name__}.")
    else:
        raise Exception(f"specialize for {type(spec).__name__}.")

def get_num_layers_to_build(config: TransformerConfig, vp_stage: Optional[int] = None) -> int:
    """
    Determine the number of transformer layers to build for the current pipeline stage.
    Args:
        config (TransformerConfig): Configuration object containing transformer model parameters.
        vp_stage (Optional[int]): Virtual pipeline stage number.

    Returns:
        int: The number of layers to be built for the current pipeline stage.
    """
    # If we have a custom PP layout, straightforwardly
    # return the number of decoders in the layout array.
    if config.pipeline_model_parallel_layout is not None:
        return config.pipeline_model_parallel_layout.get_num_layers_to_build(
            layer_type=LayerType.decoder, vp_stage=vp_stage
        )

    if (
        config.num_layers_in_first_pipeline_stage is not None
        or config.num_layers_in_last_pipeline_stage is not None
    ):

        assert not (
            config.account_for_embedding_in_pipeline_split
            or config.account_for_loss_in_pipeline_split
        ), " \
        Does not support standalone embedding stage and standalone loss stage with uneven pp"
        # Number of layers to distribute over rest of pipeline stages
        layers_to_distribute = config.num_layers
        # Number of pipeline stages left for distributing transformer layers
        pipeline_stages_left = parallel_state.get_pipeline_model_parallel_world_size()

        # If the uneven first (last) pipeline stage is enabled, remove the specified number
        # of layers to calculate the number of layers on each middle pipeline stage.
        if config.num_layers_in_first_pipeline_stage is not None:
            layers_to_distribute -= config.num_layers_in_first_pipeline_stage
            pipeline_stages_left -= 1

        if config.num_layers_in_last_pipeline_stage is not None:
            layers_to_distribute -= config.num_layers_in_last_pipeline_stage
            pipeline_stages_left -= 1

        # If pp_size <= 2, we do not have any intermediate pipeline stages, and we do not
        # need to check if the left over layers are divisible by the left over stages.
        if pipeline_stages_left > 0:
            assert (
                layers_to_distribute % pipeline_stages_left == 0
            ), "With uneven pipelineing the left over layers must be divisible by left over stages"
            num_layers_per_pipeline_rank = layers_to_distribute // pipeline_stages_left
        else:
            num_layers_per_pipeline_rank = 0

        # If the uneven first (last) pipeline stage is enabled, return the specified number
        # of layers for all virtual pipeline parallel stages within the first (last) pipeline
        # parallel stage.
        if (
            parallel_state.is_pipeline_first_stage(ignore_virtual=True)
            and config.num_layers_in_first_pipeline_stage is not None
        ):
            num_layers_per_pipeline_rank = config.num_layers_in_first_pipeline_stage

        if (
            parallel_state.is_pipeline_last_stage(ignore_virtual=True)
            and config.num_layers_in_last_pipeline_stage is not None
        ):
            num_layers_per_pipeline_rank = config.num_layers_in_last_pipeline_stage
    else:
        # Include the embedding layer and loss layer into pipeline parallelism partition
        num_layers = config.num_layers
        if config.account_for_embedding_in_pipeline_split:
            num_layers += 1

        if config.account_for_loss_in_pipeline_split:
            num_layers += 1

        assert (
            num_layers % config.pipeline_model_parallel_size == 0
        ), "num_layers should be divisible by pipeline_model_parallel_size"
        num_layers_per_pipeline_rank = num_layers // config.pipeline_model_parallel_size

    if (
        parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None
        and config.pipeline_model_parallel_size > 1
    ):
        # Interleaved pipeline parallelism:
        # Number of layers in each model chunk is the number of layers in the stage,
        # divided by the number of model chunks in a stage.
        # With 8 layers, 2 stages, and 4 model chunks, we want an assignment of
        # layers to stages like (each list is a model chunk):
        # Stage 0: [0]  [2]  [4]  [6]
        # Stage 1: [1]  [3]  [5]  [7]
        # With 8 layers, 2 stages, and 2 virtual stages, we want an assignment of
        # layers to stages like (each list is a model chunk):
        # Stage 0: [0, 1]  [4, 5]
        # Stage 1: [2, 3]  [6, 7]
        vp_size = parallel_state.get_virtual_pipeline_model_parallel_world_size()

        assert (
            num_layers_per_pipeline_rank % vp_size == 0
        ), f"num_layers_per_pipeline_rank {num_layers_per_pipeline_rank} \
            should be divisible by vp_size {vp_size}"
        num_layers_per_virtual_stage = num_layers_per_pipeline_rank // vp_size

        num_layers_to_build = num_layers_per_virtual_stage

    else:
        # Non-interleaved pipeline parallelism:
        # Each stage gets a contiguous set of layers.
        num_layers_to_build = num_layers_per_pipeline_rank

    # The embedding (or loss) layer cannot function as a standalone transformer layer
    # Reduce the number of layers to construct by 1 on the first (or last) stage if the
    # embedding (or loss) layer is included in the pipeline parallelism partition and placement.
    if (
        parallel_state.is_pipeline_first_stage(ignore_virtual=False, vp_stage=vp_stage)
        and config.account_for_embedding_in_pipeline_split
    ):
        num_layers_to_build -= 1
        assert num_layers_to_build >= 0, "Not enough layers in the first virtual pipeline stage"

    if (
        parallel_state.is_pipeline_last_stage(ignore_virtual=False, vp_stage=vp_stage)
        and config.account_for_loss_in_pipeline_split
    ):
        num_layers_to_build -= 1
        assert num_layers_to_build >= 0, "Not enough layers in the last virtual pipeline stage"

    return num_layers_to_build
