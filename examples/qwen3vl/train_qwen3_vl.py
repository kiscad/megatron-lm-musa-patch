# Mainly Adopted from https://github.com/alibaba/Pai-Megatron-Patch/blob/8949a6647cbf6b39837ad3dd911fa4aa0726895b/examples/qwen2_5_vl/pretrain_qwen.py.Below is the original copyright:
# Copyright (c) 2024 Alibaba PAI and Nvidia Megatron-LM Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import sys
import logging
from functools import partial
from copy import deepcopy
from typing import List, Optional, Tuple, Union

import torch
import torch._dynamo

from argparse import Namespace

# # For pytorch 2.6
# torch.serialization.add_safe_globals([Namespace])
sys.path.append("/mnt/seed17/001688/haoran.huang/megatron-lm-musa-patch_1014")

if os.getenv("ACCELERATOR_BACKEND", "musa") == "musa":
    import musa_patch
else:
    pass
import musa_patch
from megatron.core import parallel_state
from megatron.training.checkpointing import get_checkpoint_name # for dataloder
from megatron.core.enums import ModelType


from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.transformer.spec_utils import import_module
from megatron.core.utils import StragglerDetector

from megatron.training.utils import unwrap_model
from megatron.training import get_args, get_timers, get_tokenizer, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args

from megatron.training.yaml_arguments import core_transformer_config_from_yaml

try:
    from megatron.post_training.arguments import add_modelopt_args, modelopt_args_enabled
    from megatron.post_training.loss_func import loss_func as loss_func_modelopt
    from megatron.post_training.model_provider import model_provider as model_provider_modelopt

    has_nvidia_modelopt = True
except ImportError:
    has_nvidia_modelopt = False

from train import pretrain
from megatron.training import get_args, get_timers, print_rank_0
stimer = StragglerDetector()

#### especially for qwen2.5-vl ####
from megatron.core.num_microbatches_calculator import get_num_microbatches
torch._dynamo.config.suppress_errors = True
from megatron.core.parallel_state import get_tensor_model_parallel_rank, get_pipeline_model_parallel_world_size, get_pipeline_model_parallel_rank
from megatron.energon import (
    LimitDataset,
    RepeatDataset,
    WorkerConfig,
    get_loader,
    get_savable_loader,
    get_train_dataset,
    get_val_datasets,
)

from megatron.training.tokenizer.tokenizer import build_tokenizer
from megatron.training.global_vars import get_tokenizer

from qwen3_vl.tensor_parallel import broadcast_data

from qwen3_vl.layer_specs import (get_gpt_layer_with_transformer_engine_spec,
                                                         get_qwen3vl_vision_model_spec,
                                                         get_mlp_module_spec)
from qwen3_vl.model import Qwen3VLModel
from qwen3_vl.transformer_config import (
    Qwen3VLTransformerConfig,
    get_vision_model_config,
    get_vision_projection_config
)


from tools.datasets.qwenvl.data.dataset_helpers import TaskEncoder, print_error_handler
#### especially for qwen2.5-vl ####
IGNORE_IDX=-100
def model_provider(
    pre_process=True, post_process=True, add_encoder=True, add_decoder=True
) -> Union[Qwen3VLModel]:
    args = get_args()
    print_rank_0("start building qwen3-vl model ...")

    # Config of vit, llm and projector
    config = core_transformer_config_from_args(args, Qwen3VLTransformerConfig)
    use_te = args.transformer_impl == "transformer_engine"
    if not use_te:
        raise NotImplementedError("The Qwen3-VL model is only implemented with TransformerEngine!")

    if args.rotary_seq_len_interpolation_factor is not None or args.rotary_seq_len_interpolation_factor != 1:
        print_rank_0('Multimodal RoPE currently not support RoPE interpolation, set to None...')
        args.rotary_seq_len_interpolation_factor = None

    vision_config = get_vision_model_config(args, deepcopy(config))
    vision_config.pipeline_model_parallel_size = 1
    vision_config.first_pipeline_num_layers = None
    vision_projector_config = get_vision_projection_config(deepcopy(config), vision_config.hidden_size, args.spatial_merge_size)

    print_rank_0("building Qwen3-VL model in TE...")
    # Layer Specs of vit, llm and projector
    transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=args.num_experts,
        moe_grouped_gemm=args.moe_grouped_gemm,
        qk_layernorm=args.qk_layernorm,
        normalization=args.normalization,
    )
    vision_model_spec = get_qwen3vl_vision_model_spec()
    vision_projector_spec = get_mlp_module_spec(add_norm=False).submodules
    if args.enable_variable_seq_lengths:
        config.variable_seq_lengths = True

    model = Qwen3VLModel(
        language_transformer_config=config,
        language_transformer_layer_spec=transformer_layer_spec,
        language_vocab_size=args.padded_vocab_size,
        language_max_sequence_length=args.max_position_embeddings,

        vision_transformer_config=vision_config,
        vision_transformer_layer_spec=vision_model_spec,
        # drop_vision_class_token=False, # no use

        vision_projection_config=vision_projector_config,
        vision_projection_layer_spec=vision_projector_spec,
        vision_projection_type='mlp',
        # allow_missing_vision_projection_checkpoint= args.allow_missing_vision_projection_checkpoint,

        language_position_embedding_type=args.position_embedding_type,
        language_rotary_percent=args.rotary_percent,
        language_rotary_base=args.rotary_base,

        pre_process=pre_process,
        post_process=post_process,
        add_decoder=add_decoder,
        add_encoder=add_encoder,

        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        language_share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
    )

    model.freeze(
        freeze_language_model=args.freeze_LM,
        freeze_vision_model=args.freeze_ViT,
        freeze_vision_projection=False
    )

    # def forward_output(name):
    #     def forward_hook(module, input, output):
    #         print(f"Inside {module.__class__.__name__} forward hook")
    #         print(f"Input: {input}")  # 假设输入是个张量
    #         print(f"Output: {output}")
    #         if len(input) > 0 and input[0] != None :
    #             try:
    #                 print(input, len(input))
    #                 print("is_inf1:", torch.isinf(input[0]).any(), "is_nan1:",torch.isnan(input[0]).any())
    #                 print("max input", input[0].abs().max().item())
    #                 print("is_inf1:", torch.isinf(output[0]).any(), "is_nan1:",torch.isnan(output[0]).any())
    #                 print("max ouput",output[0].abs().max().item())
    #             except:
    #                 pass
    #         try:
    #             print("weight:", module.weight)
    #         except:
    #             pass
    #         index = 0
            
    #     return forward_hook

    # def backward_output(name):
    #    def print_backward_hook(module, grad_input, grad_output):
    #        #torch.set_printoptions(profile='full')
    #         print(module.__class__, 'backward ends output', name)
    #         print(f"Input: {grad_output}")  # 假设输入是个张量
    #         print(f"Output: {grad_input}")
    #         if len(grad_output) > 0 and grad_output[0] != None :
    #             for idx, output in enumerate(grad_output):
    #                 if output is not None and (torch.isinf(output).any() or torch.isnan(output).any()):
    #                     global_rank = torch.distributed.get_rank()
    #                     # print(module.__class__, 'backward ends', name, len(grad_output), len(grad_input))
    #                     print("output is_inf1:", torch.isinf(output).any(), "is_nan1:",torch.isnan(output).any(), output,'global_rank', global_rank)
    #         if len(grad_input) > 0 and grad_input[0] !=None:
    #             for idx, input in enumerate(grad_input):
    #                 # print(module.__class__, 'backward ends input', name, grad_input)
    #                 if input is not None and (torch.isinf(input).any() or torch.isnan(input).any()):
    #                     global_rank = torch.distributed.get_rank()
    #                     # print(module.__class__, 'backward ends input', name, len(grad_output), len(grad_input))
    #                     print("input is_inf2:", torch.isinf(input).any(), "is_nan2:",torch.isnan(input).any(),input, 'global_rank', global_rank, "idx:", idx)
    #                     if global_rank == 0:
    #                         torch.save(grad_input, f'global-{global_rank}.{name}.nan1.grad_input.pt')
    #                         torch.save(grad_output, f'global-{global_rank}.{name}.nan1.grad_output.pt')
    #                         try:
    #                             torch.save(module.weight.cpu(), f'global-{global_rank}.{name}.nan1.weight.pt')
    #                         except:
    #                             pass
    #                         exit()
            
    #         try:
    #             print("weight:", module.weight, "is_inf_w:", torch.isinf(module.weight).any())
    #         except:
    #             pass
    #                 #     for idx1, output in enumerate(grad_output):
    #                 #         torch.save(input.cpu(), f'global-{global_rank}.{name}.{idx1}.nan1.grad_output.pt')
    #                 #     for idx2, input in enumerate(grad_input):
    #                 #         torch.save(input.cpu(), f'global-{global_rank}.{name}.{idx2}.nan2.grad_input.pt')
    #    return print_backward_hook
    # # print(model)
    # for name, module in model.named_modules():
    #     # print(name, model_module)
    #     # module.register_forward_pre_hook(print_pre_forward_hook)
    #     # module.register_forward_hook(print_forward_hook)
    #     # module.register_forward_hook(forward_output(name))
    #     module.register_full_backward_hook(backward_output(name))
    def compare_tensor(a, b):
        def check_nan_inf(tensor):
            return f"nan={torch.isnan(tensor).any().item()}, inf={torch.isinf(tensor).any().item()}, max={torch.max(tensor).item()}, min={torch.min(tensor).item()}, shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}"

        def calc_diff(x: torch.Tensor, y: torch.Tensor):
            x, y = x.double(), y.double()
            denominator = (x * x + y * y).sum()
            sim = 2 * (x * y).sum() / denominator
            return 1 - sim

        def get_error(t0, t1):
            t0 = t0.float()
            t1 = t1.float()
            return (abs(t0-t1)).sum()

        def check_error(t0:torch.Tensor, t1:torch.Tensor):
            # if t0.dtype != torch.bfloat16:
            #     t0 = t0.to(torch.bfloat16)
            # if t1.dtype != torch.bfloat16:
            #     t1 = t1.to(torch.bfloat16)
            all_close = torch.allclose(t0, t1, atol= 2e-2, rtol=2e-2)
            # all_close = 0.0
            cosine_error = calc_diff(t0, t1)
            # ANSI颜色代码
            RED = '\033[91m'
            RESET = '\033[0m'
            TAG = RED if not all_close else ''
            return f"{TAG}abs_error={get_error(t0, t1):.6f}, dist_error={torch.dist(t0, t1):.6f}, all_close={all_close}, cosine_error={cosine_error:.6f}{RESET}", all_close
        return check_error(a, b)
    
    def check_fa_fwd_percision(q:torch.Tensor, k:torch.Tensor, v:torch.Tensor, is_vision_module=False):
        return
        import math
        from flash_attn.flash_attn_interface import flash_attn_func, flash_attn_varlen_func
        from transformer_engine.pytorch.attention import  UnfusedDotProductAttention
        q = q.detach().clone()
        k = k.detach().clone()
        v = v.detach().clone()
        print(f"-------- Online Forward Test -----------\n q.shape={q.shape}, k.shape={k.shape}, v.shape={v.shape}")
        if is_vision_module:
            t, h, d = q.shape
            IS_CAUSAL = False
            WINDOW_SIZE = (-1, -1)
            softmax_scale = 1.0 / math.sqrt(d)
            unfused_attention = UnfusedDotProductAttention(softmax_scale)
            fa_out = flash_attn_varlen_func(
                    q = q,
                    k = k,
                    v = v,
                    softmax_scale=softmax_scale,
                    cu_seqlens_q = None,
                    cu_seqlens_k = None,
                    max_seqlen_q = None,
                    max_seqlen_k = None,
                    causal = IS_CAUSAL,
                    window_size=WINDOW_SIZE,
                    ).view(1, t, -1)
            math_out:torch.Tensor = unfused_attention.forward(query_layer=q.unsqueeze(0), 
                                key_layer=k.unsqueeze(0),
                                value_layer=v.unsqueeze(0),
                                window_size=WINDOW_SIZE,
                                attn_mask_type = "no_mask",
                                qkv_layout="bshd_bshd_bshd") # math sdp forward
        else:
            b, s, h, d = q.shape
            IS_CAUSAL = True
            WINDOW_SIZE = (-1, 0)
            softmax_scale = 1.0 / math.sqrt(d)
            unfused_attention = UnfusedDotProductAttention(softmax_scale)
            fa_out = flash_attn_func(
                    q = q,
                    k = k,
                    v = v,
                    dropout_p=0.0,
                    softmax_scale=softmax_scale,
                    causal=IS_CAUSAL,
                    window_size=WINDOW_SIZE,
                    alibi_slopes=None,
                    deterministic=True
                    ).view(b, s, -1)
            math_out:torch.Tensor = unfused_attention.forward(query_layer=q, 
                                key_layer=k,
                                value_layer=v,
                                window_size=WINDOW_SIZE,
                                attn_mask_type='causal',
                                qkv_layout="bshd_bshd_bshd") # math sdp forward
        desc, all_close = compare_tensor(fa_out, math_out)
        if not all_close:
            r = torch.distributed.get_rank()
            tag = "vit_" if is_vision_module else "llm_" 
            torch.save(q, f"./attn_dump/rank{r}_{tag}q.pt")
            torch.save(k, f"./attn_dump/rank{r}_{tag}k.pt")
            torch.save(v, f"./attn_dump/rank{r}_{tag}v.pt")
            exit()
        print(f"----- compare FlashAttention output(is_vision_module = {is_vision_module}), {desc}")
        print(f"-------- Online Forward Test End -----------")
    def check_fa_bwd_percision(q:torch.Tensor, k:torch.Tensor, v:torch.Tensor, do:torch.Tensor, is_vision_module=False):
        return 
        import math
        from flash_attn.flash_attn_interface import flash_attn_func, flash_attn_varlen_func
        from transformer_engine.pytorch.attention import  UnfusedDotProductAttention
        q = q.detach().clone().requires_grad_(True)
        k = k.detach().clone().requires_grad_(True)
        v = v.detach().clone().requires_grad_(True)
        print(f"-------- Online Backward Test -----------\n q.shape={q.shape}, k.shape={k.shape}, v.shape={v.shape}")
        if is_vision_module:
            t, h, d = q.shape
            IS_CAUSAL = False
            WINDOW_SIZE = (-1, -1)
            softmax_scale = 1.0 / math.sqrt(d)
            unfused_attention = UnfusedDotProductAttention(softmax_scale)
            fa_out = flash_attn_varlen_func(
                    q = q,
                    k = k,
                    v = v,
                    softmax_scale=softmax_scale,
                    cu_seqlens_q = None,
                    cu_seqlens_k = None,
                    max_seqlen_q = None,
                    max_seqlen_k = None,
                    causal = IS_CAUSAL,
                    window_size=WINDOW_SIZE,
                    ).view(t, 1, -1).transpose(0, 1)
            fa_out.backward(do)
            fa_q_grad = q.grad
            fa_k_grad = k.grad
            fa_v_grad = v.grad
            q.grad, k.grad, v.grad = None, None, None
            
            q = q.detach().clone().requires_grad_(True)
            k = k.detach().clone().requires_grad_(True)
            v = v.detach().clone().requires_grad_(True)
            math_out:torch.Tensor = unfused_attention.forward(query_layer=q.unsqueeze(0), 
                                key_layer=k.unsqueeze(0),
                                value_layer=v.unsqueeze(0),
                                window_size=WINDOW_SIZE,
                                attn_mask_type = "no_mask",
                                qkv_layout="bshd_bshd_bshd").transpose(0, 1) # math sdp forward
            math_out.backward(do)
            m_q_grad = q.grad
            m_k_grad = k.grad
            m_v_grad = v.grad
            q.grad, k.grad, v.grad = None, None, None
        else:
            b, s, h, d = q.shape
            IS_CAUSAL = True
            WINDOW_SIZE = (-1, 0)
            softmax_scale = 1.0 / math.sqrt(d)
            unfused_attention = UnfusedDotProductAttention(softmax_scale)
            fa_out = flash_attn_func(
                    q = q,
                    k = k,
                    v = v,
                    dropout_p=0.0,
                    softmax_scale=softmax_scale,
                    causal=IS_CAUSAL,
                    window_size=WINDOW_SIZE,
                    alibi_slopes=None,
                    deterministic=True
                    ).view(b, s, -1).transpose(0, 1)
            fa_q_grad = q.grad
            fa_k_grad = k.grad
            fa_v_grad = v.grad
            q.grad, k.grad, v.grad = None, None, None
            
            q = q.detach().clone().requires_grad_(True)
            k = k.detach().clone().requires_grad_(True)
            v = v.detach().clone().requires_grad_(True)
            math_out:torch.Tensor = unfused_attention.forward(query_layer=q, 
                                key_layer=k,
                                value_layer=v,
                                window_size=WINDOW_SIZE,
                                attn_mask_type='causal',
                                qkv_layout="bshd_bshd_bshd").transpose(0, 1) # math sdp forward
        
            math_out.backward(do)
            m_q_grad = q.grad
            m_k_grad = k.grad
            m_v_grad = v.grad
            q.grad, k.grad, v.grad = None, None, None
        
        
        
        print(f"----- compare FlashAttention output(is_vision_module = {is_vision_module}), {compare_tensor(fa_out, math_out)}")
        print(f"----- compare FlashAttention q_grad, {compare_tensor(fa_q_grad, m_q_grad)}")
        print(f"----- compare FlashAttention k_grad, {compare_tensor(fa_k_grad, m_k_grad)}")
        print(f"----- compare FlashAttention v_grad, {compare_tensor(fa_v_grad, m_v_grad)}")
        
        print(f"-------- Online Backward Test End -----------")
        
    def get_tag(isinf, isnan):
        if isinf:
            return "?????????????????ERROR: INF"
        if isnan:
            return "!!!!!!!!!!!!!!!!!ERROR: NAN"
        return "NORMAL"
    
    def check_and_dump_tensor(tensor):
        dtype = tensor.dtype
        if dtype in [torch.long, torch.int, torch.bool]:
            tensor = tensor.float()
        isinf = torch.isinf(tensor).any()
        isnan = torch.isnan(tensor).any()
        # desc = f"data_status={get_tag(isinf, isnan)}, shape={tensor.shape}, dtype={dtype}, 10 elements={tensor.flatten()[:10].detach().clone()}"
        mean_value = torch.mean(tensor).item()
        max_value = torch.max(tensor).item()
        min_value = torch.min(tensor).item()
        std_value = torch.std(tensor).item()
        desc = f"mean={mean_value}, max={max_value}, min={min_value}, std={std_value}, first-5={tensor.flatten()[:5]}, data_status={get_tag(isinf, isnan)}, shape={tensor.shape}, dtype={dtype}"

        return desc, isinf or isnan
    
    def forward_output_sherry(name):
        def forward_hook(module, input, output):
            TRAIN_ITERATION = model.train_iteration
            hook_root = f"./hook_logs_sherry/iteration{TRAIN_ITERATION}"
            os.makedirs(hook_root, exist_ok=True)
            with  open(f'{hook_root}/hook-rank{torch.distributed.get_rank()}.log', 'a', encoding='utf-8') as f:
                print(f"[iteration {TRAIN_ITERATION}] ===== Inside {module.__class__.__name__}/{name} forward hook", file=f)
                for i, inp in enumerate(input):
                    if isinstance(inp, torch.Tensor):
                        desc, invalid_status = check_and_dump_tensor(inp)
                        print(f"[iteration {TRAIN_ITERATION}] input {i}: {desc}, invalid_status = {invalid_status}", file=f)
                    else:
                        print(f"[iteration {TRAIN_ITERATION}] input {i}: {inp}", file=f)
                # if isinstance(output, (tuple, list, torch.Tensor)):
                #     for i, out in enumerate(output):
                #         if isinstance(out, torch.Tensor):
                #             desc = check_and_dump_tensor(out)
                #             print(f"[iteration {TRAIN_ITERATION}] output {i}: {desc}", file=f)
                #         else:
                #             print(f"[iteration {TRAIN_ITERATION}] output {i}: {out}", file=f)
                if isinstance(module, FlashAttention):
                    check_fa_fwd_percision(module.query_layer, module.key_layer, module.value_layer, 
                                           is_vision_module= True if 'vision_model' in name else False)
        return forward_hook
    
    from transformer_engine.pytorch.attention import FlashAttention
    def backward_output_sherry(name):
        def backward_hook(module, grad_input, grad_output):
            TRAIN_ITERATION = model.train_iteration
            hook_root = f"./hook_logs_sherry/iteration{TRAIN_ITERATION}"
            os.makedirs(hook_root, exist_ok=True)
            with  open(f'{hook_root}/hook-rank{torch.distributed.get_rank()}.log', 'a', encoding='utf-8') as f:
                print(f"[iteration {TRAIN_ITERATION}] ===== Inside {module.__class__.__name__}/{name} backward hook", file=f)
                # check gradinput
                invalid_status = False
                for i, inp in enumerate(grad_input):
                    if isinstance(inp, torch.Tensor):
                        desc, status = check_and_dump_tensor(inp)
                        invalid_status = invalid_status or status
                        print(f"[iteration {TRAIN_ITERATION}] grad_input {i}: {desc}, invalid_status = {invalid_status}", file=f)
                    else:
                        print(f"[iteration {TRAIN_ITERATION}] grad_input {i}: {inp}", file=f)

                # check gradoutput
                for i, out in enumerate(grad_output):
                    if isinstance(out, torch.Tensor):
                        desc, status = check_and_dump_tensor(out)
                        invalid_status = invalid_status or status
                        print(f"[iteration {TRAIN_ITERATION}] grad_output {i}: {desc}, invalid_status = {invalid_status}", file=f)
                    else:
                        print(f"[iteration {TRAIN_ITERATION}] grad_output {i}: {out}", file=f)
                
                if isinstance(module, FlashAttention):
                    check_fa_bwd_percision(module.query_layer, module.key_layer, module.value_layer, 
                                           do = grad_output[0], 
                                           is_vision_module= True if 'vision_model' in name else False)
                if invalid_status and isinstance(module, FlashAttention):
                    rank = torch.distributed.get_rank()
                    print(f"[iteration {TRAIN_ITERATION}] FlashAttention module: {name} has invalid grad_output", file=f)
                    torch.save(module.query_layer, f"{hook_root}/rank{rank}.{name}.query_layer.pt")
                    torch.save(module.key_layer, f"{hook_root}/rank{rank}.{name}.key_layer.pt")
                    torch.save(module.value_layer, f"{hook_root}/rank{rank}.{name}.value_layer.pt")
                    torch.save(module.output, f"{hook_root}/rank{rank}.{name}.output.pt")

                    for i, inp in enumerate(grad_input):
                        if isinstance(inp, torch.Tensor):
                            print(f"Start save grad_input {i}")
                            torch.save(inp, f"{hook_root}/rank{rank}.{name}.grad_input{i}.pt")

                    for i, out in enumerate(grad_output):
                        if isinstance(out, torch.Tensor):
                            print(f"Start save grad_output {i}")
                            torch.save(out, f"{hook_root}/rank{rank}.{name}.grad_output{i}.pt")
                    print(f"End save {name}!!!")
                    exit()
                
                            
        return backward_hook

    # print(model)
    # setattr(model, "train_iteration", 0)
    # for name, module in model.named_modules():
    #     module:torch.nn.Module  = module
    #     if len(list(module.children())) == 0:
    #         print(f"[sherry flag info] add {name} hook!!!!")
    #         module.register_forward_hook(forward_output_sherry(name))
    #         module.register_full_backward_hook(backward_output_sherry(name))

    return model

def get_ltor_masks_and_position_ids(
        input_ids,
        image_thw_grids,
        video_thw_grids,
        target,
        pad_token,
        second_per_grid_ts,
        ignore_index=None,
        model: Qwen3VLModel = None
    ):
    """Build masks and position id for left to right model."""
    # Position ids. [3 X bs X seqlen]
    position_ids, _ = model.get_rope_index(
        input_ids=input_ids,
        image_grid_thw=image_thw_grids,
        video_grid_thw=video_thw_grids,
        attention_mask=input_ids != pad_token
    )

    # Loss mask.
    loss_mask = torch.ones(target.size(), dtype=torch.float, device=input_ids.device)
    loss_mask[target == pad_token] = 0.0  # mask paddings
    if ignore_index is not None:
        loss_mask[target == ignore_index] = 0.0  # mask prompts

    # Attention mask.
    attention_mask = None

    return attention_mask, loss_mask, position_ids

def get_batch(data_iterator, model: Qwen3VLModel = None) -> Tuple:
    """Generate a batch"""
    imgs = None
    tokens = None
    labels = None
    loss_mask = None
    attention_mask = None
    position_ids = None

    # Broadcast data.
    torch.cuda.nvtx.range_push("get_data")
    if data_iterator is not None and get_tensor_model_parallel_rank() == 0:
        data = next(data_iterator)
        # pad_token_id = get_tokenizer().pad_token_id
        pad_token_id = IGNORE_IDX
        # print(data["imgs"].shape[0])
        # while (data["target"] == pad_token_id).all() or (data["target"].shape[-1] < 986 or data["target"].shape[-1] > 1000): # for debug
        while (data["target"] == pad_token_id).all():
            logging.getLogger(__name__).warning("The current data is invalid because the target is all pad_token_id! Get next data to avoid fail, but it's better to check the data!")
            data = next(data_iterator)
    else:
        data = None


    data_text =  broadcast_data(["text"], data, torch.int64)["text"]

    target =  broadcast_data(["target"], data, torch.int64)["target"]
    # shape: num_tiles x c x h x w
    imgs = broadcast_data(["imgs"], data, torch.float32)["imgs"]

    # shape: num_tiles x c x h x w
    videos = broadcast_data(["videos"], data, torch.float32)["videos"]

    # shape: n_image_samples
    image_thw_grids = broadcast_data(["image_thw_grids"], data, torch.long)["image_thw_grids"]

    args = get_args()
    if data_text.shape[-1] == args.max_padding_length and get_pipeline_model_parallel_rank() == 0:
        torch.cuda.empty_cache()
    # shape: n_video_samples
    video_thw_grids = broadcast_data(["video_thw_grids"], data, torch.long)["video_thw_grids"]
    # shape: n_video_samples
    second_per_grid_ts = broadcast_data(['second_per_grid_ts'], data, torch.float32)['second_per_grid_ts']


    image_input_mask = broadcast_data(["image_input_mask"], data, torch.bool)["image_input_mask"]
    video_input_mask = broadcast_data(["video_input_mask"], data, torch.bool)["video_input_mask"]
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("index tokens")
    tokenizer = get_tokenizer()

    tokens = data_text.long().contiguous()
    labels = target.contiguous()

    assert tokens.shape == labels.shape, f"tokens: {tokens.shape} != labels: {labels.shape}"
    torch.cuda.nvtx.range_pop()

    # NOTE: no sequence packing in LLM inputs
    torch.cuda.nvtx.range_push("get_ltor_masks_and_position_ids")
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens, image_thw_grids, video_thw_grids, labels, pad_token=tokenizer.pad_token_id, second_per_grid_ts=second_per_grid_ts, ignore_index=IGNORE_IDX, model=model,
    )
    torch.cuda.nvtx.range_pop()

    return (
        tokens,
        labels,
        loss_mask,
        attention_mask,
        position_ids,
        imgs,
        videos,
        image_thw_grids,
        video_thw_grids,
        image_input_mask,
        video_input_mask
    )

# define spiky loss as a loss that's 10x the max loss observed
SPIKY_LOSS_FACTOR = 10


def loss_func(
    loss_mask: torch.Tensor, output_tensor: torch.Tensor, model: Optional[Qwen3VLModel] = None
):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses
        model (Qwen3VLModel, optional): The model (can be wrapped)

    Returns:
        the loss scalar for this micro-batch
        the number of non-padded tokens in this microbatch
        a dict containing reporting metrics on the loss and number of tokens across
            the data parallel ranks
    """
    args = get_args()

    if has_nvidia_modelopt and modelopt_args_enabled(args):  # [ModelOpt]
        return loss_func_modelopt(loss_mask, output_tensor, model=model)

    losses = output_tensor.view(-1).float()
    loss_mask = loss_mask.view(-1).float()
    # print(losses, loss_mask)
    loss = torch.sum(losses * loss_mask)
    # loss = torch.sum(losses)

    # Check individual rank losses are not NaN prior to DP all-reduce.
    rerun_state_machine = get_rerun_state_machine()
    if args.check_for_nan_in_loss_and_grad:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isnan,
            message="found NaN in local forward loss calculation",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=True,
        )
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isinf,
            message="found Inf in local forward loss calculation",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=True,
        )
    # Check for spiky loss
    if args.check_for_spiky_loss:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=partial(
                rerun_state_machine.is_unexpectedly_large,
                threshold=SPIKY_LOSS_FACTOR,
                context="loss",
            ),
            message="Spiky loss",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=False,
        )

    num_tokens = loss_mask.sum().clone().detach().to(torch.int)
    reporting_loss = torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])

    return (loss, num_tokens, {'lm loss': reporting_loss})


def forward_step(data_iterator, model: Qwen3VLModel):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
    """
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    global stimer
    with stimer(bdata=True):
        (
            tokens,
            labels,
            loss_mask,
            attention_mask,
            position_ids,
            imgs,
            videos,
            image_thw_grids,
            video_thw_grids,
            image_input_mask,
            video_input_mask
        ) = get_batch(data_iterator, model=unwrap_model(model))
    timers('batch-generator').stop()
    vision_data = torch.cat([imgs, videos], dim=0)
    vision_grid = torch.cat([image_thw_grids, video_thw_grids], dim=0)
    # print(vision_data, vision_grid)
    with stimer:
        # print("vision_data", vision_data.shape)
        output_tensor = model(
            input_ids = tokens,
            position_ids = position_ids,
            vision_data = vision_data,
            vision_grid_thw =  vision_grid,
            video_start_index = image_input_mask.sum().cpu().item(),
            image_input_mask = image_input_mask,
            video_input_mask = video_input_mask,
            attention_mask = attention_mask,
            labels = labels
        )

    return output_tensor, partial(loss_func, loss_mask, model=model)

def run_online_eval(model):
    """Run an evaluation benchmark during training."""
    # Do nothing.
    return []

def write_online_eval_to_tensorboard(data, iteration, writer):
    """Write online evaluation data to Tensorboard."""
    if not writer:
        return

    for item in data:
        for k, v in item.items():
            writer.add_scalar(k, v, iteration)

def datasets_provider(worker_config=None):
    """Create multimodal train, validation and test datasets."""
    args = get_args()
    dname = args.data_path[0] if type(args.data_path) is list else args.data_path
    train_dataset = get_train_dataset(
        dname,
        batch_size=args.micro_batch_size,
        task_encoder=TaskEncoder(),
        worker_config=worker_config,
        virtual_epoch_length=0,
        max_samples_per_sequence=args.max_samples_per_sequence, # sequential shuffle in a tar
        shuffle_buffer_size=args.shuffle_buffer_size, # shuffle in a sequential
        handler=print_error_handler,
        repeat=True,
        image_decode="pil",
    )
    val_datasets_without_source_datasets = None
    if args.eval_iters > 0:
        val_datasets = get_val_datasets(
            dname,
            batch_size=args.micro_batch_size,
            # This is the total number over all workers
            # limit=args.eval_iters * get_num_microbatches(),
            task_encoder=TaskEncoder(),
            worker_config=worker_config,
            handler=print_error_handler,
            image_decode="pil",
        )
        val_datasets_without_source_datasets = [
            # Limit the dataset to eval_iters * num_microbatches
            LimitDataset(
                # Repeat the inner dataset in case it's too short
                RepeatDataset(val_ds, worker_config=worker_config),
                length=args.eval_iters * get_num_microbatches(),
                worker_config=worker_config,
                reset_after_epoch=True,
            )
            for val_ds, _src_ds in val_datasets
        ]

    return train_dataset, val_datasets_without_source_datasets, None

def is_first_or_last_stage(pp_size, transformer_pipeline_model_parallel_size):
    """Check if the current pipeline parallel stage is the first or last stage."""
    if pp_size == 1:    # No pipeline parallelism.
        return True

    is_valid_rank = False
    pp_rank = get_pipeline_model_parallel_rank()
    if transformer_pipeline_model_parallel_size == 0:
        # No separate pipeline stage for the vision model. Run the dataloader on the first and last pipeline stage.
        is_valid_rank = pp_rank in (0, pp_size-1)
    elif transformer_pipeline_model_parallel_size == 1:
        # Separate pipeline stage for the vision model. Run the dataloader on the first vision and LM stage and last LM stage.
        is_valid_rank = pp_rank in (0, 1, pp_size-1)
    else:
        raise NotImplementedError("encoder-pipeline-model-parallel-size > 1 is not supported yet")

    return is_valid_rank

def is_dataloader_rank(transformer_pipeline_model_parallel_size):
    """Check if we should have the dataloader on this tensor and pipeline parallel rank."""
    # Run dataloader only on the first tensor parallel rank (will be broadcasted to others).
    is_first_rank = get_tensor_model_parallel_rank() == 0

    # NOTE(lizhiyu): when pp_size > 2
    # pp_size = get_pipeline_model_parallel_world_size()
    # is_first_rank = is_first_rank and is_first_or_last_stage(pp_size, transformer_pipeline_model_parallel_size)

    return is_first_rank

def train_valid_test_dataloaders_provider(train_val_test_num_samples):
    """Build multimodal train, validation and test dataloaders."""
    args = get_args()
    # Dataloader is only on specific ranks.
    if not is_dataloader_rank(args.transformer_pipeline_model_parallel_size):
        return None, None, None

    worker_debug_path = None
    worker_log_level = 0

    rank = parallel_state.get_data_parallel_rank()
    world_size = parallel_state.get_data_parallel_world_size()
    data_parallel_group = parallel_state.get_data_parallel_group()

    worker_config = WorkerConfig(
        rank=rank,
        world_size=world_size,
        num_workers=args.num_workers,
        data_parallel_group=data_parallel_group,
        worker_debug_path=worker_debug_path,
        worker_log_level=worker_log_level,
    )
    train_ds, valid_ds1, test_ds = datasets_provider(worker_config)

    train_dataloader = get_savable_loader(train_ds, worker_config=worker_config)
    if args.load is not None:
        if getattr(args, "dataloader_save", None):
            dp_rank = parallel_state.get_data_parallel_rank()
            data_save_name = get_checkpoint_name(
                args.dataloader_save,
                args.iteration,
                pipeline_rank=0,    # Only the first pipeline parallel rank stores the dataloader checkpoint.
                basename=f"train_dataloader_dprank{dp_rank:03d}.pt",
            )
            if os.path.exists(data_save_name):
                try:
                    dataset_state_dict = torch.load(data_save_name, map_location="cpu", weights_only=False)
                    train_dataloader.restore_state_rank(dataset_state_dict["dataloader_state_dict"])
                    print_rank_0(f"restored dataset state from {data_save_name}")
                except Exception as e:
                    print_rank_0("loading dataloader checkpoint failed. Skipping. " + str(e))

    if valid_ds1 is not None:
        valid_dataloader = [
            EnergonDataloader(get_loader(valid_ds, worker_config=worker_config))
            for valid_ds in valid_ds1
        ]
    else:
        valid_dataloader = EnergonDataloader(None)
    test_dataloader = None # NOTE: no test

    return EnergonDataloader(train_dataloader), valid_dataloader, EnergonDataloader(test_dataloader)

class EnergonDataloader:
    """A wrapper to use Megatron Energon dataloader with the Megatron-LM training loop."""
    def __init__(self, dataloader):
        self._dataloader = dataloader
        self._iter = iter(cyclic_iter(dataloader))

    def __next__(self):
        return self._iter.__next__()

    def __iter__(self):
        return self._iter.__iter__()

    def save_state(self):
        return self._dataloader.save_state_rank()


def cyclic_iter(iter):
    while True:
        for x in iter:
            yield x


def add_multimodal_extra_args(parser):
    """Extra arguments."""
    group = parser.add_argument_group(title="multimodal arguments")
    group.add_argument("--disable-vision-class-token", action="store_true", default=False, help="Disable vision class token")
    group.add_argument(
        "--dataloader-save", type=str, default=None, help="Energon dataloader state save path"
    )

    # qwen2-vl specific arguments
    group.add_argument("--extra-vocab-size", type=int, default=0)
    group.add_argument("--spatial-merge-size", type=int, default=2)
    group.add_argument("--temporal-patch-size", type=int, default=2)
    group.add_argument("--patch-size", type=int, default=16)
    group.add_argument("--max-padding-length", type=int, default=2048)
    group.add_argument("--enable-variable-seq-lengths", action="store_true", default=False, help="Enable variable sequence lengths")
    group.add_argument("--vision-root", type=str, default = None, help="The vision dirctory root path.")
    group.add_argument("--max-samples-per-sequence", type=int, default=2**31-1, help="max sequencial seqence samples in a slice")
    group.add_argument("--shuffle-buffer-size", type=int, default=0, help="the buffer size to shuffle the samples in a seqence")
    # learning rate
    group.add_argument("--vision-ration", type=float, default=0.1, help="the learning rate ration of vision(inlude merger) compared with llm")
    group.add_argument("--image-max-pixels", type=int, default=768*768, help="the maximum pixels of a single image")
    group.add_argument("--image-min-pixels", type=int, default=32*32, help="the minimum pixels of a single image")

    # vision model recompute
    group.add_argument("--vision-recompute-activations", action="store_true", default=False, help="Recompute vision model activations")
    # data processing
    group.add_argument("--no-use-system-prompt", dest="use_system_prompt", action="store_false", default=True, help="Don't use system prompt")


    # just for checkpoint conversion
    group.add_argument(
        "--convert-checkpoint-from-megatron-to-transformers",
        action="store_true",
        help=(
            "If True, convert a Megatron checkpoint to a Transformers checkpoint. "
            "If False, convert a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    group.add_argument("--freeze-LM", action="store_true", default=False, help="Freeze the language model")
    group.add_argument("--freeze-ViT", action="store_true", default=False, help="Freeze the vision model")
    group.add_argument(
        "--allow-missing-vision-projection-checkpoint",
        action="store_true",
        default=False,
        help="Allow missing vision projection checkpoint",
    )
    group.add_argument("--use-te", action="store_true", default=False, help="Use transformer engine")
    return parser


if __name__ == "__main__":
    train_valid_test_dataloaders_provider.is_distributed = True

    pretrain(
        train_valid_test_dataloaders_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={'tokenizer_type': 'Qwen2VLTokenizer'},
        extra_args_provider=add_multimodal_extra_args,
        process_non_loss_data_func=write_online_eval_to_tensorboard,
        non_loss_data_func=run_online_eval,
    )
