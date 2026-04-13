# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Pretrain Kimi-K2.5-VL on a simple multimodal dataset."""

import os
import sys
from functools import partial

import torch

if os.getenv("ACCELERATOR_BACKEND", "musa") == "musa":
    import musa_patch
else:
    import cuda_patch


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PATCH_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir, os.pardir))
MEGATRON_MULTIMODAL_DIR = os.path.abspath(
    os.path.join(PATCH_ROOT, "..", "Megatron-LM", "examples", "multimodal")
)
if MEGATRON_MULTIMODAL_DIR not in sys.path:
    sys.path.append(MEGATRON_MULTIMODAL_DIR)

from kimi_k25_vl_dataset import is_first_or_last_stage, train_valid_test_dataloaders_provider
from kimi_k25_vl_model_provider import model_provider
from multimodal_args import add_multimodal_extra_args

from megatron.core import mpu, tensor_parallel
from megatron.core.enums import ModelType
from megatron.core.parallel_state import (
    get_pipeline_model_parallel_world_size,
    get_tensor_model_parallel_rank,
    is_pipeline_last_stage,
)
from megatron.training import get_args, get_timers, pretrain
from megatron.training.utils import unwrap_model


def add_kimi_k25_vl_extra_args(parser):
    parser = add_multimodal_extra_args(parser)
    parser.set_defaults(
        vision_model_type="moonvit",
        dataloader_type="external",
        tokenizer_type="MultimodalTokenizer",
        special_tokens=["<image>"],
        image_tag_type="",
        max_num_tiles=1,
        use_tiling=False,
        use_thumbnail=False,
        disable_vision_class_token=False,
    )

    group = parser.add_argument_group(title="kimi-k25-vl arguments")
    group.add_argument("--simple-mm-train-data", type=str, default=None)
    group.add_argument("--simple-mm-valid-data", type=str, default=None)
    group.add_argument("--simple-mm-test-data", type=str, default=None)
    group.add_argument("--simple-mm-image-key", type=str, default="image")
    group.add_argument("--simple-mm-text-key", type=str, default="text")
    group.add_argument("--simple-mm-prompt-key", type=str, default="prompt")
    group.add_argument(
        "--simple-mm-default-prompt",
        type=str,
        default="Describe the image.",
    )
    group.add_argument("--kimi-vision-hidden-size", type=int, default=1152)
    group.add_argument("--kimi-vision-ffn-hidden-size", type=int, default=4304)
    group.add_argument("--kimi-vision-num-layers", type=int, default=27)
    group.add_argument("--kimi-vision-num-attention-heads", type=int, default=16)
    group.add_argument("--kimi-vision-merge-kernel-size", type=int, default=2)
    return parser


def get_batch(data_iterator):
    """Generate a multimodal batch and broadcast it across TP ranks."""

    tokens = None
    labels = None
    loss_mask = None
    attention_mask = None
    position_ids = None
    images = None
    num_image_tiles = None

    args = get_args()
    pp_size = get_pipeline_model_parallel_world_size()
    if not is_first_or_last_stage(pp_size, args.encoder_pipeline_model_parallel_size):
        return tokens, labels, loss_mask, attention_mask, position_ids, images, num_image_tiles

    if data_iterator is not None and get_tensor_model_parallel_rank() == 0:
        data = next(data_iterator)
    else:
        data = None

    tokens = tensor_parallel.broadcast_data(["tokens"], data, torch.int64)["tokens"]
    labels = tensor_parallel.broadcast_data(["labels"], data, torch.int64)["labels"]
    loss_mask = tensor_parallel.broadcast_data(["loss_mask"], data, torch.float32)["loss_mask"]
    position_ids = tensor_parallel.broadcast_data(["position_ids"], data, torch.int64)["position_ids"]
    images = tensor_parallel.broadcast_data(["imgs"], data, torch.float32)["imgs"]
    num_image_tiles = tensor_parallel.broadcast_data(["num_tiles"], data, torch.int32)["num_tiles"]

    if pp_size > 1 and is_pipeline_last_stage():
        images = None

    return tokens, labels, loss_mask, attention_mask, position_ids, images, num_image_tiles


def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
    args = get_args()

    losses = output_tensor.float()
    flat_loss_mask = loss_mask.view(-1).float()
    total_tokens = flat_loss_mask.sum()
    loss = torch.cat([torch.sum(losses.view(-1) * flat_loss_mask).view(1), total_tokens.view(1)])

    if args.context_parallel_size > 1:
        torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())

    reporting_loss = loss.clone().detach()
    torch.distributed.all_reduce(reporting_loss, group=mpu.get_data_parallel_group())
    local_num_tokens = loss[1].clone().detach().to(torch.int)
    # MUSA training_log expects the older tuple contract during training, while
    # Megatron evaluation expects a two-element tensor and calls .view(-1).
    loss_report = (reporting_loss[0], reporting_loss[1]) if torch.is_grad_enabled() else reporting_loss
    return (
        loss[0] * args.context_parallel_size,
        local_num_tokens,
        {"lm loss": loss_report},
    )


def forward_step(data_iterator, model):
    args = get_args()
    timers = get_timers()

    if args.context_parallel_size > 1:
        raise NotImplementedError("Initial Kimi-K2.5-VL pretraining path only supports context_parallel_size=1.")

    timers("batch-generator", log_level=2).start()
    (
        tokens,
        labels,
        loss_mask,
        attention_mask,
        position_ids,
        images,
        num_image_tiles,
    ) = get_batch(data_iterator)
    timers("batch-generator").stop()

    unwrapped_model = unwrap_model(model)
    output_tensor, new_loss_mask = model(
        images,
        tokens,
        position_ids,
        attention_mask,
        labels,
        loss_mask,
        image_token_index=unwrapped_model.image_token_index,
        num_image_tiles=num_image_tiles,
    )

    return output_tensor, partial(loss_func, new_loss_mask)


def kimi_embedding_ranks(pp_ranks):
    args = get_args()
    encoder_stages = args.encoder_pipeline_model_parallel_size
    last_rank = pp_ranks[-1]
    if len(pp_ranks) == 1 or pp_ranks[encoder_stages] == last_rank:
        return [last_rank]
    return [pp_ranks[encoder_stages], last_rank]


def kimi_position_embedding_ranks(pp_ranks):
    args = get_args()
    encoder_stages = args.encoder_pipeline_model_parallel_size
    last_rank = pp_ranks[-1]
    if len(pp_ranks) == 1:
        return [last_rank]
    return [pp_ranks[encoder_stages]]


if __name__ == "__main__":
    train_valid_test_dataloaders_provider.is_distributed = True

    pretrain(
        train_valid_test_dataloaders_provider,
        model_provider,
        ModelType.encoder_and_decoder,
        forward_step,
        args_defaults={
            "dataloader_type": "external",
            "tokenizer_type": "MultimodalTokenizer",
        },
        extra_args_provider=add_kimi_k25_vl_extra_args,
        get_embedding_ranks=kimi_embedding_ranks,
        get_position_embedding_ranks=kimi_position_embedding_ranks,
    )
