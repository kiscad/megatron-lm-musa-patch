import logging
import os

from megatron.core.dist_checkpointing.mapping import (
    CommonStateDict,
)

logger = logging.getLogger(__name__)

# The common state dict is intended to represent non-tensor states to be persisted in the checkpoint.
# This validation check is run in order to warn users if there is a difference across ranks, as only
# the common state dict from global rank 0 is saved and non-rank-zero common states are not persisted.
# local type不许要校验，每个rank都会保存common state dict
def _validate_common_state_dict(common_state_dict: CommonStateDict) -> None:
  pass

enable_async_ckpt = os.getenv("CKPT_DIR_PATH", "")
if enable_async_ckpt:
  print("flash ckpt enabled")
  import megatron.core.dist_checkpointing.validation
  megatron.core.dist_checkpointing.validation._validate_common_state_dict = _validate_common_state_dict