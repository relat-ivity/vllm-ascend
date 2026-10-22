# mypy: ignore-errors
import math

import torch
import vllm.model_executor.models.config
from vllm.logger import logger
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateDtypeCalculator
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.models.config import MambaModelConfig
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE, get_dtype_size, get_kv_cache_torch_dtype
from vllm.v1.core.sched.scheduler import Scheduler as _Scheduler

from vllm_ascend.utils import is_310p

_KVB_HYBRID_CONNECTOR = "UCMKvBridgeHybridConnector" 
 
 
def _using_ucm_kv_bridge(vllm_config) -> bool: 
    """Return whether the UCM hybrid KV bridge is configured.""" 
    kv_transfer_config = vllm_config.kv_transfer_config 
    if not kv_transfer_config: 
        return False 
 
    connector = kv_transfer_config.kv_connector 
    if connector == _KVB_HYBRID_CONNECTOR: 
        return True 
    if connector != "MultiConnector": 
        return False 
 
    extra_config = kv_transfer_config.kv_connector_extra_config or {} 
    return any( 
        item.get("kv_connector") == _KVB_HYBRID_CONNECTOR 
        for item in extra_config.get("connectors", ()) 
    ) 
 
 
def _qwen35_g_cache_enabled(vllm_config) -> bool: 
    """Enable the g sidecar for Qwen3.5 + UCM hybrid KV bridge in all mode."""
    hf_text_config = vllm_config.model_config.hf_text_config 
    model_type = str(getattr(hf_text_config, "model_type", "")) 
    return ( 
        not is_310p() 
        and model_type.startswith("qwen3_5") 
        and vllm_config.cache_config.mamba_cache_mode == "all" 
        and _using_ucm_kv_bridge(vllm_config) 
    ) 


@classmethod
def verify_and_update_config(cls, vllm_config) -> None:
    """
    Match attention K blocks exactly to contiguous SSM states.
    The runner stores g in V with padding between g blocks only.

    Args:
        vllm_config: vLLM Config
    """
    cache_config = vllm_config.cache_config

    # Save user-requested mode before calling parent, which may
    # downgrade "all" -> "align" because upstream model classes
    # don't declare SupportsMambaPrefixCaching for GDN yet.
    requested_mamba_cache_mode = cache_config.mamba_cache_mode

    # Enable FULL_AND_PIECEWISE by default
    MambaModelConfig.verify_and_update_config(vllm_config)

    # Restore "all" mode on NPU — we implement all-mode prefix caching
    # for GDN in vllm-ascend even though upstream hasn't merged it yet.
    if (
        requested_mamba_cache_mode == "all"
        and cache_config.enable_prefix_caching
        and cache_config.mamba_cache_mode == "align"
    ):
        cache_config.mamba_cache_mode = "all"

    cache_config = vllm_config.cache_config
    model_config = vllm_config.model_config
    parallel_config = vllm_config.parallel_config

    if cache_config.cache_dtype == "auto":
        kv_cache_dtype = model_config.dtype
    else:
        kv_cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

    kernel_block_size = 128
    model_cls, _ = ModelRegistry.resolve_model_cls(
        model_config.architecture,
        model_config=model_config,
    )

    # get mamba block size
    mamba_shapes = model_cls.get_mamba_state_shape_from_config(vllm_config)
    mamba_dtypes = model_cls.get_mamba_state_dtype_from_config(vllm_config)
    save_qwen35_g = _qwen35_g_cache_enabled(vllm_config)
    mamba_sizes = []
    for shape, dtype in zip(mamba_shapes, mamba_dtypes):
        mamba_sizes.append(math.prod(shape) * get_dtype_size(dtype))
    ssm_block_page_size, conv_block_page_size = max(mamba_sizes), min(mamba_sizes)

    # Pure linear attention models (e.g. bailing 2.5) have only SSM state,
    # no conv block. Detected by a single 3-D mamba shape (ssm only, no conv).
    # Example shape: MambaSpec(shapes=((8, 128, 128),), mamba_type='linear_attention')
    if len(mamba_shapes) == 1 and len(mamba_shapes[0]) == 3:
        conv_block_page_size = 0

    # SSM blocks stay contiguous, so their size must equal a K block exactly.
    if model_config.use_mla:
        attn_num_kv_heads = model_config.get_num_kv_heads(parallel_config)
        kv_lora_rank = model_config.hf_text_config.kv_lora_rank
        qk_rope_head_dim = model_config.hf_text_config.qk_rope_head_dim
        attn_single_token_k_page_size = kv_lora_rank * attn_num_kv_heads * get_dtype_size(kv_cache_dtype)
        attn_rope_token_page_size = qk_rope_head_dim * attn_num_kv_heads * get_dtype_size(kv_cache_dtype)
        attn_token_page_size = attn_single_token_k_page_size + attn_rope_token_page_size
    else:
        attn_num_kv_heads = model_config.get_num_kv_heads(parallel_config)
        attn_head_size = model_config.get_head_size()
        attn_single_token_k_page_size = attn_head_size * attn_num_kv_heads * get_dtype_size(kv_cache_dtype)
        attn_token_page_size = 2 * attn_head_size * attn_num_kv_heads * get_dtype_size(kv_cache_dtype)

    required_blocks = cdiv(ssm_block_page_size, kernel_block_size * attn_single_token_k_page_size)
    attn_block_size = kernel_block_size * required_blocks
    if attn_block_size % kernel_block_size or attn_single_token_k_page_size * attn_block_size != ssm_block_page_size:
        raise ValueError(
            "Cannot exactly align attention K block size with SSM state size using 128-token kernel blocks."
        )
    if cache_config.block_size != attn_block_size:
        logger.warning(
            "Setting attention block size from %s to %d tokens to exactly match "
            "the SSM state (%d bytes).",
            cache_config.block_size,
            attn_block_size,
            ssm_block_page_size,
        )
        cache_config.block_size = attn_block_size

    # compute new attention page size
    attn_page_size = cache_config.block_size * attn_token_page_size

    if save_qwen35_g:
        # The g sidecar is one value per token and local value head.
        # Compute it after block_size has been finalized above.
        cache_config.mamba_block_size = cache_config.block_size
        hf_text_config = model_config.hf_text_config
        local_num_v_heads = hf_text_config.linear_num_value_heads // parallel_config.tensor_parallel_size
        g_shape = (cache_config.mamba_block_size, local_num_v_heads)
        mamba_sizes.append(math.prod(g_shape) * get_dtype_size(torch.float32))
        v_block_page_size = attn_page_size - cache_config.block_size * attn_single_token_k_page_size
        if mamba_sizes[-1] > v_block_page_size:
            raise ValueError(
                f"Qwen3.5 g state requires {mamba_sizes[-1]} bytes per block, "
                f"but the shared V block holds {v_block_page_size} bytes."
            )

    # Keep fixed [conv, K/SSM, V/g] bands. Growing the page to fit g would
    # shift the FA K/V views rather than give g a larger per-block stride.
    base_padded_mamba_page_size = attn_page_size + conv_block_page_size
    padded_mamba_page_size = base_padded_mamba_page_size
    if cache_config.mamba_page_size_padded is None or cache_config.mamba_page_size_padded != padded_mamba_page_size:
        cache_config.mamba_page_size_padded = padded_mamba_page_size
    if (
        cache_config.mamba_cache_mode in ("align", "all")
        and (
            cache_config.enable_prefix_caching
            or _using_ucm_kv_bridge(vllm_config)
        )
    ):
        cache_config.mamba_block_size = cache_config.block_size
    else:
        cache_config.mamba_block_size = model_config.max_model_len


vllm.model_executor.models.config.HybridAttentionMambaModelConfig.verify_and_update_config = verify_and_update_config


# Enable block-aligned split for all-mode in upstream Scheduler.
# BalanceScheduler (NPU DP scheduler) already does this at L49-50,
# but the default Scheduler only enables it for "align" mode.
# Without this, scatter writes incorrect intermediate states when
# the step start position is not aligned to block boundaries.

_original_scheduler_init = _Scheduler.__init__


def _patched_scheduler_init(self, *args, **kwargs):
    _original_scheduler_init(self, *args, **kwargs)
    if self.has_mamba_layers and self.cache_config.mamba_cache_mode == "all":
        self.need_mamba_block_aligned_split = True


_Scheduler.__init__ = _patched_scheduler_init


# =============================================================================
# Patch: Remove float32 validation in linear_attention_state_dtype
# =============================================================================
# The original vLLM implementation raises ValueError when mamba_cache_dtype is
# "float32" because it was not yet tested on GPU. Ascend NPU supports fp32
# state for linear attention, so we replace the method with one that skips
# this restriction.
@classmethod  # type: ignore[misc]
def _linear_attention_state_dtype_npu(cls, model_dtype, mamba_cache_dtype):
    state_dtype = get_kv_cache_torch_dtype(mamba_cache_dtype, model_dtype)
    return (state_dtype,)


MambaStateDtypeCalculator.linear_attention_state_dtype = _linear_attention_state_dtype_npu
