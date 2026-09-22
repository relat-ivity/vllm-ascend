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
    """Enable the write-only g state only for Qwen3.5 + UCM align mode.""" 
    hf_text_config = vllm_config.model_config.hf_text_config 
    model_type = str(getattr(hf_text_config, "model_type", "")) 
    return ( 
        not is_310p() 
        and model_type.startswith("qwen3_5") 
        and vllm_config.cache_config.mamba_cache_mode == "align" 
        and _using_ucm_kv_bridge(vllm_config) 
    ) 


@classmethod
def verify_and_update_config(cls, vllm_config) -> None:
    """
    Ensure that page size of attention layers is greater than or
    equal to the mamba layers. If not, automatically set the attention
    block size to ensure that it is. If the attention page size is
    strictly greater than the mamba page size, we pad the mamba page size
    to make them equal.

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
        logger.info("Restoring mamba_cache_mode='all' (NPU all-mode prefix caching is supported in vllm-ascend).")

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

    # NOTE(zxr): because of the limit of Ascend Hardware, we need to keep
    # all cache tensors contiguous, so we align the page size of ssm_block
    # and single attn_block
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

    attn_block_size = kernel_block_size * cdiv(ssm_block_page_size, kernel_block_size * attn_single_token_k_page_size)
    assert attn_single_token_k_page_size * attn_block_size == ssm_block_page_size, (
        "Cannot align ssm_page_size and attn_page_size."
    )

    # override attention block size if either (a) the
    # user has not set it or (b) the user has set it
    # too small.
    if cache_config.block_size is None or cache_config.block_size < attn_block_size:
        cache_config.block_size = attn_block_size
        logger.info(
            "Setting attention block size to %d tokens to ensure that attention page size is >= mamba page size.",
            attn_block_size,
        )

    # compute new attention page size
    attn_page_size = cache_config.block_size * attn_token_page_size

    if save_qwen35_g: 
        # In align mode the g sidecar is one value per token in a Mamba block. 
        # Compute it after block_size has been finalized above. 
        cache_config.mamba_block_size = cache_config.block_size 
        hf_text_config = model_config.hf_text_config 
        local_num_v_heads = ( 
            hf_text_config.linear_num_value_heads 
            // parallel_config.tensor_parallel_size 
        ) 
        g_shape = (cache_config.mamba_block_size, local_num_v_heads) 
        mamba_sizes.append( 
            math.prod(g_shape) * get_dtype_size(torch.float32) 
        ) 
 
    # Prefer the existing Ascend hybrid-page padding. Grow the page only if the 
    # third state does not fit in that tail. 
    real_mamba_page_size = sum(mamba_sizes) 
    base_padded_mamba_page_size = attn_page_size + conv_block_page_size 
    padded_mamba_page_size = ( 
        max(base_padded_mamba_page_size, real_mamba_page_size) 
        if save_qwen35_g 
        else base_padded_mamba_page_size 
    )
    if (
        cache_config.mamba_page_size_padded is None
        or cache_config.mamba_page_size_padded != padded_mamba_page_size
    ):
        cache_config.mamba_page_size_padded = padded_mamba_page_size 
        if save_qwen35_g: 
            padding_size = padded_mamba_page_size - real_mamba_page_size 
            mamba_padding_pct = 100 * padding_size / padded_mamba_page_size 
            logger.info(
                "Padding mamba page size by %.2f%%; Qwen3.5 g KV cache enabled.",
                mamba_padding_pct,
            )
        else:
            mamba_padding_pct = 100 * conv_block_page_size / padded_mamba_page_size
            logger.info(
                "Padding mamba page size by %.2f%% to ensure "
                "that mamba page size and attention page size are "
                "exactly equal.",
                mamba_padding_pct,
            )
    if cache_config.enable_prefix_caching and cache_config.mamba_cache_mode in ("align", "all"):
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
