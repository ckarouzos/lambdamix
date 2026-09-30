"""Expose vLLM's Qwen3.5 text decoder with the hybrid cache interface."""

from vllm.model_executor.models.interfaces import IsHybrid
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForCausalLM,
    Qwen3_5ForConditionalGeneration,
)


class Qwen35TextForCausalLM(Qwen3_5ForCausalLM, IsHybrid):
    """Load exported Qwen3.5/3.6 text weights using vLLM's native decoder."""

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config):
        """Use the native Qwen hybrid-cache dtypes."""
        return Qwen3_5ForConditionalGeneration.get_mamba_state_dtype_from_config(vllm_config)

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config):
        """Use the native Qwen hybrid-cache shapes."""
        return Qwen3_5ForConditionalGeneration.get_mamba_state_shape_from_config(vllm_config)

    @classmethod
    def get_mamba_state_copy_func(cls):
        """Use the native Qwen hybrid-cache copy functions."""
        return Qwen3_5ForConditionalGeneration.get_mamba_state_copy_func()
