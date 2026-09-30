"""Compatibility helpers for text generation backbones."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM


class VLLMGenerator:
    """Greedy token generation for base or exported dense checkpoints."""

    def __init__(self, model_path, *, tensor_parallel_size=1, gpu_memory_utilization=0.8):
        from vllm import LLM, ModelRegistry

        config = AutoConfig.from_pretrained(model_path)
        if "Qwen3_5ForCausalLM" in (config.architectures or []):
            ModelRegistry.register_model(
                "Qwen3_5ForCausalLM", "lambdamix.evaluation.vllm:Qwen35TextForCausalLM"
            )

        self.engine = LLM(
            model=model_path,
            dtype="bfloat16",
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=3072,
            generation_config="vllm",
            seed=42,
        )

    def __call__(self, prompt_token_ids, max_new_tokens, stop_token_ids):
        """Return one greedy completion per tokenized prompt."""
        from vllm import SamplingParams

        outputs = self.engine.generate(
            [{"prompt_token_ids": ids} for ids in prompt_token_ids],
            SamplingParams(
                temperature=0.0, max_tokens=max_new_tokens, stop_token_ids=stop_token_ids
            ),
            use_tqdm=False,
        )
        return [list(output.outputs[0].token_ids) for output in outputs]


def resolve_model_source(model_name: str) -> str:
    """Resolve a local model directory or retain its Hub identifier."""
    path = Path(model_name).expanduser()
    return str(path.resolve()) if path.exists() else model_name


QWEN35_MODEL_TYPES = frozenset({"qwen3_5", "qwen3_5_moe"})
QWEN35_TEXT_MODEL_TYPES = frozenset({"qwen3_5_text", "qwen3_5_moe_text"})
QWEN35_ALL_MODEL_TYPES = QWEN35_MODEL_TYPES | QWEN35_TEXT_MODEL_TYPES


def model_type_from_config(config: Any) -> str:
    """Return the language model type for ordinary or nested VLM configs."""

    model_type = str(getattr(config, "model_type", "") or "")
    text_config = getattr(config, "text_config", None)
    text_model_type = str(getattr(text_config, "model_type", "") or "")
    if model_type in QWEN35_MODEL_TYPES:
        return model_type
    return text_model_type or model_type


def is_qwen35_generation_config(config: Any) -> bool:
    """Whether ``config`` belongs to the Qwen3.5/3.6 architecture family."""

    top_level = str(getattr(config, "model_type", "") or "")
    return (
        top_level in QWEN35_MODEL_TYPES or model_type_from_config(config) in QWEN35_ALL_MODEL_TYPES
    )


def load_text_generation_model(model_name_or_path: str, **kwargs: Any):
    """Load a text-generation model, including Qwen3.5/3.6 text-only mode."""

    config_kwargs = {
        key: kwargs[key]
        for key in ("revision", "trust_remote_code", "local_files_only", "cache_dir", "token")
        if key in kwargs and kwargs[key] is not None
    }
    config = kwargs.pop("config", None)
    if config is None:
        try:
            config = AutoConfig.from_pretrained(model_name_or_path, **config_kwargs)
        except (KeyError, ValueError) as exc:
            local_config = Path(model_name_or_path) / "config.json"
            local_type = ""
            if local_config.is_file():
                try:
                    local_type = str(json.loads(local_config.read_text()).get("model_type", ""))
                except (OSError, json.JSONDecodeError):
                    pass
            model_hint = model_name_or_path.lower().replace("_", "").replace("-", "")
            if local_type in QWEN35_MODEL_TYPES or "qwen35" in model_hint or "qwen36" in model_hint:
                raise RuntimeError(
                    "This runtime cannot parse Qwen3.5/3.6. Install the "
                    "dependencies from pyproject.toml."
                ) from exc
            raise

    if is_qwen35_generation_config(config):
        mapping = AutoModelForCausalLM._model_mapping
        try:
            mapping[config.__class__]
        except (KeyError, AttributeError, ValueError) as exc:
            raise RuntimeError(
                "Qwen3.5/3.6 requires a Transformers release whose "
                "AutoModelForCausalLM registry contains qwen3_5/qwen3_5_moe. "
                "Install the dependencies from pyproject.toml."
            ) from exc

    return AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        config=config,
        **kwargs,
    )


def load_declared_architecture_model(model_name_or_path: str, **kwargs: Any):
    """Load the architecture declared by the checkpoint configuration."""

    config_kwargs = {
        key: kwargs[key]
        for key in ("revision", "trust_remote_code", "local_files_only", "cache_dir", "token")
        if key in kwargs and kwargs[key] is not None
    }
    config = kwargs.pop("config", None)
    if config is None:
        config = AutoConfig.from_pretrained(model_name_or_path, **config_kwargs)
    declared = list(getattr(config, "architectures", None) or [])
    if declared:
        architecture = getattr(transformers, declared[0], None)
        if architecture is None:
            raise RuntimeError(
                f"Transformers does not provide the checkpoint's declared architecture {declared[0]!r}"
            )
        return architecture.from_pretrained(
            model_name_or_path,
            config=config,
            **kwargs,
        )
    return AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        config=config,
        **kwargs,
    )


def qwen35_adapter_layout(adapter_path: str | Path) -> str:
    """Return the Qwen3.5/3.6 base layout encoded by a PEFT checkpoint."""

    from safetensors import safe_open

    path = Path(adapter_path)
    weights = path / "adapter_model.safetensors" if path.is_dir() else path
    if not weights.is_file():
        raise FileNotFoundError(f"missing PEFT safetensors checkpoint: {weights}")
    with safe_open(str(weights), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
    declared = any(".language_model.layers." in key for key in keys)
    text = any(".model.layers." in key and ".language_model.layers." not in key for key in keys)
    if declared == text:
        raise ValueError(
            f"cannot determine one Qwen3.5 adapter layout from {weights}: "
            f"declared={declared}, text={text}"
        )
    return "declared" if declared else "text"


def load_adapter_compatible_base_model(
    model_name_or_path: str,
    adapter_path: str | Path,
    **kwargs: Any,
):
    """Load the base architecture whose module names match ``adapter_path``."""

    config_kwargs = {
        key: kwargs[key]
        for key in ("revision", "trust_remote_code", "local_files_only", "cache_dir", "token")
        if key in kwargs and kwargs[key] is not None
    }
    config = kwargs.pop("config", None)
    if config is None:
        config = AutoConfig.from_pretrained(model_name_or_path, **config_kwargs)
    if is_qwen35_generation_config(config):
        layout = qwen35_adapter_layout(adapter_path)
        loader = (
            load_declared_architecture_model if layout == "declared" else load_text_generation_model
        )
        print(
            f"QWEN35_ADAPTER_LAYOUT layout={layout} adapter={adapter_path}",
            flush=True,
        )
    else:
        loader = load_text_generation_model
    return loader(model_name_or_path, config=config, **kwargs)


def assert_peft_adapter_loaded(model, adapter_path: str | Path) -> dict[str, int]:
    """Fail if a non-zero serialized LoRA-B checkpoint loaded as zero/missing."""

    from safetensors import safe_open

    path = Path(adapter_path)
    weights = path / "adapter_model.safetensors" if path.is_dir() else path
    if not weights.is_file():
        raise FileNotFoundError(f"missing PEFT safetensors checkpoint: {weights}")
    with safe_open(str(weights), framework="pt", device="cpu") as handle:
        source_b_keys = [key for key in handle.keys() if ".lora_B." in key]
        source_nonzero = sum(
            torch.count_nonzero(handle.get_tensor(key)).item() for key in source_b_keys
        )
    loaded_b = [
        parameter.detach() for name, parameter in model.named_parameters() if ".lora_B." in name
    ]
    loaded_nonzero = sum(torch.count_nonzero(parameter).item() for parameter in loaded_b)
    if not source_b_keys or source_nonzero == 0:
        raise RuntimeError(f"serialized adapter has no trained LoRA-B signal: {weights}")
    if len(loaded_b) != len(source_b_keys) or loaded_nonzero != source_nonzero:
        raise RuntimeError(
            "PEFT adapter load mismatch: "
            f"source_B_tensors={len(source_b_keys)}, loaded_B_tensors={len(loaded_b)}, "
            f"source_B_nonzero={source_nonzero}, loaded_B_nonzero={loaded_nonzero}, "
            f"adapter={weights}"
        )
    report = {
        "source_B_tensors": len(source_b_keys),
        "loaded_B_tensors": len(loaded_b),
        "source_B_nonzero": source_nonzero,
        "loaded_B_nonzero": loaded_nonzero,
    }
    print(f"PEFT_ADAPTER_LOAD_VERIFIED {report}", flush=True)
    return report


def _deduplicate_token_ids(token_ids: Iterable[Any]) -> tuple[int, ...]:
    ordered: list[int] = []
    seen: set[int] = set()
    for token_id in token_ids:
        value = int(token_id)
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return tuple(ordered)


def configure_native_generation_stops(training_args, tokenizer) -> tuple[int, ...]:
    """Translate declared stop IDs into the field native HF generation uses."""

    raw_kwargs = getattr(training_args, "generation_kwargs", None) or {}
    generation_kwargs = dict(raw_kwargs)
    declared = generation_kwargs.get("stop_token_ids", ()) or ()
    native = generation_kwargs.get("eos_token_id", ()) or ()
    if isinstance(native, int):
        native = [native]
    tokenizer_eos = getattr(tokenizer, "eos_token_id", None)
    effective = _deduplicate_token_ids(
        ([] if tokenizer_eos is None else [tokenizer_eos]) + list(declared) + list(native)
    )

    tokenizer._lambdamix_generation_stop_token_ids = effective

    if bool(getattr(training_args, "use_vllm", False)):
        raise ValueError("This release uses native Transformers generation; set use_vllm=false")
    if not effective:
        return effective
    if len(effective) > 1 and bool(getattr(training_args, "mask_truncated_completions", False)):
        raise ValueError(
            "Native multi-stop generation requires mask_truncated_completions=false: "
            "pinned TRL masks against only the processing tokenizer's scalar EOS."
        )

    generation_kwargs.pop("stop_token_ids", None)
    generation_kwargs["eos_token_id"] = effective[0] if len(effective) == 1 else list(effective)
    training_args.generation_kwargs = generation_kwargs
    print(
        "NATIVE_GENERATION_TERMINATORS "
        f"processing_eos_id={tokenizer_eos} "
        f"effective_eos_token_ids={list(effective)} "
        "mask_truncated_completions=false",
        flush=True,
    )
    return effective


def _first_stop_position(sequence: Sequence[Any], stop_ids: set[int]) -> int | None:
    for position, token_id in enumerate(sequence):
        if int(token_id) in stop_ids:
            return position
    return None


def trim_completions_at_generation_stops(
    completion_ids,
    logprobs,
    stop_token_ids: Iterable[Any],
):
    """Trim each returned completion (and aligned log-probs) at its first stop."""

    stops = set(_deduplicate_token_ids(stop_token_ids))
    if not stops:
        return completion_ids, logprobs

    trimmed_ids = []
    trimmed_logprobs = [] if logprobs is not None else None
    changed = False
    for index, sequence in enumerate(completion_ids):
        stop_position = _first_stop_position(sequence, stops)
        end = None if stop_position is None else stop_position + 1
        trimmed = sequence if end is None or end == len(sequence) else sequence[:end]
        changed = changed or trimmed is not sequence
        trimmed_ids.append(trimmed)
        if logprobs is not None:
            aligned = logprobs[index]
            trimmed_aligned = aligned if end is None or end == len(aligned) else aligned[:end]
            trimmed_logprobs.append(trimmed_aligned)

    if not changed:
        return completion_ids, logprobs
    return trimmed_ids, trimmed_logprobs


def trainer_generation_stop_ids(trainer) -> tuple[int, ...]:
    """Return the frozen stop contract attached to a trainer's tokenizer."""

    processing_class = getattr(trainer, "processing_class", None)
    return tuple(getattr(processing_class, "_lambdamix_generation_stop_token_ids", ()) or ())


def repair_trl_termination_metrics(trainer, generation_output) -> None:
    """Replace pinned TRL's scalar-EOS termination metrics for a stop set."""

    stop_ids = set(trainer_generation_stop_ids(trainer))
    if len(stop_ids) <= 1:
        return

    import torch

    completion_ids = generation_output[1]
    tool_mask = generation_output[2]
    device = trainer.accelerator.device
    pad_token_id = getattr(trainer, "pad_token_id", None)
    valid_ends = stop_ids | ({int(pad_token_id)} if pad_token_id is not None else set())
    is_truncated = torch.tensor(
        [not ids or int(ids[-1]) not in valid_ends for ids in completion_ids],
        device=device,
    )
    if tool_mask is None:
        completion_lengths = torch.tensor([len(ids) for ids in completion_ids], device=device)
    else:
        completion_lengths = torch.tensor([sum(mask) for mask in tool_mask], device=device)
    agg_is_truncated = trainer.accelerator.gather(is_truncated)
    agg_completion_lengths = trainer.accelerator.gather(completion_lengths)
    terminated_lengths = agg_completion_lengths[~agg_is_truncated]
    if len(terminated_lengths) == 0:
        terminated_lengths = torch.zeros(1, device=device)

    mode = "train" if trainer.model.training else "eval"
    replacements = {
        "completions/clipped_ratio": agg_is_truncated.float().mean().item(),
        "completions/mean_terminated_length": terminated_lengths.float().mean().item(),
        "completions/min_terminated_length": terminated_lengths.float().min().item(),
        "completions/max_terminated_length": terminated_lengths.float().max().item(),
    }
    for metric, value in replacements.items():
        history = trainer._metrics[mode][metric]
        if not history:
            raise RuntimeError(f"TRL did not append expected termination metric {metric!r}")
        history[-1] = value


class GenerationStopsMixin:
    """Apply the declared termination tokens to every GRPO rollout."""

    def __init__(self, *args, **kwargs):
        configure_native_generation_stops(kwargs["args"], kwargs["processing_class"])
        super().__init__(*args, **kwargs)

    def _generate_single_turn(self, prompt_ids, images, multimodal_fields):
        fields = dict(multimodal_fields or {})
        fields.pop("mm_token_type_ids", None)
        completion_ids, logprobs, extra = super()._generate_single_turn(prompt_ids, images, fields)
        completion_ids, logprobs = trim_completions_at_generation_stops(
            completion_ids, logprobs, trainer_generation_stop_ids(self)
        )
        return completion_ids, logprobs, extra

    def _generate(self, prompts):
        output = super()._generate(prompts)
        repair_trl_termination_metrics(self, output)
        return output

    def save_model(self, output_dir=None, **kwargs):
        super().save_model(output_dir, **kwargs)
        if self.args.should_save:
            from lambdamix.run_manifest import sanitize_checkpoint_metadata

            sanitize_checkpoint_metadata(output_dir or self.args.output_dir)
