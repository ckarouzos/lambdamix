"""Shared model-loading and training setup for the paper methods."""

from __future__ import annotations

import json
import os
from math import floor

import torch
from peft import LoraConfig
from trl import GRPOConfig, ModelConfig

from lambdamix.run_manifest import update_run_manifest, write_run_manifest
from lambdamix.tasks.data import DATASET_REGISTRY, load_training_dataset
from lambdamix.tasks.rewards import REWARD_REGISTRY


def parse_training_args(arguments, training_config=GRPOConfig):
    """Parse a paper recipe and require its base model identifier."""
    import sys

    import yaml
    from trl import TrlParser

    parsed = TrlParser((arguments, training_config, ModelConfig)).parse_args_and_config()
    model_args = parsed[2]
    if not model_args.model_name_or_path:
        for index, argument in enumerate(sys.argv):
            if argument == "--config" and index + 1 < len(sys.argv):
                with open(sys.argv[index + 1]) as handle:
                    model_args.model_name_or_path = (yaml.safe_load(handle) or {}).get(
                        "model_name_or_path", ""
                    )
                break
    if not model_args.model_name_or_path:
        raise ValueError("model_name_or_path is required")
    return parsed


def load_training_data(script_args, training_args):
    """Load the requested training and evaluation splits with provenance checks."""
    if script_args.dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {script_args.dataset_name}")
    train = load_training_dataset(
        script_args.dataset_name,
        script_args.dataset_train_split,
        resume_from_checkpoint=training_args.resume_from_checkpoint,
    )
    evaluation = (
        DATASET_REGISTRY[script_args.dataset_name](script_args.dataset_test_split)
        if script_args.dataset_test_split
        else None
    )
    return train, evaluation


def get_reward_functions(names):
    """Resolve the ordered objective names used in a paper recipe."""
    try:
        return [REWARD_REGISTRY[name] for name in names]
    except KeyError as error:
        raise ValueError(f"Unknown reward: {error.args[0]}") from error


def start_run(script_args, training_args, model_args, train_dataset, eval_dataset):
    """Write the training manifest on the main process."""
    if int(os.environ.get("RANK", "0")) != 0:
        return None
    return write_run_manifest(
        output_dir=training_args.output_dir,
        script_args=script_args,
        training_args=training_args,
        model_args=model_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        resume_from_checkpoint=training_args.resume_from_checkpoint,
    )


def finish_run(trainer, training_args, manifest_path):
    """Train, save the final model, and mark the main-process manifest complete."""
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model(training_args.output_dir)
    if manifest_path is not None:
        update_run_manifest(manifest_path, "completed")


def _model_source_kwargs(model_args: ModelConfig | None) -> dict:
    """Return shared model and tokenizer loading options."""
    if model_args is None:
        return {}
    return {
        "trust_remote_code": model_args.trust_remote_code,
    }


def _reference_model_init_kwargs(training_args, model_args: ModelConfig | None) -> dict:
    """Preserve model loading options and bf16 dtype for TRL's KL reference."""

    existing = training_args.model_init_kwargs or {}
    if isinstance(existing, str):
        existing = json.loads(existing)
    if not isinstance(existing, dict):
        raise TypeError("model_init_kwargs must be a dictionary or JSON object")
    source = {
        key: value for key, value in _model_source_kwargs(model_args).items() if value is not None
    }
    return {
        "dtype": "bfloat16",
        **existing,
        **source,
    }


def _model_parallel_load_kwargs(
    *, enabled: bool, max_memory_gib: int, device_count: int | None = None
) -> dict:
    """Build the fail-closed single-process model-parallel load contract."""

    if not enabled:
        return {}
    if max_memory_gib <= 0:
        raise ValueError("model_max_memory_gib must be positive")
    count = torch.cuda.device_count() if device_count is None else device_count
    if count < 2:
        raise RuntimeError("model_device_map_auto requires at least two visible accelerators")
    return {
        "device_map": "auto",
        "max_memory": {index: f"{max_memory_gib}GiB" for index in range(count)},
        "low_cpu_mem_usage": True,
    }


def _build_peft_config(model_args: ModelConfig) -> LoraConfig | None:
    """Create a PEFT config when the parsed ModelConfig requests LoRA training."""
    if not getattr(model_args, "use_peft", False):
        return None

    target_modules = getattr(model_args, "lora_target_modules", None)
    kwargs = {
        "r": getattr(model_args, "lora_r", 8),
        "lora_alpha": getattr(model_args, "lora_alpha", 16),
        "lora_dropout": getattr(model_args, "lora_dropout", 0.0),
        "target_modules": target_modules,
        "task_type": getattr(model_args, "lora_task_type", "CAUSAL_LM"),
    }
    if target_modules and any(
        target in set(target_modules) for target in ("in_proj_qkv", "in_proj_z", "out_proj")
    ):
        kwargs["exclude_modules"] = ["visual", "mtp"]
    # These fields may not exist in all TRL/transformers versions
    for attr, key in [
        ("lora_target_parameters", "target_parameters"),
        ("lora_modules_to_save", "modules_to_save"),
        ("use_rslora", "use_rslora"),
        ("use_dora", "use_dora"),
    ]:
        value = getattr(model_args, attr, None)
        if value is not None:
            kwargs[key] = value

    return LoraConfig(**kwargs)


def load_single_process_model_parallel_policy(
    local_model_source: str,
    script_args,
    training_args: GRPOConfig,
    model_args: ModelConfig,
):
    """Load a large PEFT base across visible accelerators when requested."""

    enabled = bool(getattr(script_args, "model_device_map_auto", False))
    if not enabled:
        return local_model_source
    if training_args.use_vllm:
        raise ValueError("model_device_map_auto requires use_vllm=false")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError(
            "model_device_map_auto is single-process model parallelism; "
            "do not combine it with a multi-process launcher"
        )

    import torch

    from lambdamix.generation import load_text_generation_model

    configured_device_count = int(getattr(script_args, "model_device_count", 0))
    if configured_device_count < 0:
        raise ValueError("model_device_count must be non-negative")
    visible_device_count = torch.cuda.device_count()
    if configured_device_count and configured_device_count > visible_device_count:
        raise RuntimeError(
            "model_device_count exceeds visible accelerators: "
            f"requested={configured_device_count} visible={visible_device_count}"
        )
    policy_device_count = configured_device_count or None
    load_kwargs = {
        "torch_dtype": torch.bfloat16,
        **_model_source_kwargs(model_args),
        **_model_parallel_load_kwargs(
            enabled=True,
            max_memory_gib=int(getattr(script_args, "model_max_memory_gib", 58)),
            device_count=policy_device_count,
        ),
    }
    if model_args.attn_implementation:
        load_kwargs["attn_implementation"] = model_args.attn_implementation
    policy = load_text_generation_model(local_model_source, **load_kwargs)
    policy.config._name_or_path = local_model_source
    print(
        "ORDINARY_LORA_MODEL_SOURCE_LOCAL "
        f"path={local_model_source} policy_device_count="
        f"{configured_device_count or visible_device_count}",
        flush=True,
    )
    return policy


def estimate_grpo_steps_per_epoch(dataset_size: int, training_args) -> tuple[int, int]:
    """Return optimizer steps/epoch and unique prompts consumed per optimizer step."""
    world_size = int(os.environ.get("WORLD_SIZE", getattr(training_args, "world_size", None) or 1))
    completion_rows = (
        training_args.per_device_train_batch_size
        * max(training_args.gradient_accumulation_steps, 1)
        * max(world_size, 1)
    )
    num_generations = max(training_args.num_generations, 1)
    if completion_rows % num_generations:
        raise ValueError(
            "The global completion-row batch per optimizer step must be divisible by "
            "num_generations to define prompt coverage."
        )
    unique_prompts = completion_rows // num_generations
    if unique_prompts < 1:
        raise ValueError("GRPO must consume at least one unique prompt per optimizer step.")
    return floor(dataset_size / unique_prompts), unique_prompts


def reserve_external_reward_devices(training_args, dataset_name: str) -> int:
    """Keep a single-process policy off GCDs reserved for frozen evaluators."""

    visible = int(getattr(training_args, "n_gpu", 0))
    external_reward_datasets = {
        "pku_safety",
        "synthpai",
    }
    if (
        dataset_name in external_reward_datasets
        and visible > 1
        and int(os.environ.get("WORLD_SIZE", "1")) == 1
    ):
        training_args._n_gpu = 1
        print(
            f"Policy restricted to cuda:0; {visible - 1} visible device(s) "
            f"reserved for {dataset_name} evaluators",
            flush=True,
        )
    return visible
