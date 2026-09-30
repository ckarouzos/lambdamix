"""Portable model, data, and optimization settings for training runs."""

import importlib.metadata
import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from numbers import Integral, Real
from pathlib import Path


def public_metadata(value):
    """Remove local path prefixes and credentials from serialized metadata."""
    if is_dataclass(value):
        return public_metadata(asdict(value))
    if isinstance(value, dict):
        return {
            str(key): (
                item
                if key in {"text", "prompt", "response", "question", "answer"}
                else public_metadata(item)
            )
            for key, item in value.items()
            if not any(
                word in str(key).lower()
                for word in (
                    "password",
                    "secret",
                    "api_key",
                    "hub_token",
                    "access_token",
                    "run_name",
                )
            )
        }
    if isinstance(value, (list, tuple)):
        return [public_metadata(item) for item in value]
    if isinstance(value, Enum):
        return public_metadata(value.value)
    if isinstance(value, (str, Path)):
        text = str(value)
        if text.startswith(("/", "~")):
            for part in Path(text).parts:
                if part.startswith("models--"):
                    return part[len("models--") :].replace("--", "/")
            return Path(text).name
        return text
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        return float(value)
    if hasattr(value, "tolist"):
        return public_metadata(value.tolist())
    return str(value)


def sanitize_checkpoint_metadata(output_dir):
    """Keep portable checkpoint metadata and omit redundant training arguments."""
    root = Path(output_dir)
    for name in ("adapter_config.json", "tokenizer_config.json", "config.json"):
        path = root / name
        if path.is_file():
            path.write_text(
                json.dumps(public_metadata(json.loads(path.read_text())), indent=2) + "\n"
            )
    for name in ("training_args.bin", "README.md"):
        (root / name).unlink(missing_ok=True)


def _dataset_payload(dataset, split):
    if dataset is None:
        return None
    from lambdamix.tasks.pku import dataset_provenance

    return {"split": split, "num_rows": len(dataset), "provenance": dataset_provenance(dataset)}


def write_run_manifest(
    *,
    output_dir,
    script_args,
    training_args,
    model_args,
    train_dataset,
    eval_dataset,
    resume_from_checkpoint=None,
):
    """Record the model, selected data, and training settings."""
    fields = (
        "seed",
        "data_seed",
        "max_steps",
        "num_train_epochs",
        "learning_rate",
        "lr_scheduler_type",
        "warmup_steps",
        "weight_decay",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "num_generations",
        "beta",
        "epsilon",
        "epsilon_high",
        "loss_type",
        "temperature",
        "top_p",
        "max_completion_length",
        "max_length",
        "generation_kwargs",
        "bf16",
        "gradient_checkpointing",
        "gradient_checkpointing_kwargs",
        "use_vllm",
        "multi_objective_aggregation",
    )
    payload = {
        "status": "started",
        "resolved_config": {
            "script_args": public_metadata(script_args),
            "training_args": {
                key: public_metadata(getattr(training_args, key))
                for key in fields
                if hasattr(training_args, key)
            },
            "model_args": public_metadata(model_args),
        },
        "datasets": {
            "train": _dataset_payload(train_dataset, script_args.dataset_train_split),
            "eval": _dataset_payload(eval_dataset, script_args.dataset_test_split),
        },
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "trl", "peft", "accelerate")
        },
    }
    destination = Path(output_dir) / "run_manifest.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not resume_from_checkpoint:
        raise FileExistsError("Use a new output directory or resume the existing run")
    if resume_from_checkpoint:
        destination = destination.with_name("run_manifest_resume.json")
    destination.write_text(json.dumps(payload, indent=2) + "\n")
    return destination


def update_run_manifest(path, status, **updates):
    """Record run completion in the run manifest."""
    path = Path(path)
    payload = json.loads(path.read_text())
    payload.update(status=status, **public_metadata(updates))
    path.write_text(json.dumps(payload, indent=2) + "\n")
