#!/usr/bin/env python3
"""Evaluate MOC-LoRA, PRISM-LoRA, or HoE-LoRA on a common lambda grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoTokenizer

from lambdamix.evaluation.artifacts import _atomic_json_dump
from lambdamix.evaluation.fronts import (
    _PERSAMPLE_META,
    evaluate_model_objectives,
    evaluation_token_limit,
    prepare_eval_dataset,
)
from lambdamix.evaluation.metrics import (
    front_metrics,
    generate_simplex_grid,
)
from lambdamix.generation import (
    assert_peft_adapter_loaded,
    load_adapter_compatible_base_model,
    load_declared_architecture_model,
    resolve_model_source,
)
from lambdamix.prompting import ensure_chat_template
from lambdamix.training.hoe import HoEPolicy, load_hoe_router_state
from lambdamix.training.moc import render_moc_preference
from lambdamix.training.prism import PRISMPolicy


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _base_source(checkpoint: Path, metadata: dict) -> str:
    base = metadata.get("base_model_name_or_path")
    adapter_config = checkpoint / "adapter_config.json"
    if not base and adapter_config.is_file():
        base = _load_json(adapter_config).get("base_model_name_or_path")
    if not base:
        raise ValueError("checkpoint does not identify its base model")
    return resolve_model_source(base)


def _load_tokenizer(checkpoint: Path, base_source: str):
    source = checkpoint if (checkpoint / "tokenizer_config.json").is_file() else base_source
    tokenizer = AutoTokenizer.from_pretrained(source)
    ensure_chat_template(tokenizer, base_source)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_method(checkpoint: Path, method: str, endpoint_adapters=None):
    if method == "moc":
        metadata = _load_json(checkpoint / "moc_grpo_meta.json")
        base_source = _base_source(checkpoint, metadata)
        base = load_adapter_compatible_base_model(
            base_source,
            checkpoint,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
        )
        model = PeftModel.from_pretrained(base, checkpoint)
        assert_peft_adapter_loaded(model, checkpoint)
    elif method == "prism":
        metadata = _load_json(checkpoint / "prism_meta.json")
        base_source = _base_source(checkpoint, metadata)
        base = load_adapter_compatible_base_model(
            base_source,
            checkpoint,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
        )
        peft = PeftModel.from_pretrained(base, checkpoint)
        assert_peft_adapter_loaded(peft, checkpoint)
        tokenizer = _load_tokenizer(checkpoint, base_source)
        newline = tokenizer.encode("\n", add_special_tokens=False)[-1]
        model = PRISMPolicy(
            peft,
            num_objectives=int(metadata["num_objectives"]),
            prefix_length=int(metadata["prefix_length"]),
            gamma=float(metadata["gamma"]),
            newline_token_id=newline,
        )
        prefix_state = torch.load(
            checkpoint / "prism_prefixes.pt", map_location="cpu", weights_only=True
        )
        model.prefix_embeddings.data.copy_(
            prefix_state["prefix_embeddings"].to(model.prefix_embeddings.device)
        )
        model.eval()
        return tokenizer, model, metadata
    elif method == "hoe":
        metadata = _load_json(checkpoint / "hoe_grpo_meta.json")
        base_source = _base_source(checkpoint, metadata)
        base = load_declared_architecture_model(
            base_source, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa"
        )
        model = HoEPolicy(
            base,
            endpoint_adapters or metadata["endpoint_adapter_paths"],
            metadata["router_preferences"],
            rank=int(metadata["rank"]),
            top_k=int(metadata["top_k"]),
            dare_drop_rate=float(metadata["dare_drop_rate"]),
            pcb_ratio=float(metadata.get("pcb_ratio", 0.1)),
            router_init_scale=float(metadata.get("router_init_scale", 4.0)),
            seed=int(metadata["expert_seed"]),
        )
        routers = torch.load(checkpoint / "hoe_routers.pt", map_location="cpu", weights_only=True)
        load_hoe_router_state(model, routers)
    else:  # pragma: no cover
        raise ValueError(method)
    tokenizer = _load_tokenizer(checkpoint, base_source)
    model.eval()
    return tokenizer, model, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("moc", "prism", "hoe"), required=True)
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--benchmark")
    parser.add_argument("--split", default=None)
    parser.add_argument("--resolution", type=int, default=10)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--endpoint_adapters", nargs=2)
    args = parser.parse_args()

    tokenizer, model, metadata = load_method(args.model_path, args.method, args.endpoint_adapters)
    benchmark = args.benchmark or metadata["dataset_name"]
    reward_funcs = list(metadata["reward_funcs"])
    lambdas = generate_simplex_grid(len(reward_funcs), args.resolution)
    max_tokens = args.max_new_tokens or evaluation_token_limit(benchmark)
    dataset, resolved_split = prepare_eval_dataset(benchmark, args.split, args.max_eval_samples)
    objectives, lengths = [], []
    for index, preference in enumerate(lambdas):
        directive = None
        if args.method == "moc":
            directive = render_moc_preference(reward_funcs, preference)
        print(f"[{index + 1}/{len(lambdas)}] lambda={preference.tolist()}", flush=True)
        _PERSAMPLE_META.clear()
        _PERSAMPLE_META.update(
            {
                "point": index,
                "lambda": preference.tolist(),
                "method": args.method,
                "exp": args.model_path.name,
            }
        )
        if args.method != "moc":
            model.set_preference(preference)
        result = evaluate_model_objectives(
            model,
            tokenizer,
            reward_funcs,
            benchmark,
            split=args.split,
            max_new_tokens=max_tokens,
            batch_size=args.batch_size,
            max_eval_samples=args.max_eval_samples,
            eval_dataset=dataset,
            resolved_split=resolved_split,
            inject_directive=directive,
        )
        objectives.append(result["objective_values"])
        lengths.append(result["length_mean"])

    payload = {
        "method": metadata["display_name"],
        "method_id": metadata["method_id"],
        "benchmark": benchmark,
        "split": resolved_split,
        "reward_funcs": reward_funcs,
        "lambdas": [value.tolist() for value in lambdas],
        "objective_values": objectives,
        "length_mean": lengths,
        **front_metrics(objectives, reward_funcs, lambdas=[value.tolist() for value in lambdas]),
    }
    _atomic_json_dump(args.output_path, payload)


if __name__ == "__main__":
    main()
