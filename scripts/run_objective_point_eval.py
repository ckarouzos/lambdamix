#!/usr/bin/env python3
"""Evaluate one ordinary adapter or frozen base on pointwise objectives."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

import lambdamix  # noqa: F401
from lambdamix.evaluation.fronts import (
    _PERSAMPLE_META,
    _resolve_eval_stop_token_ids,
    evaluate_model_objectives,
)
from lambdamix.generation import (
    assert_peft_adapter_loaded,
    load_adapter_compatible_base_model,
    load_text_generation_model,
    resolve_model_source,
)
from lambdamix.prompting import CHATML_TEMPLATE, ensure_chat_template
from lambdamix.run_manifest import public_metadata


def _resolve_stop_contract(stop_token_ids: list[int] | None) -> list[int]:
    """Resolve once so generation and provenance describe the same contract."""

    return _resolve_eval_stop_token_ids(stop_token_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_kind", choices=("adapter", "base", "full"), required=True)
    parser.add_argument("--backend", choices=("transformers", "vllm"), default="transformers")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--split")
    parser.add_argument("--reward_funcs", required=True)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--stop_token_ids", type=int, nargs="*", default=None)
    parser.add_argument(
        "--require_native_chat_template",
        action="store_true",
        help="Fail instead of installing the generic ChatML fallback.",
    )
    parser.add_argument("--lambda_vec", type=float, nargs="+")
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()
    if args.backend == "vllm" and args.model_kind == "adapter":
        parser.error("Export the adapter with lambdamix-export-merged, then use --model_kind full")

    output = Path(args.output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    os.environ.setdefault("LAMBDAMIX_PERSAMPLE_DIR", str(output.parent / "persample"))
    _PERSAMPLE_META.clear()
    _PERSAMPLE_META.update({"lambda": args.lambda_vec, "method": "fixed"})

    from transformers import AutoTokenizer

    generation_backend = None
    if args.backend == "vllm":
        from lambdamix.generation import VLLMGenerator

        canonical_base = args.model_path
        base_source = resolve_model_source(args.model_path)
        tokenizer = AutoTokenizer.from_pretrained(base_source, trust_remote_code=True)
        generation_backend = VLLMGenerator(
            base_source,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        model = None
    elif args.model_kind == "adapter":
        from peft import PeftConfig, PeftModel

        peft_config = PeftConfig.from_pretrained(args.model_path)
        canonical_base = peft_config.base_model_name_or_path
        base_source = resolve_model_source(canonical_base)
        tokenizer_source = (
            args.model_path
            if (Path(args.model_path) / "tokenizer_config.json").is_file()
            else base_source
        )
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
        base = load_adapter_compatible_base_model(
            base_source,
            args.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation=args.attn_implementation,
            trust_remote_code=True,
        )
        peft_model = PeftModel.from_pretrained(base, args.model_path)
        adapter_load_report = assert_peft_adapter_loaded(peft_model, args.model_path)
        model = peft_model.merge_and_unload()
    elif args.model_kind == "base":
        canonical_base = args.model_path
        base_source = resolve_model_source(args.model_path)
        tokenizer = AutoTokenizer.from_pretrained(base_source, trust_remote_code=True)
        model = load_text_generation_model(
            base_source,
            torch_dtype=torch.bfloat16,
            attn_implementation=args.attn_implementation,
            trust_remote_code=True,
        )
    else:
        canonical_base = args.model_path
        base_source = args.model_path
        tokenizer = AutoTokenizer.from_pretrained(base_source, trust_remote_code=True)
        model = load_text_generation_model(
            base_source,
            torch_dtype=torch.bfloat16,
            attn_implementation=args.attn_implementation,
            trust_remote_code=True,
        )
    source_chat_template = getattr(tokenizer, "chat_template", None)
    ensure_chat_template(tokenizer, canonical_base, default_chatml=True)
    if args.require_native_chat_template and (
        not source_chat_template or tokenizer.chat_template == CHATML_TEMPLATE
    ):
        raise ValueError(
            "the model does not expose a native chat template; refusing the generic ChatML fallback"
        )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if model is not None:
        model.eval()
        if torch.cuda.is_available():
            model.to("cuda")

    reward_funcs = [name.strip() for name in args.reward_funcs.split(",") if name.strip()]
    resolved_stop_token_ids = _resolve_stop_contract(args.stop_token_ids)
    result = evaluate_model_objectives(
        model=model,
        tokenizer=tokenizer,
        reward_funcs=reward_funcs,
        benchmark=args.benchmark,
        split=args.split,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        max_eval_samples=args.max_eval_samples,
        stop_token_ids=resolved_stop_token_ids,
        generation_backend=generation_backend,
    )
    payload = {
        "meta": {
            "lambda": args.lambda_vec,
            "model_path": args.model_path,
            "model_kind": args.model_kind,
            "canonical_base": canonical_base,
            "resolved_base_source": base_source,
            "generation_backend": args.backend,
            "chat_template_source": "native" if source_chat_template else "fallback",
            "tokenizer_eos_token_id": tokenizer.eos_token_id,
            "model_generation_eos_token_id": getattr(
                getattr(model, "generation_config", None), "eos_token_id", None
            ),
            "explicit_stop_token_ids": resolved_stop_token_ids,
            "adapter_load_report": (adapter_load_report if args.model_kind == "adapter" else None),
        },
        "result": result,
    }
    output = Path(args.output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(public_metadata(payload), indent=2, allow_nan=False) + "\n")
    print(output)


if __name__ == "__main__":
    main()
