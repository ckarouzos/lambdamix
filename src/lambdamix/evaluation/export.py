"""Export LambdaMix or Panacea at a fixed preference as a dense model."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn


def _is_preference_wrapper(module: nn.Module) -> bool:
    """A merged LambdaMix/Panacea layer that should be flattened to its base."""
    return (
        not isinstance(module, nn.Linear)
        and hasattr(module, "base_layer")
        and hasattr(module, "set_preference")
        and isinstance(getattr(module, "base_layer"), nn.Linear)
    )


def flatten_merged_wrappers(base_model: nn.Module) -> nn.Module:
    """Replace preference-wrapper layers with their (already-merged) base layer."""
    for module in base_model.modules():
        for child_name, child in list(module.named_children()):
            if _is_preference_wrapper(child):
                setattr(module, child_name, child.base_layer)
    return base_model


def export_merged_checkpoint(
    model_path: str,
    output_dir: str,
    lambda_vec: list[float] | None = None,
    attn_implementation: str | None = None,
) -> str:
    """Materialize ``model_path`` at ``lambda_vec`` as a plain dense checkpoint."""
    model_dir = Path(model_path)
    pareto_adapter = model_dir / "pareto_adapters.pt"
    peft_config = model_dir / "adapter_config.json"
    out = Path(output_dir)

    if pareto_adapter.exists():
        if lambda_vec is None:
            raise ValueError(
                f"{model_path} is a preference-conditioned checkpoint; "
                "--lambda_vec is required to pick the preference to export."
            )
        # Load without importing the training and scoring stack.
        from lambdamix.adapters.loading import _load_pareto_components

        tokenizer, wrapper, _ = _load_pareto_components(model_path, attn_implementation)
        wrapper.merge_weights(torch.tensor(lambda_vec, dtype=torch.float32))
        base_model = flatten_merged_wrappers(wrapper.base_model)
        out.mkdir(parents=True, exist_ok=True)
        base_model.save_pretrained(str(out))
        tokenizer.save_pretrained(str(out))
        print(f"Merged preference checkpoint (lambda={lambda_vec}) saved to {out}")
        return str(out)

    if peft_config.exists():
        from peft import PeftConfig, PeftModel
        from transformers import AutoTokenizer

        from lambdamix import resolve_attn_implementation
        from lambdamix.generation import (
            assert_peft_adapter_loaded,
            load_adapter_compatible_base_model,
        )
        from lambdamix.prompting import ensure_chat_template

        cfg = PeftConfig.from_pretrained(model_path)
        base_name = cfg.base_model_name_or_path
        attn = resolve_attn_implementation(attn_implementation)
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(base_name)
        ensure_chat_template(tokenizer, base_name)
        base = load_adapter_compatible_base_model(
            base_name,
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation=attn,
        )
        peft_model = PeftModel.from_pretrained(base, model_path)
        assert_peft_adapter_loaded(peft_model, model_path)
        merged = peft_model.merge_and_unload()
        out.mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(str(out))
        tokenizer.save_pretrained(str(out))
        print(f"Merged PEFT adapter saved to {out}")
        return str(out)

    # Full checkpoint: nothing to merge.
    print(f"{model_path} is a full checkpoint; serving it directly (no merge needed).")
    return str(model_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Export a preference checkpoint at a fixed lambda as a dense HF model."
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument(
        "--lambda_vec",
        type=float,
        nargs="+",
        default=None,
        help="Preference vector for LambdaMix/Panacea checkpoints.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Destination directory for the merged dense checkpoint.",
    )
    parser.add_argument("--attn_implementation", type=str, default=None)
    parser.add_argument(
        "--print_path",
        action="store_true",
        help="Print only the resolved checkpoint path on the last line (for shell capture).",
    )
    args = parser.parse_args()

    path = export_merged_checkpoint(
        model_path=args.model_path,
        output_dir=args.output_dir,
        lambda_vec=args.lambda_vec,
        attn_implementation=args.attn_implementation,
    )
    if args.print_path:
        print(path)


if __name__ == "__main__":
    main()
