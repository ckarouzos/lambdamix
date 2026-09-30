"""Preference Vector baseline (Liang et al., EACL 2026; arXiv:2504.20106)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from lambdamix import resolve_attn_implementation
from lambdamix.evaluation.fronts import (
    _PERSAMPLE_META,
    evaluate_model_objectives,
    prepare_eval_dataset,
)
from lambdamix.evaluation.merging import compute_adapter_deltas
from lambdamix.evaluation.metrics import (
    front_metrics,
)
from lambdamix.generation import load_adapter_compatible_base_model, load_text_generation_model


def generate_simplex_coeff_grid(num_objectives: int, resolution: int) -> list[list[float]]:
    """Common-protocol simplex slice through Preference Vector intensity space."""

    if num_objectives != 2:
        raise ValueError(
            "the frozen common Preference Vector slice currently requires 2 objectives"
        )
    if resolution < 1:
        raise ValueError("resolution must be >= 1")
    return [[1.0 - index / resolution, index / resolution] for index in range(resolution + 1)]


def subtract_sparse_deltas(positive: dict, negative: dict) -> dict:
    """Return ``positive - negative`` for sparse task-vector state dictionaries."""

    result = {}
    for name in positive.keys() | negative.keys():
        pos = positive.get(name)
        neg = negative.get(name)
        if pos is None:
            result[name] = -neg.clone()
        elif neg is None:
            result[name] = pos.clone()
        else:
            if pos.shape != neg.shape:
                raise ValueError(
                    f"preference-vector shape mismatch for {name}: "
                    f"positive={tuple(pos.shape)}, negative={tuple(neg.shape)}"
                )
            result[name] = pos - neg
    return result


def build_preference_vectors(
    base_model_path: str,
    positive_adapter_paths: list[str],
    negative_adapter_paths: list[str],
    attn_implementation: str | None = None,
) -> tuple[dict, list[dict]]:
    """Realise the published contrastive construction in Eq. (1)."""
    if not positive_adapter_paths or len(positive_adapter_paths) != len(negative_adapter_paths):
        raise ValueError(
            "Preference Vector requires equally sized, non-empty positive and "
            "negative adapter lists (the published four-policy construction)."
        )
    paths = [path for pair in zip(positive_adapter_paths, negative_adapter_paths) for path in pair]
    base_state, deltas = compute_adapter_deltas(
        base_model_path, paths, attn_implementation=attn_implementation
    )
    vectors = [
        subtract_sparse_deltas(deltas[index], deltas[index + 1])
        for index in range(0, len(deltas), 2)
    ]
    return base_state, vectors


def compose_preference_vectors(
    base_state: dict, preference_vectors: list[dict], coeffs: list[float]
) -> dict:
    """Realise Eq. (2): ``theta(c) = theta_base + sum_t c_t * pv_t``."""
    if len(preference_vectors) != len(coeffs):
        raise ValueError(
            f"{len(preference_vectors)} preference vectors but {len(coeffs)} coefficients"
        )
    composed = {k: v.clone() for k, v in base_state.items()}
    for pv, c in zip(preference_vectors, coeffs):
        if c == 0.0:
            continue
        for name, diff in pv.items():
            composed[name] = composed[name] + c * diff
    return composed


def sweep_preference_vector_pareto(
    base_model_path: str,
    positive_adapter_paths: list[str],
    negative_adapter_paths: list[str],
    reward_funcs: list[str],
    benchmark: str,
    simplex_resolution: int = 10,
    split: str | None = None,
    max_new_tokens: int | None = None,
    batch_size: int = 8,
    max_eval_samples: int | None = None,
    reference_point: list[float] | None = None,
    attn_implementation: str | None = None,
) -> dict:
    """Score a Preference Vector front: build ``pv_t`` once, sweep the coeff grid."""
    if len(positive_adapter_paths) != len(reward_funcs) or len(negative_adapter_paths) != len(
        reward_funcs
    ):
        raise ValueError(
            "Preference Vector requires one positive and one negative adapter "
            f"per reward: positive={len(positive_adapter_paths)}, "
            f"negative={len(negative_adapter_paths)}, rewards={len(reward_funcs)}."
        )

    attn_impl = resolve_attn_implementation(attn_implementation)
    num_objectives = len(positive_adapter_paths)
    coeff_grid = generate_simplex_coeff_grid(num_objectives, simplex_resolution)

    # Eq. (1): build preference vectors once.
    print("Building contrastive preference vectors (positive delta - negative delta)...")
    base_state, preference_vectors = build_preference_vectors(
        base_model_path,
        positive_adapter_paths,
        negative_adapter_paths,
        attn_implementation=attn_impl,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    first_adapter = Path(positive_adapter_paths[0])
    loader = load_text_generation_model
    loader_args = [base_model_path]
    if (first_adapter / "adapter_config.json").exists():
        loader = load_adapter_compatible_base_model
        loader_args.append(first_adapter)
    model = loader(
        *loader_args,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=attn_impl,
    )
    model.eval()

    all_objectives: list[list[float]] = []
    all_coeffs: list[list[float]] = []

    eval_dataset, resolved_split = prepare_eval_dataset(benchmark, split, max_eval_samples)

    for i, coeffs in enumerate(coeff_grid):
        print(f"  [{i + 1}/{len(coeff_grid)}] coeffs = {coeffs}")
        _PERSAMPLE_META.clear()
        _PERSAMPLE_META.update(
            point=i,
            **{"lambda": list(coeffs)},
            method="preference_vector",
            exp=Path(base_model_path).name,
        )
        composed_state = compose_preference_vectors(base_state, preference_vectors, coeffs)
        model.load_state_dict(composed_state)

        result = evaluate_model_objectives(
            model=model,
            tokenizer=tokenizer,
            reward_funcs=reward_funcs,
            benchmark=benchmark,
            split=split,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            max_eval_samples=max_eval_samples,
            eval_dataset=eval_dataset,
            resolved_split=resolved_split,
        )
        resolved_split = result["split"]

        obj_values = result["objective_values"]
        all_objectives.append(obj_values)
        all_coeffs.append(coeffs)
        print(
            "    "
            + ", ".join(f"{name}={value:.4f}" for name, value in zip(reward_funcs, obj_values))
        )

    del model

    return {
        "method": "preference_vector",
        "paper": "arXiv:2504.20106",
        "construction": "contrastive_positive_minus_negative",
        "coefficient_grid_kind": "simplex",
        "benchmark": benchmark,
        "split": resolved_split,
        "reward_funcs": reward_funcs,
        "lambdas": all_coeffs,
        "objective_values": all_objectives,
        "metric_space": "task_calibrated_unclipped",
        "objective_values_space": "raw_reward_scores",
        **front_metrics(
            all_objectives, reward_funcs, lambdas=all_coeffs, reference_point=reference_point
        ),
    }


def main() -> None:

    parser = argparse.ArgumentParser(description="Evaluate a Preference Vector frontier")
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument(
        "--positive_adapter_paths",
        type=str,
        nargs="+",
        required=True,
        help="Positive-direction DPO LoRA adapters, one per objective.",
    )
    parser.add_argument(
        "--negative_adapter_paths",
        type=str,
        nargs="+",
        required=True,
        help="Negative/reversed-direction DPO LoRA adapters, one per objective.",
    )
    parser.add_argument("--reward_funcs", type=str, nargs="+", required=True)
    parser.add_argument("--benchmark", type=str, default="pku_safety")
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--simplex_resolution", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Cap eval examples per coefficient point (None = full split).",
    )
    parser.add_argument("--reference_point", type=float, nargs="+", default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default=None,
        help="Attention implementation (flash_attention_2, sdpa, eager). Auto-detects if omitted.",
    )
    args = parser.parse_args()

    results = sweep_preference_vector_pareto(
        base_model_path=args.base_model_path,
        positive_adapter_paths=args.positive_adapter_paths,
        negative_adapter_paths=args.negative_adapter_paths,
        reward_funcs=args.reward_funcs,
        benchmark=args.benchmark,
        simplex_resolution=args.simplex_resolution,
        split=args.split,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        max_eval_samples=args.max_eval_samples,
        reference_point=args.reference_point,
        attn_implementation=args.attn_implementation,
    )
    print(f"\nMethod: {results['method']} ({results['paper']})")
    print(f"Pareto front: {results['num_pareto']}/{results['num_points']} points")
    print(f"Hypervolume: {results['hypervolume']:.6f}")

    if args.output_path:
        Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output_path}")


if __name__ == "__main__":
    main()
