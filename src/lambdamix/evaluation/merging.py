"""Merge and evaluate specialist LoRA adapters."""

import argparse
import itertools
import json
import math
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from lambdamix import resolve_attn_implementation
from lambdamix.evaluation.fronts import (
    _PERSAMPLE_META,
    evaluate_model_objectives,
    prepare_eval_dataset,
)
from lambdamix.evaluation.metrics import (
    front_metrics,
    generate_simplex_grid,
)

MERGE_STRATEGIES = ("linear", "ties", "dare", "dare_ties")


def compute_adapter_deltas(
    base_model_path: str,
    adapter_paths: list[str],
    attn_implementation: str | None = None,
) -> tuple[dict, list[dict]]:
    """Compute weight deltas for each adapter relative to base model."""
    attn_impl = resolve_attn_implementation(attn_implementation)
    from lambdamix.generation import (
        assert_peft_adapter_loaded,
        load_adapter_compatible_base_model,
        load_text_generation_model,
    )

    first_adapter = Path(adapter_paths[0]) if adapter_paths else None
    if first_adapter is not None and (first_adapter / "adapter_config.json").exists():
        base_model = load_adapter_compatible_base_model(
            base_model_path,
            first_adapter,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl,
        )
    else:
        base_model = load_text_generation_model(
            base_model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl,
        )
    base_state = {k: v.clone() for k, v in base_model.state_dict().items()}
    del base_model

    delta_states = []
    for adapter_path in adapter_paths:
        # Try PEFT format first, fall back to full-model diff
        adapter_config_path = Path(adapter_path) / "adapter_config.json"
        if adapter_config_path.exists():
            adapter_model = load_adapter_compatible_base_model(
                base_model_path,
                adapter_path,
                torch_dtype=torch.bfloat16,
                attn_implementation=attn_impl,
            )
            peft_model = PeftModel.from_pretrained(adapter_model, adapter_path)
            assert_peft_adapter_loaded(peft_model, adapter_path)
            merged = peft_model.merge_and_unload()
            merged_state = merged.state_dict()
            del peft_model, adapter_model
        else:
            # Full-model save (LoRA was merged before saving)
            merged = AutoModelForCausalLM.from_pretrained(
                adapter_path,
                torch_dtype=torch.bfloat16,
            )
            merged_state = merged.state_dict()
        delta = {}
        missing_base_keys = set(base_state).difference(merged_state)
        if missing_base_keys:
            examples = sorted(missing_base_keys)[:5]
            raise RuntimeError(
                "adapter/base architecture mismatch during composition: "
                f"{len(missing_base_keys)} base keys are absent after merge; "
                f"examples={examples}, adapter={adapter_path}"
            )
        for name in base_state:
            diff = merged_state[name] - base_state[name]
            if diff.abs().sum() > 0:
                delta[name] = diff
        delta_states.append(delta)
        del merged, merged_state

    return base_state, delta_states


def apply_ties(delta_states: list[dict], top_k: float = 0.2) -> list[dict]:
    """TIES-Merging: Trim, Elect Sign, keep sign-consistent values."""
    all_names = set()
    for delta in delta_states:
        all_names.update(delta.keys())

    processed = [{} for _ in delta_states]

    for name in sorted(all_names):
        present = [
            (i, delta_states[i][name].clone())
            for i in range(len(delta_states))
            if name in delta_states[i]
        ]
        if not present:
            continue

        trimmed = []
        for i, d in present:
            flat = d.abs().float().flatten()
            if flat.numel() > 2**24:  # ~16M elements — quantile limit
                idx = torch.randperm(flat.numel(), device=flat.device)[: 2**24]
                threshold = flat[idx].quantile(1 - top_k).to(d.dtype)
            else:
                threshold = flat.quantile(1 - top_k).to(d.dtype)
            trimmed.append((i, d * (d.abs() >= threshold)))

        stacked = torch.stack([d for _, d in trimmed])
        sign_votes = (stacked.sign() * stacked.abs()).sum(dim=0).sign()

        for idx, (i, d) in enumerate(trimmed):
            mask = d.sign() == sign_votes
            processed[i][name] = d * mask

    return processed


def apply_dare(
    delta_states: list[dict],
    drop_rate: float = 0.9,
    seed: int = 42,
) -> list[dict]:
    """DARE: Drop And REscale delta parameters."""
    processed = []
    for t, delta in enumerate(delta_states):
        new_delta = {}
        gen = torch.Generator().manual_seed(seed + t)
        for name, tensor in delta.items():
            mask = torch.bernoulli(
                torch.ones_like(tensor, dtype=torch.float32) * (1 - drop_rate),
                generator=gen,
            ).to(tensor.dtype)
            new_delta[name] = tensor * mask / (1 - drop_rate)
        processed.append(new_delta)
    return processed


def sweep_mapreduce_pareto(
    base_model_path: str,
    adapter_paths: list[str],
    reward_funcs: list[str],
    benchmark: str,
    resolution: int = 10,
    split: str | None = None,
    max_new_tokens: int | None = None,
    batch_size: int = 8,
    max_eval_samples: int | None = None,
    reference_point: list[float] | None = None,
    attn_implementation: str | None = None,
    merge_strategy: str = "linear",
    ties_top_k: float = 0.2,
    dare_drop_rate: float = 0.9,
    coeff_values: list[float] | None = None,
    device_map: str | dict = "auto",
    merge_seed: int = 42,
    point_indices: list[int] | None = None,
) -> dict:
    """Sweep weight vectors and evaluate merged models."""
    if merge_strategy not in MERGE_STRATEGIES:
        raise ValueError(f"Unknown merge strategy: {merge_strategy}")
    if len(adapter_paths) != len(reward_funcs):
        raise ValueError(
            f"adapter_paths has {len(adapter_paths)} entries but reward_funcs has "
            f"{len(reward_funcs)}. These must match."
        )

    attn_impl = resolve_attn_implementation(attn_implementation)
    num_objectives = len(adapter_paths)
    if coeff_values is not None:
        if (
            merge_strategy != "linear"
            or not coeff_values
            or not all(math.isfinite(v) for v in coeff_values)
        ):
            raise ValueError(
                "Task arithmetic requires linear deltas and finite nonempty coefficients"
            )
        lambda_grid = [list(v) for v in itertools.product(coeff_values, repeat=num_objectives)]
    else:
        lambda_grid = [v.tolist() for v in generate_simplex_grid(num_objectives, resolution)]
    selected = list(range(len(lambda_grid))) if point_indices is None else point_indices
    if (
        not selected
        or len(set(selected)) != len(selected)
        or any(i < 0 or i >= len(lambda_grid) for i in selected)
    ):
        raise ValueError("Invalid or duplicate point indices")
    method = "task_arithmetic" if coeff_values is not None else f"mapreduce_{merge_strategy}"
    strategy_label = {
        "linear": "Rewarded Soups",
        "ties": "TIES-Merging",
        "dare": "DARE",
        "dare_ties": "DARE+TIES",
    }.get(merge_strategy, merge_strategy)
    print(
        f"MapReduce sweep ({strategy_label}): {len(lambda_grid)} weight vectors on {benchmark}..."
    )

    # Compute deltas once, apply merge strategy once
    print("Computing adapter deltas...")
    base_state, delta_states = compute_adapter_deltas(
        base_model_path,
        adapter_paths,
        attn_implementation=attn_impl,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(merge_seed)
        if merge_strategy in {"dare", "dare_ties"}:
            delta_states = apply_dare(delta_states, dare_drop_rate, seed=merge_seed)
        if merge_strategy in {"ties", "dare_ties"}:
            delta_states = apply_ties(delta_states, ties_top_k)

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Load model once, swap state dicts for each lambda
    from lambdamix.generation import load_adapter_compatible_base_model, load_text_generation_model

    first_adapter = Path(adapter_paths[0])
    loader = load_text_generation_model
    loader_args = [base_model_path]
    if (first_adapter / "adapter_config.json").exists():
        loader = load_adapter_compatible_base_model
        loader_args.append(first_adapter)
    model = loader(
        *loader_args,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        attn_implementation=attn_impl,
    )
    model.eval()

    all_objectives = []
    all_lambdas = []

    eval_dataset, resolved_split = prepare_eval_dataset(benchmark, split, max_eval_samples)

    for i in selected:
        weights = lambda_grid[i]
        print(f"  [{i + 1}/{len(lambda_grid)}] weights = {weights}")
        _PERSAMPLE_META.clear()
        _PERSAMPLE_META.update(
            point=i, **{"lambda": weights}, method=method, exp=Path(base_model_path).name
        )

        # Build merged state dict on CPU, load_state_dict copies to GPU
        merged_state = {k: v.clone() for k, v in base_state.items()}
        for delta, w in zip(delta_states, weights):
            for name, diff in delta.items():
                merged_state[name] = merged_state[name] + w * diff
        model.load_state_dict(merged_state, strict=True)
        del merged_state

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
        all_lambdas.append(weights)
        print(
            "    "
            + ", ".join(f"{name}={value:.4f}" for name, value in zip(reward_funcs, obj_values))
        )

    del model

    return {
        "method": method,
        "merge_strategy": merge_strategy,
        "merge_seed": merge_seed,
        "coefficient_grid_kind": "cartesian" if coeff_values is not None else "simplex",
        "coeff_values": coeff_values,
        "point_indices": selected,
        "complete_grid": len(selected) == len(lambda_grid),
        "grid_size": len(lambda_grid),
        "benchmark": benchmark,
        "split": resolved_split,
        "reward_funcs": reward_funcs,
        "lambdas": all_lambdas,
        "objective_values": all_objectives,
        "metric_space": "task_calibrated_unclipped",
        "objective_values_space": "raw_reward_scores",
        **front_metrics(
            all_objectives,
            reward_funcs,
            lambdas=all_lambdas if coeff_values is None else None,
            reference_point=reference_point,
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate a MapReduce LoRA frontier")
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--adapter_paths", type=str, nargs="+", required=True)
    parser.add_argument("--reward_funcs", type=str, nargs="+", required=True)
    parser.add_argument("--benchmark", type=str, default="gsm8k_ifeval")
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--resolution", type=int, default=10)
    parser.add_argument(
        "--coeff_values",
        type=float,
        nargs="+",
        default=None,
        help="Task arithmetic: independent delta coefficients instead of simplex weights.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Cap eval examples per lambda point (None = full split). Speeds up merge sweeps.",
    )
    parser.add_argument("--reference_point", type=float, nargs="+", default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default=None,
        help="Attention implementation (flash_attention_2, sdpa, eager). Auto-detects if omitted.",
    )
    parser.add_argument(
        "--merge_strategy",
        type=str,
        default="linear",
        choices=list(MERGE_STRATEGIES),
        help="Merge strategy: linear (Rewarded Soups), ties, dare, dare_ties",
    )
    parser.add_argument(
        "--ties_top_k",
        type=float,
        default=0.2,
        help="TIES: fraction of parameters to keep by magnitude (default 0.2)",
    )
    parser.add_argument(
        "--dare_drop_rate",
        type=float,
        default=0.9,
        help="DARE: fraction of parameters to drop (default 0.9)",
    )
    args = parser.parse_args()

    results = sweep_mapreduce_pareto(
        base_model_path=args.base_model_path,
        adapter_paths=args.adapter_paths,
        reward_funcs=args.reward_funcs,
        benchmark=args.benchmark,
        split=args.split,
        resolution=args.resolution,
        coeff_values=args.coeff_values,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        max_eval_samples=args.max_eval_samples,
        reference_point=args.reference_point,
        attn_implementation=args.attn_implementation,
        merge_strategy=args.merge_strategy,
        ties_top_k=args.ties_top_k,
        dare_drop_rate=args.dare_drop_rate,
    )

    print(f"\nStrategy: {results['merge_strategy']}")
    print(f"Pareto front: {results['num_pareto']}/{results['num_points']} points")
    print(f"Hypervolume: {results['hypervolume']:.6f}")

    if args.output_path:
        Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output_path}")


if __name__ == "__main__":
    main()
