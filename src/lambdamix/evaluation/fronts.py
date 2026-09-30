"""Preference sweeps and shared scoring for the paper's three training tasks."""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn

from lambdamix.adapters.loading import _load_pareto_components
from lambdamix.evaluation.artifacts import _atomic_json_dump
from lambdamix.evaluation.metrics import (
    front_metrics,
    generate_simplex_grid,
)
from lambdamix.prompting import render_chat_prompt
from lambdamix.tasks.data import DATASET_REGISTRY
from lambdamix.tasks.math import math_verify_equal
from lambdamix.tasks.rewards import REWARD_REGISTRY, extract_xml_answer

EVAL_SAMPLE_COUNTS = {
    "gsm8k_ifeval": 300,
    "gsm8k_ifeval_nativecot": 300,
    "pku_safety": 1354,
    "synthpai": 111,
}


def evaluation_token_limit(benchmark: str) -> int:
    """Return the paper's completion-token cap for the selected task."""
    return 512 if benchmark == "synthpai" else 1024


SUPPORTED_OBJECTIVES = {
    "correctness",
    "ifeval_constraint",
    "pku_harmlessness",
    "pku_helpfulness",
    "synthpai_privacy",
    "synthpai_utility",
}


# Save prompt-level scores and responses when LAMBDAMIX_PERSAMPLE_DIR is set.
_PERSAMPLE_META: dict = {}


def _dump_persample(
    out_dir: str,
    reward_funcs: list[str],
    decoded: list[str],
    completion_ids_list,
    batch_objectives: torch.Tensor,
    meta: dict,
    example_ids: list[str] | None = None,
) -> None:
    """Append one JSONL line per completion in this batch."""
    try:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
    except (FileExistsError, PermissionError, OSError):
        if not os.path.isdir(out_dir):
            raise
    point = meta.get("point")
    fname = "persample.jsonl" if point is None else f"point{int(point):02d}.jsonl"
    path = Path(out_dir) / fname
    lengths = [len(ids) for ids in completion_ids_list]
    if example_ids is not None and len(example_ids) != len(decoded):
        raise ValueError("Per-sample prompt identity count does not match generated responses")
    rows = []
    for k, text in enumerate(decoded):
        rows.append(
            {
                "len": int(lengths[k]),
                "answer_present": int(bool(extract_xml_answer(text))),
                "rewards": {
                    name: float(batch_objectives[k, m]) for m, name in enumerate(reward_funcs)
                },
                "text": text,
                **({"example_id": example_ids[k]} if example_ids is not None else {}),
            }
        )
    mode = "a" if path in _dump_persample._opened else "w"  # type: ignore[attr-defined]
    _dump_persample._opened.add(path)  # type: ignore[attr-defined]
    header = {
        "point": point,
        "lambda": meta.get("lambda"),
        "reward_funcs": reward_funcs,
        "method": meta.get("method"),
        "exp": meta.get("exp"),
    }
    with open(path, mode) as f:
        if mode == "w":
            f.write(json.dumps({"__meta__": header}) + "\n")
        for r in rows:
            f.write(json.dumps(r) + "\n")


_dump_persample._opened = set()  # type: ignore[attr-defined]


def _resolve_eval_stop_token_ids(stop_token_ids: list[int] | None = None) -> list[int]:
    """Resolve the explicit evaluation stop contract deterministically."""
    if stop_token_ids is None:
        raw = os.environ.get("LAMBDAMIX_EVAL_STOP_TOKEN_IDS", "")
        stop_token_ids = [int(value.strip()) for value in raw.split(",") if value.strip()]
    return list(dict.fromkeys(int(value) for value in stop_token_ids))


DEFAULT_EVAL_SPLIT = {
    "gsm8k_ifeval": "test",
    "gsm8k_ifeval_nativecot": "test",
    "pku_safety": "test",
    "synthpai": "test",
}


def _load_eval_dataset(target: str, split: str | None):
    """Load a paper task using its held-out evaluation split by default."""
    if target not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown evaluation target {target!r}. Available: {sorted(DATASET_REGISTRY)}"
        )
    eval_split = split or DEFAULT_EVAL_SPLIT[target]
    if target == "pku_safety" and eval_split != "test":
        raise ValueError("PKU evaluation requires the prompt-disjoint test split")
    return DATASET_REGISTRY[target](eval_split), eval_split


def _math_correctness_reward(prompts, completions, answer, **kwargs) -> list[float]:
    """Math correctness objective using math-verify equivalence."""
    responses = [completion[0]["content"] for completion in completions]
    extracted = [extract_xml_answer(r) for r in responses]
    return [1.0 if math_verify_equal(pred, gt) else 0.0 for pred, gt in zip(extracted, answer)]


OBJECTIVE_EVALUATORS = {"correctness": _math_correctness_reward}


def _evaluate_objectives_for_batch(
    reward_funcs: list[str],
    prompts,
    answers,
    completions,
    completion_ids_list,
    batch_kwargs: dict,
    max_completion_length: int,
) -> torch.Tensor:
    """Evaluate one batch of completions against the configured objective functions."""
    reward_kwargs = {k: v for k, v in batch_kwargs.items() if k != "prompt"}
    reward_kwargs.setdefault("answer", answers)

    per_objective = []
    for reward_name in reward_funcs:
        if reward_name not in SUPPORTED_OBJECTIVES:
            raise ValueError(f"Unsupported Pareto objective '{reward_name}'.")

        evaluator = OBJECTIVE_EVALUATORS.get(reward_name, REWARD_REGISTRY[reward_name])
        values = evaluator(
            prompts=prompts,
            completions=completions,
            completion_ids=completion_ids_list,
            max_tokens=max_completion_length,
            **reward_kwargs,
        )
        scores = torch.tensor(values, dtype=torch.float64)
        if scores.shape != (len(completions),):
            raise ValueError(
                f"{reward_name}: expected {len(completions)} scores, got {tuple(scores.shape)}"
            )
        if not torch.isfinite(scores).all():
            bad = (~torch.isfinite(scores)).nonzero().flatten().tolist()
            ids = batch_kwargs.get("example_id")
            identity = [ids[i] for i in bad] if ids is not None else bad
            raise ValueError(f"{reward_name}: nonfinite evaluator scores for samples {identity}")
        per_objective.append(scores)

    return torch.stack(per_objective, dim=1)


def _infer_model_device(model: nn.Module) -> torch.device:
    """Best-effort device inference for generation/evaluation."""
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


def prepare_eval_dataset(
    benchmark: str,
    split: str | None = None,
    max_eval_samples: int | None = None,
) -> tuple[object, str]:
    """Load + subsample the eval dataset ONCE so a sweep can reuse it per point."""
    eval_dataset, resolved_split = _load_eval_dataset(benchmark, split)
    if max_eval_samples is None:
        max_eval_samples = EVAL_SAMPLE_COUNTS.get(benchmark)
    if max_eval_samples is not None and max_eval_samples <= 0:
        raise ValueError("max_eval_samples must be positive")
    if max_eval_samples and max_eval_samples < len(eval_dataset):
        eval_dataset = eval_dataset.select(range(max_eval_samples))
    return eval_dataset, resolved_split


def _trim_completion_row(
    ids: list[int],
    eos_token_id,
    pad_token_id,
    stop_token_ids: list[int] | None = None,
) -> list[int]:
    """Cut a batch-padded HF ``generate`` row down to the sample's TRUE completion tokens."""
    stop_ids = {eos_token_id, pad_token_id, *(stop_token_ids or [])}
    stop_ids.discard(None)
    for k, t in enumerate(ids):
        if t in stop_ids:
            return ids[:k]
    return ids


def evaluate_model_objectives(
    model: nn.Module | None,
    tokenizer,
    reward_funcs: list[str],
    benchmark: str,
    split: str | None = None,
    max_new_tokens: int | None = None,
    batch_size: int = 8,
    max_eval_samples: int | None = None,
    eval_dataset: object | None = None,
    resolved_split: str | None = None,
    inject_directive: str | None = None,
    stop_token_ids: list[int] | None = None,
    generation_backend=None,
) -> dict:
    """Evaluate a model on the configured reward dimensions for one target dataset."""
    if eval_dataset is None:
        eval_dataset, resolved_split = prepare_eval_dataset(benchmark, split, max_eval_samples)

    from lambdamix.prompting import dataset_requests_no_thinking, force_no_thinking

    if dataset_requests_no_thinking(eval_dataset):
        force_no_thinking(tokenizer)

    max_new_tokens = max_new_tokens or evaluation_token_limit(benchmark)
    model_device = _infer_model_device(model) if generation_backend is None else "cpu"

    stop_token_ids = _resolve_eval_stop_token_ids(stop_token_ids)

    objective_sum = torch.zeros(len(reward_funcs), dtype=torch.float64)
    total = 0

    length_sum = 0

    for j in range(0, len(eval_dataset), batch_size):
        batch = eval_dataset[j : j + batch_size]
        prompts = batch["prompt"]
        answers = batch.get("answer", [""] * len(prompts))

        if inject_directive is not None:
            from lambdamix.prompting import inject_preference_into_prompt

            render_prompts = [
                inject_preference_into_prompt(prompt, inject_directive) for prompt in prompts
            ]
        else:
            render_prompts = prompts
        texts = [render_chat_prompt(tokenizer, prompt) for prompt in render_prompts]

        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(model_device)

        generation_eos_ids = list(stop_token_ids)
        if tokenizer.eos_token_id is not None:
            generation_eos_ids.append(int(tokenizer.eos_token_id))
        generation_eos_ids = list(dict.fromkeys(generation_eos_ids))

        if generation_backend is not None:
            prompt_ids = [
                row[mask.bool()].tolist()
                for row, mask in zip(inputs["input_ids"], inputs["attention_mask"])
            ]
            generated = generation_backend(prompt_ids, max_new_tokens, generation_eos_ids)
        else:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    eos_token_id=(generation_eos_ids or None),
                    pad_token_id=tokenizer.pad_token_id,
                )
            generated = outputs[:, inputs["input_ids"].shape[1] :].tolist()

        completion_ids_list = [
            _trim_completion_row(
                row, tokenizer.eos_token_id, tokenizer.pad_token_id, stop_token_ids
            )
            for row in generated
        ]
        decoded = tokenizer.batch_decode(completion_ids_list, skip_special_tokens=True)

        completions = [[{"role": "assistant", "content": text}] for text in decoded]

        batch_objectives = _evaluate_objectives_for_batch(
            reward_funcs=reward_funcs,
            prompts=prompts,
            answers=answers,
            completions=completions,
            completion_ids_list=completion_ids_list,
            batch_kwargs=batch,
            max_completion_length=max_new_tokens,
        )
        _ps_dir = os.environ.get("LAMBDAMIX_PERSAMPLE_DIR")
        if _ps_dir:
            _dump_persample(
                _ps_dir,
                reward_funcs,
                decoded,
                completion_ids_list,
                batch_objectives,
                _PERSAMPLE_META,
                example_ids=batch.get("example_id"),
            )

        _lengths = [len(ids) for ids in completion_ids_list]
        length_sum += sum(_lengths)

        objective_sum += batch_objectives.sum(dim=0).double()
        total += batch_objectives.shape[0]

    if total == 0:
        raise RuntimeError(f"No evaluation examples produced for target '{benchmark}'.")

    objective_mean = objective_sum / total
    from lambdamix.tasks.pku import dataset_provenance

    provenance = dataset_provenance(eval_dataset)
    return {
        "benchmark": benchmark,
        "split": resolved_split,
        "reward_funcs": reward_funcs,
        "objective_values": objective_mean.tolist(),
        "total": total,
        **({"dataset_provenance": provenance} if provenance else {}),
        "length_mean": length_sum / total,
    }


def sweep_pareto_front(
    model_path: str,
    benchmark: str,
    num_objectives: int | None = None,
    resolution: int = 10,
    split: str | None = None,
    max_new_tokens: int | None = None,
    batch_size: int = 8,
    reference_point: list[float] | None = None,
    attn_implementation: str | None = None,
    max_eval_samples: int | None = None,
) -> dict:
    """Sweep preference vectors and evaluate actual objective values."""
    tokenizer, pareto_model, checkpoint_config = _load_pareto_components(
        model_path, attn_implementation
    )

    # Auto-detect benchmark from checkpoint if it was trained on a different dataset
    checkpoint_dataset = checkpoint_config.get("dataset_name")
    if checkpoint_dataset and checkpoint_dataset != benchmark:
        print(
            f"NOTE: checkpoint trained on '{checkpoint_dataset}', overriding benchmark '{benchmark}' → '{checkpoint_dataset}'"
        )
        benchmark = checkpoint_dataset

    reward_funcs = checkpoint_config.get("reward_funcs")
    if not reward_funcs:
        raise ValueError(
            "Checkpoint does not declare reward_funcs. Re-train or re-save with updated metadata."
        )
    _NAME_MAP = {}
    normalized = []
    for name in reward_funcs:
        name = name.removesuffix("_reward_func")
        name = _NAME_MAP.get(name, name)
        normalized.append(name)
    reward_funcs = normalized

    if num_objectives is None:
        num_objectives = len(reward_funcs)
    if num_objectives != len(reward_funcs):
        raise ValueError(
            f"num_objectives={num_objectives} does not match checkpoint reward funcs {reward_funcs}."
        )

    generation_length = max_new_tokens or evaluation_token_limit(benchmark)
    lambda_grid = generate_simplex_grid(num_objectives, resolution)

    all_objectives = []
    all_lambdas = []
    all_length_mean: list[float] = []

    if max_eval_samples:
        print(f"Using {max_eval_samples} eval samples (subsampled)")
    eval_dataset, resolved_split = prepare_eval_dataset(benchmark, split, max_eval_samples)

    print(f"Sweeping {len(lambda_grid)} preference vectors on {benchmark}:{split or 'auto'} ...")

    for i, lambda_vec in enumerate(lambda_grid):
        print(f"  [{i + 1}/{len(lambda_grid)}] lambda = {lambda_vec.tolist()}", flush=True)

        _PERSAMPLE_META.clear()
        _PERSAMPLE_META.update(
            {
                "point": i,
                "lambda": lambda_vec.tolist(),
                "method": "pareto",
                "exp": Path(model_path).name,
            }
        )
        pareto_model.set_preference(lambda_vec)

        objective_result = evaluate_model_objectives(
            model=pareto_model,
            tokenizer=tokenizer,
            reward_funcs=reward_funcs,
            benchmark=benchmark,
            split=split,
            max_new_tokens=generation_length,
            batch_size=batch_size,
            max_eval_samples=max_eval_samples,
            eval_dataset=eval_dataset,
            resolved_split=resolved_split,
        )

        all_objectives.append(objective_result["objective_values"])
        all_lambdas.append(lambda_vec.tolist())
        all_length_mean.append(objective_result["length_mean"])
        resolved_split = objective_result["split"]
        print(
            "    "
            + ", ".join(
                f"{name}={value:.4f}"
                for name, value in zip(reward_funcs, objective_result["objective_values"])
            )
        )
        print(f"    len_mean={objective_result['length_mean']:.1f}  ")

    return {
        "benchmark": benchmark,
        "split": resolved_split,
        "reward_funcs": reward_funcs,
        "lambdas": all_lambdas,
        "objective_values": all_objectives,
        "length_mean": all_length_mean,
        **front_metrics(
            all_objectives, reward_funcs, lambdas=all_lambdas, reference_point=reference_point
        ),
    }


def _load_prompt_conditioned_components(model_path: str, attn_implementation: str | None):
    """Load (tokenizer, model, metadata) for a prompt-conditioned checkpoint."""
    import json

    from peft import PeftModel
    from transformers import AutoTokenizer

    from lambdamix import resolve_attn_implementation
    from lambdamix.generation import assert_peft_adapter_loaded, load_adapter_compatible_base_model
    from lambdamix.prompting import ensure_chat_template

    meta_path = Path(model_path) / "lambdamix_promptcond_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Missing prompt-conditioned metadata sidecar: {meta_path}. "
            "Was this checkpoint written by PromptConditionedGRPOTrainer.save_model?"
        )
    with open(meta_path) as f:
        metadata = json.load(f)

    adapter_config_path = Path(model_path) / "adapter_config.json"
    base_model_name = metadata.get("base_model_name_or_path")
    if not base_model_name and adapter_config_path.exists():
        with open(adapter_config_path) as f:
            base_model_name = json.load(f).get("base_model_name_or_path")
    if not base_model_name:
        raise ValueError(
            "Checkpoint is missing base_model_name_or_path (neither metadata sidecar "
            "nor adapter_config.json declares it)."
        )

    from lambdamix.generation import resolve_model_source

    base_model_source = resolve_model_source(base_model_name)
    metadata["_resolved_base_model_source"] = base_model_source

    attn = attn_implementation or resolve_attn_implementation()
    base_model = load_adapter_compatible_base_model(
        base_model_source,
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn,
    )
    model = PeftModel.from_pretrained(base_model, model_path)
    assert_peft_adapter_loaded(model, model_path)
    model.eval()

    tokenizer_source = (
        model_path if (Path(model_path) / "tokenizer_config.json").exists() else base_model_source
    )
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    except (ValueError, KeyError):
        tokenizer = AutoTokenizer.from_pretrained(base_model_source)
    ensure_chat_template(tokenizer, base_model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, model, metadata


def sweep_prompt_conditioned_front(
    model_path: str,
    benchmark: str,
    resolution: int = 10,
    split: str | None = None,
    max_new_tokens: int | None = None,
    batch_size: int = 8,
    reference_point: list[float] | None = None,
    attn_implementation: str | None = None,
    max_eval_samples: int | None = None,
) -> dict:
    """Evaluate the prompt-conditioned baseline across preference weights."""
    from lambdamix.prompting import render_preference_directive

    tokenizer, model, metadata = _load_prompt_conditioned_components(
        model_path, attn_implementation
    )

    if torch.cuda.is_available():
        model.to("cuda")

    checkpoint_dataset = metadata.get("dataset_name")
    if checkpoint_dataset and checkpoint_dataset != benchmark:
        print(
            f"NOTE: checkpoint trained on '{checkpoint_dataset}', overriding "
            f"benchmark '{benchmark}' -> '{checkpoint_dataset}'"
        )
        benchmark = checkpoint_dataset

    reward_funcs = metadata.get("reward_funcs")
    if not reward_funcs:
        raise ValueError("Metadata sidecar does not declare reward_funcs.")
    reward_funcs = [name.removesuffix("_reward_func") for name in reward_funcs]
    num_objectives = len(reward_funcs)

    generation_length = max_new_tokens or evaluation_token_limit(benchmark)
    lambda_grid = generate_simplex_grid(num_objectives, resolution)

    eval_dataset, resolved_split = prepare_eval_dataset(benchmark, split, max_eval_samples)

    all_objectives = []
    all_lambdas = []
    all_length_mean: list[float] = []
    print(
        f"Prompt-conditioned sweep: {len(lambda_grid)} preference vectors on "
        f"{benchmark}:{split or 'auto'} ..."
    )
    for i, lambda_vec in enumerate(lambda_grid):
        directive = render_preference_directive(reward_funcs, lambda_vec.tolist())
        print(f"  [{i + 1}/{len(lambda_grid)}] lambda = {lambda_vec.tolist()}", flush=True)
        _PERSAMPLE_META.clear()
        _PERSAMPLE_META.update(
            {
                "point": i,
                "lambda": lambda_vec.tolist(),
                "method": "prompt",
                "exp": Path(model_path).name,
            }
        )
        objective_result = evaluate_model_objectives(
            model=model,
            tokenizer=tokenizer,
            reward_funcs=reward_funcs,
            benchmark=benchmark,
            split=split,
            max_new_tokens=generation_length,
            batch_size=batch_size,
            max_eval_samples=max_eval_samples,
            eval_dataset=eval_dataset,
            resolved_split=resolved_split,
            inject_directive=directive,
        )
        all_objectives.append(objective_result["objective_values"])
        all_lambdas.append(lambda_vec.tolist())
        all_length_mean.append(objective_result["length_mean"])
        resolved_split = objective_result["split"]
        print(
            "    "
            + ", ".join(
                f"{name}={value:.4f}"
                for name, value in zip(reward_funcs, objective_result["objective_values"])
            )
        )
        print(f"    len_mean={objective_result['length_mean']:.1f}  ")

    return {
        "benchmark": benchmark,
        "split": resolved_split,
        "conditioning": "prompt",
        "generation_backend": "hf",
        "reward_funcs": reward_funcs,
        "lambdas": all_lambdas,
        "objective_values": all_objectives,
        **front_metrics(
            all_objectives, reward_funcs, lambdas=all_lambdas, reference_point=reference_point
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Pareto front sweep evaluation")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--benchmark", type=str, default="gsm8k_ifeval")
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--num_objectives", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--reference_point", type=float, nargs="+", default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default=None,
        help="Attention implementation (flash_attention_2, sdpa, eager). Auto-detects if omitted.",
    )
    parser.add_argument(
        "--max_eval_samples",
        "--subsample",
        type=int,
        default=None,
        dest="max_eval_samples",
        help="Subsample the eval dataset to this many examples for faster sweeps. ",
    )
    parser.add_argument(
        "--prompt_conditioned",
        action="store_true",
        default=False,
        help="Evaluate a single adapter using preference directives in the system prompt.",
    )
    args = parser.parse_args()

    if args.prompt_conditioned:
        results = sweep_prompt_conditioned_front(
            model_path=args.model_path,
            benchmark=args.benchmark,
            resolution=args.resolution,
            split=args.split,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            reference_point=args.reference_point,
            attn_implementation=args.attn_implementation,
            max_eval_samples=args.max_eval_samples,
        )
    else:
        results = sweep_pareto_front(
            model_path=args.model_path,
            benchmark=args.benchmark,
            num_objectives=args.num_objectives,
            resolution=args.resolution,
            split=args.split,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            reference_point=args.reference_point,
            attn_implementation=args.attn_implementation,
            max_eval_samples=args.max_eval_samples,
        )

    print(f"\nPareto front: {results['num_pareto']}/{results['num_points']} points")
    print(f"Hypervolume: {results['hypervolume']:.6f}")

    if args.output_path:
        _atomic_json_dump(Path(args.output_path), results)
        print(f"Results saved to {args.output_path}")


if __name__ == "__main__":
    main()
