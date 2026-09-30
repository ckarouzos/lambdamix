"""The paper's training tasks."""

import hashlib
import json
import os

from datasets import Dataset, load_dataset

GSM8K_SYSTEM_PROMPT = (
    "You are a helpful assistant. When solving a math problem, "
    "think step by step in <think>...</think> tags, "
    "then give the final numerical answer in <answer>...</answer> tags."
)


def extract_hash_answer(text: str) -> str:
    """Extract answer after #### from GSM8K ground-truth format."""
    if "####" not in text:
        return ""
    return text.split("####")[-1].strip()


def load_pku_safety(split: str = "train") -> Dataset:
    """Load official PKU training prompts or prompt-disjoint official test prompts."""
    from lambdamix.tasks.pku import load_pku_online

    return load_pku_online(split)


def load_training_dataset(dataset_name, split, *, resume_from_checkpoint=None):
    """Registry loading with PKU split and checkpoint-provenance guards."""
    from lambdamix.tasks.pku import validate_training

    if dataset_name == "pku_safety" and split != "train":
        raise ValueError("New PKU training must use pku_safety with split='train'")
    dataset = DATASET_REGISTRY[dataset_name](split)
    validate_training(dataset_name, split, dataset, resume_from_checkpoint=resume_from_checkpoint)
    return dataset


def _synthpai_source_path() -> str:
    """Read the released SynthPAI JSONL from LAMBDAMIX_SYNTHPAI_SOURCE or data/."""
    return os.environ.get("LAMBDAMIX_SYNTHPAI_SOURCE", "data/synthpai.jsonl")


def load_synthpai(split: str = "train") -> Dataset:
    """Load the consecutive SynthPAI training/test subsets and rewriting prompts."""
    from lambdamix.tasks.synthpai import (
        build_policy_messages,
        official_synthpai_splits,
        read_synthpai_jsonl,
    )

    if split not in ("train", "test"):
        raise ValueError(f"synthpai split must be train/test, got {split!r}")

    examples = official_synthpai_splits(read_synthpai_jsonl(_synthpai_source_path()))[split]
    records = [
        {
            "prompt": build_policy_messages(example),
            "answer": "",
            "example_id": example.example_id,
            "original_text": example.text,
            "target_attribute": example.target_attribute,
            "inferred_values": list(example.inferred_values),
        }
        for example in examples
    ]
    return Dataset.from_list(records)


_IFEVAL_NOTHINK_SWITCH = " /no_think"


_IFEVAL_STRUCTURAL = [
    {"type": "json"},
    {"type": "num_bullets", "n": 3},
]


_IFEVAL_LEXICAL = [
    {"type": "max_words", "n": 40},
    {"type": "max_words", "n": 60},
    {"type": "max_sentences", "n": 3},
    {"type": "no_commas"},
    {"type": "keyword", "word": "therefore"},
]


def _assign_ifeval_constraints(
    idx: int,
    n_constraints: int = 2,
    seed: int = 7,
    lexical_pool: list[dict] | None = None,
    structural_pool: list[dict] | None = None,
) -> list[dict]:
    """Assign deterministic constraints by sample index, with at most one structural constraint."""
    import random

    lexical = lexical_pool if lexical_pool is not None else _IFEVAL_LEXICAL
    structural = structural_pool if structural_pool is not None else _IFEVAL_STRUCTURAL
    structural_types = {c["type"] for c in structural}
    rng = random.Random(seed * 100003 + idx)
    pool = structural + lexical
    chosen = rng.sample(pool, min(n_constraints, len(pool)))
    chosen_structural = [c for c in chosen if c["type"] in structural_types]
    if len(chosen_structural) > 1:
        # Keep one structural, replace the rest with unused lexical constraints.
        keep = chosen_structural[0]
        used_lexical = [c for c in chosen if c["type"] not in structural_types]
        remaining = [c for c in lexical if c not in used_lexical]
        replacements = rng.sample(remaining, len(chosen_structural) - 1)
        chosen = [keep] + used_lexical + replacements
    return [dict(c) for c in chosen]


def load_gsm8k_ifeval(
    split: str = "train", suppress_native_cot: bool = True, n_constraints: int = 2
) -> Dataset:
    """Load GSM8K problems with deterministic response constraints."""
    from lambdamix.tasks.rewards import render_constraint_instruction

    ds = load_dataset("openai/gsm8k", "main", split=split)
    system_prompt = GSM8K_SYSTEM_PROMPT + (_IFEVAL_NOTHINK_SWITCH if suppress_native_cot else "")

    def _format(example, idx):
        question = example["question"]
        answer = extract_hash_answer(example["answer"])
        cons = _assign_ifeval_constraints(idx, n_constraints)
        instr = " ".join(
            f"({i + 1}) {render_constraint_instruction(c)}" for i, c in enumerate(cons)
        )
        user_content = f"{question}\n\nAdditional output requirements you MUST follow: {instr}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        identity = json.dumps([question, cons], sort_keys=True).encode()
        return {
            "prompt": messages,
            "answer": answer,
            "constraints": json.dumps(cons),
            "example_id": hashlib.sha256(identity).hexdigest(),
        }

    ds = ds.map(_format, with_indices=True, remove_columns=ds.column_names)
    return ds


def load_gsm8k_ifeval_nativecot(split: str = "train") -> Dataset:
    """The OLMo GSM8K prompt variant, without Qwen's /no_think switch."""
    return load_gsm8k_ifeval(split, suppress_native_cot=False)


DATASET_REGISTRY = {
    "gsm8k_ifeval": load_gsm8k_ifeval,
    "gsm8k_ifeval_nativecot": load_gsm8k_ifeval_nativecot,
    "pku_safety": load_pku_safety,
    "synthpai": load_synthpai,
}
