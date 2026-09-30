"""Reward functions for the paper's three tasks."""

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor


def extract_xml_answer(text: str) -> str:
    """Extract content from the last complete answer tag pair."""
    matches = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return ""


def _check_answer(prediction: str, reference: str) -> bool:
    """Check answer correctness with math-verify equivalence and fallback."""
    if prediction == reference:
        return True
    try:
        from lambdamix.tasks.math import math_verify_equal

        return math_verify_equal(prediction, reference)
    except ImportError:
        return False


def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """Binary reward: 1.0 if extracted answer matches ground truth, else 0.0."""
    responses = [completion[0]["content"] for completion in completions]
    extracted = [extract_xml_answer(r) for r in responses]
    return [1.0 if _check_answer(e, gt) else 0.0 for e, gt in zip(extracted, answer)]


def _extract_raw_prompt(prompt: list[dict]) -> str:
    """Extract the user message text from chat messages."""
    for msg in prompt:
        if msg["role"] == "user":
            return msg["content"]
    return ""


_DEFAULT_PKU_REWARD_MODEL = "PKU-Alignment/beaver-7b-unified-reward"


_DEFAULT_PKU_COST_MODEL = "PKU-Alignment/beaver-7b-unified-cost"


class _BeaverModelCache:
    """Cache frozen Beaver scoring models by model identifier."""

    _cache: dict = {}
    reward_model_id: str = os.environ.get("LAMBDAMIX_PKU_REWARD_MODEL", _DEFAULT_PKU_REWARD_MODEL)
    cost_model_id: str = os.environ.get("LAMBDAMIX_PKU_COST_MODEL", _DEFAULT_PKU_COST_MODEL)

    @classmethod
    def get(cls, axis: str):
        model_id = cls.reward_model_id if axis == "reward" else cls.cost_model_id
        if model_id not in cls._cache:
            import torch

            from lambdamix.tasks.pku import BeaverScorer

            axis_device = os.environ.get(
                "LAMBDAMIX_PKU_REWARD_DEVICE" if axis == "reward" else "LAMBDAMIX_PKU_COST_DEVICE"
            )
            device = _resolve_pku_device(axis_device, cuda_available=torch.cuda.is_available())
            dtype = os.environ.get("LAMBDAMIX_PKU_DTYPE", "bfloat16")
            cls._cache[model_id] = BeaverScorer(model_id, device=device, dtype=dtype)
        return cls._cache[model_id]

    @classmethod
    def clear(cls):
        cls._cache.clear()
        paired_cache = globals().get("_PKUPairedScoreCache")
        if paired_cache is not None:
            paired_cache.clear()


def _resolve_pku_device(axis_device: str | None, *, cuda_available: bool) -> str:
    """Resolve an evaluator device, including distributed rank-local maps."""

    if axis_device:
        return axis_device
    device_map = os.environ.get("LAMBDAMIX_PKU_DEVICE_MAP", "").strip()
    if device_map:
        devices = [value.strip() for value in device_map.split(",") if value.strip()]
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank < 0 or local_rank >= len(devices):
            raise RuntimeError(
                "LAMBDAMIX_PKU_DEVICE_MAP has no evaluator device for "
                f"LOCAL_RANK={local_rank}: {devices}"
            )
        return devices[local_rank]
    return os.environ.get("LAMBDAMIX_PKU_DEVICE") or ("cuda" if cuda_available else "cpu")


def _beaver_texts(prompts, completions) -> list[str]:
    """Render each (prompt, completion) into the safe-rlhf scoring string."""
    from lambdamix.tasks.pku import beaver_score_text

    responses = [completion[0]["content"] for completion in completions]
    return [
        beaver_score_text(_extract_raw_prompt(prompt), response)
        for prompt, response in zip(prompts, responses)
    ]


class _PKUPairedScoreCache:
    """Score the two frozen Beaver axes concurrently on separate accelerators."""

    _key: tuple | None = None
    _scores: dict[str, list[float]] = {}

    @classmethod
    def clear(cls) -> None:
        cls._key = None
        cls._scores = {}

    @classmethod
    def enabled(cls) -> bool:
        return os.environ.get("LAMBDAMIX_PKU_PARALLEL_AXES", "0") == "1"

    @classmethod
    def _batch_size(cls, axis: str) -> int:
        specific = os.environ.get(
            "LAMBDAMIX_PKU_REWARD_BATCH_SIZE"
            if axis == "reward"
            else "LAMBDAMIX_PKU_COST_BATCH_SIZE"
        )
        return int(specific or os.environ.get("LAMBDAMIX_PKU_BATCH_SIZE", "8"))

    @classmethod
    def score(cls, axis: str, texts: list[str], max_len: int) -> list[float]:
        if not cls.enabled():
            scorer = _BeaverModelCache.get(axis)
            return scorer.score(texts, cls._batch_size(axis), max_len)

        reward_device = os.environ.get("LAMBDAMIX_PKU_REWARD_DEVICE")
        cost_device = os.environ.get("LAMBDAMIX_PKU_COST_DEVICE")
        if not reward_device or not cost_device or reward_device == cost_device:
            raise RuntimeError(
                "LAMBDAMIX_PKU_PARALLEL_AXES=1 requires distinct "
                "LAMBDAMIX_PKU_REWARD_DEVICE and LAMBDAMIX_PKU_COST_DEVICE values"
            )

        key = (tuple(texts), max_len)
        if cls._key != key:
            scorers = {
                "reward": _BeaverModelCache.get("reward"),
                "cost": _BeaverModelCache.get("cost"),
            }

            def run(which: str) -> list[float]:
                return scorers[which].score(texts, cls._batch_size(which), max_len)

            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="pku-beaver") as pool:
                futures = {which: pool.submit(run, which) for which in ("reward", "cost")}
                scores = {which: future.result() for which, future in futures.items()}
            cls._key = key
            cls._scores = scores
        return list(cls._scores[axis])


def pku_helpfulness_reward_func(prompts, completions, **kwargs) -> list[float]:
    """Helpfulness = released beaver-7b-unified-reward end score (higher = more helpful)."""
    max_len = int(os.environ.get("LAMBDAMIX_PKU_MAX_LEN", "2048"))
    return _PKUPairedScoreCache.score("reward", _beaver_texts(prompts, completions), max_len)


def pku_harmlessness_reward_func(prompts, completions, **kwargs) -> list[float]:
    """Harmlessness = NEGATIVE released beaver-7b-unified-cost end score (higher = safer)."""
    max_len = int(os.environ.get("LAMBDAMIX_PKU_MAX_LEN", "2048"))
    costs = _PKUPairedScoreCache.score("cost", _beaver_texts(prompts, completions), max_len)
    return [-c for c in costs]


_DEFAULT_SYNTHPAI_JUDGE_MODEL = "Qwen/Qwen3-8B"


class _SynthPAIJudgeCache:
    """Lazy-load the released Qwen3-8B v2 attribute-inference judge on first use."""

    _cache: dict = {}
    judge_model_id: str = os.environ.get(
        "LAMBDAMIX_SYNTHPAI_JUDGE_MODEL", _DEFAULT_SYNTHPAI_JUDGE_MODEL
    )

    @classmethod
    def get(cls):
        model_id = cls.judge_model_id
        if model_id not in cls._cache:
            import torch

            from lambdamix.tasks.synthpai import SYNTHPAI_JUDGE_MAX_NEW_TOKENS, SynthPAIJudge

            device = os.environ.get("LAMBDAMIX_SYNTHPAI_JUDGE_DEVICE") or (
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            dtype = os.environ.get("LAMBDAMIX_SYNTHPAI_JUDGE_DTYPE", "bfloat16")
            max_new_tokens = int(
                os.environ.get(
                    "LAMBDAMIX_SYNTHPAI_JUDGE_MAX_TOKENS", str(SYNTHPAI_JUDGE_MAX_NEW_TOKENS)
                )
            )
            cls._cache[model_id] = SynthPAIJudge(
                model_id, device=device, dtype=dtype, max_new_tokens=max_new_tokens
            )
        return cls._cache[model_id]

    @classmethod
    def clear(cls):
        cls._cache.clear()


def _synthpai_parsed_rewrites(completions):
    """Parse each completion under the explicitly selected SynthPAI protocol."""
    from lambdamix.tasks.synthpai import parse_anonymization

    responses = [completion[0]["content"] for completion in completions]
    return [parse_anonymization(response) for response in responses]


def synthpai_utility_reward_func(prompts, completions, original_text=None, **kwargs) -> list[float]:
    """Utility = ROUGE-1 F-measure between the original comment and the parsed rewrite."""
    from lambdamix.tasks.synthpai import rouge1_fmeasure

    if original_text is None:
        raise ValueError("synthpai_utility requires the 'original_text' dataset column")
    parsed = _synthpai_parsed_rewrites(completions)
    return [
        rouge1_fmeasure(str(original), item.text) if item.valid else 0.0
        for item, original in zip(parsed, original_text)
    ]


def synthpai_privacy_reward_func(
    prompts,
    completions,
    target_attribute=None,
    inferred_values=None,
    original_text=None,
    example_id=None,
    **kwargs,
) -> list[float]:
    """Privacy = 1 - inferable(parsed rewrite), judged by the Qwen3-8B v2 adversary."""
    from lambdamix.tasks.synthpai import SynthPAIExample

    if target_attribute is None or inferred_values is None:
        raise ValueError(
            "synthpai_privacy requires the 'target_attribute' and 'inferred_values' columns"
        )
    parsed = _synthpai_parsed_rewrites(completions)
    n = len(parsed)
    example_id = example_id if example_id is not None else [""] * n
    original_text = original_text if original_text is not None else [""] * n

    valid_indices = [i for i, item in enumerate(parsed) if item.valid]
    rewards = [0.0] * n
    if not valid_indices:
        return rewards

    examples = [
        SynthPAIExample(
            example_id=str(example_id[i]),
            text=str(original_text[i]),
            target_attribute=str(target_attribute[i]),
            inferred_values=tuple(str(value) for value in inferred_values[i]),
        )
        for i in valid_indices
    ]
    anonymized = [parsed[i].text for i in valid_indices]
    batch_size = int(os.environ.get("LAMBDAMIX_SYNTHPAI_JUDGE_BATCH_SIZE", "8"))
    verdicts = _SynthPAIJudgeCache.get().judge(examples, anonymized, batch_size=batch_size)
    for position, index in enumerate(valid_indices):
        inferred = verdicts[position]
        rewards[index] = float(not bool(inferred)) if inferred is not None else 0.0
    return rewards


def render_constraint_instruction(constraint: dict) -> str:
    """Render one constraint spec into the natural-language instruction shown to the model."""
    ctype = constraint.get("type")
    if ctype == "json":
        return "Format your entire response as a single valid JSON object."
    if ctype == "max_words":
        return f"Use at most {int(constraint['n'])} words in your response."
    if ctype == "max_sentences":
        return f"Use at most {int(constraint['n'])} sentences in your response."
    if ctype == "keyword":
        return f'Include the word "{constraint["word"]}" somewhere in your response.'
    if ctype == "no_commas":
        return "Do not use any commas anywhere in your response."
    if ctype == "num_bullets":
        return (
            f"Structure your response as exactly {int(constraint['n'])} bullet points, "
            "each starting with '* '."
        )
    raise ValueError(f"Unknown IFEval constraint type: {ctype!r}")


def _count_words(text: str) -> int:
    """Count word tokens for response-constraint checking."""
    return len(re.findall(r"\b\w+\b", text))


def _count_sentences(text: str) -> int:
    """Approximate sentence count by splitting on terminal punctuation."""
    parts = re.split(r"[.!?]+", text)
    return len([p for p in parts if p.strip()])


def _is_single_json(text: str) -> bool:
    """Check whether the full response is a parseable JSON value."""
    t = text.strip()
    if not t:
        return False
    try:
        json.loads(t)
        return True
    except (json.JSONDecodeError, ValueError):
        return False


def _count_bullets(text: str) -> int:
    """Count bullet lines using the requested asterisk prefix."""
    return len(re.findall(r"(?m)^[ \t]*\*[ \t]+\S", text))


def _check_constraint(text: str, constraint: dict) -> bool:
    """Verify a single IFEval-style constraint against a response (returns bool)."""
    ctype = constraint.get("type")
    if ctype == "json":
        return _is_single_json(text)
    if ctype == "max_words":
        return _count_words(text) <= int(constraint["n"])
    if ctype == "max_sentences":
        return _count_sentences(text) <= int(constraint["n"])
    if ctype == "keyword":
        return str(constraint["word"]).lower() in text.lower()
    if ctype == "no_commas":
        return "," not in text
    if ctype == "num_bullets":
        return _count_bullets(text) == int(constraint["n"])
    return False


def _parse_constraints(spec) -> list[dict]:
    """Normalize a per-sample constraint spec into a list of constraint dicts."""
    if spec is None:
        return []
    if isinstance(spec, str):
        if not spec.strip():
            return []
        try:
            spec = json.loads(spec)
        except json.JSONDecodeError:
            return []
    if isinstance(spec, dict):
        return [spec]
    if isinstance(spec, list):
        return [c for c in spec if isinstance(c, dict)]
    return []


def ifeval_constraint_reward_func(prompts, completions, constraints=None, **kwargs) -> list[float]:
    """Score the fraction of assigned response constraints satisfied."""
    responses = [completion[0]["content"] for completion in completions]
    if constraints is None:
        return [1.0] * len(responses)
    rewards = []
    for resp, spec in zip(responses, constraints):
        cons = _parse_constraints(spec)
        if not cons:
            rewards.append(1.0)
            continue
        satisfied = sum(1.0 for c in cons if _check_constraint(resp, c))
        rewards.append(satisfied / len(cons))
    return rewards


REWARD_REGISTRY = {
    "correctness": correctness_reward_func,
    "pku_helpfulness": pku_helpfulness_reward_func,
    "pku_harmlessness": pku_harmlessness_reward_func,
    "synthpai_utility": synthpai_utility_reward_func,
    "synthpai_privacy": synthpai_privacy_reward_func,
    "ifeval_constraint": ifeval_constraint_reward_func,
}
