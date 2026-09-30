"""Prompt rendering helpers for chat-style datasets."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any

from transformers import AutoTokenizer

CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)

CHATML_EOS = "<|im_end|>"

_FORCE_NOTHINK_PREFIX = "{%- set enable_thinking = false %}\n"
# Data-layer sentinel that a dataset loader appended to signal "no native CoT".
_NOTHINK_SENTINEL = "/no_think"


def force_no_thinking(tokenizer) -> bool:
    """Force ``enable_thinking=false`` in a hybrid-thinking chat template (Qwen3)."""
    template = getattr(tokenizer, "chat_template", None)
    if not template or "enable_thinking" not in template:
        return False
    if template.startswith(_FORCE_NOTHINK_PREFIX):
        return True
    tokenizer.chat_template = _FORCE_NOTHINK_PREFIX + template
    return True


def _prompt_text(prompt: Any) -> str:
    """Flatten a dataset ``prompt`` (str / message list / message dict) to text."""
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, dict):
        return str(prompt.get("content", ""))
    if isinstance(prompt, list):
        return " ".join(
            str(m.get("content", "")) if isinstance(m, dict) else str(m) for m in prompt
        )
    return str(prompt)


def dataset_requests_no_thinking(dataset) -> bool:
    """True if the dataset's prompts carry the ``/no_think`` switch (no-CoT intent)."""
    try:
        row = dataset[0]
    except Exception:
        return False
    prompt = row.get("prompt") if isinstance(row, dict) else None
    if prompt is None:
        return False
    return _NOTHINK_SENTINEL in _prompt_text(prompt)


def ensure_chat_template(
    tokenizer,
    fallback_source: str | None = None,
    *,
    default_chatml: bool = False,
    align_eos_to_chatml: bool = True,
) -> None:
    """Populate ``tokenizer.chat_template`` from a fallback tokenizer if needed."""
    if getattr(tokenizer, "chat_template", None):
        return

    if fallback_source is not None:
        try:
            fallback_tokenizer = AutoTokenizer.from_pretrained(fallback_source)
            fallback_template = getattr(fallback_tokenizer, "chat_template", None)
            if fallback_template:
                tokenizer.chat_template = fallback_template
                return
        except Exception:
            pass

    if default_chatml:
        tokenizer.chat_template = CHATML_TEMPLATE
        if align_eos_to_chatml:
            _align_eos_to_chatml(tokenizer)


def prepare_training_tokenizer(
    tokenizer,
    fallback_source: str | None = None,
    *,
    preserve_native_eos: bool = False,
    generation_kwargs: dict[str, Any] | None = None,
    train_dataset: Any | None = None,
) -> None:
    """Install a training chat template with backend-correct EOS semantics."""
    if train_dataset is not None and dataset_requests_no_thinking(train_dataset):
        force_no_thinking(tokenizer)

    ensure_chat_template(
        tokenizer,
        fallback_source,
        default_chatml=True,
        align_eos_to_chatml=not preserve_native_eos,
    )

    if getattr(tokenizer, "chat_template", None) != CHATML_TEMPLATE:
        return

    im_end_id = tokenizer.convert_tokens_to_ids(CHATML_EOS)
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if im_end_id is None or im_end_id == unk_id:
        return

    raw_stop_ids = (generation_kwargs or {}).get("stop_token_ids", [])
    stop_ids = {int(token_id) for token_id in raw_stop_ids}
    if preserve_native_eos and int(im_end_id) not in stop_ids:
        raise ValueError(
            "Template-less model with native-EOS preservation requires the ChatML terminator "
            f"{CHATML_EOS!r} (token id {im_end_id}) in generation_kwargs.stop_token_ids; "
            "the processing tokenizer deliberately retains the model-native EOS."
        )
    print(
        "TRAINING_TERMINATORS "
        f"preserve_native_eos={preserve_native_eos} "
        f"processing_eos={tokenizer.eos_token!r} "
        f"processing_eos_id={tokenizer.eos_token_id} "
        f"chatml_stop_id={im_end_id} "
        f"explicit_stop_ids={sorted(stop_ids)}",
        flush=True,
    )


def _align_eos_to_chatml(tokenizer) -> None:
    """Point the tokenizer EOS at the ChatML turn terminator (``<|im_end|>``)."""
    im_end_id = tokenizer.convert_tokens_to_ids(CHATML_EOS)
    unk_id = getattr(tokenizer, "unk_token_id", None)
    # convert_tokens_to_ids returns the unk id (or None) for unknown tokens.
    if im_end_id is None or im_end_id == unk_id:
        return
    if tokenizer.eos_token_id == im_end_id:
        return
    tokenizer.eos_token = CHATML_EOS


def _coerce_messages(messages: Any) -> list[dict[str, Any]]:
    """Normalize common prompt payload variants into a flat chat message list."""
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]

    if isinstance(messages, dict):
        return [
            {
                "role": str(messages.get("role", "user")).strip() or "user",
                "content": str(messages.get("content", "")),
            }
        ]

    if not isinstance(messages, list):
        return [{"role": "user", "content": str(messages)}]

    normalized: list[dict[str, Any]] = []
    for item in messages:
        if isinstance(item, dict):
            normalized.append(
                {
                    "role": str(item.get("role", "user")).strip() or "user",
                    "content": str(item.get("content", "")),
                }
            )
        elif isinstance(item, list):
            normalized.extend(_coerce_messages(item))
        else:
            normalized.append({"role": "user", "content": str(item)})
    return normalized


def render_chat_prompt(
    tokenizer,
    messages: Any,
    *,
    add_generation_prompt: bool = True,
    fallback_source: str | None = None,
) -> str:
    """Render chat messages using the training tokenizer template."""
    normalized_messages = _coerce_messages(messages)
    ensure_chat_template(tokenizer, fallback_source)

    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("Evaluation requires the checkpoint's training chat template")
    return tokenizer.apply_chat_template(
        normalized_messages, tokenize=False, add_generation_prompt=add_generation_prompt
    )


OBJECTIVE_LABELS = {"correctness": "correctness"}

_WEIGHT_DECIMALS = 2


def objective_label(name: str) -> str:
    """Map a reward-function registry key to its natural-language label."""
    return OBJECTIVE_LABELS.get(name, name)


def render_preference_directive(reward_funcs: Sequence[str], weights: Sequence[float]) -> str:
    """Render a preference vector as a short, deterministic system-prompt directive."""
    if len(reward_funcs) != len(weights):
        raise ValueError(
            f"reward_funcs ({len(reward_funcs)}) and weights ({len(weights)}) "
            "must have equal length."
        )
    parts = [
        f"{objective_label(name)} {float(w):.{_WEIGHT_DECIMALS}f}"
        for name, w in zip(reward_funcs, weights)
    ]
    return "Priorities for this response (weights sum to 1): " + ", ".join(parts) + "."


def inject_preference_into_prompt(prompt, directive: str) -> list[dict]:
    """Return a NEW chat-message list with ``directive`` folded into the system turn."""
    if isinstance(prompt, list):
        messages = copy.deepcopy(prompt)
    else:
        messages = [{"role": "user", "content": str(prompt)}]

    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "system":
            existing = str(msg.get("content", "")).strip()
            msg["content"] = f"{existing}\n\n{directive}".strip()
            return messages

    return [{"role": "system", "content": directive}, *messages]
