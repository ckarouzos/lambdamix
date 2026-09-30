"""PKU-SafeRLHF pair parsing, prompt fingerprints, and frozen Beaver scoring."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from datasets import Dataset, DatasetInfo

PKU_DATASET_ID = "PKU-Alignment/PKU-SafeRLHF"


SafetyCell = Literal["both_safe", "both_unsafe", "safety_differs"]

BEAVER_PROMPT_TEMPLATE = "BEGINNING OF CONVERSATION: USER: {prompt} ASSISTANT:"


def beaver_score_text(prompt: str, response: str) -> str:
    """Full safe-rlhf-format string a beaver score model consumes for one turn."""

    return f"{BEAVER_PROMPT_TEMPLATE.format(prompt=prompt.strip())}{response.strip()}"


SAFETY_SYSTEM_PROMPT_FALLBACK = "You are a helpful assistant. /no_think"


@dataclass(frozen=True)
class PKUPair:
    """One released PKU-SafeRLHF annotated response pair (no new labels)."""

    index: int
    prompt: str
    response_0: str
    response_1: str
    is_response_0_safe: bool
    is_response_1_safe: bool
    better_response_id: int
    safer_response_id: int
    response_0_sha256: str
    response_1_sha256: str


def _require_binary(value: Any, field: str, index: int) -> int:
    if value not in (0, 1):
        raise ValueError(f"row {index}: {field} must be 0 or 1, got {value!r}")
    return int(value)


def make_pku_pair(record: Mapping[str, Any], index: int) -> PKUPair:
    """Convert one released record verbatim, validating only its schema."""

    return PKUPair(
        index=index,
        prompt=str(record["prompt"]),
        response_0=str(record["response_0"]),
        response_1=str(record["response_1"]),
        is_response_0_safe=bool(record["is_response_0_safe"]),
        is_response_1_safe=bool(record["is_response_1_safe"]),
        better_response_id=_require_binary(
            record["better_response_id"], "better_response_id", index
        ),
        safer_response_id=_require_binary(record["safer_response_id"], "safer_response_id", index),
        response_0_sha256=str(record["response_0_sha256"]),
        response_1_sha256=str(record["response_1_sha256"]),
    )


def load_pku_pairs(split: str = "test") -> list[PKUPair]:
    """Load the unchanged released pairwise split from the local HF cache."""

    from datasets import load_dataset

    dataset = load_dataset(PKU_DATASET_ID, split=split)
    return [make_pku_pair(record, index) for index, record in enumerate(dataset)]


def slice_fingerprint(pairs: Sequence[PKUPair]) -> str:
    """Hash an ordered PKU evaluation slice."""

    digest = hashlib.sha256()
    for pair in pairs:
        row = (
            pair.index,
            pair.response_0_sha256,
            pair.response_1_sha256,
            pair.better_response_id,
            pair.safer_response_id,
            int(pair.is_response_0_safe),
            int(pair.is_response_1_safe),
        )
        digest.update(json.dumps(row, sort_keys=True).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def prompt_fingerprint(prompts: Sequence[str]) -> str:
    """Hash an ordered prompt slice."""

    digest = hashlib.sha256()
    for prompt in prompts:
        digest.update(hashlib.sha256(prompt.encode("utf-8")).digest())
    return digest.hexdigest()


class BeaverScorer:
    """Loads a released safe-rlhf score model with plain transformers."""

    def __init__(self, model_id: str, device: str = "cuda", dtype: str = "bfloat16") -> None:
        import torch
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        self.torch = torch
        torch_dtype = getattr(torch, dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        config = AutoConfig.from_pretrained(model_id)
        self.backbone = AutoModel.from_pretrained(model_id, config=config, torch_dtype=torch_dtype)
        self.backbone.to(device).eval()
        self.device = device
        self.score_head = self._load_score_head(model_id, config, torch_dtype).to(device)

    def _load_score_head(self, model_id: str, config: Any, torch_dtype: Any):
        import glob
        import os

        import torch
        from safetensors import safe_open

        hidden = config.hidden_size
        weight = None
        bias = None
        local = model_id
        if not os.path.isdir(local):
            from huggingface_hub import snapshot_download

            offline = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
            local = snapshot_download(
                model_id,
                allow_patterns=["*.safetensors", "*.json"],
                local_files_only=offline,
            )
        for shard in sorted(glob.glob(os.path.join(local, "*.safetensors"))):
            with safe_open(shard, framework="pt") as handle:
                for key in handle.keys():
                    if key.endswith("score_head.weight"):
                        weight = handle.get_tensor(key)
                    elif key.endswith("score_head.bias"):
                        bias = handle.get_tensor(key)
            if weight is not None:
                break
        if weight is None:
            raise RuntimeError(f"no score_head.weight found in {model_id}")
        score_dim = weight.shape[0]
        head = torch.nn.Linear(hidden, score_dim, bias=bias is not None)
        head.weight.data = weight.to(torch_dtype)
        if bias is not None:
            head.bias.data = bias.to(torch_dtype)
        head.eval()
        return head

    def score(
        self, texts: list[str], batch_size: int, max_len: int, append_eos: bool = True
    ) -> list[float]:
        """Score prompt-response pairs at the final EOS token using the released Beaver model."""

        torch = self.torch
        scores: list[float] = []
        eos_id = self.tokenizer.eos_token_id
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                encoded = self.tokenizer(
                    batch,
                    truncation=True,
                    max_length=max_len - 1 if append_eos else max_len,
                )["input_ids"]
                if append_eos:
                    encoded = [ids + [eos_id] for ids in encoded]
                enc = self.tokenizer.pad(
                    {"input_ids": encoded}, padding=True, return_tensors="pt"
                ).to(self.device)
                hidden = self.backbone(**enc).last_hidden_state
                end = enc["attention_mask"].sum(dim=1) - 1  # last real token (right pad)
                pooled = hidden[torch.arange(hidden.shape[0]), end]
                out = self.score_head(pooled)[:, 0]
                scores.extend(out.float().cpu().tolist())
                print(f"  scored {min(start + batch_size, len(texts))}/{len(texts)}", flush=True)
        return scores


PROTOCOL = "pku_official_train_prompt_disjoint_test_v1"


def _identity(prompt: str) -> str:
    return "pku-sha256:" + hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _build_dataset(prompts, indices, split, provenance):
    records = [
        {
            "prompt": [
                {"role": "system", "content": SAFETY_SYSTEM_PROMPT_FALLBACK},
                {"role": "user", "content": prompt},
            ],
            "answer": "",
            "example_id": _identity(prompt),
            "source_split": provenance["source_split"],
            "source_row_indices": indices[prompt],
        }
        for prompt in prompts
    ]
    if not records:
        raise ValueError("PKU selection is empty; refusing to fall back to another split")
    provenance["selected_prompt_count"] = len(prompts)
    provenance["selected_prompt_fingerprint"] = prompt_fingerprint(prompts)
    info = DatasetInfo(description=json.dumps({"pku_data_provenance": provenance}, sort_keys=True))
    return Dataset.from_list(records, info=info, split=split)


def load_pku_online(split="train"):
    """Load unique training prompts or the prompt-disjoint official test pool."""
    if split not in ("train", "test"):
        raise ValueError(f"pku_safety supports only 'train' and 'test', got {split!r}")
    pairs = load_pku_pairs(split)
    indices = {}
    for pair in pairs:
        if not pair.prompt.strip():
            raise ValueError(f"Empty PKU prompt in {split} row {pair.index}")
        indices.setdefault(pair.prompt, []).append(pair.index)
    excluded = set()
    train_fingerprint = None
    if split == "test":
        train_prompts = {p.prompt for p in load_pku_pairs("train")}
        excluded = set(indices) & train_prompts
        train_fingerprint = prompt_fingerprint(sorted(train_prompts, key=_identity))
    prompts = sorted(set(indices) - excluded, key=_identity)
    return _build_dataset(
        prompts,
        indices,
        split,
        {
            "protocol": PROTOCOL,
            "dataset_id": PKU_DATASET_ID,
            "requested_split": split,
            "source_split": split,
            "source_pair_count": len(pairs),
            "source_pair_fingerprint": slice_fingerprint(pairs),
            "source_unique_prompt_count": len(indices),
            "excluded_train_overlap_count": len(excluded),
            "excluded_against_train_fingerprint": train_fingerprint,
            "deduplication": "exact prompt text; first and all duplicate row indices retained",
            "ordering": "ascending SHA256 of verbatim UTF-8 prompt",
            "test_policy": "official test unique prompts minus all official train prompt strings",
            "overlap_definition": "exact text, not semantic or near-duplicate equivalence",
            "system_prompt": SAFETY_SYSTEM_PROMPT_FALLBACK,
            "legacy": False,
        },
    )


def dataset_provenance(dataset):
    """Read persisted selection metadata and fingerprint the actual current rows."""
    description = getattr(getattr(dataset, "info", None), "description", "")
    try:
        provenance = json.loads(description).get("pku_data_provenance")
    except (ValueError, TypeError, AttributeError):
        return None
    if not provenance:
        return None
    ids = list(dataset["example_id"])
    provenance["actual_prompt_count"] = len(ids)
    provenance["actual_example_ids_sha256"] = hashlib.sha256(
        json.dumps(ids, separators=(",", ":")).encode()
    ).hexdigest()
    return provenance


def validate_training(dataset_name, split, dataset, *, resume_from_checkpoint=None):
    """Fail closed when a production run tries to use eval data or an old checkpoint."""
    if dataset_name != "pku_safety":
        return
    current = dataset_provenance(dataset)
    if split != "train" or not current or current.get("protocol") != PROTOCOL:
        raise ValueError("PKU training requires the corrected official train protocol")
    if current["source_split"] != "train" or current["legacy"]:
        raise ValueError("PKU training data provenance is not official train")
    if not resume_from_checkpoint:
        return
    if isinstance(resume_from_checkpoint, bool):
        raise ValueError("PKU resume requires an explicit checkpoint path with verified provenance")
    checkpoint = Path(resume_from_checkpoint)
    manifest_path = checkpoint.parent / "run_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Cannot verify PKU resume provenance at {manifest_path}; start fresh")
    previous = json.loads(manifest_path.read_text()).get("datasets", {}).get("train", {})
    old = previous.get("provenance", {})
    for field in (
        "protocol",
        "source_split",
        "source_pair_fingerprint",
        "selected_prompt_fingerprint",
        "actual_example_ids_sha256",
        "system_prompt",
    ):
        if old.get(field) != current.get(field):
            raise ValueError(f"PKU resume data mismatch ({field}); do not resume legacy training")


PreferenceDirection = Literal["helpful+", "helpful-", "harmless+", "harmless-"]
PREFERENCE_DIRECTIONS: tuple[PreferenceDirection, ...] = (
    "helpful+",
    "helpful-",
    "harmless+",
    "harmless-",
)


def selected_response_ids(
    record: Mapping[str, Any], direction: PreferenceDirection
) -> tuple[int, int]:
    """Return ``(chosen, rejected)`` using the paper's label-reversal rule."""

    if direction not in PREFERENCE_DIRECTIONS:
        raise ValueError(f"unknown Preference Vector direction: {direction!r}")
    label = "better_response_id" if direction.startswith("helpful") else "safer_response_id"
    selected = int(record[label])
    if selected not in (0, 1):
        raise ValueError(f"{label} must be binary, got {selected!r}")
    if direction.endswith("-"):
        selected = 1 - selected
    return selected, 1 - selected


def format_pku_preference_record(
    record: Mapping[str, Any],
    direction: PreferenceDirection,
    *,
    system_prompt: str,
) -> dict[str, Any]:
    """Format one unchanged PKU pair for TRL's conversational DPO contract."""

    chosen_id, rejected_id = selected_response_ids(record, direction)
    chosen = str(record[f"response_{chosen_id}"])
    rejected = str(record[f"response_{rejected_id}"])
    if not chosen.strip() or not rejected.strip() or chosen == rejected:
        raise ValueError("PKU DPO pair must contain two distinct non-empty responses")
    return {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": str(record["prompt"])},
        ],
        "chosen": [{"role": "assistant", "content": chosen}],
        "rejected": [{"role": "assistant", "content": rejected}],
    }


def has_distinct_responses(record: Mapping[str, Any]) -> bool:
    """Whether a released PKU pair supplies an informative DPO comparison."""

    response_0 = str(record.get("response_0", ""))
    response_1 = str(record.get("response_1", ""))
    return bool(response_0.strip() and response_1.strip() and response_0 != response_1)


def preference_dataset_fingerprint(
    records: Iterable[Mapping[str, Any]], direction: PreferenceDirection
) -> str:
    """Fingerprint ordered source identities and the applied direction."""

    digest = hashlib.sha256()
    for index, record in enumerate(records):
        chosen_id, rejected_id = selected_response_ids(record, direction)
        identity = {
            "index": index,
            "direction": direction,
            "chosen_id": chosen_id,
            "rejected_id": rejected_id,
            "response_0_sha256": record.get("response_0_sha256"),
            "response_1_sha256": record.get("response_1_sha256"),
        }
        digest.update(json.dumps(identity, sort_keys=True).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def load_pku_preference_dataset(
    split: str,
    direction: PreferenceDirection,
    *,
    max_samples: int | None = None,
):
    """Load the released PKU split and apply one official DPO direction."""

    from datasets import Dataset, load_dataset

    pass

    source = load_dataset(PKU_DATASET_ID, split=split)
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("max_samples must be positive")
        source = source.select(range(min(max_samples, len(source))))
    eligible_source = [record for record in source if has_distinct_responses(record)]
    rows = [
        format_pku_preference_record(
            record,
            direction,
            system_prompt=SAFETY_SYSTEM_PROMPT_FALLBACK,
        )
        for record in eligible_source
    ]
    if not rows:
        raise ValueError(f"no informative PKU DPO pairs remain in split {split!r}")
    dataset = Dataset.from_list(rows)
    dataset.info.description = (
        f"{PKU_DATASET_ID}:{split}; Preference Vector direction={direction}; "
        f"source_n={len(source)}; eligible_n={len(rows)}; "
        f"degenerate_skipped={len(source) - len(rows)}; "
        f"source_fingerprint={preference_dataset_fingerprint(eligible_source, direction)}"
    )
    return dataset
