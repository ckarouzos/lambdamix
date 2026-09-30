"""Explicit initialization contracts for ordinary PEFT LoRA controls."""

from __future__ import annotations

import json
import math
from typing import Any

import torch


def _named_lora_factors(model):
    """Yield PEFT linear LoRA A/B factors without depending on PEFT internals."""

    for module_name, module in model.named_modules():
        lora_a = getattr(module, "lora_A", None)
        lora_b = getattr(module, "lora_B", None)
        if lora_a is None or lora_b is None:
            continue
        for adapter_name, a_module in lora_a.items():
            if adapter_name not in lora_b:
                raise RuntimeError(
                    f"ordinary LoRA module {module_name!r} is missing B for adapter "
                    f"{adapter_name!r}"
                )
            a_weight = getattr(a_module, "weight", None)
            b_weight = getattr(lora_b[adapter_name], "weight", None)
            if a_weight is None or b_weight is None or a_weight.ndim != 2 or b_weight.ndim != 2:
                raise RuntimeError(
                    "rank-scaled ordinary-LoRA control currently supports linear A/B "
                    f"factors only; got module={module_name!r} adapter={adapter_name!r}"
                )
            yield module_name, adapter_name, a_weight, b_weight


def apply_ordinary_lora_a_init(
    model,
    mode: str = "peft",
    *,
    resume_from_checkpoint: str | None = None,
) -> dict[str, Any]:
    """Apply the declared A-factor scale to an already wrapped PEFT model."""

    normalized = str(mode).strip().lower().replace("-", "_")
    aliases = {"standard": "peft", "peft_scale": "peft", "wide": "rank_scaled"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"peft", "rank_scaled"}:
        raise ValueError(f"ordinary_lora_a_init must be 'peft' or 'rank_scaled', got {mode!r}")

    if resume_from_checkpoint:
        contract = {
            "mode": normalized,
            "applied": False,
            "reason": "checkpoint_resume_restores_trained_factors",
            "resume_from_checkpoint": str(resume_from_checkpoint),
        }
        print(f"ORDINARY_LORA_A_INIT {json.dumps(contract, sort_keys=True)}", flush=True)
        return contract

    factors = list(_named_lora_factors(model))
    if not factors:
        raise RuntimeError(
            "ordinary_lora_a_init was requested but no PEFT linear LoRA factors were found"
        )

    rows = []
    with torch.no_grad():
        for module_name, adapter_name, a_weight, b_weight in factors:
            if torch.count_nonzero(b_weight).item() != 0:
                raise RuntimeError(
                    "ordinary LoRA B must be exactly zero before applying the A-init "
                    f"contract: module={module_name!r} adapter={adapter_name!r}"
                )
            rank, in_features = map(int, a_weight.shape)
            multiplier = math.sqrt(in_features / rank)
            before_rms = float(torch.sqrt(torch.mean(a_weight.float().square())))
            if normalized == "rank_scaled":
                a_weight.mul_(multiplier)
            after_rms = float(torch.sqrt(torch.mean(a_weight.float().square())))
            rows.append(
                {
                    "module": module_name,
                    "adapter": adapter_name,
                    "rank": rank,
                    "in_features": in_features,
                    "multiplier": multiplier if normalized == "rank_scaled" else 1.0,
                    "before_rms": before_rms,
                    "after_rms": after_rms,
                }
            )

    contract = {
        "mode": normalized,
        "applied": normalized == "rank_scaled",
        "same_initial_function": True,
        "factor_count": len(rows),
        "input_dimensions": sorted({row["in_features"] for row in rows}),
        "rank_values": sorted({row["rank"] for row in rows}),
        "mean_before_rms": sum(row["before_rms"] for row in rows) / len(rows),
        "mean_after_rms": sum(row["after_rms"] for row in rows) / len(rows),
        "multipliers": sorted({row["multiplier"] for row in rows}),
    }
    print(f"ORDINARY_LORA_A_INIT {json.dumps(contract, sort_keys=True)}", flush=True)
    return contract
