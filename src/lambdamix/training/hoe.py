"""Hierarchical Experts (HoE) router with group-relative policy updates."""

from __future__ import annotations

import inspect
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from safetensors.torch import load_file as load_safetensors
from transformers import AutoTokenizer
from trl import GRPOTrainer

from lambdamix.generation import (
    GenerationStopsMixin,
    load_declared_architecture_model,
    resolve_model_source,
)
from lambdamix.prompting import prepare_training_tokenizer
from lambdamix.run_manifest import (
    public_metadata,
    sanitize_checkpoint_metadata,
)
from lambdamix.training.common import (
    _model_source_kwargs,
    _reference_model_init_kwargs,
    finish_run,
    get_reward_functions,
    load_training_data,
    parse_training_args,
    start_run,
)
from lambdamix.training.lambdamix import ParetoExperimentArguments

HOE_METHOD_ID = "hoe_lora_grpo_adaptation"
HOE_PAPER = (
    "https://proceedings.iclr.cc/paper_files/paper/2026/file/"
    "d191ba4c8923ed8fd8935b7c98658b5f-Paper-Conference.pdf"
)


def register_supported_logit_slicing(model_kwarg_keys, model: nn.Module):
    """Expose ``logits_to_keep`` through the HoE wrapper when supported."""

    try:
        supports_slicing = "logits_to_keep" in inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        supports_slicing = False
    if not supports_slicing or "logits_to_keep" in model_kwarg_keys:
        return model_kwarg_keys
    if hasattr(model_kwarg_keys, "add"):
        model_kwarg_keys.add("logits_to_keep")
    elif hasattr(model_kwarg_keys, "append"):
        model_kwarg_keys.append("logits_to_keep")
    else:
        model_kwarg_keys = list(model_kwarg_keys)
        model_kwarg_keys.append("logits_to_keep")
    return model_kwarg_keys


def preference_interpolation_weights(
    preference: torch.Tensor, anchors: torch.Tensor
) -> torch.Tensor:
    """Piecewise-linear barycentric interpolation on a bi-objective simplex."""

    p = preference.detach().float().reshape(-1)
    a = anchors.detach().float()
    if p.numel() != 2 or a.ndim != 2 or a.shape[1] != 2:
        raise ValueError("HoE interpolation currently supports two objectives")
    if torch.any(a < 0) or not torch.allclose(
        a.sum(1), torch.ones(len(a), device=a.device, dtype=a.dtype), atol=1e-5
    ):
        raise ValueError("router anchors must lie on the simplex")
    order = torch.argsort(a[:, 0])
    x = p[0].clamp(0.0, 1.0)
    sorted_x = a[order, 0]
    weights = torch.zeros(len(a), dtype=torch.float32, device=a.device)
    if x <= sorted_x[0]:
        weights[order[0]] = 1.0
        return weights
    if x >= sorted_x[-1]:
        weights[order[-1]] = 1.0
        return weights
    upper_pos = int(torch.searchsorted(sorted_x, x, right=False))
    lower_pos = upper_pos - 1
    lower_x, upper_x = sorted_x[lower_pos], sorted_x[upper_pos]
    upper_weight = (x - lower_x) / (upper_x - lower_x).clamp_min(1e-12)
    weights[order[lower_pos]] = 1.0 - upper_weight
    weights[order[upper_pos]] = upper_weight
    return weights


def online_mirror_descent_update(
    log_weights: torch.Tensor,
    preference: torch.Tensor,
    achieved_rewards: torch.Tensor,
    reference_point: torch.Tensor,
    step_size: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One HoE log-space online mirror-descent update."""

    updated = log_weights.float() + float(step_size) * preference.float() * (
        reference_point.float() - achieved_rewards.float()
    )
    weights = torch.softmax(updated, dim=-1)
    return updated, weights


def factorize_delta(delta: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Rank-r factors ``A,B`` satisfying ``A@B ~= delta.T``."""

    if delta.ndim != 2:
        raise ValueError("LoRA delta must be a matrix")
    u, singular, vh = torch.linalg.svd(delta.float(), full_matrices=False)
    kept = min(rank, singular.numel())
    root = singular[:kept].sqrt()
    a = vh[:kept].T * root.unsqueeze(0)
    b = (u[:, :kept] * root.unsqueeze(0)).T
    return a.to(delta.dtype), b.to(delta.dtype)


def _pcb_normalize(tensor: torch.Tensor, dim: int, *, epsilon: float = 1e-12) -> torch.Tensor:
    """Min-max normalization used by PCB, made finite for constant slices."""

    minimum = tensor.amin(dim=dim, keepdim=True)
    maximum = tensor.amax(dim=dim, keepdim=True)
    spread = maximum - minimum
    normalized = (tensor - minimum) / spread.clamp_min(epsilon)
    return torch.where(spread > epsilon, normalized, torch.zeros_like(normalized))


def _pcb_clamp(
    tensor: torch.Tensor, *, min_ratio: float = 0.0, max_ratio: float = 0.0
) -> torch.Tensor:
    """PCB's row-wise order-statistic clamp."""

    if tensor.ndim != 2:
        raise ValueError("PCB clamp expects a [tasks, parameters] matrix")
    if not 0 <= min_ratio < 1 or not 0 <= max_ratio < 1:
        raise ValueError("PCB clamp ratios must lie in [0,1)")
    width = tensor.shape[1]
    if width == 0:
        raise ValueError("PCB cannot merge empty task vectors")
    lower_index = min(int(width * min_ratio), width - 1)
    upper_index = max(int(width * (1.0 - max_ratio) - 1), 0)
    if lower_index > upper_index:
        raise ValueError("PCB clamp ratios select an empty interval")
    ordered = torch.sort(tensor, dim=1).values
    lower = ordered[:, lower_index].unsqueeze(1)
    upper = ordered[:, upper_index].unsqueeze(1)
    return torch.maximum(torch.minimum(tensor, upper), lower)


def pcb_merge(task_vectors: torch.Tensor, *, pcb_ratio: float = 0.1) -> torch.Tensor:
    """Parameter Competition Balancing merge from the released PCB equations."""

    if task_vectors.ndim != 2 or task_vectors.shape[0] < 2:
        raise ValueError("PCB expects at least two flattened task vectors")
    if not 0 < pcb_ratio <= 1:
        raise ValueError("pcb_ratio must lie in (0,1]")
    values = task_vectors.float()
    task_count = values.shape[0]
    absolute = _pcb_clamp(values.abs(), min_ratio=0.0001, max_ratio=0.0001)
    clamped = values.sign() * absolute
    self_balance = _pcb_normalize(absolute, dim=1).square()
    self_activation = torch.exp(task_count * self_balance)
    cross_activation = torch.tanh(values * values.sum(dim=0, keepdim=True))
    task_balance = self_activation * cross_activation
    retained = _pcb_clamp(task_balance, min_ratio=1.0 - pcb_ratio)
    scale = _pcb_normalize(retained, dim=1)
    denominator = scale.sum(dim=0).clamp_min(1e-12)
    return (clamped * scale).sum(dim=0) / denominator


def dare_task_vectors(
    tensors: list[torch.Tensor], *, drop_rate: float, seed: int
) -> list[torch.Tensor]:
    """Apply deterministic DARE drop-and-rescale to task vectors."""

    if len(tensors) < 2 or not 0 <= drop_rate < 1:
        raise ValueError("DARE requires aligned task vectors and drop_rate in [0,1)")
    shape = tensors[0].shape
    if any(tensor.shape != shape for tensor in tensors):
        raise ValueError("DARE task vectors must have identical shapes")
    processed = []
    for offset, tensor in enumerate(tensors):
        generator = torch.Generator(device="cpu").manual_seed(seed + offset)
        keep = torch.rand(tensor.shape, generator=generator) >= drop_rate
        processed.append(tensor * keep.to(tensor.device, tensor.dtype) / (1.0 - drop_rate))
    return processed


def pcb_dare_midpoint(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    drop_rate: float,
    pcb_ratio: float,
    seed: int,
) -> torch.Tensor:
    """Construct HoE's equal-preference expert: DARE + PCB, before task-SVD."""

    sparse = dare_task_vectors([first, second], drop_rate=drop_rate, seed=seed)
    stacked = torch.stack([tensor.reshape(-1) for tensor in sparse], dim=0)
    return pcb_merge(stacked, pcb_ratio=pcb_ratio).reshape_as(first).to(first.dtype)


def pcb_dare_module_midpoints(
    first: dict[str, torch.Tensor],
    second: dict[str, torch.Tensor],
    modules: list[str],
    *,
    drop_rate: float,
    pcb_ratio: float,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Apply DARE+PCB once to globally flattened model task vectors."""

    if not modules or any(name not in first or name not in second for name in modules):
        raise ValueError("PCB module lists must be non-empty and shared by both endpoints")
    flat_first = torch.cat([first[name].reshape(-1) for name in modules])
    flat_second = torch.cat([second[name].reshape(-1) for name in modules])
    merged = pcb_dare_midpoint(
        flat_first,
        flat_second,
        drop_rate=drop_rate,
        pcb_ratio=pcb_ratio,
        seed=seed,
    )
    result: dict[str, torch.Tensor] = {}
    offset = 0
    for name in modules:
        count = first[name].numel()
        result[name] = merged[offset : offset + count].reshape_as(first[name])
        offset += count
    if offset != merged.numel():  # pragma: no cover - construction invariant
        raise RuntimeError("PCB module split did not consume the merged task vector")
    return result


def _load_adapter_state(path: str) -> tuple[dict[str, torch.Tensor], dict]:
    root = Path(path)
    config = json.loads((root / "adapter_config.json").read_text())
    safe = root / "adapter_model.safetensors"
    binary = root / "adapter_model.bin"
    if safe.is_file():
        state = load_safetensors(str(safe), device="cpu")
    elif binary.is_file():
        state = torch.load(binary, map_location="cpu", weights_only=True)
    else:
        raise FileNotFoundError(f"no PEFT adapter weights below {root}")
    return state, config


def load_lora_factor_pairs(
    path: str,
) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], float]:
    """Load compact LoRA ``(A, B)`` factors without densifying the adapter."""

    state, config = _load_adapter_state(path)
    adapter_rank = int(config["r"])
    alpha = float(config.get("lora_alpha", adapter_rank))
    use_rslora = bool(config.get("use_rslora", False))
    scaling = alpha / (adapter_rank**0.5 if use_rslora else adapter_rank)
    factors: dict[str, dict[str, torch.Tensor]] = {}
    markers = (
        "lora_A.default.weight",
        "lora_B.default.weight",
        "lora_A.weight",
        "lora_B.weight",
    )
    for key, value in state.items():
        marker = next((candidate for candidate in markers if key.endswith(candidate)), None)
        if marker is None:
            continue
        module = key[: -len(marker)].rstrip(".")
        for prefix in ("base_model.model.", "base_model.", "model."):
            if module.startswith(prefix):
                module = module[len(prefix) :]
                break
        kind = "A" if "lora_A" in marker else "B"
        factors.setdefault(module, {})[kind] = value.float()
    result: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for module, pair in factors.items():
        if set(pair) != {"A", "B"}:
            raise ValueError(f"incomplete LoRA factors for {module}")
        result[module] = (pair["A"], pair["B"])
    if not result:
        raise ValueError(f"adapter {path} contained no ordinary LoRA factors")
    return result, scaling


def load_lora_dense_deltas(path: str) -> dict[str, torch.Tensor]:
    """Load a PEFT LoRA adapter as dense per-module deltas on CPU."""

    pairs, scaling = load_lora_factor_pairs(path)
    return {module: (b @ a) * scaling for module, (a, b) in pairs.items()}


def _retain_largest(
    retained: torch.Tensor | None, values: torch.Tensor, count: int
) -> torch.Tensor:
    """Keep the globally largest ``count`` values with bounded memory."""

    candidate_count = min(int(count), values.numel())
    candidates = torch.topk(values.reshape(-1), candidate_count).values
    if retained is not None:
        candidates = torch.cat((retained, candidates))
    if candidates.numel() > count:
        candidates = torch.topk(candidates, int(count)).values
    return candidates


def _randomized_rank_factorization(
    delta: torch.Tensor, rank: int, *, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic low-memory rank-r task-SVD for a dense midpoint."""

    minimum = min(delta.shape)
    if rank >= minimum or delta.numel() <= 1_000_000:
        return factorize_delta(delta, rank)
    q = min(minimum, max(int(rank) + 8, int(rank)))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        u, singular, v = torch.svd_lowrank(delta.float(), q=q, niter=4)
    order = torch.argsort(singular, descending=True)[:rank]
    root = singular[order].sqrt()
    a = v[:, order] * root.unsqueeze(0)
    b = (u[:, order] * root.unsqueeze(0)).T
    return a.to(delta.dtype), b.to(delta.dtype)


def build_memory_bounded_hoe_experts(
    first_path: str,
    second_path: str,
    modules: list[str],
    *,
    rank: int,
    drop_rate: float,
    pcb_ratio: float,
    seed: int,
) -> dict[str, list[tuple[torch.Tensor, torch.Tensor]]]:
    """Build endpoint/midpoint factors with bounded host memory."""

    if not 0 <= drop_rate < 1:
        raise ValueError("DARE drop_rate must lie in [0,1)")
    first, first_scale = load_lora_factor_pairs(first_path)
    second, second_scale = load_lora_factor_pairs(second_path)
    if not modules or any(name not in first or name not in second for name in modules):
        raise ValueError("HoE module lists must be non-empty and shared by both endpoints")
    shapes: dict[str, tuple[int, int]] = {}
    offsets: dict[str, tuple[int, int]] = {}
    total = 0
    for name in modules:
        first_a, first_b = first[name]
        second_a, second_b = second[name]
        shape = (first_b.shape[0], first_a.shape[1])
        if shape != (second_b.shape[0], second_a.shape[1]):
            raise ValueError(f"endpoint shape mismatch for {name}")
        shapes[name] = shape
        count = shape[0] * shape[1]
        offsets[name] = (total, total + count)
        total += count
    if total < 1:
        raise ValueError("HoE adapters contain no dense task-vector elements")

    gib = total * 4 / (1024**3)
    print(
        f"HOE_MEMORY_BOUNDED_INIT modules={len(modules)} dense_elements={total} "
        f"one_task_gib={gib:.3f}",
        flush=True,
    )
    task_vectors = torch.empty((2, total), dtype=torch.float32)
    generators = [
        torch.Generator(device="cpu").manual_seed(int(seed) + offset) for offset in range(2)
    ]
    upper_index = max(int(total * (1.0 - 0.0001)) - 1, 0)
    upper_keep = total - upper_index
    upper_candidates: list[torch.Tensor | None] = [None, None]
    zero_counts = [0, 0]

    for name in modules:
        start, end = offsets[name]
        for task, (pairs, scaling) in enumerate(((first, first_scale), (second, second_scale))):
            a, b = pairs[name]
            dense = (b @ a) * scaling
            keep = torch.rand(dense.shape, generator=generators[task]) >= drop_rate
            dense.mul_(keep).div_(1.0 - drop_rate)
            flat = dense.reshape(-1)
            task_vectors[task, start:end].copy_(flat)
            zero_counts[task] += int(torch.count_nonzero(flat == 0).item())
            upper_candidates[task] = _retain_largest(upper_candidates[task], flat.abs(), upper_keep)

    lower_index = min(int(total * 0.0001), total - 1)
    lower = []
    upper = []
    for task in range(2):
        if zero_counts[task] <= lower_index:
            raise RuntimeError(
                "memory-bounded PCB expected DARE zeros to cover the 0.01% "
                f"absolute clamp for task {task}: zeros={zero_counts[task]} "
                f"lower_index={lower_index}"
            )
        lower.append(0.0)
        candidates = upper_candidates[task]
        if candidates is None or candidates.numel() != upper_keep:
            raise RuntimeError("incomplete PCB upper order statistic")
        upper.append(float(candidates.min().item()))
    del upper_candidates

    balance_negative = [0, 0]
    balance_zero = [0, 0]
    balance_max = [float("-inf"), float("-inf")]
    for name in modules:
        start, end = offsets[name]
        values = task_vectors[:, start:end]
        cross_sum = values[0] + values[1]
        module_balances = []
        for task in range(2):
            absolute = values[task].abs().clamp(min=lower[task], max=upper[task])
            span = max(upper[task] - lower[task], 1e-12)
            self_balance = (absolute - lower[task]) / span
            row = torch.exp(2.0 * self_balance.square()) * torch.tanh(values[task] * cross_sum)
            module_balances.append(row)
            balance_negative[task] += int(torch.count_nonzero(row < 0).item())
            balance_zero[task] += int(torch.count_nonzero(row == 0).item())
            balance_max[task] = max(balance_max[task], float(row.max().item()))
        for task, row in enumerate(module_balances):
            task_vectors[task, start:end].copy_(row)

    retained_index = min(int(total * (1.0 - pcb_ratio)), total - 1)
    retained_lower = []
    for task in range(2):
        negative = balance_negative[task]
        zero = balance_zero[task]
        if not negative <= retained_index < negative + zero:
            raise RuntimeError(
                "memory-bounded PCB requires its globally retained lower bound "
                "to be the DARE zero mass; refusing an approximate selection: "
                f"task={task} negative={negative} zero={zero} "
                f"retained_index={retained_index}"
            )
        retained_lower.append(0.0)

    generators = [
        torch.Generator(device="cpu").manual_seed(int(seed) + offset) for offset in range(2)
    ]
    experts: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {}
    for module_index, name in enumerate(modules):
        start, end = offsets[name]
        clamped_values = []
        for task, (pairs, scaling) in enumerate(((first, first_scale), (second, second_scale))):
            endpoint_a, endpoint_b = pairs[name]
            dense = (endpoint_b @ endpoint_a) * scaling
            keep = torch.rand(dense.shape, generator=generators[task]) >= drop_rate
            dense.mul_(keep).div_(1.0 - drop_rate)
            absolute = dense.abs().clamp(min=lower[task], max=upper[task])
            clamped_values.append((dense.sign() * absolute).reshape(-1))
        scales = []
        for task in range(2):
            retained = task_vectors[task, start:end].clamp(
                min=retained_lower[task], max=balance_max[task]
            )
            span = max(balance_max[task] - retained_lower[task], 1e-12)
            scales.append((retained - retained_lower[task]) / span)
        scale_matrix = torch.stack(scales)
        denominator = scale_matrix.sum(dim=0).clamp_min(1e-12)
        midpoint = (torch.stack(clamped_values) * scale_matrix).sum(dim=0) / denominator
        midpoint = midpoint.reshape(shapes[name])
        midpoint_factors = _randomized_rank_factorization(
            midpoint, rank, seed=int(seed) + 10_000 + module_index
        )
        first_a, first_b = first[name]
        second_a, second_b = second[name]
        endpoint_first = (first_a.T.contiguous(), (first_b.T * first_scale).contiguous())
        endpoint_second = (second_a.T.contiguous(), (second_b.T * second_scale).contiguous())
        experts[name] = [endpoint_first, midpoint_factors, endpoint_second]
    del task_vectors
    return experts


class HoELoRALinear(nn.Module):
    """Frozen LoRA experts, lightweight router experts, and preference routing."""

    def __init__(
        self,
        base_layer: nn.Linear,
        expert_deltas: list[torch.Tensor | tuple[torch.Tensor, torch.Tensor]],
        expert_preferences: torch.Tensor,
        router_preferences: torch.Tensor,
        *,
        rank: int,
        top_k: int = 2,
        router_init_scale: float = 4.0,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.num_experts = len(expert_deltas)
        self.top_k = min(int(top_k), self.num_experts)
        self.register_buffer("expert_preferences", expert_preferences.float().clone())
        self.register_buffer("router_preferences", router_preferences.float().clone())
        all_preferences = torch.cat([expert_preferences, router_preferences], dim=0)
        self.register_buffer("preference", all_preferences.float().mean(0), persistent=False)
        if expert_preferences.shape != (self.num_experts, 2):
            raise ValueError("HoE expects one bi-objective preference per LoRA expert")
        if router_preferences.ndim != 2 or router_preferences.shape[1] != 2:
            raise ValueError("HoE router preferences must be bi-objective vectors")
        if len(router_preferences) < 1:
            raise ValueError("HoE requires at least one router expert")
        pairwise = torch.cdist(all_preferences.float(), all_preferences.float())
        duplicate = torch.triu(pairwise < 1e-6, diagonal=1).any()
        if bool(duplicate):
            raise ValueError("HoE expert and router preferences must be distinct")
        self.expert_a = nn.ParameterList()
        self.expert_b = nn.ParameterList()
        for delta in expert_deltas:
            if isinstance(delta, tuple):
                a, b = delta
                if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
                    raise ValueError("pre-factorized HoE expert has incompatible A/B shapes")
                if a.shape[0] != base_layer.in_features or b.shape[1] != base_layer.out_features:
                    raise ValueError("pre-factorized HoE expert does not match its base layer")
            else:
                a, b = factorize_delta(delta, rank)
            self.expert_a.append(
                nn.Parameter(
                    a.to(device=base_layer.weight.device, dtype=base_layer.weight.dtype),
                    requires_grad=False,
                )
            )
            self.expert_b.append(
                nn.Parameter(
                    b.to(device=base_layer.weight.device, dtype=base_layer.weight.dtype),
                    requires_grad=False,
                )
            )
        router_count = len(router_preferences)
        self.router_weight = nn.Parameter(
            torch.zeros(
                router_count,
                self.num_experts,
                base_layer.in_features,
                dtype=torch.float32,
                device=base_layer.weight.device,
            )
        )
        initial_bias = -float(router_init_scale) * torch.cdist(
            router_preferences.float(), expert_preferences.float(), p=1
        )
        self.router_bias = nn.Parameter(initial_bias.to(base_layer.weight.device))
        nearest = torch.topk(
            torch.cdist(router_preferences.float(), expert_preferences.float()),
            k=self.top_k,
            largest=False,
            dim=1,
        ).indices
        router_mask = torch.full(
            (router_count, self.num_experts), float("-inf"), dtype=torch.float32
        )
        router_mask.scatter_(1, nearest.cpu(), 0.0)
        self.register_buffer("router_expert_mask", router_mask)
        self._adapter_enabled = True

    def set_preference(self, preference: torch.Tensor) -> None:
        self.preference = preference.detach().float().reshape(-1).to(self.preference.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(x)
        if not self._adapter_enabled:
            return base
        component_preferences = torch.cat([self.expert_preferences, self.router_preferences], dim=0)
        component_weights = preference_interpolation_weights(
            self.preference, component_preferences
        ).to(x.device)
        static_weights = component_weights[: self.num_experts]
        router_component_weights = component_weights[self.num_experts :]
        router_logits = (
            torch.einsum("...i,rei->...re", x.float(), self.router_weight) + self.router_bias
        )
        router_logits = router_logits + self.router_expert_mask.to(router_logits.device)
        router_votes = torch.softmax(router_logits, dim=-1)
        mixture = static_weights + torch.einsum(
            "r,...re->...e", router_component_weights, router_votes
        )
        mixture = mixture.to(base.dtype)
        expert_outputs = []
        for a, b in zip(self.expert_a, self.expert_b):
            expert_outputs.append((x.to(a.dtype) @ a @ b).to(base.dtype))
        stacked = torch.stack(expert_outputs, dim=-2)
        return base + (stacked * mixture.unsqueeze(-1)).sum(dim=-2)


class HoEPolicy(nn.Module):
    """Frozen base/expert policy whose only trainables are router banks."""

    def __init__(
        self,
        base_model: nn.Module,
        endpoint_adapter_paths: list[str],
        router_preferences: list[list[float]],
        *,
        rank: int = 8,
        top_k: int = 2,
        dare_drop_rate: float = 0.9,
        pcb_ratio: float = 0.1,
        router_init_scale: float = 4.0,
        seed: int = 42,
    ):
        super().__init__()
        if len(endpoint_adapter_paths) != 2:
            raise ValueError("bi-objective HoE requires exactly two endpoint adapters")
        self.base_model = base_model
        self.endpoint_adapter_paths = list(endpoint_adapter_paths)
        self.router_preferences = torch.tensor(router_preferences, dtype=torch.float32)
        self.expert_preferences = torch.tensor(
            [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]], dtype=torch.float32
        )
        self.hoe_rank = int(rank)
        self.hoe_top_k = int(top_k)
        self.hoe_dare_drop_rate = float(dare_drop_rate)
        self.hoe_pcb_ratio = float(pcb_ratio)
        self.hoe_router_init_scale = float(router_init_scale)
        self.hoe_seed = int(seed)
        first, _ = load_lora_factor_pairs(endpoint_adapter_paths[0])
        second, _ = load_lora_factor_pairs(endpoint_adapter_paths[1])
        common = sorted(set(first) & set(second))
        if not common:
            raise ValueError("endpoint adapters have no common LoRA modules")
        dense_elements = sum(first[name][1].shape[0] * first[name][0].shape[1] for name in common)
        if dense_elements <= 10_000_000:
            dense_first = load_lora_dense_deltas(endpoint_adapter_paths[0])
            dense_second = load_lora_dense_deltas(endpoint_adapter_paths[1])
            midpoints = pcb_dare_module_midpoints(
                dense_first,
                dense_second,
                common,
                drop_rate=dare_drop_rate,
                pcb_ratio=pcb_ratio,
                seed=seed,
            )
            expert_factors = {
                name: [dense_first[name], midpoints[name], dense_second[name]] for name in common
            }
        else:
            expert_factors = build_memory_bounded_hoe_experts(
                endpoint_adapter_paths[0],
                endpoint_adapter_paths[1],
                common,
                rank=rank,
                drop_rate=dare_drop_rate,
                pcb_ratio=pcb_ratio,
                seed=seed,
            )
        replacements: list[HoELoRALinear] = []
        names: list[str] = []
        modules = dict(base_model.named_modules())
        for logical_name in common:
            candidates = [
                name for name in modules if name == logical_name or name.endswith(logical_name)
            ]
            candidates = [name for name in candidates if isinstance(modules[name], nn.Linear)]
            if len(candidates) != 1:
                raise ValueError(
                    f"cannot uniquely match expert module {logical_name!r}: {candidates}"
                )
            name = candidates[0]
            wrapper = HoELoRALinear(
                modules[name],
                expert_factors[logical_name],
                self.expert_preferences,
                self.router_preferences,
                rank=rank,
                top_k=top_k,
                router_init_scale=router_init_scale,
            )
            if "." in name:
                parent_name, attribute = name.rsplit(".", 1)
                parent = modules[parent_name]
            else:
                attribute = name
                parent = base_model
            setattr(parent, attribute, wrapper)
            replacements.append(wrapper)
            names.append(name)
        self._hoe_layers = nn.ModuleList(replacements)
        self._hoe_layer_names = names
        for parameter in self.base_model.parameters():
            parameter.requires_grad_(False)
        for layer in self._hoe_layers:
            layer.router_weight.requires_grad_(True)
            layer.router_bias.requires_grad_(True)

    @property
    def config(self):
        return self.base_model.config

    @property
    def peft_config(self) -> dict:
        """Expose TRL's exact no-copy reference protocol."""

        return {}

    @contextmanager
    def disable_adapter(self):
        """Temporarily expose the frozen base policy for KL scoring."""

        previous = [layer._adapter_enabled for layer in self._hoe_layers]
        for layer in self._hoe_layers:
            layer._adapter_enabled = False
        try:
            yield
        finally:
            for layer, enabled in zip(self._hoe_layers, previous):
                layer._adapter_enabled = enabled

    def set_preference(self, preference: torch.Tensor) -> None:
        for layer in self._hoe_layers:
            layer.set_preference(preference)

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.base_model.generate(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)


def hoe_router_state_dict(policy: HoEPolicy) -> dict[str, torch.Tensor]:
    """Return one canonical copy of every trainable HoE router tensor."""

    state: dict[str, torch.Tensor] = {}
    for index, layer in enumerate(policy._hoe_layers):
        for parameter_name in ("router_weight", "router_bias"):
            state[f"_hoe_layers.{index}.{parameter_name}"] = (
                getattr(layer, parameter_name).detach().cpu().clone()
            )
    return state


def load_hoe_router_state(policy: HoEPolicy, state: dict[str, torch.Tensor]) -> None:
    """Restore canonical or legacy alias-heavy HoE router checkpoints."""

    invalid = sorted(
        key for key in state if not (key.endswith(".router_weight") or key.endswith(".router_bias"))
    )
    if invalid:
        raise RuntimeError(f"HoE router checkpoint contains non-router keys: {invalid}")

    missing: list[str] = []
    for index, (layer_name, layer) in enumerate(zip(policy._hoe_layer_names, policy._hoe_layers)):
        for parameter_name in ("router_weight", "router_bias"):
            canonical = f"_hoe_layers.{index}.{parameter_name}"
            candidates = [
                canonical,
                f"base_model.{layer_name}.{parameter_name}",
            ]
            key = next((candidate for candidate in candidates if candidate in state), None)
            if key is None:
                suffix = f".{layer_name}.{parameter_name}"
                matches = [name for name in state if name.endswith(suffix)]
                if len(matches) == 1:
                    key = matches[0]
            if key is None:
                missing.append(canonical)
                continue
            source = state[key]
            target = getattr(layer, parameter_name)
            if not isinstance(source, torch.Tensor) or source.shape != target.shape:
                shape = getattr(source, "shape", None)
                raise RuntimeError(
                    f"HoE router checkpoint shape mismatch for {canonical}: "
                    f"expected {tuple(target.shape)}, found {shape}"
                )
            target.data.copy_(source.to(device=target.device, dtype=target.dtype))
    if missing:
        raise RuntimeError(f"HoE router checkpoint is missing tensors: {missing}")


class HoEGRPOTrainer(GenerationStopsMixin, GRPOTrainer):
    """Group-relative adaptation of HoE router PPO with OMD scalarization."""

    def __init__(
        self,
        *,
        hoe_model: HoEPolicy,
        reward_func_names: list[str],
        dataset_name: str | None,
        router_preferences: list[list[float]],
        omd_step_size: float = 0.1,
        reference_point: list[float] | None = None,
        **kwargs,
    ):
        args = kwargs.get("args")
        if args is not None and getattr(args, "use_vllm", False):
            raise ValueError("HoE routers require native generation")
        self.hoe_model = hoe_model
        self._reward_func_names = list(reward_func_names)
        self._dataset_name = dataset_name
        self._anchors = torch.tensor(router_preferences, dtype=torch.float32)
        self._omd_log_weights = torch.zeros_like(self._anchors)
        self._omd_step_size = float(omd_step_size)
        self._reference = torch.tensor(
            reference_point or [1.0] * len(reward_func_names), dtype=torch.float32
        )
        self._rewards_per_func: torch.Tensor | None = None
        self._current_router = 0
        kwargs["model"] = hoe_model
        requested_beta = float(args.beta) if args is not None else 0.0
        if args is not None:
            args.beta = 0.0
        try:
            super().__init__(**kwargs)
        finally:
            if args is not None:
                args.beta = requested_beta
        self.beta = requested_beta
        self.model_kwarg_keys = register_supported_logit_slicing(
            self.model_kwarg_keys, hoe_model.base_model
        )
        if self.ref_model is not None:
            raise RuntimeError("HoE no-copy KL initialization unexpectedly allocated a reference")
        print(
            "HOE_KL_REFERENCE mode=adapter_disabled_no_copy "
            f"beta={requested_beta:g} ref_model_allocated=false",
            flush=True,
        )

    def _calculate_rewards(self, *args, **kwargs):
        rewards = super()._calculate_rewards(*args, **kwargs)
        self._rewards_per_func = rewards.clone()
        return rewards

    def _generate_and_score_completions(self, inputs):
        mode = "train" if self.model.training else "eval"
        router = (
            self.state.global_step % len(self._anchors)
            if mode == "train"
            else len(self._anchors) // 2
        )
        self._current_router = router
        preference = self._anchors[router]
        self.hoe_model.set_preference(preference)
        output = super()._generate_and_score_completions(inputs)
        if self._rewards_per_func is None:
            raise RuntimeError("HoE did not receive the per-objective reward matrix")
        rewards = torch.nan_to_num(self._rewards_per_func.float())
        groups = rewards.reshape(-1, self.args.num_generations, rewards.shape[1])
        objective_adv = (groups - groups.mean(1, keepdim=True)) / (
            groups.std(1, keepdim=True) + 1e-4
        )
        objective_adv = objective_adv.reshape_as(rewards)
        updated, weights = online_mirror_descent_update(
            self._omd_log_weights[router],
            preference,
            rewards.mean(0).detach().cpu(),
            self._reference,
            self._omd_step_size,
        )
        self._omd_log_weights[router] = updated.detach().cpu()
        advantages = (objective_adv * weights.to(objective_adv.device)).sum(1)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-4)
        local_size = len(output["advantages"])
        start = self.accelerator.process_index * local_size
        output["advantages"] = advantages[start : start + local_size]
        self._metrics[mode]["hoe/router_anchor"].append(float(router))
        for index, weight in enumerate(weights):
            self._metrics[mode][f"hoe/omd_weight_{index}"].append(float(weight))
        return output

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        self.hoe_model.set_preference(self._anchors[self._current_router])
        return super().compute_loss(
            model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
        )

    def save_model(self, output_dir: str | None = None, **kwargs):
        if not self.args.should_save:
            return
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        state = hoe_router_state_dict(self.hoe_model)
        torch.save(state, os.path.join(output_dir, "hoe_routers.pt"))
        torch.save(
            {
                "version": 1,
                "omd_log_weights": self._omd_log_weights.detach().cpu().clone(),
            },
            os.path.join(output_dir, "hoe_optimization_state.pt"),
        )
        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)
        metadata = {
            "method_id": HOE_METHOD_ID,
            "display_name": "HoE-LoRA (GRPO adaptation)",
            "paper": HOE_PAPER,
            "adaptation": "PPO value heads replaced by group-relative advantages",
            "endpoint_adapter_paths": self.hoe_model.endpoint_adapter_paths,
            "expert_preferences": self.hoe_model.expert_preferences.tolist(),
            "router_preferences": self._anchors.tolist(),
            "rank": self.hoe_model.hoe_rank,
            "top_k": self.hoe_model.hoe_top_k,
            "dare_drop_rate": self.hoe_model.hoe_dare_drop_rate,
            "pcb_ratio": self.hoe_model.hoe_pcb_ratio,
            "router_init_scale": self.hoe_model.hoe_router_init_scale,
            "expert_seed": self.hoe_model.hoe_seed,
            "omd_step_size": self._omd_step_size,
            "reference_point": self._reference.tolist(),
            "reward_funcs": self._reward_func_names,
            "dataset_name": self._dataset_name,
            "max_completion_length": self.args.max_completion_length,
            "layer_names": self.hoe_model._hoe_layer_names,
            "base_model_name_or_path": getattr(self.hoe_model.config, "_name_or_path", None),
        }
        sanitize_checkpoint_metadata(output_dir)
        with open(os.path.join(output_dir, "hoe_grpo_meta.json"), "w") as handle:
            json.dump(public_metadata(metadata), handle, indent=2)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        """Restore all HoE-specific state before Trainer resumes its state."""

        router_path = os.path.join(resume_from_checkpoint, "hoe_routers.pt")
        optimization_path = os.path.join(resume_from_checkpoint, "hoe_optimization_state.pt")
        if os.path.exists(router_path) or os.path.exists(optimization_path):
            missing = [
                path for path in (router_path, optimization_path) if not os.path.exists(path)
            ]
            if missing:
                raise FileNotFoundError(
                    "HoE continuation requires both router and optimization state; "
                    f"missing: {missing}"
                )
            router_state = torch.load(router_path, map_location="cpu", weights_only=True)
            load_hoe_router_state(self.hoe_model, router_state)
            optimization_state = torch.load(
                optimization_path, map_location="cpu", weights_only=True
            )
            if not isinstance(optimization_state, dict) or optimization_state.get("version") != 1:
                raise RuntimeError("unsupported HoE optimization-state checkpoint")
            omd = optimization_state.get("omd_log_weights")
            if not isinstance(omd, torch.Tensor) or omd.shape != self._omd_log_weights.shape:
                shape = getattr(omd, "shape", None)
                raise RuntimeError(
                    "HoE OMD-state shape mismatch: "
                    f"expected {tuple(self._omd_log_weights.shape)}, found {shape}"
                )
            if not torch.isfinite(omd).all():
                raise RuntimeError("HoE OMD state contains non-finite values")
            self._omd_log_weights.copy_(
                omd.to(
                    device=self._omd_log_weights.device,
                    dtype=self._omd_log_weights.dtype,
                )
            )
            print(
                f"Resumed HoE routers and OMD state from {resume_from_checkpoint}",
                flush=True,
            )
        else:
            super()._load_from_checkpoint(resume_from_checkpoint, model)


@dataclass
class HoEExperimentArguments(ParetoExperimentArguments):
    hoe_endpoint_adapters: list[str] = field(default_factory=list)
    hoe_router_preferences: list[list[float]] = field(default_factory=lambda: [[0.75, 0.25]])
    hoe_rank: int = field(default=8)
    hoe_top_k: int = field(default=2)
    hoe_dare_drop_rate: float = field(default=0.9)
    hoe_pcb_ratio: float = field(default=0.1)
    hoe_router_init_scale: float = field(default=4.0)
    hoe_omd_step_size: float = field(default=0.1)
    hoe_reference_point: list[float] = field(default_factory=list)


def _resolve_endpoint(identifier: str) -> str:
    """Resolve an expert adapter path directly or through configured search roots."""

    direct = Path(identifier)
    if (direct / "adapter_config.json").is_file():
        return str(direct)
    roots = os.environ.get("LAMBDAMIX_HOE_ENDPOINT_ROOTS", "").split(":")
    matches = [
        Path(root) / identifier
        for root in roots
        if root and (Path(root) / identifier / "adapter_config.json").is_file()
    ]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"HoE endpoint {identifier!r} resolved to {len(matches)} checkpoints; "
            "set LAMBDAMIX_HOE_ENDPOINT_ROOTS to a colon-separated, unambiguous list"
        )
    return str(matches[0])


def _reserve_non_policy_devices(training_args) -> int:
    """Keep single-process HoE off evaluator GCDs and out of DataParallel."""

    visible = int(training_args.n_gpu)
    if visible > 1 and int(os.environ.get("WORLD_SIZE", "1")) == 1:
        training_args._n_gpu = 1
        print(
            f"HoE policy restricted to cuda:0; {visible - 1} visible device(s) "
            "reserved for frozen evaluators",
            flush=True,
        )
    return visible


def main() -> None:
    script_args, training_args, model_args = parse_training_args(HoEExperimentArguments)

    if len(script_args.hoe_endpoint_adapters) != 2:
        raise ValueError("hoe_endpoint_adapters must name the two specialist endpoints")

    _reserve_non_policy_devices(training_args)

    train_dataset, eval_dataset = load_training_data(script_args, training_args)
    rewards = get_reward_functions(script_args.reward_funcs)
    if len(rewards) != 2 or script_args.num_objectives != 2:
        raise ValueError("HoE is bi-objective")

    local_model_source = resolve_model_source(model_args.model_name_or_path)
    training_args.model_init_kwargs = _reference_model_init_kwargs(training_args, model_args)
    tokenizer = AutoTokenizer.from_pretrained(
        local_model_source, **_model_source_kwargs(model_args)
    )
    prepare_training_tokenizer(
        tokenizer,
        local_model_source,
        preserve_native_eos=True,
        generation_kwargs=getattr(training_args, "generation_kwargs", None),
        train_dataset=train_dataset,
    )
    model = load_declared_architecture_model(
        local_model_source,
        torch_dtype=torch.bfloat16,
        attn_implementation=model_args.attn_implementation or "sdpa",
        trust_remote_code=model_args.trust_remote_code,
    )
    model.config._name_or_path = local_model_source
    policy_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(policy_device)
    print(f"HOE_MODEL_SOURCE_LOCAL path={local_model_source}", flush=True)
    endpoint_paths = [_resolve_endpoint(value) for value in script_args.hoe_endpoint_adapters]
    hoe_model = HoEPolicy(
        model,
        endpoint_paths,
        script_args.hoe_router_preferences,
        rank=script_args.hoe_rank,
        top_k=script_args.hoe_top_k,
        dare_drop_rate=script_args.hoe_dare_drop_rate,
        pcb_ratio=script_args.hoe_pcb_ratio,
        router_init_scale=script_args.hoe_router_init_scale,
        seed=training_args.seed,
    )
    trainer = HoEGRPOTrainer(
        hoe_model=hoe_model,
        reward_func_names=list(script_args.reward_funcs),
        dataset_name=script_args.dataset_name,
        router_preferences=script_args.hoe_router_preferences,
        omd_step_size=script_args.hoe_omd_step_size,
        reference_point=script_args.hoe_reference_point or None,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        reward_funcs=rewards,
        processing_class=tokenizer,
    )

    manifest_path = start_run(script_args, training_args, model_args, train_dataset, eval_dataset)
    finish_run(trainer, training_args, manifest_path)


if __name__ == "__main__":
    main()
