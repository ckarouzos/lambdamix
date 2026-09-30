"""ParetoPO (Pareto Ranking Policy Optimization) advantage computation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import torch
from trl import GRPOTrainer, ScriptArguments

from lambdamix.generation import GenerationStopsMixin, resolve_model_source
from lambdamix.training.common import (
    _build_peft_config,
    _reference_model_init_kwargs,
    finish_run,
    get_reward_functions,
    load_single_process_model_parallel_policy,
    load_training_data,
    parse_training_args,
    reserve_external_reward_devices,
    start_run,
)


def _dominates(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Return True iff objective vector ``a`` Pareto-dominates ``b`` (maximization)."""
    return bool(torch.all(a >= b)) and bool(torch.any(a > b))


def non_dominated_rank(objs: torch.Tensor) -> torch.Tensor:
    """Assign 1-indexed non-dominated ranks to a set of objective vectors."""
    n = objs.shape[0]
    ranks = torch.zeros(n, dtype=torch.long)
    remaining = list(range(n))
    current = 1
    while remaining:
        front = []
        for i in remaining:
            dominated = False
            for j in remaining:
                if i != j and _dominates(objs[j], objs[i]):
                    dominated = True
                    break
            if not dominated:
                front.append(i)
        for i in front:
            ranks[i] = current
        front_set = set(front)
        remaining = [i for i in remaining if i not in front_set]
        current += 1
    return ranks


def _normalize_groups(
    values: torch.Tensor, num_generations: int, mode: str, eps: float = 1e-4
) -> torch.Tensor:
    """Per-group normalization of a (batch,) signal into a GRPO advantage."""
    n = values.shape[0]
    G = num_generations
    if mode == "none" or n == 0 or n % G != 0:
        return values
    num_prompts = n // G
    grouped = values.view(num_prompts, G)
    mean = grouped.mean(dim=1, keepdim=True)
    if mode == "group_center":
        out = grouped - mean
    elif mode == "group_zscore":
        std = grouped.std(dim=1, keepdim=True)
        out = (grouped - mean) / (std + eps)
    else:
        raise ValueError(f"Unknown normalization mode: {mode!r}")
    return out.reshape(-1)


def pareto_ranking_advantages(
    rewards_per_func: torch.Tensor,
    weights: torch.Tensor,
    num_generations: int,
    beta: float = 0.5,
    normalize: str = "group_zscore",
    eps: float = 1e-4,
) -> torch.Tensor:
    """ParetoPO Stage-2 advantages (Eq. 7-9), one advantage per completion."""
    device = rewards_per_func.device
    out_dtype = rewards_per_func.dtype
    rewards = torch.nan_to_num(rewards_per_func).float().cpu()
    w = weights.detach().float().cpu()

    n = rewards.shape[0]
    G = num_generations
    if n == 0 or n % G != 0:
        # Degenerate batch — fall back to a scalarized group advantage.
        rw = (rewards * w.unsqueeze(0)).sum(dim=1)
        return _normalize_groups(rw.to(device=device, dtype=out_dtype), G, normalize, eps)

    num_prompts = n // G
    adv = torch.zeros(n, dtype=torch.float32)
    for p in range(num_prompts):
        sl = slice(p * G, (p + 1) * G)
        objs = rewards[sl]  # (G, m)
        ranks = non_dominated_rank(objs).float()  # 1..Nrank
        n_rank = ranks.max()
        a_base = n_rank - ranks + 1.0  # Eq. 7
        rw = (objs * w.unsqueeze(0)).sum(dim=1)  # r_w = w . r
        r_min, r_max = rw.min(), rw.max()
        if (r_max - r_min) < eps:
            r_hat = torch.full((G,), 0.5)  # Eq. 8 degenerate case
        else:
            r_hat = (rw - r_min) / (r_max - r_min)
        adv[sl] = a_base + beta * (r_hat - 0.5)  # Eq. 9

    return _normalize_groups(adv.to(device=device, dtype=out_dtype), G, normalize, eps)


def hypervolume_2d(points: torch.Tensor, ref_point: tuple[float, float]) -> float:
    """2-D hypervolume dominated by ``points`` w.r.t. a lower-left ``ref_point``."""
    rx, ry = ref_point
    pts = [(float(x), float(y)) for x, y in points if x > rx and y > ry]
    if not pts:
        return 0.0
    # Keep the non-dominated staircase.
    pts.sort(key=lambda pt: (pt[0], pt[1]))
    nd: list[tuple[float, float]] = []
    best_y = -math.inf
    for x, y in reversed(pts):  # x descending
        if y > best_y:
            nd.append((x, y))
            best_y = y
    # nd: x descending, y ascending. Sweep the strip formula.
    hv = 0.0
    prev_y = ry
    for x, y in nd:
        hv += (x - rx) * (y - prev_y)
        prev_y = y
    return hv


class ParetoHVTracker:
    """Maintains the Pareto archive B and the smoothed hypervolume-gain signal."""

    def __init__(
        self,
        num_objectives: int = 2,
        ref_point: tuple[float, float] | None = None,
        gamma: float = 0.5,
    ):
        if num_objectives != 2:
            raise ValueError(
                "ParetoHVTracker implements the 2-D hypervolume used by Testbed 1; "
                f"got num_objectives={num_objectives}."
            )
        self.num_objectives = num_objectives
        self.ref_point = ref_point if ref_point is not None else (0.0, 0.0)
        self.gamma = gamma
        self.archive: list[tuple[float, float]] = []
        self.prev_hv: float = 0.0
        self.ema: float = 0.0

    def _insert(self, point: tuple[float, float]) -> None:
        """Insert ``point`` into the archive, keeping only non-dominated members."""
        px, py = point
        for ax, ay in self.archive:
            if ax >= px and ay >= py and (ax > px or ay > py):
                return  # dominated by an existing member
        self.archive = [
            (ax, ay)
            for ax, ay in self.archive
            if not (px >= ax and py >= ay and (px > ax or py > ay))
        ]
        self.archive.append(point)

    def update(self, batch_objs: torch.Tensor) -> float:
        """Add a batch of objective vectors, update the smoothed HV gain, return r_pareto."""
        objs = torch.nan_to_num(batch_objs).float().cpu()
        for row in objs:
            self._insert((float(row[0]), float(row[1])))
        hv = hypervolume_2d(self.archive, self.ref_point)
        d_hv = hv - self.prev_hv
        self.prev_hv = hv
        # Eq. 5 — smoothed hypervolume gain.
        self.ema = self.gamma * self.ema + (1.0 - self.gamma) * d_hv
        # Eq. 6 — meta-level reward amplification.
        return 0.5 + 1.5 * math.tanh(self.ema)

    def state_dict(self) -> dict:
        return {
            "archive": list(self.archive),
            "prev_hv": self.prev_hv,
            "ema": self.ema,
            "ref_point": self.ref_point,
            "gamma": self.gamma,
        }

    def load_state_dict(self, state: dict) -> None:
        self.archive = [tuple(p) for p in state.get("archive", [])]
        self.prev_hv = state.get("prev_hv", 0.0)
        self.ema = state.get("ema", 0.0)
        self.ref_point = tuple(state.get("ref_point", self.ref_point))
        self.gamma = state.get("gamma", self.gamma)


def hv_scalarized_advantages(
    rewards_per_func: torch.Tensor,
    weights: torch.Tensor,
    num_generations: int,
    r_pareto: float,
    normalize: str = "group_center",
    eps: float = 1e-4,
) -> torch.Tensor:
    """ParetoPO Stage-1 advantages: HV-amplified scalarized reward (Eq. 6)."""
    device = rewards_per_func.device
    out_dtype = rewards_per_func.dtype
    rewards = torch.nan_to_num(rewards_per_func).float().cpu()
    w = weights.detach().float().cpu()
    rw = (rewards * w.unsqueeze(0)).sum(dim=1) * float(r_pareto)
    return _normalize_groups(rw.to(device=device, dtype=out_dtype), num_generations, normalize, eps)


class ParetoPOTrainer(GenerationStopsMixin, GRPOTrainer):
    """GRPOTrainer with the ParetoPO two-stage advantage (arXiv:2606.16111)."""

    def __init__(
        self,
        paretopo_beta: float = 0.5,
        paretopo_gamma: float = 0.5,
        paretopo_stage1_steps: int = 0,
        paretopo_ref_point: tuple[float, float] | None = None,
        paretopo_stage2_normalize: str = "group_zscore",
        paretopo_stage1_normalize: str = "group_center",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._pp_beta = paretopo_beta
        self._pp_stage1_steps = paretopo_stage1_steps
        self._pp_stage2_norm = paretopo_stage2_normalize
        self._pp_stage1_norm = paretopo_stage1_normalize
        num_obj = len(self.reward_funcs)
        self._pp_hv = ParetoHVTracker(
            num_objectives=num_obj, ref_point=paretopo_ref_point, gamma=paretopo_gamma
        )
        # Log the two stages exactly once each.
        self._pp_logged_stage: set[int] = set()

    def _calculate_rewards(self, *args, **kwargs):
        rewards_per_func = super()._calculate_rewards(*args, **kwargs)
        self._rewards_per_func = rewards_per_func.clone()
        return rewards_per_func

    def _generate_and_score_completions(self, inputs):
        output = super()._generate_and_score_completions(inputs)

        rewards = self._rewards_per_func
        weights = self.reward_weights.to(rewards.device)
        step = int(self.state.global_step)

        r_pareto = self._pp_hv.update(rewards) if step < self._pp_stage1_steps else 1.0

        if step < self._pp_stage1_steps:
            stage = 1
            advantages = hv_scalarized_advantages(
                rewards,
                weights,
                self.num_generations,
                r_pareto,
                normalize=self._pp_stage1_norm,
            )
        else:
            stage = 2
            advantages = pareto_ranking_advantages(
                rewards,
                weights,
                self.num_generations,
                beta=self._pp_beta,
                normalize=self._pp_stage2_norm,
            )

        if stage not in self._pp_logged_stage and self.accelerator.is_main_process:
            self._pp_logged_stage.add(stage)
            print(
                f"[ParetoPO] entering Stage {stage} at global_step={step} "
                f"(stage1_steps={self._pp_stage1_steps}, r_pareto={r_pareto:.4f})",
                flush=True,
            )

        # Slice full (gathered) advantages back to this process's shard.
        local_size = len(output["advantages"])
        start = self.accelerator.process_index * local_size
        output["advantages"] = advantages[start : start + local_size]
        return output

    def save_model(self, output_dir=None, **kwargs):
        super().save_model(output_dir, **kwargs)
        if self.args.should_save:
            path = Path(output_dir or self.args.output_dir) / "paretopo_state.json"
            path.write_text(json.dumps(self._pp_hv.state_dict()))

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        super()._load_from_checkpoint(resume_from_checkpoint, model)
        path = Path(resume_from_checkpoint) / "paretopo_state.json"
        if not path.exists() and self._pp_stage1_steps:
            raise FileNotFoundError("ParetoPO resume requires paretopo_state.json")
        if path.exists():
            self._pp_hv.load_state_dict(json.loads(path.read_text()))


@dataclass
class ParetoPOArguments(ScriptArguments):
    """Script arguments for the ParetoPO baseline (config-driven)."""

    reward_funcs: list[str] = field(
        default_factory=lambda: ["correctness", "ifeval_constraint"],
        metadata={"help": "Objective reward function names from REWARD_REGISTRY (>=2)."},
    )
    num_objectives: int = field(
        default=2, metadata={"help": "Number of objectives (Testbed 1 = 2)."}
    )
    model_device_map_auto: bool = field(
        default=False,
        metadata={"help": "Shard a large frozen base across visible accelerators."},
    )
    model_max_memory_gib: int = field(
        default=58,
        metadata={"help": "Per-accelerator HBM ceiling for model_device_map_auto."},
    )
    paretopo_beta: float = field(
        default=0.5, metadata={"help": "Within-rank refinement scaling (Eq. 9; paper 0.5)."}
    )
    paretopo_gamma: float = field(
        default=0.5, metadata={"help": "Hypervolume-gain EMA smoothing (Eq. 5; paper 0.5)."}
    )
    paretopo_stage1_steps: int = field(
        default=0,
        metadata={
            "help": "Optimizer steps of Stage-1 HV-guided scalarization before switching "
            "to Stage-2 Pareto-ranking (paper: one epoch). 0 = pure Stage 2."
        },
    )
    paretopo_ref_point: list[float] | None = field(
        default=None,
        metadata={
            "help": "2-D hypervolume reference (lower-left). Default (0,0) for [0,1] rewards."
        },
    )
    paretopo_stage2_normalize: str = field(
        default="group_zscore",
        metadata={
            "help": "Per-group normalization of Stage-2 advantages (group_zscore/center/none)."
        },
    )
    paretopo_stage1_normalize: str = field(
        default="group_center",
        metadata={
            "help": "Per-group normalization of Stage-1 advantages (group_center/zscore/none)."
        },
    )


def main() -> None:
    script_args, training_args, model_args = parse_training_args(ParetoPOArguments)
    reserve_external_reward_devices(training_args, script_args.dataset_name)
    if not getattr(model_args, "use_peft", False):
        raise ValueError("ParetoPO-LoRA requires use_peft: true")

    train_dataset, eval_dataset = load_training_data(script_args, training_args)
    reward_functions = get_reward_functions(script_args.reward_funcs)
    if len(reward_functions) < 2:
        raise ValueError("ParetoPO requires at least two reward functions")

    local_model_source = resolve_model_source(model_args.model_name_or_path)
    training_args.model_init_kwargs = _reference_model_init_kwargs(training_args, model_args)

    from transformers import AutoTokenizer

    from lambdamix.prompting import prepare_training_tokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        local_model_source,
        trust_remote_code=model_args.trust_remote_code,
    )
    prepare_training_tokenizer(
        tokenizer,
        model_args.model_name_or_path,
        generation_kwargs=training_args.generation_kwargs,
        train_dataset=train_dataset,
    )

    ref_point = None
    if script_args.paretopo_ref_point is not None:
        values = list(script_args.paretopo_ref_point)
        if len(values) != 2:
            raise ValueError("The current ParetoPO implementation requires a 2-D reference")
        ref_point = (float(values[0]), float(values[1]))

    manifest_path = start_run(script_args, training_args, model_args, train_dataset, eval_dataset)

    policy_model = load_single_process_model_parallel_policy(
        local_model_source, script_args, training_args, model_args
    )
    trainer = ParetoPOTrainer(
        model=policy_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        reward_funcs=reward_functions,
        processing_class=tokenizer,
        peft_config=_build_peft_config(model_args),
        paretopo_beta=script_args.paretopo_beta,
        paretopo_gamma=script_args.paretopo_gamma,
        paretopo_stage1_steps=script_args.paretopo_stage1_steps,
        paretopo_ref_point=ref_point,
        paretopo_stage2_normalize=script_args.paretopo_stage2_normalize,
        paretopo_stage1_normalize=script_args.paretopo_stage1_normalize,
    )
    finish_run(trainer, training_args, manifest_path)


if __name__ == "__main__":
    main()
