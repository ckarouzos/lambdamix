"""MOC-style controllability for a parameter-matched LoRA GRPO policy."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import torch
from peft import LoraConfig
from trl import GRPOTrainer

from lambdamix.adapters.lambdamix import PreferenceSampler
from lambdamix.generation import GenerationStopsMixin
from lambdamix.prompting import inject_preference_into_prompt, prepare_training_tokenizer
from lambdamix.run_manifest import public_metadata
from lambdamix.training.common import (
    _reference_model_init_kwargs,
    finish_run,
    get_reward_functions,
    load_training_data,
    parse_training_args,
    reserve_external_reward_devices,
    start_run,
)
from lambdamix.training.lambdamix import ParetoExperimentArguments

MOC_METHOD_ID = "moc_lora_grpo_adaptation"
MOC_PAPER = "https://arxiv.org/abs/2604.04497"


class RunningRewardNormalizer:
    """MOC Eq. 17: (r - running_mean) / (2 * running_std) + 1."""

    def __init__(self, objectives):
        self.mean = torch.zeros(objectives, dtype=torch.float64)
        self.var = torch.ones(objectives, dtype=torch.float64)
        self.count = 1e-4

    def normalize(self, rewards, *, update=True):
        values = rewards.detach().to(device="cpu", dtype=torch.float64)
        if values.ndim != 2 or values.shape[1] != len(self.mean) or not len(values):
            raise ValueError("Invalid running-reward batch dimensions")
        if not torch.isfinite(values).all():
            raise ValueError("Nonfinite running-reward batch")
        if update:
            n = len(values)
            delta = values.mean(0) - self.mean
            total = self.count + n
            combined = (
                self.var * self.count
                + values.var(0, unbiased=False) * n
                + delta.square() * self.count * n / total
            )
            self.mean = self.mean + delta * n / total
            self.var = (combined / total).clamp_min(0)
            self.count = total
        mean = self.mean.to(device=rewards.device, dtype=rewards.dtype)
        std = self.var.sqrt().clamp_min(1e-8).to(device=rewards.device, dtype=rewards.dtype)
        return (rewards - mean) / (2 * std) + 1

    def state_dict(self):
        return dict(mean=self.mean.tolist(), var=self.var.tolist(), count=self.count)

    def load_state_dict(self, state):
        mean, var = (
            torch.tensor(state["mean"], dtype=torch.float64),
            torch.tensor(state["var"], dtype=torch.float64),
        )
        count = float(state["count"])
        if (
            mean.shape != self.mean.shape
            or var.shape != self.var.shape
            or not torch.isfinite(mean).all()
            or not torch.isfinite(var).all()
            or (var < 0).any()
            or not 0 < count < float("inf")
        ):
            raise ValueError("Invalid MOC running-normalizer checkpoint")
        self.mean, self.var, self.count = mean, var, count


def render_moc_preference(reward_names: list[str], preference: torch.Tensor) -> str:
    """Render the preference prefix used by MOC, e.g."""

    values = preference.detach().float().cpu().reshape(-1).tolist()
    if len(reward_names) != len(values):
        raise ValueError("reward_names and preference must have the same length")
    fields = " ".join(f"<R{i + 1}> {value:.6f}" for i, value in enumerate(values))
    legend = "; ".join(f"R{i + 1}={name}" for i, name in enumerate(reward_names))
    return f"{fields}\nPreference objectives: {legend}."


def normalize_reward_vector(
    rewards: torch.Tensor,
    lower_bounds: torch.Tensor,
    upper_bounds: torch.Tensor,
) -> torch.Tensor:
    """Map reward coordinates to [0, 1] using frozen, declared bounds."""

    if rewards.ndim != 2:
        raise ValueError("rewards must have shape (samples, objectives)")
    lower = lower_bounds.to(device=rewards.device, dtype=rewards.dtype).reshape(1, -1)
    upper = upper_bounds.to(device=rewards.device, dtype=rewards.dtype).reshape(1, -1)
    if lower.shape[1] != rewards.shape[1] or upper.shape[1] != rewards.shape[1]:
        raise ValueError("reward-bound dimensionality does not match rewards")
    span = upper - lower
    if torch.any(span <= 0):
        raise ValueError("every MOC reward upper bound must exceed its lower bound")
    return ((rewards - lower) / span).clamp(0.0, 1.0)


def group_standardized_objective_advantages(
    rewards: torch.Tensor,
    num_generations: int,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Return per-objective GRPO advantages with shape ``(B*G, T)``."""

    if rewards.ndim != 2 or rewards.shape[0] % num_generations:
        raise ValueError("reward rows must be divisible by num_generations")
    groups = rewards.reshape(-1, num_generations, rewards.shape[1])
    mean = groups.mean(dim=1, keepdim=True)
    std = groups.std(dim=1, keepdim=True)
    return ((groups - mean) / (std + eps)).reshape_as(rewards)


def min_norm_two_objective_weight(
    performance_vector: torch.Tensor,
    control_vector: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Closed-form MOC Algorithm 2 weight on the performance objective."""

    v = performance_vector.reshape(-1).float()
    v_bar = control_vector.reshape(-1).float()
    if v.shape != v_bar.shape:
        raise ValueError("the two surrogate vectors must have identical shapes")
    vv = torch.dot(v, v)
    bb = torch.dot(v_bar, v_bar)
    vb = torch.dot(v, v_bar)
    if vb >= vv:
        return v.new_tensor(1.0)
    if vb >= bb:
        return v.new_tensor(0.0)
    denom = torch.dot(v - v_bar, v - v_bar)
    if denom <= eps:
        return v.new_tensor(0.5)
    return (torch.dot(v_bar - v, v_bar) / denom).clamp(0.0, 1.0)


@dataclass(frozen=True)
class MOCAdvantageResult:
    advantages: torch.Tensor
    performance_advantages: torch.Tensor
    control_advantages: torch.Tensor
    performance_weight: torch.Tensor
    reward_mse: torch.Tensor
    control_active: torch.Tensor


def compute_moc_grpo_advantages(
    rewards_per_func: torch.Tensor,
    preference: torch.Tensor,
    num_generations: int,
    *,
    tolerance: float = 0.05,
    lower_bounds: torch.Tensor | None = None,
    upper_bounds: torch.Tensor | None = None,
    normalized_rewards: torch.Tensor | None = None,
) -> MOCAdvantageResult:
    """Compute the MOC two-objective surrogate using group-relative critics."""

    rewards = rewards_per_func.float()
    if not torch.isfinite(rewards).all():
        raise ValueError("Nonfinite MOC training rewards; refusing imputation")
    if rewards.ndim != 2:
        raise ValueError("rewards_per_func must have shape (samples, objectives)")
    objectives = rewards.shape[1]
    p = preference.to(device=rewards.device, dtype=rewards.dtype).reshape(-1)
    if p.numel() != objectives:
        raise ValueError("preference dimensionality does not match rewards")
    if torch.any(p < 0) or not torch.isclose(p.sum(), p.new_tensor(1.0), atol=1e-5):
        raise ValueError("MOC preference must lie on the probability simplex")
    lower = torch.zeros(objectives, device=rewards.device) if lower_bounds is None else lower_bounds
    upper = torch.ones(objectives, device=rewards.device) if upper_bounds is None else upper_bounds
    if normalized_rewards is None:
        normalized_rewards = normalize_reward_vector(rewards, lower, upper)
    elif normalized_rewards.shape != rewards.shape or not torch.isfinite(normalized_rewards).all():
        raise ValueError("Invalid supplied normalized MOC rewards")
    objective_adv = group_standardized_objective_advantages(normalized_rewards, num_generations)

    achieved = normalized_rewards.mean(dim=0)
    residual = achieved - p
    reward_mse = residual.square().mean()
    control_active = reward_mse > float(tolerance)

    performance = (objective_adv * p.unsqueeze(0)).sum(dim=1)
    control = -(2.0 / objectives) * (objective_adv * residual.unsqueeze(0)).sum(dim=1)
    if not bool(control_active):
        control = torch.zeros_like(control)
        c = performance.new_tensor(1.0)
    else:
        c = min_norm_two_objective_weight(performance, control).to(performance.dtype)
    combined = c * performance + (1.0 - c) * control
    combined = (combined - combined.mean()) / (combined.std() + 1e-4)
    return MOCAdvantageResult(
        advantages=combined,
        performance_advantages=performance,
        control_advantages=control,
        performance_weight=c,
        reward_mse=reward_mse,
        control_active=control_active,
    )


class MOCGRPOTrainer(GenerationStopsMixin, GRPOTrainer):
    """Parameter-matched MOC prompt conditioning with GRPO critics."""

    def __init__(
        self,
        *,
        preference_sampler: PreferenceSampler,
        reward_func_names: list[str],
        dataset_name: str | None = None,
        tolerance: float = 0.05,
        reward_lower_bounds: list[float] | None = None,
        reward_upper_bounds: list[float] | None = None,
        reward_normalization: str = "frozen_bounds",
        **kwargs,
    ):
        args = kwargs.get("args")
        if args is not None and getattr(args, "use_vllm", False):
            raise ValueError(
                "MOC-LoRA prompt conditioning requires use_vllm=false unless the "
                "conditioned prompts are forwarded to the rollout engine."
            )
        self.preference_sampler = preference_sampler
        self._reward_func_names = list(reward_func_names)
        self._dataset_name = dataset_name
        self._moc_tolerance = float(tolerance)
        n = len(reward_func_names)
        if reward_normalization not in {"running", "frozen_bounds"}:
            raise ValueError("Unknown MOC reward normalization")
        if reward_normalization == "running" and (reward_lower_bounds or reward_upper_bounds):
            raise ValueError("Running MOC normalization must not specify clipping bounds")
        self._moc_reward_normalization = reward_normalization
        self._moc_running = RunningRewardNormalizer(n)
        self._moc_lower = torch.tensor(reward_lower_bounds or [0.0] * n)
        self._moc_upper = torch.tensor(reward_upper_bounds or [1.0] * n)
        self._rewards_per_func: torch.Tensor | None = None
        self._current_lambda: torch.Tensor | None = None
        super().__init__(**kwargs)

    def _calculate_rewards(self, *args, **kwargs):
        rewards = super()._calculate_rewards(*args, **kwargs)
        self._rewards_per_func = rewards.clone()
        return rewards

    def _generate_and_score_completions(self, inputs):
        mode = "train" if self.model.training else "eval"
        preference = (
            self.preference_sampler.step_lambda(self.state.global_step)
            if mode == "train"
            else self.preference_sampler.get_uniform()
        )
        self._current_lambda = preference
        directive = render_moc_preference(self._reward_func_names, preference)
        for row in inputs:
            row["prompt"] = inject_preference_into_prompt(row["prompt"], directive)

        output = super()._generate_and_score_completions(inputs)
        if self._rewards_per_func is None:
            raise RuntimeError("MOC-LoRA did not receive the per-objective reward matrix")
        normalized = None
        if self._moc_reward_normalization == "running":
            conditioned = self._rewards_per_func.float()
            normalized = self._moc_running.normalize(conditioned, update=mode == "train")
        result = compute_moc_grpo_advantages(
            self._rewards_per_func,
            preference,
            self.args.num_generations,
            tolerance=self._moc_tolerance,
            lower_bounds=self._moc_lower,
            upper_bounds=self._moc_upper,
            normalized_rewards=normalized,
        )
        local_size = len(output["advantages"])
        start = self.accelerator.process_index * local_size
        output["advantages"] = result.advantages[start : start + local_size]
        self._metrics[mode]["moc/reward_mse"].append(float(result.reward_mse))
        self._metrics[mode]["moc/performance_weight"].append(float(result.performance_weight))
        self._metrics[mode]["moc/control_active"].append(float(result.control_active))
        if normalized is not None:
            self._metrics[mode]["moc/running_count"].append(self._moc_running.count)
        return output

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        super()._load_from_checkpoint(resume_from_checkpoint, model=model)
        if self._moc_reward_normalization == "running":
            meta = json.loads((Path(resume_from_checkpoint) / "moc_grpo_meta.json").read_text())
            if meta.get("reward_normalization") != "running":
                raise ValueError("Cannot resume running-normalized MOC from a different protocol")
            self._moc_running.load_state_dict(meta["running_normalizer"])

    def _write_metadata(self, output_dir: str) -> None:
        if not self.args.should_save:
            return
        os.makedirs(output_dir, exist_ok=True)
        payload = {
            "method_id": MOC_METHOD_ID,
            "display_name": "MOC-LoRA (GRPO adaptation)",
            "paper": MOC_PAPER,
            "adaptation": "PPO value-head advantages replaced by group-relative advantages",
            "conditioning": "MOC preference prefix",
            "reward_funcs": self._reward_func_names,
            "dataset_name": self._dataset_name,
            "max_completion_length": self.args.max_completion_length,
            "tolerance": self._moc_tolerance,
            "reward_normalization": self._moc_reward_normalization,
            "reward_normalization_formula": "(r - running_mean) / (2 * running_std) + 1"
            if self._moc_reward_normalization == "running"
            else "clip((r-lower)/(upper-lower),0,1)",
            "running_normalizer": self._moc_running.state_dict()
            if self._moc_reward_normalization == "running"
            else None,
            "reward_lower_bounds": self._moc_lower.tolist()
            if self._moc_reward_normalization == "frozen_bounds"
            else None,
            "reward_upper_bounds": self._moc_upper.tolist()
            if self._moc_reward_normalization == "frozen_bounds"
            else None,
            "base_model_name_or_path": getattr(self.model.config, "_name_or_path", None),
        }
        with open(os.path.join(output_dir, "moc_grpo_meta.json"), "w") as handle:
            json.dump(public_metadata(payload), handle, indent=2)

    def save_model(self, output_dir: str | None = None, **kwargs):
        output_dir = output_dir or self.args.output_dir
        super().save_model(output_dir, **kwargs)
        if self.args.should_save:
            self._write_metadata(output_dir)


@dataclass
class MOCExperimentArguments(ParetoExperimentArguments):
    """Extra command-line fields mixed into the common experiment schema."""

    moc_tolerance: float = field(default=0.05)
    moc_reward_lower_bounds: list[float] = field(default_factory=list)
    moc_reward_upper_bounds: list[float] = field(default_factory=list)
    moc_reward_normalization: str = field(default="frozen_bounds")


def configure_model_initialization(training_args, model_args):
    """Pass dtype and model loading settings to TRL's string-model loader."""
    kwargs = _reference_model_init_kwargs(training_args, model_args)
    if model_args.attn_implementation is not None:
        kwargs["attn_implementation"] = model_args.attn_implementation
    if training_args.bf16 and str(kwargs["dtype"]) not in {"bfloat16", "torch.bfloat16"}:
        raise ValueError("BF16 MOC configuration conflicts with model loading dtype")
    training_args.model_init_kwargs = kwargs
    return kwargs


def main() -> None:
    script_args, training_args, model_args = parse_training_args(MOCExperimentArguments)
    if (
        script_args.dataset_name == "pku_safety"
        and script_args.moc_reward_normalization != "running"
    ):
        raise ValueError(
            "Fresh PKU MOC requires published running normalization; raw [0,1] clipping is invalid"
        )
    reserve_external_reward_devices(training_args, script_args.dataset_name)

    if training_args.use_vllm:
        raise ValueError("MOC-LoRA currently requires native generation (use_vllm=false)")

    configure_model_initialization(training_args, model_args)

    train_dataset, eval_dataset = load_training_data(script_args, training_args)
    reward_functions = get_reward_functions(script_args.reward_funcs)
    if len(reward_functions) != script_args.num_objectives:
        raise ValueError("num_objectives must equal len(reward_funcs)")

    target_modules = script_args.pareto_target_modules or ["q_proj", "v_proj"]
    lora_config = LoraConfig(
        r=script_args.pareto_r,
        lora_alpha=script_args.pareto_alpha,
        lora_dropout=script_args.pareto_dropout,
        target_modules=target_modules,
        exclude_modules=script_args.pareto_exclude_modules or None,
        task_type="CAUSAL_LM",
    )
    total_steps = training_args.max_steps if training_args.max_steps > 0 else 1000
    steps_per_epoch = 0
    if script_args.preference_sampling == "coverage":
        effective_batch = training_args.per_device_train_batch_size * max(
            training_args.gradient_accumulation_steps, 1
        )
        steps_per_epoch = len(train_dataset) // effective_batch if effective_batch else 0
    sampler = PreferenceSampler(
        num_objectives=script_args.num_objectives,
        total_steps=total_steps,
        num_samples=script_args.num_preference_samples,
        temperature=script_args.annealing_temperature,
        strategy=script_args.preference_sampling,
        steps_per_epoch=steps_per_epoch,
    )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    )
    prepare_training_tokenizer(
        tokenizer,
        model_args.model_name_or_path,
        preserve_native_eos=True,
        generation_kwargs=getattr(training_args, "generation_kwargs", None),
        train_dataset=train_dataset,
    )
    trainer = MOCGRPOTrainer(
        preference_sampler=sampler,
        reward_func_names=list(script_args.reward_funcs),
        dataset_name=script_args.dataset_name,
        tolerance=script_args.moc_tolerance,
        reward_lower_bounds=script_args.moc_reward_lower_bounds or None,
        reward_upper_bounds=script_args.moc_reward_upper_bounds or None,
        reward_normalization=script_args.moc_reward_normalization,
        model=model_args.model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        reward_funcs=reward_functions,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    manifest_path = start_run(script_args, training_args, model_args, train_dataset, eval_dataset)
    finish_run(trainer, training_args, manifest_path)


if __name__ == "__main__":
    main()
