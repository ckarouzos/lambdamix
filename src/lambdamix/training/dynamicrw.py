"""Gradient-influence Dynamic Reward Weighting (Lu et al., TACL 2026)."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
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

_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


def grpo_objective_advantages(
    rewards: torch.Tensor,
    num_generations: int,
    scale_rewards: str = "group",
) -> torch.Tensor:
    """Return one ordinary-GRPO advantage column per reward objective."""
    if rewards.ndim != 2:
        raise ValueError("rewards must have shape (batch, objectives)")
    if num_generations < 2 or rewards.shape[0] % num_generations:
        raise ValueError("batch size must be divisible by num_generations >= 2")
    if scale_rewards not in {"group", "batch", "none"}:
        raise ValueError("scale_rewards must be 'group', 'batch', or 'none'")

    clean = torch.nan_to_num(rewards)
    groups = clean.view(-1, num_generations, clean.shape[1])
    centered = groups - groups.mean(dim=1, keepdim=True)
    if scale_rewards == "group":
        centered = centered / (groups.std(dim=1, keepdim=True) + 1e-4)
    elif scale_rewards == "batch":
        centered = centered / (clean.std(dim=0).view(1, 1, -1) + 1e-4)
    return centered.view_as(clean)


def grpo_scalarized_advantages(
    rewards: torch.Tensor,
    weights: torch.Tensor,
    num_generations: int,
    scale_rewards: str = "group",
) -> torch.Tensor:
    """Scalarize raw rewards first, then apply TRL's GRPO transform."""
    if rewards.ndim != 2 or weights.ndim != 1 or rewards.shape[1] != len(weights):
        raise ValueError("rewards/weights have incompatible shapes")
    scalar = torch.nan_to_num(rewards) @ weights.to(rewards.device, rewards.dtype)
    groups = scalar.view(-1, num_generations)
    advantages = groups - groups.mean(dim=1, keepdim=True)
    if scale_rewards == "group":
        advantages = advantages / (groups.std(dim=1, keepdim=True) + 1e-4)
    elif scale_rewards == "batch":
        advantages = advantages / (scalar.std() + 1e-4)
    elif scale_rewards != "none":
        raise ValueError("scale_rewards must be 'group', 'batch', or 'none'")
    return advantages.reshape(-1)


def select_gradient_probe_parameters(
    model: torch.nn.Module,
    target_layer_ids: Sequence[int] | None,
) -> list[tuple[str, torch.nn.Parameter]]:
    """Select trainable parameters in the official gradient-probe layers."""
    target = None if target_layer_ids is None else {int(index) for index in target_layer_ids}
    selected: list[tuple[str, torch.nn.Parameter]] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        match = _LAYER_RE.search(name)
        if target is not None and (match is None or int(match.group(1)) not in target):
            continue
        selected.append((name, parameter))
    if not selected:
        requested = "all trainable layers" if target is None else sorted(target)
        raise RuntimeError(f"No trainable parameters found for DRW probe layers {requested}")
    return selected


def multi_device_squared_l2_norm(values: list[torch.Tensor]) -> torch.Tensor:
    """Return a global squared L2 norm without reducing across CUDA devices."""
    total = torch.zeros((), dtype=torch.float64)
    for value in values:
        total.add_(torch.sum(value.float() * value.float()).double().cpu())
    return total


class GradientInfluenceWeightAdapter:
    """Stateful implementation of the official multiplicative simplex update."""

    def __init__(
        self,
        num_objectives: int,
        regularization_factor: float = 1e-5,
        init_weights: Sequence[float] | None = None,
    ) -> None:
        if num_objectives < 2:
            raise ValueError("Gradient DynamicRW requires at least two objectives")
        if regularization_factor <= 0 or not math.isfinite(float(regularization_factor)):
            raise ValueError("regularization_factor must be positive")
        self.num_objectives = int(num_objectives)
        self.regularization_factor = float(regularization_factor)
        if init_weights is None:
            self.weights = torch.full((num_objectives,), 1.0 / num_objectives, dtype=torch.float64)
        else:
            weights = torch.as_tensor(init_weights, dtype=torch.float64)
            if weights.shape != (num_objectives,) or torch.any(weights <= 0):
                raise ValueError("init_weights must be a strictly positive length-K vector")
            self.weights = weights / weights.sum()
        self.last_gram = torch.zeros((num_objectives, num_objectives), dtype=torch.float64)
        self.last_influence = torch.zeros(num_objectives, dtype=torch.float64)
        self.last_update_factor = torch.ones(num_objectives, dtype=torch.float64)
        self.num_updates = 0

    def update(self, gram: torch.Tensor, learning_rate: float) -> torch.Tensor:
        """Apply the official ``softmax(log(w) + lr * rowsum(G) / mu)`` rule."""
        matrix = torch.as_tensor(gram, dtype=torch.float64, device="cpu")
        if matrix.shape != (self.num_objectives, self.num_objectives):
            raise ValueError("gram has the wrong shape")
        if not torch.isfinite(matrix).all():
            raise ValueError("gram must be finite")
        if learning_rate < 0 or not math.isfinite(float(learning_rate)):
            raise ValueError("learning_rate must be finite and non-negative")

        influence = matrix.sum(dim=-1)
        scores = float(learning_rate) * influence / self.regularization_factor
        log_weights = torch.log(self.weights.clamp_min(torch.finfo(torch.float64).tiny))
        self.weights = torch.softmax(log_weights + scores, dim=-1)
        self.last_gram = matrix.clone()
        self.last_influence = influence.clone()
        self.last_update_factor = torch.exp(scores.clamp(max=700.0))
        self.num_updates += 1
        return self.weights.clone()

    def get_weights(self, *, device=None, dtype=torch.float32) -> torch.Tensor:
        return self.weights.to(device=device, dtype=dtype)

    def state_dict(self) -> dict:
        return {
            "weights": self.weights.tolist(),
            "last_gram": self.last_gram.tolist(),
            "last_influence": self.last_influence.tolist(),
            "last_update_factor": self.last_update_factor.tolist(),
            "num_updates": self.num_updates,
        }

    def load_state_dict(self, state: dict) -> None:
        weights = torch.as_tensor(state["weights"], dtype=torch.float64)
        if weights.shape != (self.num_objectives,) or torch.any(weights <= 0):
            raise ValueError("checkpoint has invalid DynamicRW weights")
        self.weights = weights / weights.sum()
        self.last_gram = torch.as_tensor(
            state.get("last_gram", self.last_gram), dtype=torch.float64
        )
        self.last_influence = torch.as_tensor(
            state.get("last_influence", self.last_influence), dtype=torch.float64
        )
        self.last_update_factor = torch.as_tensor(
            state.get("last_update_factor", self.last_update_factor), dtype=torch.float64
        )
        self.num_updates = int(state.get("num_updates", 0))

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.state_dict(), indent=2) + "\n")

    def load_json(self, path: str | Path) -> None:
        self.load_state_dict(json.loads(Path(path).read_text()))


class GradientDynamicRewardWeightingTrainer(GenerationStopsMixin, GRPOTrainer):
    """TRL GRPO trainer with the official gradient-influence weight update."""

    def __init__(
        self,
        *args,
        drw_regularization_factor: float = 1e-5,
        drw_target_layer_ids: Sequence[int] | None = tuple(range(10, 26)),
        drw_grad_clip: float = 1.0,
        drw_init_weights: Sequence[float] | None = None,
        drw_resume_path: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if drw_grad_clip <= 0:
            raise ValueError("drw_grad_clip must be positive")
        self._gradient_drw = GradientInfluenceWeightAdapter(
            len(self.reward_funcs),
            regularization_factor=drw_regularization_factor,
            init_weights=drw_init_weights,
        )
        self._drw_target_layer_ids = (
            None if drw_target_layer_ids is None else tuple(int(x) for x in drw_target_layer_ids)
        )
        self._drw_grad_clip = float(drw_grad_clip)
        self._drw_gradient_accumulator: list[list[torch.Tensor]] | None = None
        self._drw_microsteps = 0
        self._rewards_per_func: torch.Tensor | None = None
        if drw_resume_path:
            state_path = Path(drw_resume_path) / "gradient_drw_state.json"
            if state_path.exists():
                self._gradient_drw.load_json(state_path)

    def _calculate_rewards(self, *args, **kwargs):
        rewards = super()._calculate_rewards(*args, **kwargs)
        self._rewards_per_func = rewards.clone()
        return rewards

    def _generate_and_score_completions(self, inputs):
        output = super()._generate_and_score_completions(inputs)
        if self._rewards_per_func is None:
            raise RuntimeError("DynamicRW did not receive per-objective rewards")
        rewards = self._rewards_per_func
        objective_advantages = grpo_objective_advantages(
            rewards, self.num_generations, self.scale_rewards
        )
        weights = self._gradient_drw.get_weights(device=rewards.device, dtype=rewards.dtype)
        scalarized = grpo_scalarized_advantages(
            rewards, weights, self.num_generations, self.scale_rewards
        )
        local_size = len(output["advantages"])
        start = self.accelerator.process_index * local_size
        output["drw_objective_advantages"] = objective_advantages[start : start + local_size]
        output["advantages"] = scalarized[start : start + local_size]
        return output

    def _probe_policy_losses(
        self,
        model,
        inputs,
        objective_advantages: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Compute all objective probe losses from one shared policy forward."""
        if objective_advantages.ndim != 2:
            raise ValueError("objective_advantages must have shape (batch, objectives)")
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        per_token_logps, _ = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            completion_ids.size(1),
            compute_entropy=False,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            image_sizes=inputs.get("image_sizes"),
        )
        old_logps = inputs.get("old_per_token_logps")
        old_logps = per_token_logps.detach() if old_logps is None else old_logps
        log_ratio = per_token_logps - old_logps
        if self.importance_sampling_level == "sequence":
            log_ratio = (
                (log_ratio * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
            ).unsqueeze(-1)
        elif self.importance_sampling_level != "token":
            raise ValueError(
                f"Unsupported importance_sampling_level={self.importance_sampling_level}"
            )
        ratio = torch.exp(log_ratio)
        clipped = torch.clamp(ratio, 1 - self.epsilon_low, 1 + self.epsilon_high)
        if self.args.delta is not None:
            ratio = torch.clamp(ratio, max=self.args.delta)
        per_token_kl = None
        if self.beta != 0.0:
            ref_logps = inputs["ref_per_token_logps"]
            diff = ref_logps - per_token_logps
            per_token_kl = torch.exp(diff) - diff - 1

        losses: list[torch.Tensor] = []
        for advantages in objective_advantages.unbind(dim=1):
            loss = -torch.min(ratio * advantages.unsqueeze(1), clipped * advantages.unsqueeze(1))
            if per_token_kl is not None:
                loss = loss + self.beta * per_token_kl

            if self.loss_type == "grpo":
                reduced = (
                    (loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
                ).mean()
            elif self.loss_type == "bnpo":
                reduced = (loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
            elif self.loss_type == "dr_grpo":
                reduced = (loss * completion_mask).sum() / (
                    loss.size(0) * self.max_completion_length
                )
            elif self.loss_type == "dapo":
                normalizer = inputs["num_items_in_batch"] / self.accelerator.num_processes
                reduced = (loss * completion_mask).sum() / normalizer
            else:
                raise ValueError(f"Unsupported loss_type={self.loss_type}")
            losses.append(reduced)
        return losses

    def _objective_gradients(self, model, inputs) -> list[list[torch.Tensor]]:
        unwrapped = self.accelerator.unwrap_model(model)
        selected = select_gradient_probe_parameters(unwrapped, self._drw_target_layer_ids)
        parameters = [parameter for _, parameter in selected]
        objective_advantages = inputs["drw_objective_advantages"]
        losses = self._probe_policy_losses(unwrapped, inputs, objective_advantages)
        gradients: list[list[torch.Tensor]] = []
        for objective, loss in enumerate(losses):
            values = torch.autograd.grad(
                loss,
                parameters,
                retain_graph=objective + 1 < len(losses),
                create_graph=False,
                allow_unused=True,
            )
            gradients.append(
                [
                    torch.zeros_like(parameter, memory_format=torch.preserve_format)
                    if value is None
                    else value.detach()
                    for parameter, value in zip(parameters, values)
                ]
            )
        return gradients

    def _accumulate_objective_gradients(self, gradients: list[list[torch.Tensor]]) -> None:
        if self._drw_gradient_accumulator is None:
            self._drw_gradient_accumulator = [
                [value.float().clone() for value in objective] for objective in gradients
            ]
        else:
            for accumulated, objective in zip(self._drw_gradient_accumulator, gradients):
                for target, value in zip(accumulated, objective):
                    target.add_(value.float())
        self._drw_microsteps += 1

    def _optimizer_learning_rate(self) -> float:
        scheduler = getattr(self, "lr_scheduler", None)
        if scheduler is not None:
            rates = scheduler.get_last_lr()
            if rates:
                return float(rates[0])
        optimizer = getattr(self, "optimizer", None)
        if optimizer is not None and optimizer.param_groups:
            return float(optimizer.param_groups[0]["lr"])
        return float(self.args.learning_rate)

    def _finish_gradient_influence_update(self) -> None:
        gradients = self._drw_gradient_accumulator
        if gradients is None:
            return
        if self.accelerator.num_processes > 1:
            for objective in gradients:
                for index, value in enumerate(objective):
                    objective[index] = self.accelerator.reduce(value, reduction="mean")

        for objective in gradients:
            norm_sq = multi_device_squared_l2_norm(objective)
            norm = torch.sqrt(norm_sq).item()
            if norm > self._drw_grad_clip:
                scale = self._drw_grad_clip / (norm + 1e-12)
                for value in objective:
                    value.mul_(scale)

        count = len(gradients)
        gram = torch.zeros((count, count), dtype=torch.float64)
        for row in range(count):
            for column in range(row, count):
                dot = sum(
                    torch.sum(left * right).double().cpu()
                    for left, right in zip(gradients[row], gradients[column])
                )
                gram[row, column] = dot
                gram[column, row] = dot
        weights = self._gradient_drw.update(gram, self._optimizer_learning_rate())
        self._log_gradient_drw_metrics(weights)
        self._drw_gradient_accumulator = None
        self._drw_microsteps = 0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("Gradient DynamicRW does not support return_outputs")
        if "drw_objective_advantages" not in inputs:
            raise RuntimeError("Missing objective advantages for DynamicRW gradient probe")
        gradients = self._objective_gradients(model, inputs)
        self._accumulate_objective_gradients(gradients)
        accumulation_steps = int(max(1, self.current_gradient_accumulation_steps))
        if self._drw_microsteps >= accumulation_steps:
            self._finish_gradient_influence_update()
        return super().compute_loss(
            model,
            inputs,
            return_outputs=False,
            num_items_in_batch=num_items_in_batch,
        )

    def _log_gradient_drw_metrics(self, weights: torch.Tensor) -> None:
        try:
            mode = "train" if self.model.training else "eval"
            metrics = self._metrics[mode]
            for index, value in enumerate(weights.tolist()):
                metrics[f"gradient_drw/weight_{index}"].append(float(value))
                metrics[f"gradient_drw/influence_{index}"].append(
                    float(self._gradient_drw.last_influence[index])
                )
            metrics["gradient_drw/updates"].append(float(self._gradient_drw.num_updates))
        except (AttributeError, KeyError, TypeError):
            pass

    def _save_checkpoint(self, *args, **kwargs):
        result = super()._save_checkpoint(*args, **kwargs)
        try:
            checkpoint = Path(self.args.output_dir) / f"checkpoint-{self.state.global_step}"
            if checkpoint.exists() and self.accelerator.is_main_process:
                self._gradient_drw.save_json(checkpoint / "gradient_drw_state.json")
        except (AttributeError, OSError):
            pass
        return result


@dataclass
class GradientDynamicRWArguments(ScriptArguments):
    """Script-level arguments not represented by TRL's GRPO config."""

    reward_funcs: list[str] = field(default_factory=lambda: ["correctness", "ifeval_constraint"])
    drw_regularization_factor: float = 1e-5
    drw_target_layer_ids: list[int] = field(default_factory=lambda: list(range(10, 26)))
    drw_grad_clip: float = 1.0
    drw_init_weights: list[float] | None = None
    model_device_map_auto: bool = False
    model_max_memory_gib: int = 58


def main() -> None:
    script_args, training_args, model_args = parse_training_args(GradientDynamicRWArguments)
    reserve_external_reward_devices(training_args, script_args.dataset_name)
    if len(script_args.reward_funcs) < 2:
        raise ValueError("Gradient DynamicRW requires at least two reward functions")
    if not getattr(model_args, "use_peft", False):
        raise ValueError(
            "The paper's Gradient DynamicRW row is parameter-matched LoRA; set use_peft: true"
        )
    if training_args.scale_rewards != "group":
        raise ValueError("The paper recipe requires scale_rewards: group")

    train_dataset, eval_dataset = load_training_data(script_args, training_args)
    reward_functions = get_reward_functions(script_args.reward_funcs)

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

    manifest_path = start_run(script_args, training_args, model_args, train_dataset, eval_dataset)

    policy_model = load_single_process_model_parallel_policy(
        local_model_source, script_args, training_args, model_args
    )
    trainer = GradientDynamicRewardWeightingTrainer(
        model=policy_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        reward_funcs=reward_functions,
        processing_class=tokenizer,
        peft_config=_build_peft_config(model_args),
        drw_regularization_factor=script_args.drw_regularization_factor,
        drw_target_layer_ids=script_args.drw_target_layer_ids,
        drw_grad_clip=script_args.drw_grad_clip,
        drw_init_weights=script_args.drw_init_weights,
        drw_resume_path=training_args.resume_from_checkpoint,
    )
    finish_run(trainer, training_args, manifest_path)


if __name__ == "__main__":
    main()
