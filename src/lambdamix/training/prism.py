"""PRISM policy decomposition for group-relative multi-reward RL."""

from __future__ import annotations

import inspect
import json
import os
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.utils.modeling import is_peft_model
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer
from trl import GRPOTrainer

from lambdamix.adapters.lambdamix import PreferenceSampler
from lambdamix.generation import GenerationStopsMixin, load_declared_architecture_model
from lambdamix.prompting import prepare_training_tokenizer
from lambdamix.run_manifest import (
    public_metadata,
    sanitize_checkpoint_metadata,
)
from lambdamix.training.common import (
    _model_source_kwargs,
    finish_run,
    get_reward_functions,
    load_training_data,
    parse_training_args,
    start_run,
)
from lambdamix.training.lambdamix import ParetoExperimentArguments

PRISM_METHOD_ID = "prism_lora"
PRISM_PAPER = "https://arxiv.org/abs/2607.29246"


def _supports_logits_to_keep(model: torch.nn.Module) -> bool:
    """Return whether the underlying causal LM supports last-token logits."""

    target = model.get_base_model() if hasattr(model, "get_base_model") else model
    return "logits_to_keep" in inspect.signature(target.forward).parameters


def compose_prism_logits(
    positive_logits: torch.Tensor,
    negative_logits: torch.Tensor,
    preference: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Compose PRISM branch logits under ``sum(alpha)-gamma=1``."""

    if positive_logits.ndim < 2:
        raise ValueError("positive_logits must include objective and vocabulary axes")
    objectives = positive_logits.shape[0]
    if negative_logits.shape != positive_logits.shape[1:]:
        raise ValueError("negative logits must match one positive branch")
    p = preference.to(positive_logits.device, positive_logits.dtype)
    if p.ndim == 1:
        if p.numel() != objectives:
            raise ValueError("preference dimensionality mismatch")
        p = p.reshape(objectives, *([1] * (positive_logits.ndim - 1)))
    elif p.ndim == 2:
        if positive_logits.ndim != 3 or p.shape != positive_logits.shape[:2]:
            raise ValueError("batched preference must have shape (objectives, batch)")
        p = p.unsqueeze(-1)
    else:
        raise ValueError("preference must be a vector or objective-by-batch matrix")
    if gamma < 0:
        raise ValueError("gamma must be non-negative")
    return ((1.0 + gamma) * p * positive_logits).sum(dim=0) - gamma * negative_logits


def prism_positive_advantages(advantages: torch.Tensor, center: float = 0.5) -> torch.Tensor:
    """PRISM Eq. 7 positive-branch shaping: ``sigmoid(A_k)-c``."""

    if not 0.0 < center < 1.0:
        raise ValueError("PRISM advantage center must lie strictly between zero and one")
    return torch.sigmoid(advantages) - center


def prism_negative_advantages(advantages: torch.Tensor, center: float = 0.5) -> torch.Tensor:
    """PRISM Eq. 8 negative branch: ``c-geomean(sigmoid(A_k))``."""

    if advantages.ndim != 2:
        raise ValueError("advantages must have shape (samples, objectives)")
    probs = torch.sigmoid(advantages).clamp_min(torch.finfo(torch.float32).tiny)
    geometric_mean = torch.exp(torch.log(probs).mean(dim=1))
    return center - geometric_mean


def prism_branch_advantages(
    rewards_per_func: torch.Tensor,
    num_generations: int,
    *,
    center: float = 0.5,
) -> torch.Tensor:
    """Return shaped advantages in positive-then-negative branch order."""

    rewards = torch.nan_to_num(rewards_per_func.float())
    if rewards.ndim != 2 or rewards.shape[0] % num_generations:
        raise ValueError("reward rows must be divisible by num_generations")
    groups = rewards.reshape(-1, num_generations, rewards.shape[1])
    normalized = (groups - groups.mean(dim=1, keepdim=True)) / (
        groups.std(dim=1, keepdim=True) + 1e-4
    )
    normalized = normalized.reshape_as(rewards)
    positive = prism_positive_advantages(normalized, center)
    negative = prism_negative_advantages(normalized, center).unsqueeze(1)
    return torch.cat([positive, negative], dim=1)


def _top_p_sample(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    if temperature <= 0:
        return logits.argmax(dim=-1)
    logits = logits / temperature
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        filtered = torch.full_like(logits, float("-inf"))
        filtered.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
        logits = filtered
    return torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1).squeeze(-1)


@contextmanager
def _temporarily_freeze_trainable_parameters(module: nn.Module):
    trainable = [parameter for parameter in module.parameters() if parameter.requires_grad]
    for parameter in trainable:
        parameter.requires_grad_(False)
    try:
        yield
    finally:
        for parameter in trainable:
            parameter.requires_grad_(True)


class PRISMPolicy(nn.Module):
    """Shared LoRA policy with T positive and one negative soft-prefix branch."""

    def __init__(
        self,
        base_model: nn.Module,
        *,
        num_objectives: int,
        prefix_length: int = 8,
        gamma: float = 0.1,
        newline_token_id: int,
    ):
        super().__init__()
        if num_objectives < 2 or prefix_length < 1:
            raise ValueError("PRISM requires >=2 objectives and a non-empty prefix")
        self.base_model = base_model
        self.num_objectives = int(num_objectives)
        self.num_branches = self.num_objectives + 1
        self.prefix_length = int(prefix_length)
        self.gamma = float(gamma)
        embedding = base_model.get_input_embeddings().weight.detach()[newline_token_id]
        initial = embedding.reshape(1, 1, -1).repeat(self.num_branches, self.prefix_length, 1)
        self.prefix_embeddings = nn.Parameter(initial.clone().to(torch.bfloat16))
        self.register_buffer(
            "_preference", torch.ones(num_objectives) / num_objectives, persistent=False
        )

    @property
    def config(self):
        return self.base_model.config

    def set_preference(self, preference: torch.Tensor) -> None:
        p = preference.detach().float().reshape(-1)
        if p.numel() != self.num_objectives or torch.any(p < 0):
            raise ValueError("invalid PRISM preference")
        if not torch.isclose(p.sum(), p.new_tensor(1.0), atol=1e-5):
            raise ValueError("PRISM preference must lie on the simplex")
        self._preference = p.to(self._preference.device)

    def _prefixed_inputs(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, branch: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token_embeddings = self.base_model.get_input_embeddings()(input_ids)
        prefix = self.prefix_embeddings[branch].to(token_embeddings.dtype)
        prefix = prefix.unsqueeze(0).expand(input_ids.shape[0], -1, -1)
        embeddings = torch.cat([prefix, token_embeddings], dim=1)
        prefix_mask = torch.ones(
            input_ids.shape[0],
            self.prefix_length,
            device=attention_mask.device,
            dtype=attention_mask.dtype,
        )
        return embeddings, torch.cat([prefix_mask, attention_mask], dim=1)

    def branch_completion_logps(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        completion_length: int,
        branch: int,
        *,
        freeze_backbone: bool = False,
    ) -> torch.Tensor:
        """Token log-probabilities for one soft-prefix policy branch."""

        embeddings, mask = self._prefixed_inputs(input_ids, attention_mask, branch)

        def forward_branch(embeddings, mask):
            checkpointing = bool(getattr(self.base_model, "is_gradient_checkpointing", False))
            if freeze_backbone and checkpointing:
                self.base_model.gradient_checkpointing_disable()
            context = (
                _temporarily_freeze_trainable_parameters(self.base_model)
                if freeze_backbone
                else nullcontext()
            )
            try:
                with context:
                    return self.base_model(
                        inputs_embeds=embeddings,
                        attention_mask=mask,
                        use_cache=False,
                        return_dict=True,
                    ).logits
            finally:
                if freeze_backbone and checkpointing:
                    self.base_model.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={"use_reentrant": False}
                    )

        if (
            freeze_backbone
            and self.training
            and getattr(self.base_model, "is_gradient_checkpointing", False)
        ):
            from torch.utils.checkpoint import checkpoint

            logits = checkpoint(forward_branch, embeddings, mask, use_reentrant=False)
        else:
            logits = forward_branch(embeddings, mask)
        start = self.prefix_length + input_ids.shape[1] - completion_length - 1
        token_logits = logits[:, start : start + completion_length, :]
        targets = input_ids[:, -completion_length:]
        return (
            F.log_softmax(token_logits.float(), dim=-1)
            .gather(-1, targets.unsqueeze(-1))
            .squeeze(-1)
        )

    @torch.no_grad()
    def generate(self, input_ids, attention_mask=None, generation_config=None, **kwargs):
        """Generate once from the PRISM mixed logits and broadcast tokens to branches."""

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        config = generation_config or getattr(self.base_model, "generation_config", None)
        max_new_tokens = int(
            kwargs.get("max_new_tokens") or getattr(config, "max_new_tokens", None) or 64
        )
        do_sample = bool(kwargs.get("do_sample", getattr(config, "do_sample", True)))
        temperature = float(
            kwargs.get("temperature") or getattr(config, "temperature", None) or 1.0
        )
        top_p = float(kwargs.get("top_p") or getattr(config, "top_p", None) or 1.0)
        if not do_sample:
            temperature = 0.0
        eos = kwargs.get("eos_token_id", getattr(config, "eos_token_id", None))
        eos_ids = set(
            int(x) for x in (eos if isinstance(eos, (list, tuple)) else [eos]) if x is not None
        )
        pad_token_id = int(kwargs.get("pad_token_id", getattr(config, "pad_token_id", None)) or 0)

        branches = self.num_branches
        batch = input_ids.shape[0]
        token_embeddings = self.base_model.get_input_embeddings()(input_ids)
        expanded_tokens = token_embeddings.unsqueeze(0).expand(branches, -1, -1, -1)
        prefixes = (
            self.prefix_embeddings.to(token_embeddings.dtype).unsqueeze(1).expand(-1, batch, -1, -1)
        )
        embeds = torch.cat([prefixes, expanded_tokens], dim=2).reshape(
            branches * batch, self.prefix_length + input_ids.shape[1], -1
        )
        prefix_mask = torch.ones(
            branches,
            batch,
            self.prefix_length,
            device=input_ids.device,
            dtype=attention_mask.dtype,
        )
        masks = torch.cat(
            [prefix_mask, attention_mask.unsqueeze(0).expand(branches, -1, -1)], dim=2
        ).reshape(branches * batch, -1)
        forward_kwargs = {
            "inputs_embeds": embeds,
            "attention_mask": masks,
            "use_cache": True,
            "return_dict": True,
        }
        if _supports_logits_to_keep(self.base_model):
            forward_kwargs["logits_to_keep"] = 1
        output = self.base_model(**forward_kwargs)
        cache = output.past_key_values
        next_logits = output.logits[:, -1, :].reshape(branches, batch, -1)
        preference = self._preference.to(input_ids.device)
        generated: list[torch.Tensor] = []
        finished = torch.zeros(batch, dtype=torch.bool, device=input_ids.device)
        for step in range(max_new_tokens):
            mixed = compose_prism_logits(
                next_logits[: self.num_objectives], next_logits[-1], preference, self.gamma
            )
            sampled = _top_p_sample(mixed, temperature, top_p)
            sampled = torch.where(finished, sampled.new_full((), pad_token_id), sampled)
            generated.append(sampled)
            if eos_ids:
                just_finished = torch.zeros_like(finished)
                for eos_id in eos_ids:
                    just_finished |= sampled == eos_id
                finished |= just_finished
                if bool(finished.all()):
                    break
            repeated = sampled.unsqueeze(0).expand(branches, -1).reshape(-1, 1)
            masks = torch.cat(
                [masks, torch.ones(branches * batch, 1, device=masks.device, dtype=masks.dtype)],
                dim=1,
            )
            forward_kwargs = {
                "input_ids": repeated,
                "attention_mask": masks,
                "past_key_values": cache,
                "use_cache": True,
                "return_dict": True,
            }
            if _supports_logits_to_keep(self.base_model):
                forward_kwargs["logits_to_keep"] = 1
            output = self.base_model(**forward_kwargs)
            cache = output.past_key_values
            next_logits = output.logits[:, -1, :].reshape(branches, batch, -1)
        completion = torch.stack(generated, dim=1) if generated else input_ids[:, :0]
        return torch.cat([input_ids, completion], dim=1)

    def forward(self, *args, completion_length=None, **kwargs):
        if completion_length is None:
            return self.base_model(*args, **kwargs)
        return torch.stack(
            [
                self.branch_completion_logps(
                    kwargs["input_ids"],
                    kwargs["attention_mask"],
                    completion_length,
                    branch,
                    freeze_backbone=branch == self.num_objectives,
                )
                for branch in range(self.num_branches)
            ]
        )

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)


class PRISMGRPOTrainer(GenerationStopsMixin, GRPOTrainer):
    """TRL integration for positive/negative PRISM branch updates."""

    def __init__(
        self,
        *,
        prism_model: PRISMPolicy,
        preference_sampler: PreferenceSampler,
        reward_func_names: list[str],
        dataset_name: str | None = None,
        advantage_center: float = 0.5,
        **kwargs,
    ):
        args = kwargs.get("args")
        if args is not None:
            if getattr(args, "use_vllm", False):
                raise ValueError("PRISM mixed-logit generation is incompatible with vLLM")
            if int(getattr(args, "num_iterations", 1)) != 1:
                raise ValueError("PRISM-LoRA currently requires num_iterations=1")
        self.prism_model = prism_model
        self.preference_sampler = preference_sampler
        self._reward_func_names = list(reward_func_names)
        self._dataset_name = dataset_name
        self._advantage_center = float(advantage_center)
        self._rewards_per_func: torch.Tensor | None = None
        self._current_lambda: torch.Tensor | None = None
        kwargs["model"] = prism_model
        if not is_peft_model(prism_model.base_model):
            raise ValueError("PRISM-LoRA requires a PEFT base model")

        requested_beta = float(args.beta) if args is not None else 0.0
        if args is not None:
            args.beta = 0.0
        try:
            super().__init__(**kwargs)
        finally:
            if args is not None:
                args.beta = requested_beta
        self.beta = requested_beta

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
        self.prism_model.set_preference(preference)
        output = super()._generate_and_score_completions(inputs)
        if self._rewards_per_func is None:
            raise RuntimeError("PRISM did not receive the per-objective reward matrix")
        branch_adv = prism_branch_advantages(
            self._rewards_per_func,
            self.args.num_generations,
            center=self._advantage_center,
        )
        local_size = len(output["advantages"])
        start = self.accelerator.process_index * local_size
        output["prism_branch_advantages"] = branch_adv[start : start + local_size]
        output["advantages"] = (
            branch_adv[:, : self.prism_model.num_objectives]
            * preference.to(branch_adv.device).unsqueeze(0)
        ).sum(1)[start : start + local_size]
        return output

    def _branch_loss(self, token_logps, advantages, completion_mask, ref_logps=None):
        per_token_loss = -token_logps * advantages.unsqueeze(1)
        if ref_logps is not None and self.beta != 0:
            delta = ref_logps - token_logps
            per_token_loss = per_token_loss + self.beta * (torch.exp(delta) - delta - 1)
        if self.loss_type == "dapo":
            normalizer = completion_mask.sum().clamp_min(1)
            return (per_token_loss * completion_mask).sum() / normalizer
        return (
            (per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp_min(1)
        ).mean()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("PRISMGRPOTrainer does not return model outputs")
        prompt_ids, completion_ids = inputs["prompt_ids"], inputs["completion_ids"]
        prompt_mask, completion_mask = inputs["prompt_mask"], inputs["completion_mask"]
        ids = torch.cat([prompt_ids, completion_ids], dim=1)
        mask = torch.cat([prompt_mask, completion_mask], dim=1)
        branch_adv = inputs["prism_branch_advantages"]
        reference = inputs.get("ref_per_token_logps")
        branch_logps = model(
            input_ids=ids, attention_mask=mask, completion_length=completion_ids.shape[1]
        )
        losses = [
            self._branch_loss(logps, branch_adv[:, branch], completion_mask, reference)
            for branch, logps in enumerate(branch_logps)
        ]
        return torch.stack(losses).mean() / max(int(self.current_gradient_accumulation_steps), 1)

    def save_model(self, output_dir: str | None = None, **kwargs):
        if not self.args.should_save:
            return
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.prism_model.base_model.save_pretrained(output_dir)
        torch.save(
            {"prefix_embeddings": self.prism_model.prefix_embeddings.detach().cpu()},
            os.path.join(output_dir, "prism_prefixes.pt"),
        )
        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)
        metadata = {
            "method_id": PRISM_METHOD_ID,
            "display_name": "PRISM-LoRA",
            "paper": PRISM_PAPER,
            "num_objectives": self.prism_model.num_objectives,
            "prefix_length": self.prism_model.prefix_length,
            "gamma": self.prism_model.gamma,
            "advantage_center": self._advantage_center,
            "reward_funcs": self._reward_func_names,
            "dataset_name": self._dataset_name,
            "max_completion_length": self.args.max_completion_length,
            "base_model_name_or_path": getattr(self.prism_model.config, "_name_or_path", None),
        }
        sanitize_checkpoint_metadata(output_dir)
        with open(os.path.join(output_dir, "prism_meta.json"), "w") as handle:
            json.dump(public_metadata(metadata), handle, indent=2)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict

        weights = load_peft_weights(resume_from_checkpoint, device="cpu")
        set_peft_model_state_dict(self.prism_model.base_model, weights)
        saved = torch.load(
            os.path.join(resume_from_checkpoint, "prism_prefixes.pt"),
            map_location="cpu",
            weights_only=True,
        )["prefix_embeddings"]
        if saved.shape != self.prism_model.prefix_embeddings.shape:
            raise ValueError("PRISM checkpoint prefix shape does not match the policy")
        with torch.no_grad():
            self.prism_model.prefix_embeddings.copy_(saved)


@dataclass
class PRISMExperimentArguments(ParetoExperimentArguments):
    prism_prefix_length: int = field(default=8)
    prism_gamma: float = field(default=0.1)
    prism_advantage_center: float = field(default=0.5)


def main() -> None:
    script_args, training_args, model_args = parse_training_args(PRISMExperimentArguments)

    if training_args.use_vllm:
        raise ValueError("PRISM-LoRA requires native mixed-logit generation")

    train_dataset, eval_dataset = load_training_data(script_args, training_args)
    rewards = get_reward_functions(script_args.reward_funcs)
    if len(rewards) != script_args.num_objectives:
        raise ValueError("num_objectives must equal len(reward_funcs)")

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, **_model_source_kwargs(model_args)
    )
    prepare_training_tokenizer(
        tokenizer,
        model_args.model_name_or_path,
        preserve_native_eos=True,
        generation_kwargs=getattr(training_args, "generation_kwargs", None),
        train_dataset=train_dataset,
    )
    newline_ids = tokenizer.encode("\n", add_special_tokens=False)
    if not newline_ids:
        raise RuntimeError("tokenizer produced no token for PRISM newline initialization")

    load_kwargs = {
        "torch_dtype": torch.bfloat16,
        "attn_implementation": model_args.attn_implementation or "sdpa",
        **_model_source_kwargs(model_args),
    }
    model = load_declared_architecture_model(model_args.model_name_or_path, **load_kwargs)
    lora_config = LoraConfig(
        r=script_args.pareto_r,
        lora_alpha=script_args.pareto_alpha,
        lora_dropout=script_args.pareto_dropout,
        target_modules=script_args.pareto_target_modules or ["q_proj", "v_proj"],
        exclude_modules=script_args.pareto_exclude_modules or None,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    prism_model = PRISMPolicy(
        model,
        num_objectives=script_args.num_objectives,
        prefix_length=script_args.prism_prefix_length,
        gamma=script_args.prism_gamma,
        newline_token_id=newline_ids[-1],
    )

    total_steps = training_args.max_steps if training_args.max_steps > 0 else 1000
    effective_batch = training_args.per_device_train_batch_size * max(
        training_args.gradient_accumulation_steps, 1
    )
    steps_per_epoch = (
        len(train_dataset) // effective_batch
        if script_args.preference_sampling == "coverage" and effective_batch
        else 0
    )
    sampler = PreferenceSampler(
        num_objectives=script_args.num_objectives,
        total_steps=total_steps,
        num_samples=script_args.num_preference_samples,
        temperature=script_args.annealing_temperature,
        strategy=script_args.preference_sampling,
        steps_per_epoch=steps_per_epoch,
    )
    trainer = PRISMGRPOTrainer(
        prism_model=prism_model,
        preference_sampler=sampler,
        reward_func_names=list(script_args.reward_funcs),
        dataset_name=script_args.dataset_name,
        advantage_center=script_args.prism_advantage_center,
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
