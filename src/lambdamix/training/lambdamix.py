"""LambdaMix trainer with preference-conditioned adapters and objective advantages."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from math import floor

import torch
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments

from lambdamix.adapters.lambdamix import ParetoLoRAConfig, ParetoLoRAModel, PreferenceSampler
from lambdamix.generation import (
    GenerationStopsMixin,
    load_text_generation_model,
    resolve_model_source,
)
from lambdamix.training.advantages import compute_gdpo_advantages
from lambdamix.training.common import (
    _model_parallel_load_kwargs,
    _model_source_kwargs,
    _reference_model_init_kwargs,
    estimate_grpo_steps_per_epoch,
    finish_run,
    get_reward_functions,
    load_training_data,
    parse_training_args,
    reserve_external_reward_devices,
    start_run,
)

_TRL_HAS_AGGREGATION = hasattr(GRPOConfig, "multi_objective_aggregation")


@dataclass
class ParetoExperimentArguments(ScriptArguments):
    """Extended script arguments for PaLoRA experiments."""

    reward_funcs: list[str] = field(
        default_factory=lambda: ["correctness", "ifeval_constraint"],
        metadata={"help": "List of reward function names from REWARD_REGISTRY."},
    )
    num_objectives: int = field(
        default=2,
        metadata={"help": "Number of objectives (must match len(reward_funcs))."},
    )
    pareto_r: int = field(
        default=8,
        metadata={"help": "LoRA rank for each PaLoRA adapter."},
    )
    pareto_alpha: int = field(
        default=16,
        metadata={"help": "LoRA alpha for PaLoRA scaling."},
    )
    pareto_target_modules: list[str] | None = field(
        default=None,
        metadata={"help": "Target modules for PaLoRA. Defaults to ['q_proj', 'v_proj']."},
    )
    pareto_dropout: float = field(
        default=0.0,
        metadata={"help": "Dropout for PaLoRA adapters."},
    )
    pareto_adapter_dtype: str = field(
        default="auto",
        metadata={
            "help": (
                "Trainable LambdaMix dtype: auto, bf16, fp16, or fp32. "
                "Qwen3.5/3.6 FP8 bases must use bf16."
            )
        },
    )
    pareto_a_init: str = field(
        default="rank_scaled",
        metadata={
            "help": (
                "LambdaMix LoRA-A initialization: 'rank_scaled' is the declared "
                "full-method component; 'peft' is the standard-LoRA ablation."
            )
        },
    )
    ordinary_lora_a_init: str = field(
        default="peft",
        metadata={
            "help": (
                "A initialization for plain-PEFT baselines that reuse this argument "
                "schema: peft or rank_scaled. LambdaMix itself uses pareto_a_init."
            )
        },
    )
    pareto_exclude_modules: list[str] = field(
        default_factory=list,
        metadata={
            "help": "Module path fragments excluded from LambdaMix wrapping (e.g. visual, mtp)."
        },
    )
    model_device_map_auto: bool = field(
        default=False,
        metadata={
            "help": (
                "Shard the frozen base over all visible accelerators in one process. "
                "This is a native-generation large-model path and is incompatible "
                "with vLLM and multi-process launch."
            )
        },
    )
    model_max_memory_gib: int = field(
        default=58,
        metadata={"help": "Per-accelerator HBM ceiling for model_device_map_auto."},
    )
    model_device_count: int = field(
        default=0,
        metadata={
            "help": (
                "Number of leading visible accelerators available to "
                "model_device_map_auto; 0 uses every visible accelerator. "
                "Set this when later devices are reserved for frozen evaluators."
            )
        },
    )
    num_preference_samples: int = field(
        default=4,
        metadata={
            "help": "Number of preference anchors (M) for the sampling schedule. "
            "With M=1 per step, this controls the schedule density, not batch size."
        },
    )
    annealing_temperature: float = field(
        default=1.0,
        metadata={"help": "Temperature (Q) for deterministic annealing."},
    )
    preference_sampling: str = field(
        default="coverage",
        metadata={"help": "Paper preference schedule: coverage."},
    )
    adapter_preference_conditioning: str = field(
        default="requested",
        metadata={
            "help": (
                "Method ablation: 'requested' mixes adapters with the sampled lambda; "
                "'uniform' holds parameter mixing at the simplex centre."
            )
        },
    )
    advantage_preference_conditioning: str = field(
        default="requested",
        metadata={
            "help": (
                "Method ablation: 'requested' weights objective advantages with the sampled "
                "lambda; 'uniform' holds optimization weights at the simplex centre."
            )
        },
    )


if not _TRL_HAS_AGGREGATION:

    @dataclass
    class ParetoExperimentArguments(ParetoExperimentArguments):  # type: ignore[no-redef]
        multi_objective_aggregation: str = field(
            default="normalize_then_sum",
            metadata={
                "help": (
                    "Aggregation mode for multi-objective rewards. "
                    "Handled here for compatibility with TRL versions that do not expose "
                    "GRPOConfig.multi_objective_aggregation."
                )
            },
        )


def resolve_conditioning_preferences(
    requested_lambda: torch.Tensor,
    *,
    adapter_mode: str,
    advantage_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the parameter-mixing and objective-optimization preferences."""
    for label, mode in (("adapter_mode", adapter_mode), ("advantage_mode", advantage_mode)):
        if mode not in {"requested", "uniform"}:
            raise ValueError(f"{label} must be 'requested' or 'uniform', got {mode!r}")
    uniform = torch.ones_like(requested_lambda) / requested_lambda.shape[-1]
    adapter = requested_lambda if adapter_mode == "requested" else uniform
    advantage = requested_lambda if advantage_mode == "requested" else uniform
    return adapter, advantage


class ParetoGRPOTrainer(GenerationStopsMixin, GRPOTrainer):
    """Sample a preference, generate with its adapter mixture, and optimize weighted advantages."""

    def __init__(
        self,
        pareto_config: ParetoLoRAConfig,
        preference_sampler: PreferenceSampler,
        model_args: ModelConfig | None = None,
        dataset_name: str | None = None,
        adapter_preference_conditioning: str = "requested",
        advantage_preference_conditioning: str = "requested",
        model_device_map_auto: bool = False,
        model_max_memory_gib: int = 58,
        **kwargs,
    ):
        training_args = kwargs.get("args")
        if training_args is None:
            raise ValueError("ParetoGRPOTrainer requires explicit GRPOConfig args.")
        if model_device_map_auto:
            if training_args.use_vllm:
                raise ValueError("model_device_map_auto requires use_vllm=false")
            if int(os.environ.get("WORLD_SIZE", "1")) != 1:
                raise ValueError(
                    "model_device_map_auto is single-process model parallelism; "
                    "do not combine it with a multi-process launcher"
                )

        for label, value in (
            ("adapter_preference_conditioning", adapter_preference_conditioning),
            ("advantage_preference_conditioning", advantage_preference_conditioning),
        ):
            if value not in {"requested", "uniform"}:
                raise ValueError(f"{label} must be 'requested' or 'uniform', got {value!r}")
        self.adapter_preference_conditioning = adapter_preference_conditioning
        self.advantage_preference_conditioning = advantage_preference_conditioning

        # Load and wrap model with PaLoRA before passing to parent
        model_name = kwargs.pop("model", None)
        local_model_source = None
        if isinstance(model_name, str) and model_name:
            load_kwargs = {
                "torch_dtype": torch.bfloat16,
                **_model_source_kwargs(model_args),
                **_model_parallel_load_kwargs(
                    enabled=model_device_map_auto,
                    max_memory_gib=model_max_memory_gib,
                ),
            }
            if model_args is not None:
                if model_args.attn_implementation:
                    load_kwargs["attn_implementation"] = model_args.attn_implementation
            try:
                model = load_text_generation_model(model_name, **load_kwargs)
            except ImportError:
                load_kwargs["attn_implementation"] = "sdpa"
                model = load_text_generation_model(model_name, **load_kwargs)
        else:
            model = model_name

        if isinstance(model_name, str) and model_name:
            local_model_source = resolve_model_source(model_name)
            model.config._name_or_path = local_model_source
            training_args.model_init_kwargs = _reference_model_init_kwargs(
                training_args, model_args
            )
            print(
                f"Model source: {local_model_source}",
                flush=True,
            )

        if pareto_config.base_model_name_or_path is None and isinstance(model_name, str):
            pareto_config.base_model_name_or_path = model_name

        self.pareto_model = ParetoLoRAModel(model, pareto_config)
        if model_device_map_auto:
            self.pareto_model.place_adapters_with_base_layers()
        if local_model_source is not None:
            self.pareto_model.name_or_path = local_model_source
        self.preference_sampler = preference_sampler
        self._rewards_per_func: torch.Tensor | None = None
        self._dataset_name = dataset_name
        self._current_lambda: torch.Tensor | None = None
        self._current_requested_lambda: torch.Tensor | None = None

        trainable, total = self.pareto_model.trainable_parameters()
        print(
            f"PaLoRA: {trainable:,} trainable / {total:,} total parameters "
            f"({100 * trainable / total:.2f}%)"
        )

        if "processing_class" not in kwargs and isinstance(model_name, str) and model_name:
            from lambdamix.prompting import prepare_training_tokenizer

            tokenizer_kwargs = _model_source_kwargs(model_args)
            _tok = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
            prepare_training_tokenizer(
                _tok,
                model_name,
                preserve_native_eos=False,
                generation_kwargs=training_args.generation_kwargs,
                train_dataset=kwargs.get("train_dataset"),
            )
            kwargs["processing_class"] = _tok

        kwargs["model"] = self.pareto_model
        requested_beta = float(training_args.beta)
        training_args.beta = 0.0
        try:
            super().__init__(**kwargs)
        finally:
            training_args.beta = requested_beta
        self.beta = requested_beta
        print(
            "LAMBDAMIX_KL_REFERENCE mode=adapter_disabled_no_copy "
            f"beta={requested_beta:g} ref_model_allocated={self.ref_model is not None}",
            flush=True,
        )

    def _calculate_rewards(self, *args, **kwargs):
        """Override to stash per-function rewards for GDPO advantage computation."""
        rewards_per_func = super()._calculate_rewards(*args, **kwargs)
        self._rewards_per_func = rewards_per_func.clone()
        return rewards_per_func

    def _generate_and_score_completions(self, inputs):
        """Generate completions and replace advantages with GDPO-weighted ones."""

        # Sample preference for this step
        mode = "train" if self.model.training else "eval"
        if mode == "train":
            lambda_vec = self.preference_sampler.step_lambda(self.state.global_step)
        else:
            lambda_vec = self.preference_sampler.get_uniform()

        requested_lambda = lambda_vec
        adapter_lambda, advantage_lambda = resolve_conditioning_preferences(
            requested_lambda,
            adapter_mode=self.adapter_preference_conditioning,
            advantage_mode=self.advantage_preference_conditioning,
        )
        self._current_requested_lambda = requested_lambda
        self._current_lambda = adapter_lambda
        self.pareto_model.set_preference(adapter_lambda)

        # Generate completions for the selected preference.
        output = super()._generate_and_score_completions(inputs)

        # Replace GRPO advantages with GDPO advantages weighted by lambda
        if self._rewards_per_func is not None:
            full_advantages = compute_gdpo_advantages(
                self._rewards_per_func,
                advantage_lambda.to(self._rewards_per_func.device),
                self.args.num_generations,
                aggregation=getattr(self.args, "multi_objective_aggregation", "normalize_then_sum"),
            )

            local_size = len(output["advantages"])
            start = self.accelerator.process_index * local_size
            output["advantages"] = full_advantages[start : start + local_size]

        return output

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Set the active scalar or per-sample preference before policy loss."""
        if self._current_lambda is not None:
            active_preferences = self._current_lambda.to(self.accelerator.device)
            self.pareto_model.set_preference(active_preferences)

        if "advantages" in inputs:
            self._last_advantages = inputs["advantages"].detach().clone()

        loss = super().compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )

        return loss

    def _save_adapters(self, output_dir: str):
        """Save PaLoRA adapters and tokenizer to output_dir."""
        import os

        os.makedirs(output_dir, exist_ok=True)
        adapter_path = os.path.join(output_dir, "pareto_adapters.pt")
        metadata = {
            "reward_funcs": list(self.reward_func_names),
            "dataset_name": self._dataset_name,
            "max_completion_length": self.args.max_completion_length,
            "adapter_preference_conditioning": self.adapter_preference_conditioning,
            "advantage_preference_conditioning": self.advantage_preference_conditioning,
        }
        self.pareto_model.save_pareto_adapters(
            adapter_path,
            metadata=metadata,
        )
        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)
        from lambdamix.run_manifest import sanitize_checkpoint_metadata

        sanitize_checkpoint_metadata(output_dir)
        print(f"PaLoRA checkpoint saved to {output_dir}")

    def save_model(self, output_dir: str = None, **kwargs):
        """Save PaLoRA adapters and tokenizer (skip full base model)."""
        if output_dir is None:
            output_dir = self.args.output_dir
        if self.args.should_save:
            self._save_adapters(output_dir)

    def _save(self, output_dir=None, state_dict=None):
        """Save adapter weights without serializing the full base model."""
        if output_dir is None:
            output_dir = self.args.output_dir
        if self.args.should_save:
            self._save_adapters(output_dir)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        """Resume PaLoRA training from a checkpoint written by ``_save_adapters``."""
        import os

        adapter_path = os.path.join(resume_from_checkpoint, "pareto_adapters.pt")
        if os.path.exists(adapter_path):
            self.pareto_model.load_pareto_adapters(adapter_path)
            print(f"Resumed PaLoRA adapters from {adapter_path}")
        else:
            super()._load_from_checkpoint(resume_from_checkpoint, model)


def main():

    script_args, training_args, model_args = parse_training_args(ParetoExperimentArguments)
    reserve_external_reward_devices(training_args, script_args.dataset_name)
    conditioning_modes = {
        "adapter": script_args.adapter_preference_conditioning,
        "advantage": script_args.advantage_preference_conditioning,
    }
    for label, value in conditioning_modes.items():
        if value not in {"requested", "uniform"}:
            raise ValueError(
                f"{label}_preference_conditioning must be requested or uniform, got {value!r}"
            )

    aggregation = getattr(script_args, "multi_objective_aggregation", None) or getattr(
        training_args, "multi_objective_aggregation", "sum_then_normalize"
    )
    if aggregation not in ("normalize_then_sum", "sum_then_normalize", "weighted_sum"):
        raise ValueError(
            "PaLoRA training requires multi_objective_aggregation in "
            "{normalize_then_sum (default, GDPO), sum_then_normalize (ablation: scalarize "
            "before normalization)} so rollout advantages match the objective weighting."
        )
    if aggregation == "weighted_sum":
        aggregation = "sum_then_normalize"
        if hasattr(training_args, "multi_objective_aggregation"):
            training_args.multi_objective_aggregation = aggregation
        if hasattr(script_args, "multi_objective_aggregation"):
            script_args.multi_objective_aggregation = aggregation

    # Resolve dataset
    train_dataset, eval_dataset = load_training_data(script_args, training_args)

    # Configure reward model directory for model-based rewards (QA)

    # Resolve reward functions
    reward_functions = get_reward_functions(script_args.reward_funcs)

    # Infer num_objectives from reward functions if not set
    num_objectives = script_args.num_objectives
    if num_objectives != len(reward_functions):
        print(
            f"Warning: num_objectives ({num_objectives}) != len(reward_funcs) "
            f"({len(reward_functions)}). Using len(reward_funcs)."
        )
        num_objectives = len(reward_functions)

    # Create PaLoRA config
    pareto_config = ParetoLoRAConfig(
        num_objectives=num_objectives,
        r=script_args.pareto_r,
        lora_alpha=script_args.pareto_alpha,
        target_modules=script_args.pareto_target_modules,
        lora_dropout=script_args.pareto_dropout,
        base_model_name_or_path=model_args.model_name_or_path,
        adapter_dtype=script_args.pareto_adapter_dtype,
        a_init=script_args.pareto_a_init,
        exclude_modules=script_args.pareto_exclude_modules,
        snapshot_base_weights=False,
    )

    dataset_size = len(train_dataset) if train_dataset is not None else 0
    steps_per_epoch, unique_prompts = estimate_grpo_steps_per_epoch(dataset_size, training_args)
    total_steps = (
        training_args.max_steps
        if training_args.max_steps > 0
        else max(floor(training_args.num_train_epochs * steps_per_epoch), 1)
    )
    if script_args.preference_sampling == "coverage":
        print(
            f"Coverage strategy: {steps_per_epoch} steps/epoch "
            f"({dataset_size} examples / {unique_prompts} unique prompts per optimizer step; "
            f"num_generations={training_args.num_generations})"
        )

    preference_sampler = PreferenceSampler(
        num_objectives=num_objectives,
        total_steps=total_steps,
        num_samples=script_args.num_preference_samples,
        temperature=script_args.annealing_temperature,
        strategy=script_args.preference_sampling,
        steps_per_epoch=steps_per_epoch,
    )

    manifest_path = start_run(script_args, training_args, model_args, train_dataset, eval_dataset)

    # Create PaLoRA trainer
    trainer = ParetoGRPOTrainer(
        pareto_config=pareto_config,
        preference_sampler=preference_sampler,
        model_args=model_args,
        dataset_name=script_args.dataset_name,
        adapter_preference_conditioning=script_args.adapter_preference_conditioning,
        advantage_preference_conditioning=script_args.advantage_preference_conditioning,
        model_device_map_auto=script_args.model_device_map_auto,
        model_max_memory_gib=script_args.model_max_memory_gib,
        model=model_args.model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        reward_funcs=reward_functions,
    )

    # Train
    finish_run(trainer, training_args, manifest_path)


if __name__ == "__main__":
    main()
