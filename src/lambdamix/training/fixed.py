"""Training entry point for fixed-preference GRPO and GDPO LoRA policies."""

from dataclasses import dataclass, field

from trl import GRPOConfig, GRPOTrainer, ScriptArguments

from lambdamix.adapters.initialization import apply_ordinary_lora_a_init
from lambdamix.generation import (
    GenerationStopsMixin,
    configure_native_generation_stops,
    resolve_model_source,
)
from lambdamix.training.advantages import compute_gdpo_advantages
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


class NativeGRPOTrainer(GenerationStopsMixin, GRPOTrainer):
    """Standard GRPO with shared native-generation stop handling."""


class GDPOTrainer(NativeGRPOTrainer):
    """GRPOTrainer with GDPO normalize-then-sum advantage computation."""

    def _calculate_rewards(self, *args, **kwargs):
        rewards_per_func = super()._calculate_rewards(*args, **kwargs)
        self._rewards_per_func = rewards_per_func.clone()
        return rewards_per_func

    def _generate_and_score_completions(self, inputs):
        output = super()._generate_and_score_completions(inputs)

        # Replace GRPO advantages with GDPO advantages
        advantages = compute_gdpo_advantages(
            self._rewards_per_func,
            self.reward_weights.to(self._rewards_per_func.device),
            self.num_generations,
        )

        # Slice to local process (super already sliced, we need the same slice)
        local_size = len(output["advantages"])
        start = self.accelerator.process_index * local_size
        output["advantages"] = advantages[start : start + local_size]

        return output


_TRL_HAS_AGGREGATION = hasattr(GRPOConfig, "multi_objective_aggregation")


if _TRL_HAS_AGGREGATION:

    @dataclass
    class ExperimentArguments(ScriptArguments):
        """Extended script arguments for experiment configuration."""

        reward_funcs: list[str] = field(
            default_factory=lambda: ["correctness", "ifeval_constraint"],
            metadata={"help": "List of reward function names from REWARD_REGISTRY."},
        )
        model_device_map_auto: bool = field(
            default=False,
            metadata={"help": "Shard a large frozen base across visible accelerators."},
        )
        model_max_memory_gib: int = field(
            default=58,
            metadata={"help": "Per-accelerator HBM ceiling for model_device_map_auto."},
        )
        ordinary_lora_a_init: str = field(
            default="peft",
            metadata={"help": "Ordinary LoRA A initialization: peft or rank_scaled."},
        )
else:

    @dataclass
    class ExperimentArguments(ScriptArguments):  # type: ignore[no-redef]
        """Extended script arguments for experiment configuration."""

        reward_funcs: list[str] = field(
            default_factory=lambda: ["correctness", "ifeval_constraint"],
            metadata={"help": "List of reward function names from REWARD_REGISTRY."},
        )
        multi_objective_aggregation: str = field(
            default="sum_then_normalize",
            metadata={
                "help": (
                    "Aggregation mode for multi-objective rewards. "
                    "Handled here for compatibility with TRL versions that do not expose "
                    "GRPOConfig.multi_objective_aggregation."
                )
            },
        )
        model_device_map_auto: bool = field(
            default=False,
            metadata={"help": "Shard a large frozen base across visible accelerators."},
        )
        model_max_memory_gib: int = field(
            default=58,
            metadata={"help": "Per-accelerator HBM ceiling for model_device_map_auto."},
        )
        ordinary_lora_a_init: str = field(
            default="peft",
            metadata={"help": "Ordinary LoRA A initialization: peft or rank_scaled."},
        )


def _resolve_aggregation(script_args: ExperimentArguments, training_args: GRPOConfig) -> str:
    """Resolve multi-objective aggregation across TRL versions."""
    return (
        getattr(script_args, "multi_objective_aggregation", None)
        or getattr(training_args, "multi_objective_aggregation", None)
        or "sum_then_normalize"
    )


def verify_trainer_reward_weights(training_args, trainer, num_rewards: int) -> list[float]:
    """Fail closed if TRL did not install the parsed fixed-policy weights."""

    import torch

    configured = getattr(training_args, "reward_weights", None)
    expected = (
        torch.ones(num_rewards, dtype=torch.float32)
        if configured is None
        else torch.as_tensor(configured, dtype=torch.float32)
    )
    actual = torch.as_tensor(trainer.reward_weights, dtype=torch.float32).detach().cpu()
    expected = expected.detach().cpu()
    if actual.shape != expected.shape or not torch.allclose(actual, expected, rtol=0.0, atol=1e-7):
        raise RuntimeError(
            "TRL reward-weight installation mismatch: "
            f"configured={expected.tolist()} trainer={actual.tolist()}"
        )
    verified = [float(value) for value in actual.tolist()]
    print(f"REWARD_WEIGHTS_VERIFIED weights={verified}", flush=True)
    return verified


def main():

    script_args, training_args, model_args = parse_training_args(ExperimentArguments)
    reserve_external_reward_devices(training_args, script_args.dataset_name)

    # Resolve dataset
    train_dataset, eval_dataset = load_training_data(script_args, training_args)

    # Configure reward model directory for model-based rewards (QA)

    # Resolve reward functions
    reward_functions = get_reward_functions(script_args.reward_funcs)

    aggregation = _resolve_aggregation(script_args, training_args)
    use_gdpo = aggregation == "normalize_then_sum" and len(reward_functions) > 1
    trainer_cls = GDPOTrainer if use_gdpo else NativeGRPOTrainer

    local_model_source = resolve_model_source(model_args.model_name_or_path)
    training_args.model_init_kwargs = _reference_model_init_kwargs(training_args, model_args)

    manifest_path = start_run(script_args, training_args, model_args, train_dataset, eval_dataset)

    from transformers import AutoTokenizer

    from lambdamix.prompting import prepare_training_tokenizer

    _tokenizer = AutoTokenizer.from_pretrained(
        local_model_source,
        trust_remote_code=model_args.trust_remote_code,
    )
    prepare_training_tokenizer(
        _tokenizer,
        model_args.model_name_or_path,
        preserve_native_eos=False,
        generation_kwargs=training_args.generation_kwargs,
        train_dataset=train_dataset,
    )
    configure_native_generation_stops(training_args, _tokenizer)

    policy_model = load_single_process_model_parallel_policy(
        local_model_source, script_args, training_args, model_args
    )
    trainer_kwargs = {
        "model": policy_model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "reward_funcs": reward_functions,
        "processing_class": _tokenizer,
    }
    peft_config = _build_peft_config(model_args)
    if peft_config is not None:
        trainer_kwargs["peft_config"] = peft_config

    trainer = trainer_cls(
        **trainer_kwargs,
    )
    if peft_config is not None:
        apply_ordinary_lora_a_init(
            trainer.model,
            getattr(script_args, "ordinary_lora_a_init", "peft"),
            resume_from_checkpoint=training_args.resume_from_checkpoint,
        )
    verify_trainer_reward_weights(training_args, trainer, len(reward_functions))

    # Train (with checkpoint resumption support)
    finish_run(trainer, training_args, manifest_path)


if __name__ == "__main__":
    main()
