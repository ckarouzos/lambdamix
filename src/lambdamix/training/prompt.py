"""Train a LoRA policy with preference directives and GDPO advantages."""

from __future__ import annotations

import os

from peft import LoraConfig
from trl import GRPOTrainer

from lambdamix.adapters.initialization import apply_ordinary_lora_a_init
from lambdamix.adapters.lambdamix import PreferenceSampler
from lambdamix.generation import GenerationStopsMixin, resolve_model_source
from lambdamix.prompting import inject_preference_into_prompt, render_preference_directive
from lambdamix.run_manifest import public_metadata
from lambdamix.training.advantages import compute_gdpo_advantages
from lambdamix.training.common import (
    _reference_model_init_kwargs,
    estimate_grpo_steps_per_epoch,
    finish_run,
    get_reward_functions,
    load_single_process_model_parallel_policy,
    load_training_data,
    parse_training_args,
    reserve_external_reward_devices,
    start_run,
)
from lambdamix.training.lambdamix import ParetoExperimentArguments


class PromptConditionedGRPOTrainer(GenerationStopsMixin, GRPOTrainer):
    """Condition a GRPO policy on preferences through system-prompt directives."""

    def __init__(
        self,
        preference_sampler: PreferenceSampler,
        reward_func_names: list[str],
        dataset_name: str | None = None,
        aggregation: str = "normalize_then_sum",
        **kwargs,
    ):
        self.preference_sampler = preference_sampler
        # Objective names in reward-matrix column order (for the directive labels).
        self._reward_func_names = list(reward_func_names)

        self._aggregation = aggregation
        self._dataset_name = dataset_name
        self._rewards_per_func = None
        self._current_lambda = None

        model_name = kwargs.get("model")
        if "processing_class" not in kwargs and isinstance(model_name, str) and model_name:
            from transformers import AutoTokenizer

            from lambdamix.prompting import prepare_training_tokenizer

            tok = AutoTokenizer.from_pretrained(model_name)
            args = kwargs.get("args")
            prepare_training_tokenizer(
                tok,
                model_name,
                generation_kwargs=getattr(args, "generation_kwargs", None),
                train_dataset=kwargs.get("train_dataset"),
            )
            kwargs["processing_class"] = tok

        args = kwargs.get("args")
        has_beta = args is not None and hasattr(args, "beta")
        requested_beta = float(args.beta) if has_beta else 0.0
        if has_beta:
            args.beta = 0.0
        try:
            super().__init__(**kwargs)
        finally:
            if has_beta:
                args.beta = requested_beta
        self.beta = requested_beta
        if getattr(self, "ref_model", None) is not None:
            raise RuntimeError(
                "Prompt-conditioned no-copy KL initialization unexpectedly "
                "allocated a reference model"
            )
        print(
            "PROMPT_KL_REFERENCE mode=adapter_disabled_no_copy "
            f"beta={requested_beta:g} ref_model_allocated=false",
            flush=True,
        )

    def _calculate_rewards(self, *args, **kwargs):
        """Stash per-objective rewards for the lambda-weighted advantage recompute."""
        rewards_per_func = super()._calculate_rewards(*args, **kwargs)
        self._rewards_per_func = rewards_per_func.clone()
        return rewards_per_func

    def _generate_and_score_completions(self, inputs):
        """Inject lambda into the prompt, then replace advantages with GDPO ones."""
        mode = "train" if self.model.training else "eval"
        if mode == "train":
            lambda_vec = self.preference_sampler.step_lambda(self.state.global_step)
        else:
            lambda_vec = self.preference_sampler.get_uniform()
        self._current_lambda = lambda_vec

        directive = render_preference_directive(self._reward_func_names, lambda_vec.tolist())
        for x in inputs:
            x["prompt"] = inject_preference_into_prompt(x["prompt"], directive)

        output = super()._generate_and_score_completions(inputs)

        if self._rewards_per_func is not None:
            full_advantages = compute_gdpo_advantages(
                self._rewards_per_func,
                lambda_vec.to(self._rewards_per_func.device),
                self.args.num_generations,
                aggregation=self._aggregation,
            )

            local_size = len(output["advantages"])
            start = self.accelerator.process_index * local_size
            output["advantages"] = full_advantages[start : start + local_size]

        return output

    def _write_metadata(self, output_dir: str):
        """Persist reward funcs / dataset so the eval sweep can reconstruct the run."""
        import json

        os.makedirs(output_dir, exist_ok=True)
        meta = {
            "conditioning": "prompt",
            "reward_funcs": list(self._reward_func_names),
            "dataset_name": self._dataset_name,
            "max_completion_length": self.args.max_completion_length,
            "multi_objective_aggregation": self._aggregation,
            "base_model_name_or_path": getattr(
                getattr(self.model, "config", None), "_name_or_path", None
            ),
        }
        with open(os.path.join(output_dir, "lambdamix_promptcond_meta.json"), "w") as f:
            json.dump(public_metadata(meta), f, indent=2)

    def save_model(self, output_dir: str = None, **kwargs):
        """Save the PEFT adapter + tokenizer (via TRL) plus our metadata sidecar."""
        if output_dir is None:
            output_dir = self.args.output_dir
        super().save_model(output_dir, **kwargs)
        if self.args.should_save:
            self._write_metadata(output_dir)


def main():
    script_args, training_args, model_args = parse_training_args(ParetoExperimentArguments)
    reserve_external_reward_devices(training_args, script_args.dataset_name)

    aggregation = getattr(script_args, "multi_objective_aggregation", None) or getattr(
        training_args, "multi_objective_aggregation", "normalize_then_sum"
    )
    if aggregation not in ("normalize_then_sum", "sum_then_normalize", "weighted_sum"):
        raise ValueError(
            "Prompt-conditioned training requires multi_objective_aggregation in "
            "{normalize_then_sum (default, GDPO), sum_then_normalize (ablation)} so the "
            "rollout advantages match the objective weighting used by LambdaMix."
        )
    if aggregation == "weighted_sum":
        aggregation = "sum_then_normalize"
        if hasattr(training_args, "multi_objective_aggregation"):
            training_args.multi_objective_aggregation = aggregation
        if hasattr(script_args, "multi_objective_aggregation"):
            script_args.multi_objective_aggregation = aggregation

    # Resolve dataset
    train_dataset, eval_dataset = load_training_data(script_args, training_args)

    # Resolve reward functions
    reward_functions = get_reward_functions(script_args.reward_funcs)

    num_objectives = script_args.num_objectives
    if num_objectives != len(reward_functions):
        print(
            f"Warning: num_objectives ({num_objectives}) != len(reward_funcs) "
            f"({len(reward_functions)}). Using len(reward_funcs)."
        )
        num_objectives = len(reward_functions)

    lora_config = LoraConfig(
        r=script_args.pareto_r,
        lora_alpha=script_args.pareto_alpha,
        lora_dropout=script_args.pareto_dropout,
        target_modules=script_args.pareto_target_modules or ["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )

    # Preference sampler: identical schedule to LambdaMix (coverage rotation / annealing).
    total_steps = training_args.max_steps if training_args.max_steps > 0 else 1000
    steps_per_epoch = 0
    if script_args.preference_sampling == "coverage":
        dataset_size = len(train_dataset) if train_dataset is not None else 0
        if dataset_size > 0:
            steps_per_epoch, unique_prompts = estimate_grpo_steps_per_epoch(
                dataset_size, training_args
            )
            print(
                f"Coverage strategy: {steps_per_epoch} steps/epoch "
                f"({dataset_size} examples / {unique_prompts} unique prompts per optimizer step)"
            )

    preference_sampler = PreferenceSampler(
        num_objectives=num_objectives,
        total_steps=total_steps,
        num_samples=script_args.num_preference_samples,
        temperature=script_args.annealing_temperature,
        strategy=script_args.preference_sampling,
        steps_per_epoch=steps_per_epoch,
    )

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
        preserve_native_eos=True,
        generation_kwargs=training_args.generation_kwargs,
        train_dataset=train_dataset,
    )

    policy_model = load_single_process_model_parallel_policy(
        local_model_source, script_args, training_args, model_args
    )
    trainer = PromptConditionedGRPOTrainer(
        preference_sampler=preference_sampler,
        reward_func_names=list(script_args.reward_funcs),
        dataset_name=script_args.dataset_name,
        aggregation=aggregation,
        model=policy_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        reward_funcs=reward_functions,
        peft_config=lora_config,
        processing_class=tokenizer,
    )
    apply_ordinary_lora_a_init(
        trainer.model,
        script_args.ordinary_lora_a_init,
        resume_from_checkpoint=training_args.resume_from_checkpoint,
    )

    manifest_path = start_run(script_args, training_args, model_args, train_dataset, eval_dataset)

    finish_run(trainer, training_args, manifest_path)


if __name__ == "__main__":
    main()
