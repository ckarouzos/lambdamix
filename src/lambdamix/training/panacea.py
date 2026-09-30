"""Panacea-augmented GRPO trainer with orthogonality regularization."""

from __future__ import annotations

import os

import torch
from transformers import AutoTokenizer
from trl import GRPOTrainer, ModelConfig

from lambdamix.adapters.lambdamix import PreferenceSampler
from lambdamix.adapters.panacea import PanaceaConfig, PanaceaModel
from lambdamix.generation import GenerationStopsMixin, load_text_generation_model
from lambdamix.prompting import prepare_training_tokenizer
from lambdamix.training.advantages import compute_gdpo_advantages
from lambdamix.training.common import (
    _model_source_kwargs,
    finish_run,
    get_reward_functions,
    load_training_data,
    parse_training_args,
    start_run,
)
from lambdamix.training.lambdamix import ParetoExperimentArguments


class PanaceaGRPOTrainer(GenerationStopsMixin, GRPOTrainer):
    """GRPO trainer with Panacea preference-conditioned adapters."""

    def __init__(
        self,
        panacea_config: PanaceaConfig,
        preference_sampler: PreferenceSampler,
        model_args: ModelConfig | None = None,
        dataset_name: str | None = None,
        **kwargs,
    ):
        args = kwargs.get("args")
        if args is not None and getattr(args, "use_vllm", False):
            raise ValueError(
                "Panacea requires use_vllm=false so rollout generation uses the same "
                "preference-conditioned model state as optimization."
            )

        model_name = kwargs.pop("model", None)
        if isinstance(model_name, str) and model_name:
            load_kwargs = {"torch_dtype": torch.bfloat16, **_model_source_kwargs(model_args)}
            if model_args is not None and model_args.attn_implementation:
                load_kwargs["attn_implementation"] = model_args.attn_implementation
            try:
                model = load_text_generation_model(model_name, **load_kwargs)
            except ImportError:
                load_kwargs["attn_implementation"] = "sdpa"
                model = load_text_generation_model(model_name, **load_kwargs)
        else:
            model = model_name

        if panacea_config.base_model_name_or_path is None and isinstance(model_name, str):
            panacea_config.base_model_name_or_path = model_name

        self.preference_model = PanaceaModel(model, panacea_config)
        self.preference_sampler = preference_sampler
        self._ortho_reg_weight = panacea_config.ortho_reg_weight
        self._rewards_per_func: torch.Tensor | None = None
        self._current_lambda: torch.Tensor | None = None
        self._dataset_name = dataset_name

        trainable, total = self.preference_model.trainable_parameters()
        print(
            f"Panacea: {trainable:,} trainable / {total:,} total parameters "
            f"({100 * trainable / total:.2f}%)"
        )
        print(
            f"  shared_rank={panacea_config.shared_rank}, "
            f"num_objectives={panacea_config.num_objectives}, "
            f"total_rank={panacea_config.total_rank}, "
            f"ortho_reg={panacea_config.ortho_reg_weight}"
        )

        if "processing_class" not in kwargs and isinstance(model_name, str) and model_name:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, **_model_source_kwargs(model_args)
            )
            tokenizer.padding_side = "left"
            prepare_training_tokenizer(
                tokenizer,
                model_name,
                preserve_native_eos=True,
                generation_kwargs=getattr(args, "generation_kwargs", None),
                train_dataset=kwargs.get("train_dataset"),
            )
            kwargs["processing_class"] = tokenizer

        kwargs["model"] = self.preference_model
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
        rewards_per_func = super()._calculate_rewards(*args, **kwargs)
        self._rewards_per_func = rewards_per_func.clone()
        return rewards_per_func

    def _generate_and_score_completions(self, inputs):
        """Generate for one lambda per step (M=1 design, matching PaLoRA)."""
        mode = "train" if self.model.training else "eval"
        if mode == "train":
            samples = self.preference_sampler.sample(self.state.global_step)
            if not samples:
                samples = [self.preference_sampler.get_uniform()]
            # Rotate through preference samples each step
            lambda_vec = samples[self.state.global_step % len(samples)]
        else:
            lambda_vec = self.preference_sampler.get_uniform()

        self._current_lambda = lambda_vec
        self.preference_model.set_preference(lambda_vec)

        output = super()._generate_and_score_completions(inputs)

        # Replace GRPO advantages with GDPO advantages weighted by lambda
        if self._rewards_per_func is not None:
            full_advantages = compute_gdpo_advantages(
                self._rewards_per_func,
                lambda_vec.to(self._rewards_per_func.device),
                self.args.num_generations,
            )
            local_size = len(output["advantages"])
            start = self.accelerator.process_index * local_size
            output["advantages"] = full_advantages[start : start + local_size]

        return output

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Lambda is set in _generate_and_score_completions via self._current_lambda
        if hasattr(self, "_current_lambda") and self._current_lambda is not None:
            self.preference_model.set_preference(self._current_lambda.to(self.accelerator.device))

        loss = super().compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )

        # Add orthogonality regularization
        if self._ortho_reg_weight > 0 and self.model.training:
            ortho_loss = self.preference_model.ortho_reg_loss()
            if return_outputs:
                loss = (loss[0] + self._ortho_reg_weight * ortho_loss, loss[1])
            else:
                loss = loss + self._ortho_reg_weight * ortho_loss

        return loss

    def _save_adapters(self, output_dir: str):

        os.makedirs(output_dir, exist_ok=True)
        adapter_path = os.path.join(output_dir, "pareto_adapters.pt")
        self.preference_model.save_pareto_adapters(
            adapter_path,
            metadata={
                "reward_funcs": list(self.reward_func_names),
                "dataset_name": self._dataset_name,
                "max_completion_length": self.args.max_completion_length,
            },
        )
        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)
        from lambdamix.run_manifest import sanitize_checkpoint_metadata

        sanitize_checkpoint_metadata(output_dir)
        print(f"Panacea checkpoint saved to {output_dir}")

    def save_model(self, output_dir: str = None, **kwargs):
        if output_dir is None:
            output_dir = self.args.output_dir
        if self.args.should_save:
            self._save_adapters(output_dir)

    def _save(self, output_dir=None, state_dict=None):
        if output_dir is None:
            output_dir = self.args.output_dir
        if self.args.should_save:
            self._save_adapters(output_dir)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        """Restore a Panacea adapter checkpoint before Trainer state reload."""

        adapter_path = os.path.join(resume_from_checkpoint, "pareto_adapters.pt")
        if os.path.exists(adapter_path):
            self.preference_model.load_pareto_adapters(adapter_path)
            print(f"Resumed Panacea adapters from {adapter_path}")
        else:
            super()._load_from_checkpoint(resume_from_checkpoint, model)


def main():
    script_args, training_args, model_args = parse_training_args(ParetoExperimentArguments)

    aggregation = getattr(script_args, "multi_objective_aggregation", None) or getattr(
        training_args, "multi_objective_aggregation", "sum_then_normalize"
    )
    if aggregation != "normalize_then_sum":
        raise ValueError(
            "Panacea training requires multi_objective_aggregation=normalize_then_sum."
        )

    train_dataset, eval_dataset = load_training_data(script_args, training_args)

    reward_functions = get_reward_functions(script_args.reward_funcs)

    num_objectives = script_args.num_objectives
    if num_objectives != len(reward_functions):
        print(
            f"Warning: num_objectives ({num_objectives}) != len(reward_funcs) "
            f"({len(reward_functions)}). Using len(reward_funcs)."
        )
        num_objectives = len(reward_functions)

    panacea_config = PanaceaConfig(
        num_objectives=num_objectives,
        shared_rank=script_args.pareto_r,
        lora_alpha=script_args.pareto_alpha,
        target_modules=script_args.pareto_target_modules,
        lora_dropout=script_args.pareto_dropout,
        ortho_reg_weight=getattr(script_args, "ortho_reg_weight", 0.1),
        base_model_name_or_path=model_args.model_name_or_path,
        adapter_dtype=script_args.pareto_adapter_dtype,
        exclude_modules=script_args.pareto_exclude_modules,
    )

    total_steps = training_args.max_steps if training_args.max_steps > 0 else 1000
    preference_sampler = PreferenceSampler(
        num_objectives=num_objectives,
        total_steps=total_steps,
        num_samples=script_args.num_preference_samples,
        temperature=script_args.annealing_temperature,
        strategy=script_args.preference_sampling,
    )

    manifest_path = start_run(script_args, training_args, model_args, train_dataset, eval_dataset)

    trainer = PanaceaGRPOTrainer(
        panacea_config=panacea_config,
        preference_sampler=preference_sampler,
        model_args=model_args,
        dataset_name=script_args.dataset_name,
        model=model_args.model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        reward_funcs=reward_functions,
    )

    finish_run(trainer, training_args, manifest_path)


if __name__ == "__main__":
    main()
