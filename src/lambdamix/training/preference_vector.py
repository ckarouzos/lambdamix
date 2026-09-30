"""Train one directional DPO adapter for Preference Vector (EACL 2026)."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

from trl import DPOConfig, DPOTrainer, ScriptArguments

from lambdamix.generation import resolve_model_source
from lambdamix.run_manifest import (
    sanitize_checkpoint_metadata,
)
from lambdamix.tasks.pku import PREFERENCE_DIRECTIONS, load_pku_preference_dataset
from lambdamix.training.common import (
    _build_peft_config,
    _reference_model_init_kwargs,
    finish_run,
    parse_training_args,
    start_run,
)


@dataclass
class PreferenceVectorDPOArguments(ScriptArguments):
    """Dataset construction arguments for one published preference direction."""

    preference_direction: str = field(
        default="helpful+",
        metadata={"help": "One of helpful+, helpful-, harmless+, harmless-."},
    )
    max_train_samples: int | None = field(
        default=None,
        metadata={"help": "Optional deterministic prefix cap; None uses the full split."},
    )
    max_eval_samples: int | None = field(
        default=512,
        metadata={"help": "Deterministic evaluation prefix cap."},
    )
    pv_warmup_ratio: float = field(
        default=0.03,
        metadata={"help": "Published DPO warmup fraction, converted to steps."},
    )


class PreferenceVectorDPOTrainer(DPOTrainer):
    """Save directional adapters with portable metadata."""

    def save_model(self, output_dir=None, **kwargs):
        super().save_model(output_dir, **kwargs)
        if self.args.should_save:
            sanitize_checkpoint_metadata(output_dir or self.args.output_dir)


def main() -> None:
    script_args, training_args, model_args = parse_training_args(
        PreferenceVectorDPOArguments, DPOConfig
    )
    direction = script_args.preference_direction
    if direction not in PREFERENCE_DIRECTIONS:
        raise ValueError(
            f"preference_direction must be one of {PREFERENCE_DIRECTIONS}, got {direction!r}"
        )
    if script_args.dataset_name != "pku_preference_vector":
        raise ValueError(
            "The currently qualified published construction uses dataset_name: "
            "pku_preference_vector"
        )
    if not getattr(model_args, "use_peft", False):
        raise ValueError(
            "The paper's Preference Vector comparison is parameter-matched LoRA; set use_peft: true"
        )

    train_dataset = load_pku_preference_dataset(
        script_args.dataset_train_split,
        direction,  # type: ignore[arg-type]
        max_samples=script_args.max_train_samples,
    )
    if not 0.0 <= script_args.pv_warmup_ratio < 1.0:
        raise ValueError("pv_warmup_ratio must be in [0, 1)")
    world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
    effective_batch = (
        training_args.per_device_train_batch_size
        * training_args.gradient_accumulation_steps
        * world_size
    )
    steps_per_epoch = math.ceil(len(train_dataset) / effective_batch)
    total_steps = math.ceil(steps_per_epoch * training_args.num_train_epochs)
    training_args.warmup_steps = math.ceil(script_args.pv_warmup_ratio * total_steps)
    eval_dataset = None
    if script_args.dataset_test_split and training_args.eval_strategy != "no":
        eval_dataset = load_pku_preference_dataset(
            script_args.dataset_test_split,
            direction,  # type: ignore[arg-type]
            max_samples=script_args.max_eval_samples,
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
        train_dataset=train_dataset,
    )

    manifest_path = start_run(script_args, training_args, model_args, train_dataset, eval_dataset)

    trainer = PreferenceVectorDPOTrainer(
        model=local_model_source,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=_build_peft_config(model_args),
    )
    finish_run(trainer, training_args, manifest_path)


if __name__ == "__main__":
    main()
