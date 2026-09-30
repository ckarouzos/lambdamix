"""Load LambdaMix and Panacea checkpoints for inference without importing trainers."""

from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from lambdamix import resolve_attn_implementation
from lambdamix.adapters.lambdamix import ParetoLoRAConfig, ParetoLoRAModel
from lambdamix.generation import load_text_generation_model, resolve_model_source
from lambdamix.prompting import ensure_chat_template


def _load_pareto_components(
    model_path: str,
    attn_implementation: str | None,
) -> tuple[AutoTokenizer, nn.Module, dict]:
    """Load tokenizer, preference-conditioned wrapper, and saved metadata."""
    adapter_path = Path(model_path) / "pareto_adapters.pt"
    if not adapter_path.exists():
        raise FileNotFoundError(f"Missing preference-adapter checkpoint: {adapter_path}")

    checkpoint = torch.load(adapter_path, map_location="cpu", weights_only=True)
    config_dict = dict(checkpoint["config"])
    checkpoint_type = config_dict.get("type", "pareto_lora")
    base_model_name = config_dict.get("base_model_name_or_path")
    if not base_model_name:
        raise ValueError(
            "Checkpoint is missing base_model_name_or_path metadata. "
            "Re-save the checkpoint with the updated trainer."
        )

    base_model_source = resolve_model_source(base_model_name)
    config_dict["_resolved_base_model_source"] = base_model_source
    tokenizer_source = (
        model_path if (Path(model_path) / "tokenizer_config.json").exists() else base_model_source
    )
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    except (ValueError, KeyError):
        # Fallback: saved tokenizer_config.json may have invalid tokenizer_class (e.g. "TokenizersBackend")
        tokenizer = AutoTokenizer.from_pretrained(base_model_source)
    ensure_chat_template(tokenizer, base_model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    attn_impl = resolve_attn_implementation(attn_implementation)
    base_model = load_text_generation_model(
        base_model_source,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=attn_impl,
    )

    if checkpoint_type == "panacea":
        from lambdamix.adapters.panacea import PanaceaConfig, PanaceaModel

        preference_config = PanaceaConfig(
            num_objectives=config_dict["num_objectives"],
            shared_rank=config_dict["shared_rank"],
            lora_alpha=config_dict["lora_alpha"],
            target_modules=config_dict["target_modules"],
            lora_dropout=config_dict["lora_dropout"],
            ortho_reg_weight=config_dict.get("ortho_reg_weight", 0.1),
            base_model_name_or_path=base_model_source,
            adapter_dtype=config_dict.get("adapter_dtype", "auto"),
            exclude_modules=config_dict.get("exclude_modules", []),
        )
        preference_model = PanaceaModel(base_model, preference_config)
    else:
        preference_config = ParetoLoRAConfig(
            num_objectives=config_dict["num_objectives"],
            r=config_dict["r"],
            lora_alpha=config_dict["lora_alpha"],
            target_modules=config_dict["target_modules"],
            lora_dropout=config_dict["lora_dropout"],
            base_model_name_or_path=base_model_source,
            adapter_dtype=config_dict.get("adapter_dtype", "auto"),
            a_init=config_dict.get("a_init", "rank_scaled"),
            exclude_modules=config_dict.get("exclude_modules", []),
        )
        preference_model = ParetoLoRAModel(base_model, preference_config)

    preference_model.load_pareto_adapters(str(adapter_path))
    if hasattr(preference_model, "place_adapters_with_base_layers"):
        preference_model.place_adapters_with_base_layers()
    preference_model.eval()

    return tokenizer, preference_model, config_dict
