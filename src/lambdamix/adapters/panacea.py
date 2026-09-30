"""Panacea: SVD-LoRA with shared + preference-specific singular values."""

from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn as nn

from lambdamix.adapters.lambdamix import _resolve_adapter_dtype
from lambdamix.run_manifest import public_metadata


@dataclass
class PanaceaConfig:
    """Configuration for Panacea adapters."""

    num_objectives: int = 2
    shared_rank: int = 8  # k: preference-agnostic dimensions
    lora_alpha: int = 16
    target_modules: list[str] | None = None
    lora_dropout: float = 0.0
    ortho_reg_weight: float = 0.1  # orthogonality regularization on U, V
    base_model_name_or_path: str | None = None
    adapter_dtype: str = "auto"
    exclude_modules: list[str] | None = None

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj"]
        if self.exclude_modules is None:
            self.exclude_modules = []

    @property
    def total_rank(self) -> int:
        """Total rank = shared (k) + preference-specific (m)."""
        return self.shared_rank + self.num_objectives


class PanaceaLinear(nn.Module):
    """Linear layer with Panacea-style SVD adaptation."""

    def __init__(
        self,
        base_layer: nn.Linear,
        num_objectives: int,
        shared_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        adapter_dtype: str = "auto",
    ):
        super().__init__()
        self.base_layer = base_layer
        self.num_objectives = num_objectives
        self.shared_rank = shared_rank
        self.total_rank = shared_rank + num_objectives
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / self.total_rank

        in_features = base_layer.in_features
        out_features = base_layer.out_features
        dtype = _resolve_adapter_dtype(adapter_dtype, base_layer.weight.dtype)

        # Shared singular vectors
        self.U = nn.Parameter(torch.empty(out_features, self.total_rank, dtype=dtype))
        self.V = nn.Parameter(torch.empty(in_features, self.total_rank, dtype=dtype))

        # Shared (preference-agnostic) singular values
        self.sigma_shared = nn.Parameter(torch.empty(shared_rank, dtype=dtype))

        # Per-layer preference scaling factor (single scalar s)
        self.pref_scale = nn.Parameter(torch.ones(1, dtype=dtype))

        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()
        self._lambda: torch.Tensor | None = None
        self._adapter_enabled = True
        self._merged = False

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.U, mean=0.0, std=0.02)
        nn.init.normal_(self.V, mean=0.0, std=0.02)
        nn.init.zeros_(self.sigma_shared)
        nn.init.zeros_(self.pref_scale)

    def set_preference(self, lambda_vec: torch.Tensor):
        self._lambda = lambda_vec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)

        if not self._adapter_enabled or self._merged:
            return base_out

        if self._lambda is None:
            lam = (
                torch.ones(self.num_objectives, device=x.device, dtype=x.dtype)
                / self.num_objectives
            )
        else:
            lam = self._lambda.to(device=x.device, dtype=self.pref_scale.dtype)

        # Build full singular value vector: [σ_shared, s·λ]
        pref_sigma = self.pref_scale * lam
        sigma = torch.cat([self.sigma_shared, pref_sigma])

        # x @ V * sigma -> (batch, total_rank), then @ U^T -> (batch, out)
        x_dropped = self.lora_dropout(x).to(self.V.dtype)
        lora_out = (x_dropped @ self.V * sigma.unsqueeze(0)) @ self.U.T

        return base_out + self.scaling * lora_out.to(base_out.dtype)

    def ortho_reg_loss(self) -> torch.Tensor:
        """Orthogonality regularization: ||U^T U - I||_F + ||V^T V - I||_F."""
        eye = torch.eye(self.total_rank, device=self.U.device, dtype=self.U.dtype)
        loss_u = torch.norm(self.U.T @ self.U - eye, p="fro")
        loss_v = torch.norm(self.V.T @ self.V - eye, p="fro")
        return loss_u + loss_v


class PanaceaModel(nn.Module):
    """Wraps a base model with Panacea layers."""

    def __init__(self, base_model: nn.Module, config: PanaceaConfig):
        super().__init__()
        self.base_model = base_model
        self.panacea_config = config
        # Alias for compatibility with PaLoRA trainer/eval
        self.pareto_config = config
        self._panacea_layers: list[PanaceaLinear] = []

        self._replace_target_layers()
        self._original_weights: dict[int, torch.Tensor] = {
            id(layer): layer.base_layer.weight.data.clone() for layer in self._panacea_layers
        }
        self.freeze_base()

    @property
    def peft_config(self) -> dict:
        """Expose TRL's adapter-context protocol without claiming PEFT type."""
        return {}

    def _replace_target_layers(self):
        for name, module in self.base_model.named_modules():
            if any(
                name == excluded or name.startswith(f"{excluded}.") or f".{excluded}." in name
                for excluded in self.panacea_config.exclude_modules
            ):
                continue
            for target_name in self.panacea_config.target_modules:
                if (name == target_name or name.endswith(f".{target_name}")) and isinstance(
                    module, nn.Linear
                ):
                    parent_name, attr_name = self._split_name(name)
                    parent = self._get_module(parent_name)
                    panacea_linear = PanaceaLinear(
                        base_layer=module,
                        num_objectives=self.panacea_config.num_objectives,
                        shared_rank=self.panacea_config.shared_rank,
                        lora_alpha=self.panacea_config.lora_alpha,
                        lora_dropout=self.panacea_config.lora_dropout,
                        adapter_dtype=self.panacea_config.adapter_dtype,
                    )
                    setattr(parent, attr_name, panacea_linear)
                    self._panacea_layers.append(panacea_linear)
                    break
        if not self._panacea_layers:
            raise ValueError("No nn.Linear modules matched Panacea targets")

    def _split_name(self, name: str) -> tuple[str, str]:
        parts = name.rsplit(".", 1)
        if len(parts) == 1:
            return "", parts[0]
        return parts[0], parts[1]

    def _get_module(self, name: str) -> nn.Module:
        if not name:
            return self.base_model
        module = self.base_model
        for attr in name.split("."):
            module = getattr(module, attr)
        return module

    def set_preference(self, lambda_vec: torch.Tensor):
        for layer in self._panacea_layers:
            layer.set_preference(lambda_vec)

    @contextmanager
    def disable_adapter(self):
        """Temporarily expose the frozen base policy for KL reference scoring."""
        previous = [layer._adapter_enabled for layer in self._panacea_layers]
        for layer in self._panacea_layers:
            layer._adapter_enabled = False
        try:
            yield
        finally:
            for layer, enabled in zip(self._panacea_layers, previous):
                layer._adapter_enabled = enabled

    def freeze_base(self):
        for name, param in self.base_model.named_parameters():
            if not any(k in name for k in (".U", ".V", ".sigma_shared", ".pref_scale")):
                param.requires_grad = False

    def ortho_reg_loss(self) -> torch.Tensor:
        """Mean AdaLoRA-style orthogonality penalty across adapter factors."""
        if not self._panacea_layers:
            return torch.zeros((), device=next(self.parameters()).device)
        return torch.stack([layer.ortho_reg_loss() for layer in self._panacea_layers]).sum() / (
            2 * len(self._panacea_layers)
        )

    def place_adapters_with_base_layers(self) -> None:
        for layer in self._panacea_layers:
            device = layer.base_layer.weight.device
            for parameter in (layer.U, layer.V, layer.sigma_shared, layer.pref_scale):
                parameter.data = parameter.data.to(device=device)

    def save_pareto_adapters(self, path: str, metadata: dict | None = None):
        adapter_state = {}
        for name, param in self.base_model.named_parameters():
            if any(k in name for k in (".U", ".V", ".sigma_shared", ".pref_scale")):
                adapter_state[name] = param.data.clone()
        config_payload = {
            "num_objectives": self.panacea_config.num_objectives,
            "shared_rank": self.panacea_config.shared_rank,
            "lora_alpha": self.panacea_config.lora_alpha,
            "target_modules": self.panacea_config.target_modules,
            "lora_dropout": self.panacea_config.lora_dropout,
            "adapter_dtype": self.panacea_config.adapter_dtype,
            "exclude_modules": self.panacea_config.exclude_modules,
            "ortho_reg_weight": self.panacea_config.ortho_reg_weight,
            "base_model_name_or_path": (
                self.panacea_config.base_model_name_or_path
                or getattr(self.base_model.config, "_name_or_path", None)
            ),
            "type": "panacea",
        }
        if metadata:
            config_payload.update(metadata)
        torch.save(
            {
                "adapter_state_dict": adapter_state,
                "config": public_metadata(config_payload),
            },
            path,
        )

    def load_pareto_adapters(self, path: str):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        adapter_state = checkpoint["adapter_state_dict"]
        model_state = self.base_model.state_dict()
        model_state.update(adapter_state)
        self.base_model.load_state_dict(model_state)

    def merge_weights(self, lambda_vec: torch.Tensor) -> nn.Module:
        self.set_preference(lambda_vec)
        for layer in self._panacea_layers:
            lam = lambda_vec.to(
                device=layer.pref_scale.device,
                dtype=layer.pref_scale.dtype,
            )
            pref_sigma = layer.pref_scale * lam
            sigma = torch.cat([layer.sigma_shared, pref_sigma]).to(
                device=layer.U.device,
                dtype=layer.U.dtype,
            )
            delta = (layer.U * sigma.unsqueeze(0)) @ layer.V.T
            merged_weight = (
                self._original_weights[id(layer)]
                .clone()
                .to(
                    device=layer.base_layer.weight.device,
                    dtype=layer.base_layer.weight.dtype,
                )
            )
            merged_weight += layer.scaling * delta.to(merged_weight.dtype)
            layer.base_layer.weight.data = merged_weight
            layer._merged = True
        return self.base_model

    def unmerge(self) -> None:
        """Restore the immutable base weights after a dense evaluation sync."""
        for layer in self._panacea_layers:
            layer._merged = False
            layer.base_layer.weight.data.copy_(
                self._original_weights[id(layer)].to(layer.base_layer.weight.device)
            )

    def trainable_parameters(self) -> tuple[int, int]:
        trainable = sum(p.numel() for p in self.base_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.base_model.parameters())
        return trainable, total

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.base_model.generate(*args, **kwargs)

    @property
    def config_base(self):
        return self.base_model.config

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)
