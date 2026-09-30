"""PaLoRA: Pareto Low-Rank Adapters for multi-objective RL."""

import math
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn as nn

from lambdamix.run_manifest import public_metadata


@dataclass
class ParetoLoRAConfig:
    """Configuration for PaLoRA adapters."""

    num_objectives: int = 2
    r: int = 8
    lora_alpha: int = 16
    target_modules: list[str] | None = None
    lora_dropout: float = 0.0
    base_model_name_or_path: str | None = None
    adapter_dtype: str = "auto"
    a_init: str = "rank_scaled"
    exclude_modules: list[str] | None = None
    snapshot_base_weights: bool = True

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj"]
        if self.exclude_modules is None:
            self.exclude_modules = []


def _resolve_adapter_dtype(value: str, base_dtype: torch.dtype) -> torch.dtype:
    """Resolve the trainable adapter dtype without ever defaulting to FP8."""

    normalized = str(value).lower().replace("torch.", "")
    if normalized == "auto":
        if base_dtype in {torch.float16, torch.bfloat16, torch.float32, torch.float64}:
            return base_dtype
        return torch.bfloat16
    choices = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if normalized not in choices:
        raise ValueError(
            "adapter_dtype must be one of auto, float16/fp16, bfloat16/bf16, or float32/fp32"
        )
    return choices[normalized]


class ParetoLoRALinear(nn.Module):
    """Linear layer with T sets of LoRA adapters, one per objective."""

    def __init__(
        self,
        base_layer: nn.Linear,
        num_objectives: int,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        adapter_dtype: str = "auto",
        a_init: str = "rank_scaled",
    ):
        super().__init__()
        self.base_layer = base_layer
        self.num_objectives = num_objectives
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        self.a_init = a_init

        in_features = base_layer.in_features
        out_features = base_layer.out_features
        dtype = _resolve_adapter_dtype(adapter_dtype, base_layer.weight.dtype)

        # T sets of (A, B) matrices — match base layer dtype to avoid fp32 upcasting
        self.lora_A = nn.ParameterList(
            [nn.Parameter(torch.empty(in_features, r, dtype=dtype)) for _ in range(num_objectives)]
        )
        self.lora_B = nn.ParameterList(
            [nn.Parameter(torch.empty(r, out_features, dtype=dtype)) for _ in range(num_objectives)]
        )

        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()

        self._lambda: torch.Tensor | None = None
        self._adapter_enabled = True
        self._merged = False

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize the transposed A factor explicitly, and B with zeros."""
        if self.a_init == "peft":
            bound = 1.0 / math.sqrt(self.lora_A[0].shape[0])
        elif self.a_init == "rank_scaled":
            bound = 1.0 / math.sqrt(self.r)
        else:
            raise ValueError(
                f"LambdaMix a_init must be 'peft' or 'rank_scaled', got {self.a_init!r}."
            )
        for A in self.lora_A:
            nn.init.uniform_(A, -bound, bound)
        for B in self.lora_B:
            nn.init.zeros_(B)

    def set_preference(self, lambda_vec: torch.Tensor):
        """Store a global ``(T,)`` or per-sample ``(batch,T)`` preference."""
        if lambda_vec.ndim not in (1, 2) or lambda_vec.shape[-1] != self.num_objectives:
            raise ValueError(
                "PaLoRA preference must have shape (num_objectives,) or (batch, num_objectives)."
            )
        self._lambda = lambda_vec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)

        if not self._adapter_enabled or self._merged:
            return base_out

        if self._lambda is None:
            # Uniform preference by default
            lam = (
                torch.ones(self.num_objectives, device=x.device, dtype=base_out.dtype)
                / self.num_objectives
            )
        else:
            lam = self._lambda.to(device=x.device, dtype=base_out.dtype)

        if lam.ndim == 2 and lam.shape[0] != x.shape[0]:
            raise ValueError(
                f"Per-sample PaLoRA preferences have batch {lam.shape[0]}, "
                f"but the activation batch is {x.shape[0]}."
            )

        # Compute weighted sum of LoRA outputs
        lora_out = torch.zeros_like(base_out)
        x_dropped = self.lora_dropout(x)
        for t in range(self.num_objectives):
            # x_dropped @ A_t @ B_t
            adapter_input = x_dropped.to(self.lora_A[t].dtype)
            objective_out = adapter_input @ self.lora_A[t] @ self.lora_B[t]
            objective_out = objective_out.to(base_out.dtype)
            if lam.ndim == 1:
                coefficient = lam[t]
            else:
                coefficient = lam[:, t].view(lam.shape[0], *([1] * (objective_out.ndim - 1)))
            lora_out = lora_out + coefficient * objective_out

        return base_out + self.scaling * lora_out


class ParetoLoRAModel(nn.Module):
    """Wraps a base model, replacing target linear layers with ParetoLoRALinear."""

    def __init__(self, base_model: nn.Module, config: ParetoLoRAConfig):
        super().__init__()
        self.base_model = base_model
        self.pareto_config = config
        self._pareto_layers: list[ParetoLoRALinear] = []
        self._pareto_layer_names: list[str] = []

        self._replace_target_layers()
        self._original_weights: dict[int, torch.Tensor] = {}
        if config.snapshot_base_weights:
            self._snapshot_original_weights()
        self.freeze_base()

    def _snapshot_original_weights(self) -> None:
        """Lazily retain pre-merge weights, exactly once per wrapped layer."""
        for layer in self._pareto_layers:
            key = id(layer)
            if key not in self._original_weights:
                self._original_weights[key] = layer.base_layer.weight.data.clone()

    def _replace_target_layers(self):
        """Replace target linear layers with ParetoLoRALinear."""
        for name, module in self.base_model.named_modules():
            if any(
                name == excluded or name.startswith(f"{excluded}.") or f".{excluded}." in name
                for excluded in self.pareto_config.exclude_modules
            ):
                continue
            for target_name in self.pareto_config.target_modules:
                matches = name == target_name or name.endswith(f".{target_name}")
                if matches and isinstance(module, nn.Linear):
                    parent_name, attr_name = self._split_name(name)
                    parent = self._get_module(parent_name)
                    pareto_linear = ParetoLoRALinear(
                        base_layer=module,
                        num_objectives=self.pareto_config.num_objectives,
                        r=self.pareto_config.r,
                        lora_alpha=self.pareto_config.lora_alpha,
                        lora_dropout=self.pareto_config.lora_dropout,
                        adapter_dtype=self.pareto_config.adapter_dtype,
                        a_init=self.pareto_config.a_init,
                    )
                    setattr(parent, attr_name, pareto_linear)
                    self._pareto_layers.append(pareto_linear)
                    self._pareto_layer_names.append(name)
                    break  # Don't double-match
        if not self._pareto_layers:
            raise ValueError(
                "No nn.Linear modules matched LambdaMix targets "
                f"{self.pareto_config.target_modules!r}; exclusions="
                f"{self.pareto_config.exclude_modules!r}."
            )

    def _split_name(self, name: str) -> tuple[str, str]:
        """Split 'a.b.c' into ('a.b', 'c')."""
        parts = name.rsplit(".", 1)
        if len(parts) == 1:
            return "", parts[0]
        return parts[0], parts[1]

    def _get_module(self, name: str) -> nn.Module:
        """Get a submodule by dotted name."""
        if not name:
            return self.base_model
        module = self.base_model
        for attr in name.split("."):
            module = getattr(module, attr)
        return module

    def set_preference(self, lambda_vec: torch.Tensor):
        """Propagate preference vector to all PaLoRA layers."""
        for layer in self._pareto_layers:
            layer.set_preference(lambda_vec)

    @property
    def peft_config(self) -> dict:
        """Expose TRL's no-copy adapter-reference protocol."""
        return {}

    @contextmanager
    def disable_adapter(self):
        """Temporarily expose the frozen base policy for KL reference scoring."""
        previous = [layer._adapter_enabled for layer in self._pareto_layers]
        for layer in self._pareto_layers:
            layer._adapter_enabled = False
        try:
            yield
        finally:
            for layer, enabled in zip(self._pareto_layers, previous):
                layer._adapter_enabled = enabled

    def freeze_base(self):
        """Freeze all non-LoRA parameters."""
        for name, param in self.base_model.named_parameters():
            if "lora_A" not in name and "lora_B" not in name:
                param.requires_grad = False

    def save_pareto_adapters(self, path: str, metadata: dict | None = None):
        """Save only the LoRA adapter weights."""
        adapter_state = {}
        for name, param in self.base_model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                adapter_state[name] = param.data.clone()
        config_payload = {
            "num_objectives": self.pareto_config.num_objectives,
            "r": self.pareto_config.r,
            "lora_alpha": self.pareto_config.lora_alpha,
            "target_modules": self.pareto_config.target_modules,
            "lora_dropout": self.pareto_config.lora_dropout,
            "adapter_dtype": self.pareto_config.adapter_dtype,
            "a_init": self.pareto_config.a_init,
            "exclude_modules": self.pareto_config.exclude_modules,
            "base_model_name_or_path": (
                self.pareto_config.base_model_name_or_path
                or getattr(self.base_model.config, "_name_or_path", None)
            ),
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
        """Load LoRA adapter weights from file."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        adapter_state = checkpoint["adapter_state_dict"]
        model_state = self.base_model.state_dict()
        model_state.update(adapter_state)
        self.base_model.load_state_dict(model_state)

    def merge_weights(self, lambda_vec: torch.Tensor) -> nn.Module:
        """Merge one preference into the base weights until unmerge is called."""
        if lambda_vec.ndim != 1 or lambda_vec.shape[0] != self.pareto_config.num_objectives:
            raise ValueError("PaLoRA merging requires one fixed preference vector.")
        self._snapshot_original_weights()
        self.set_preference(lambda_vec)
        for layer in self._pareto_layers:
            if layer.base_layer.weight.dtype not in {
                torch.float16,
                torch.bfloat16,
                torch.float32,
                torch.float64,
            }:
                raise RuntimeError(
                    "Dense LambdaMix merging is not supported for a quantized/FP8 "
                    f"base weight ({layer.base_layer.weight.dtype}). Use dynamic native "
                    "generation or an adapter-native inference backend."
                )
            lam = lambda_vec.to(layer.base_layer.weight.device)
            # Start from original weights to avoid accumulation on repeated calls
            merged_weight = self._original_weights[id(layer)].clone().to(lam.device)
            for t in range(layer.num_objectives):
                # B_t^T @ A_t^T = (A_t @ B_t)^T
                delta = (layer.lora_A[t] @ layer.lora_B[t]).T
                merged_weight += layer.scaling * lam[t] * delta
            layer.base_layer.weight.data = merged_weight
            layer._merged = True

        return self.base_model

    def unmerge(self) -> None:
        """Restore every wrapped layer's base weight to its pre-merge snapshot."""
        for layer in self._pareto_layers:
            layer._merged = False
            if id(layer) not in self._original_weights:
                continue
            original = self._original_weights[id(layer)]
            layer.base_layer.weight.data = original.clone().to(layer.base_layer.weight.device)

    def trainable_parameters(self) -> tuple[int, int]:
        """Returns (trainable_params, total_params)."""
        trainable = sum(p.numel() for p in self.base_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.base_model.parameters())
        return trainable, total

    def place_adapters_with_base_layers(self) -> None:
        """Place each adapter beside its base layer without moving a sharded model."""

        for layer in self._pareto_layers:
            device = layer.base_layer.weight.device
            for parameter in (*layer.lora_A, *layer.lora_B):
                parameter.data = parameter.data.to(device=device)

    @property
    def pareto_layer_names(self) -> tuple[str, ...]:
        """Exact wrapped module names for architecture/runtime audits."""

        return tuple(self._pareto_layer_names)

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.base_model.generate(*args, **kwargs)

    @property
    def config_base(self):
        """Access the base model's config."""
        return self.base_model.config

    def __getattr__(self, name: str):
        """Delegate attribute access to base_model for compatibility."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)


class PreferenceSampler:
    """Samples preference vectors from the T-simplex for PaLoRA training."""

    def __init__(
        self,
        num_objectives: int,
        total_steps: int,
        num_samples: int = 4,
        temperature: float = 1.0,
        strategy: str = "coverage",
        steps_per_epoch: int = 0,
    ):
        self.num_objectives = num_objectives
        self.total_steps = max(total_steps, 1)
        self.num_samples = num_samples
        self.temperature = temperature
        if strategy != "coverage":
            raise ValueError("Only the paper coverage schedule is supported")
        self.strategy = strategy
        self.steps_per_epoch = steps_per_epoch  # For coverage strategy

        # Pre-generate deterministic anchor points on simplex edges + center
        self._anchors = self._generate_anchors()

    def _generate_anchors(self) -> list[torch.Tensor]:
        """Generate deterministic anchor points with enough variety for the requested M."""
        T = self.num_objectives
        anchors = []

        # Simplex corners.
        for i in range(T):
            e = torch.zeros(T)
            e[i] = 1.0
            anchors.append(e)

        # Uniform center.
        anchors.append(torch.ones(T) / T)

        # Add reproducible interior points when M exceeds corners+center.
        extra = max(self.num_samples - len(anchors), 0)
        if extra > 0:
            dist = torch.distributions.Dirichlet(torch.ones(T))
            with torch.random.fork_rng():
                torch.manual_seed(0)
                for point in dist.sample((extra,)):
                    anchors.append(point)
        return anchors

    def sample(self, global_step: int) -> list[torch.Tensor]:
        """Return M preference vectors for the current training step."""
        return self._sample_deterministic(global_step)

    def _sample_deterministic(self, global_step: int) -> list[torch.Tensor]:
        """Deterministic annealing: lambda_m(tau) = tilde_lambda_m^(tau/Q) / sum."""
        tau = global_step / self.total_steps  # Progress in [0, 1]
        exponent = tau / self.temperature  # tau / Q

        samples = []
        for m in range(self.num_samples):
            # Cycle through anchors
            anchor = self._anchors[m % len(self._anchors)]
            annealed = (anchor + 1e-8) ** exponent
            annealed = annealed / annealed.sum()
            samples.append(annealed)
        return samples

    def step_lambda(self, global_step: int) -> torch.Tensor:
        """Return the single preference vector to condition on at ``global_step``."""
        samples = self.sample(global_step)
        epoch = global_step // self.steps_per_epoch if self.steps_per_epoch > 0 else 0
        within_epoch = (
            global_step % self.steps_per_epoch if self.steps_per_epoch > 0 else global_step
        )
        return samples[(within_epoch + epoch) % len(samples)]

    def get_uniform(self) -> torch.Tensor:
        """Return uniform preference vector (1/T, ..., 1/T)."""
        return torch.ones(self.num_objectives) / self.num_objectives
