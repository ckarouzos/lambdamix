"""Group reward-decoupled normalization and weighted advantages."""

import torch


def compute_gdpo_advantages(
    rewards_per_func: torch.Tensor,
    weights: torch.Tensor,
    num_generations: int,
    aggregation: str = "normalize_then_sum",
) -> torch.Tensor:
    """Compute multi-objective advantages."""
    device = rewards_per_func.device
    weights = weights.to(device)
    if not torch.isfinite(rewards_per_func).all():
        bad = (~torch.isfinite(rewards_per_func)).nonzero().tolist()[:8]
        raise ValueError(
            f"Nonfinite training rewards at [row, objective] {bad}; refusing imputation"
        )
    if not torch.isfinite(weights).all():
        raise ValueError("Nonfinite objective weights")
    batch_size, num_funcs = rewards_per_func.shape
    num_prompts = batch_size // num_generations

    if aggregation not in {"normalize_then_sum", "sum_then_normalize", "weighted_sum"}:
        raise ValueError(f"Unsupported multi-objective aggregation: {aggregation!r}.")

    if aggregation in {"sum_then_normalize", "weighted_sum"}:
        # Ablation: weighted sum of RAW rewards (no per-objective normalization).
        pre_bn = (rewards_per_func * weights.unsqueeze(0)).sum(dim=1)
    else:
        normalized = torch.zeros_like(rewards_per_func)
        for i in range(num_funcs):
            obj_rewards = rewards_per_func[:, i]
            groups = obj_rewards.view(num_prompts, num_generations)
            mean = groups.mean(dim=1, keepdim=True)
            std = groups.std(dim=1, keepdim=True)
            normalized[:, i] = ((groups - mean) / (std + 1e-4)).view(-1)

        pre_bn = (normalized * weights.unsqueeze(0)).sum(dim=1)

    advantages = (pre_bn - pre_bn.mean()) / (pre_bn.std() + 1e-4)

    if not torch.isfinite(advantages).all():
        raise ValueError("Nonfinite advantages after reward aggregation/normalization")

    return advantages.detach()
