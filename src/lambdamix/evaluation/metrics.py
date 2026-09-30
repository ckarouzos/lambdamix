"""Shared objective calibration and preference-front metrics."""

import torch

DEFAULT_REFERENCE_POINT = {
    "correctness": 0.0,
    "pku_helpfulness": 0.0,
    "pku_harmlessness": 0.0,
    "synthpai_utility": 0.0,
    "synthpai_privacy": 0.0,
    "ifeval_constraint": 0.0,
}


REWARD_RANGES = {
    "correctness": (0.0, 1.0),
    "pku_helpfulness": (-9.0, 9.0),
    "pku_harmlessness": (-9.0, 9.0),
    "synthpai_utility": (0.0, 1.0),
    "synthpai_privacy": (0.0, 1.0),
    "ifeval_constraint": (0.0, 1.0),
}


def normalize_objectives(values: torch.Tensor, reward_funcs: list[str]) -> torch.Tensor:
    """Affinely scale objectives using frozen calibration ranges."""
    mins = torch.tensor(
        [REWARD_RANGES[f][0] for f in reward_funcs], dtype=values.dtype, device=values.device
    )
    maxs = torch.tensor(
        [REWARD_RANGES[f][1] for f in reward_funcs], dtype=values.dtype, device=values.device
    )
    ranges = maxs - mins
    ranges = ranges.clamp(min=1e-8)  # avoid division by zero
    return (values - mins) / ranges


def normalization_diagnostics(values: torch.Tensor, reward_funcs: list[str]) -> dict:
    """Disclose calibration-bound extrapolation without changing any metric."""
    bounds = [
        [float(REWARD_RANGES[name][0]), float(REWARD_RANGES[name][1])] for name in reward_funcs
    ]
    below_counts = []
    above_counts = []
    raw_min = []
    raw_max = []
    for index, (lower, upper) in enumerate(bounds):
        column = values[:, index]
        below_counts.append(int((column < lower).sum().item()))
        above_counts.append(int((column > upper).sum().item()))
        raw_min.append(float(column.min().item()))
        raw_max.append(float(column.max().item()))
    return {
        "normalization_bounds": bounds,
        "normalization_extrapolation": {
            "present": bool(any(below_counts) or any(above_counts)),
            "below_counts": below_counts,
            "above_counts": above_counts,
            "raw_min": raw_min,
            "raw_max": raw_max,
            "clipped": False,
        },
    }


def generate_simplex_grid(num_objectives: int, resolution: int) -> list[torch.Tensor]:
    """Generate the paper two-objective preference grid."""
    if num_objectives != 2 or resolution < 1:
        raise ValueError("The paper grid requires two objectives and a positive resolution")
    return [torch.tensor([i / resolution, 1 - i / resolution]) for i in range(resolution + 1)]


def find_pareto_front(objective_values: torch.Tensor) -> torch.Tensor:
    """Return boolean mask of non-dominated solutions (all objectives maximized)."""
    n = objective_values.shape[0]
    is_pareto = torch.ones(n, dtype=torch.bool)

    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i == j or not is_pareto[j]:
                continue
            if (objective_values[j] >= objective_values[i]).all() and (
                objective_values[j] > objective_values[i]
            ).any():
                is_pareto[i] = False
                break

    return is_pareto


def compute_hypervolume(
    objective_values: torch.Tensor, reference_point: torch.Tensor | None = None
) -> float:
    """Compute hypervolume indicator for the Pareto front."""
    if not torch.isfinite(objective_values).all():
        raise ValueError("Nonfinite objective values cannot define hypervolume")
    if reference_point is not None and not torch.isfinite(reference_point).all():
        raise ValueError("Nonfinite hypervolume reference point")
    pareto_mask = find_pareto_front(objective_values)
    pareto_points = objective_values[pareto_mask].numpy()

    if len(pareto_points) == 0:
        return 0.0

    if reference_point is None:
        reference_point = torch.zeros(objective_values.shape[1])
    ref = reference_point.numpy()

    num_obj = objective_values.shape[1]

    if num_obj == 1:
        best = float(pareto_points[:, 0].max())
        return max(0.0, best - float(ref[0]))

    if num_obj == 2:
        return _hypervolume_2d(pareto_points, ref)

    raise ValueError("The paper evaluates at most two objectives")


def _hypervolume_2d(points, ref) -> float:
    """Exact 2D hypervolume via sorting."""
    valid = points[(points[:, 0] > ref[0]) & (points[:, 1] > ref[1])]
    if len(valid) == 0:
        return 0.0

    sorted_idx = valid[:, 0].argsort()[::-1]
    sorted_pts = valid[sorted_idx]

    hv = 0.0
    prev_y = ref[1]
    for pt in sorted_pts:
        hv += (pt[0] - ref[0]) * (pt[1] - prev_y)
        prev_y = max(prev_y, pt[1])

    return float(hv)


def compute_inner_product(
    lambdas: list[list[float]] | torch.Tensor,
    objective_values: torch.Tensor,
) -> float:
    """Inner-product (IP) steering metric (Panacea, Zhong et al., 2024)."""
    lam = torch.as_tensor(lambdas, dtype=objective_values.dtype)
    r = objective_values
    if lam.shape != r.shape:
        return float("nan")
    ips = (lam * r).sum(dim=1)
    return float(ips.mean())


def infer_reference_point(reward_funcs: list[str]) -> torch.Tensor:
    """Infer a conservative reference point from reward ranges."""
    return torch.tensor(
        [DEFAULT_REFERENCE_POINT[name] for name in reward_funcs], dtype=torch.float32
    )


def front_metrics(objective_values, reward_funcs, *, lambdas=None, reference_point=None) -> dict:
    """Compute the shared calibrated metrics for an evaluated preference front."""
    values = torch.as_tensor(objective_values, dtype=torch.float64).detach().cpu()
    normalized = normalize_objectives(values, reward_funcs)
    mask = find_pareto_front(normalized)
    reference = (
        torch.as_tensor(reference_point, dtype=torch.float32)
        if reference_point is not None
        else infer_reference_point(reward_funcs)
    )
    result = {
        "reference_point": reference.tolist(),
        "pareto_front_indices": mask.nonzero().squeeze(-1).tolist(),
        "hypervolume": compute_hypervolume(normalized, reference),
        "num_points": len(values),
        "num_pareto": int(torch.unique(values[mask], dim=0).shape[0]),
        "span": (values.max(dim=0).values - values.min(dim=0).values).tolist(),
        **normalization_diagnostics(values, reward_funcs),
    }
    if lambdas is not None:
        weights = torch.as_tensor(lambdas, dtype=values.dtype)
        if weights.shape != values.shape:
            raise ValueError("Preference and score arrays must have the same shape")
        result["inner_product"] = compute_inner_product(weights, normalized)
        result["reversals"] = []
        for axis in range(values.shape[1]):
            order = torch.argsort(weights[:, axis], stable=True)
            weight_steps = torch.diff(weights[order, axis])
            score_steps = torch.diff(values[order, axis])
            result["reversals"].append(int(((weight_steps > 0) & (score_steps < -1e-6)).sum()))
    return result
