#!/usr/bin/env python3
"""Assemble fixed-policy evaluations into the common front and sidecar format."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lambdamix.evaluation.artifacts import _atomic_json_dump, read_sidecar
from lambdamix.evaluation.metrics import (
    front_metrics,
)


def _read_point(directory):
    """Validate one point's means, preference, and ordered prompt identifiers."""
    payload = json.loads((directory / "results.json").read_text())
    result = payload["result"]
    preference = payload["meta"].get("lambda")
    if preference is None:
        raise ValueError("Evaluate each specialist with --lambda_vec")
    meta, rows = read_sidecar(directory / "persample/persample.jsonl")
    rewards = result["reward_funcs"]
    if meta["reward_funcs"] != rewards or meta["lambda"] != preference:
        raise ValueError("Point metadata and sidecar disagree")
    ids = [row.get("example_id") for row in rows]
    if not all(ids) or len(set(ids)) != len(ids):
        raise ValueError("Each prompt must have a unique example_id")
    scores = np.asarray([[row["rewards"][key] for key in rewards] for row in rows], dtype=float)
    if scores.shape != (result["total"], len(rewards)) or not np.isfinite(scores).all():
        raise ValueError("Invalid point score matrix")
    if not np.allclose(scores.mean(0), result["objective_values"], atol=1e-6, rtol=0):
        raise ValueError("Sidecar means do not match the recorded point")
    if len(preference) != len(rewards) or min(preference) < 0 or not np.isclose(sum(preference), 1):
        raise ValueError("Specialist preferences must lie on the objective simplex")
    return result, preference, rows, ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Use a new output directory")
    points = [_read_point(directory) for directory in args.point_dirs]
    if len(points) < 2:
        raise ValueError("A specialist front requires at least two points")
    first = points[0][0]
    for result, _, _, ids in points[1:]:
        if ids != points[0][3]:
            raise ValueError("Specialists have different prompt IDs/order")
        for key in ("benchmark", "split", "reward_funcs", "total"):
            if result[key] != first[key]:
                raise ValueError(f"Specialists disagree on {key}")
    points.sort(key=lambda point: point[1][0])
    lambdas = [point[1] for point in points]
    if len({tuple(value) for value in lambdas}) != len(lambdas):
        raise ValueError("Duplicate specialist preferences")
    objectives = torch.tensor(
        [point[0]["objective_values"] for point in points], dtype=torch.float64
    )
    result = {
        "benchmark": first["benchmark"],
        "split": first["split"],
        "reward_funcs": first["reward_funcs"],
        "lambdas": lambdas,
        "objective_values": objectives.tolist(),
        **front_metrics(objectives, first["reward_funcs"], lambdas=lambdas),
    }
    sidecars = args.output_dir / "persample"
    sidecars.mkdir(parents=True)
    for index, (_, preference, rows, _) in enumerate(points):
        header = {
            "__meta__": {
                "point": index,
                "lambda": preference,
                "reward_funcs": first["reward_funcs"],
            }
        }
        (sidecars / f"point{index:02d}.jsonl").write_text(
            "\n".join(json.dumps(row) for row in [header, *rows]) + "\n"
        )
    _atomic_json_dump(args.output_dir / "results.json", result)


if __name__ == "__main__":
    main()
