"""Read per-example scores and write evaluation results."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def read_sidecar(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read one evaluation point and its per-example scores."""
    meta: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if "__meta__" in value:
                if meta is not None:
                    raise ValueError(f"{path}:{line_number}: repeated metadata")
                meta = dict(value["__meta__"])
            else:
                if "rewards" not in value:
                    raise ValueError(f"{path}:{line_number}: missing rewards")
                rows.append(value)
    if meta is None or not rows:
        raise ValueError(f"{path}: missing metadata or rows")
    return meta, rows


def _atomic_json_dump(path: Path, payload: dict, *, indent: int | None = 2) -> None:
    """Durably replace ``path`` with one complete JSON document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with open(temporary, "w") as handle:
        json.dump(payload, handle, indent=indent, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
