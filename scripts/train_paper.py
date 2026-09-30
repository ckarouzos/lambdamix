"""Launch a paper configuration with its configured process count and task protocol."""

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args, overrides = parser.parse_known_args()
    root = Path(__file__).resolve().parents[1]
    config = args.config.resolve()
    entries = json.loads((root / "configs/paper/index.json").read_text())
    entry = next((row for row in entries if root / row["config"] == config), None)
    if entry is None:
        parser.error("configuration is not listed in configs/paper/index.json")
    processes = entry["processes"]
    if processes == 1:
        command = [sys.executable, "-m", entry["module"]]
    else:
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node",
            str(processes),
            "--module",
            entry["module"],
        ]
    command += ["--config", str(config), *overrides]
    env = dict(os.environ, **entry["environment"])
    for name, value in entry["environment"].items():
        print(f"{name}={shlex.quote(value)}", flush=True)
    print(shlex.join(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, cwd=root, env=env, check=True)


if __name__ == "__main__":
    main()
