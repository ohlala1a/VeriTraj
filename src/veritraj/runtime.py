"""Run configuration, metadata and output files."""

from __future__ import annotations

import json
import platform
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml


def load_yaml(path: str | Path) -> dict:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a configuration mapping in {path}")
    return value


def write_json(path: str | Path, value) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(target)


def init_run(output_dir: str | Path, config: dict, seed: int) -> Path:
    output = Path(output_dir)
    if (output / "run.json").exists():
        raise FileExistsError(f"Run already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                cwd=Path(__file__).resolve().parents[2], check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    write_json(output / "run.json", {"created_utc": datetime.now(timezone.utc).isoformat(),
                                    "python": platform.python_version(), "git_commit": commit,
                                    "seed": seed, "config": config})
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    random.seed(seed)
    return output
