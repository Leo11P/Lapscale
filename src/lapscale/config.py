"""Configuration: one YAML with a section per stage. Precedence: CLI flag > --config file > built-in default."""
from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: str | Path | None) -> dict:
    if path is None:
        return {}
    d = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(d, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    return d


def section(cfg: dict, *keys: str) -> dict:
    """Nested section with '-' keys turned into '_' (YAML style -> argparse dest)."""
    d = cfg
    for k in keys:
        d = d.get(k) or {}
    return {str(k).replace("-", "_"): v for k, v in d.items() if not isinstance(v, dict)}
