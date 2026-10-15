"""YAML config loading with `_base_` inheritance and dotted CLI overrides."""
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def _deep_merge(base: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str) -> Dict[str, Any]:
    p = Path(path)
    with open(p) as f:
        cfg = yaml.safe_load(f) or {}
    merged: Dict[str, Any] = {}
    for b in cfg.pop("_base_", []):
        merged = _deep_merge(merged, load_config(str(p.parent / b)))
    return _deep_merge(merged, cfg)


def load_many(paths: List[str], overrides: Optional[List[str]] = None) -> Dict[str, Any]:
    """Merge several config files left-to-right, then apply `a.b.c=value` overrides."""
    cfg: Dict[str, Any] = {}
    for p in paths:
        cfg = _deep_merge(cfg, load_config(p))
    for ov in overrides or []:
        key, val = ov.split("=", 1)
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(val)
    return cfg
