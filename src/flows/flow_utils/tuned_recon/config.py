from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

def normalize_protocol_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Fill in protocol defaults for a nested flow config."""
    data = deepcopy(cfg)
    if "data" not in data or "reconstruction" not in data:
        raise ValueError("Flow config must define the nested 'data' and 'reconstruction' sections.")
    data.setdefault("protocol", {}).setdefault("mode", "standard")
    data.setdefault("protocol", {}).setdefault("smoke_test", data.get("smoke", {}).get("enabled", False))
    data.setdefault("smoke", {}).setdefault("enabled", bool(data["protocol"].get("smoke_test", False)))
    data.setdefault("smoke", {}).setdefault("overrides", {})
    data.setdefault("reconstruction", {}).setdefault("prior_mesh_scale", {"enabled": False})
    data["reconstruction"]["prior_mesh_scale"].setdefault("enabled", False)
    return data
