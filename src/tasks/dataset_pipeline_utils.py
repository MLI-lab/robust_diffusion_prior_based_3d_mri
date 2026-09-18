from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict

from omegaconf import DictConfig, ListConfig, OmegaConf

def resolve_path(raw_cfg: Any, path_resolver: Callable[[Any], Any]) -> Any:
    if raw_cfg is None:
        return None
    return path_resolver(raw_cfg)


def stable_config_hash(*cfgs: Any, length: int = 8) -> str:
    payload = []
    for cfg in cfgs:
        payload.append(OmegaConf.to_container(cfg, resolve=False) if OmegaConf.is_config(cfg) else cfg)
    raw = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(raw).hexdigest()[:length]


def ensure_dir(path: str | Path) -> str:
    os.makedirs(path, exist_ok=True)
    return str(path)


def write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    os.makedirs(path.parent, exist_ok=True)
    tmp = path.with_name(path.stem + f".{os.getpid()}.tmp{path.suffix}")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)
    os.replace(tmp, path)


def concrete_path_resolver(local_cache_path: str | None = None) -> Callable[[Any], Any]:
    def _resolve(param: Any) -> Any:
        if param is None:
            return None
        if isinstance(param, ListConfig):
            return [_resolve(item) for item in list(param)]
        if isinstance(param, list):
            return [_resolve(item) for item in param]
        if isinstance(param, str) and local_cache_path is not None:
            path = Path(param)
            if not path.is_absolute():
                return str(Path(local_cache_path) / path)
            return param
        if isinstance(param, DictConfig):
            if "default" in param and len(param) == 1:
                return param["default"]
            raise ValueError(
                "Cluster-keyed paths are no longer resolved inside tasks. "
                "Resolve them in main.py/flow and pass concrete paths to the task."
            )
        if isinstance(param, dict):
            if "default" in param and len(param) == 1:
                return param["default"]
            raise ValueError(
                "Cluster-keyed paths are no longer resolved inside tasks. "
                "Resolve them in main.py/flow and pass concrete paths to the task."
            )
        return param

    return _resolve


def cache_base_from_subfolder(local_cache_path: str, subfolder: str | None, default_subfolder: str) -> Path:
    subfolder_path = Path(str(subfolder or default_subfolder))
    if subfolder_path.is_absolute():
        return subfolder_path
    return Path(local_cache_path) / subfolder_path
