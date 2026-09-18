
from pydantic import BaseModel, ConfigDict, model_serializer
from omegaconf import DictConfig, OmegaConf, ListConfig
from typing import Dict
import dict_hash
import hashlib


class CacheableDictConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    cfg: DictConfig

    @model_serializer
    def ser(self) -> dict:
        # Convert to a plain dict (with interpolation resolved if needed)
        return dict_hash.sha256(OmegaConf.to_container(self.cfg, resolve=True))

class CacheableDict(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    cfg: Dict

    @model_serializer
    def ser(self) -> dict:
        # Convert to a plain dict (with interpolation resolved if needed)
        return dict_hash.sha256(self.cfg)

class CacheableListConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    cfg: ListConfig

    @model_serializer
    def ser(self) -> str:
        # Convert to a plain dict (with interpolation resolved if needed)
        return hashlib.sha256(OmegaConf.to_yaml(self.cfg).encode("utf-8")).hexdigest()


class CacheableList(BaseModel):
    """Serialisable wrapper for a plain Python list (e.g., HP override strings)."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    cfg: list

    @model_serializer
    def ser(self) -> str:
        import json
        return hashlib.sha256(json.dumps(self.cfg, sort_keys=True, default=str).encode()).hexdigest()


def hydra_config_to_cacheable_dict(cfg: DictConfig) -> CacheableDictConfig:
    """
    Convert a Hydra DictConfig to a CacheableDictConfig.
    """
    ret = {}
    for key, value in cfg.items():
        if isinstance(value, DictConfig):
            ret[key + "_cfg"] = CacheableDictConfig(cfg=value)
        elif isinstance(value, ListConfig):
            ret[key + "_cfg"] = CacheableListConfig(cfg=value)
        else:
            ret[key] = value
    return ret
