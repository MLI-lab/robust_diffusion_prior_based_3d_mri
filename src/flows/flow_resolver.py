from functools import partial
from omegaconf import DictConfig, ListConfig, OmegaConf

from src.prefect.sweeping import normalize_range_notation


def _normalize_flow_kwargs(args):
    normalized = {}
    for key, value in args.items():
        if key in {"wandb_cfg"}:
            normalized[key] = value
            continue
        if isinstance(value, (DictConfig, ListConfig)):
            value = OmegaConf.to_container(value, resolve=True)
        normalized[key] = normalize_range_notation(value)
    return normalized

def resolve_flow(options, name : str, **args):
    args = _normalize_flow_kwargs(args)
    if name == "tuned_recon_protocol_flow":
        from src.flows.tuned_recon_protocol_flow import tuned_recon_protocol_flow

        infra = {
            "wandb_cfg": args.pop("wandb_cfg"),
            "local_cache_path": args.pop("local_cache_path"),
            "bart_path": args.pop("bart_path", None),
        }
        return partial(tuned_recon_protocol_flow.with_options(**options), cfg=args, **infra)
    else:
        raise ValueError(f"Flow {name} is not defined or not implemented.")
