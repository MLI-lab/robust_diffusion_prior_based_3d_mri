
from collections.abc import Mapping, Sequence
from typing import Dict, List, Any, Tuple
from itertools import product
import re
from pandas import DataFrame

_RANGE_EXPR = re.compile(
    r"^range\(\s*(-?\d+)\s*,\s*(-?\d+)(?:\s*,\s*(-?\d+))?\s*\)$"
)


def _parse_range_expr(value: str) -> List[int] | None:
    match = _RANGE_EXPR.fullmatch(value)
    if match is None:
        return None

    start = int(match.group(1))
    end = int(match.group(2))
    step = int(match.group(3)) if match.group(3) is not None else 1

    if step == 0:
        raise ValueError(f"Invalid sweep range expression '{value}': step cannot be 0")

    return list(range(start, end, step))


def normalize_range_notation(value: Any) -> Any:
    """Recursively expand `range(start, stop[, step])` strings into lists."""
    if isinstance(value, str):
        parsed = _parse_range_expr(value)
        return parsed if parsed is not None else value

    if isinstance(value, Mapping):
        return {k: normalize_range_notation(v) for k, v in value.items()}

    if isinstance(value, list):
        return [normalize_range_notation(v) for v in value]

    if isinstance(value, tuple):
        return tuple(normalize_range_notation(v) for v in value)

    return value


def _normalize_sweep_values(value: Any) -> List[Any]:
    if isinstance(value, str):
        parsed = _parse_range_expr(value)
        return parsed if parsed is not None else [value]

    if isinstance(value, Sequence):
        return list(value)

    return [value]


def get_hydra_sweep_combos(overrides: Dict[str, Any]) -> List[List[Tuple[str, Any]]]:
    """
        Cartesian products for overrides of the form:
        - +param1: [value1, value2]
        - +param2: [value3, value4]
        - +param3: range(0,4)      # Python-style stop-exclusive -> 0,1,2,3
        - +param4: range(0,4,1)    # Python-style stop-exclusive -> 0,1,2,3

        # Output: [[+param1=value1, +param2=value3], [+param1=value1, +param2=value4], [+param1=value2, +param2=value3], [+param1=value2, +param2=value4]]
        Output: [[(+param1, value1), (+param2, value3)], [(+param1, value1), (+param2, value4)], [(+param1, value2), (+param2, value3)], [(+param1, value2), (+param2, value4)]]

        A blank/null `overrides` (e.g. an empty `sweep:` key in YAML) is treated the same as `{}`.
    """
    if overrides is None:
        overrides = {}
    keys = list(overrides.keys())
    values = [_normalize_sweep_values(overrides[key]) for key in keys]
    # return [ [f"{k}={v}" for k, v in zip(overrides.keys(), combo)] for combo in product(*values)]
    return [ [ (k,v) for k, v in zip(overrides.keys(), combo)] for combo in product(*values)]

def get_prefixed_sweep_combo(prefix : str, sweep_combo: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Adds a prefix to each key in the sweep combo."""
    return [(f"{prefix}.{k}", v) for k, v in sweep_combo]

def _format_hydra_override_value(value: Any) -> str:
    if isinstance(value, str) and "," in value:
        return "'" + value.replace("'", "\\'") + "'"
    return str(value)


def get_hydra_overwrite_list(sweep_combo: List[Tuple[str, Any]]) -> List[str]:
    return [f"{k}={_format_hydra_override_value(v)}" for k, v in sweep_combo]

def get_hydra_overwrite_str(sweep_combo: List[Tuple[str, str]]) -> str:
    return ",".join(get_hydra_overwrite_list(sweep_combo))

def get_hydra_overwrite_str_short(sweep_combo: List[Tuple[str, str]]) -> str:
    """Like get_hydra_overwrite_str but strips the Hydra modifier prefix (++/+/~) and the dotted namespace, keeping only the leaf key name and value.
    """
    parts = []
    for k, v in sweep_combo:
        label_prefix = ""
        key = k
        if ":" in key:
            label_prefix, key = key.split(":", 1)
            label_prefix += ":"
        # strip Hydra modifiers
        key = key.lstrip("+~-")
        # keep only last dotted segment
        key = key.rsplit(".", 1)[-1]
        parts.append(f"{label_prefix}{key}={v}")
    return ",".join(parts)
def get_pandas_dataframe_from_sweep_combos(sweep_combos: List[List[Tuple[str, str]]]) -> DataFrame:
    """
    Converts a list of sweep combinations into a pandas DataFrame.
    """
    if len(sweep_combos) == 0:
        return DataFrame()

    column_names = [key for key, val in sweep_combos[0]]
    data = {key: [] for key in column_names}

    for combo in sweep_combos:
        for key, value in combo:
            if key not in data:
                data[key] = []
            data[key].append(value)

    # Convert to DataFrame
    print(f"Data for DataFrame: {data}")
    print(f"Column names: {column_names}")
    return DataFrame(data, columns=column_names)