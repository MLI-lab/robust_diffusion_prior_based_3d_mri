import os
from typing import Dict

# def get_path_by_cluster_name(param, cfg):
    # cluster_name = os.environ.get("CLUSTER_NAME", cfg.cluster_name)
    # if param is None:
        # return None
    # elif cluster_name in param:
        # return param[cluster_name]
    # elif "default" in param:
        # return param["default"]
    # else:
        # raise ValueError(f"Cluster name {cluster_name} not found in {param}.")

from omegaconf import DictConfig, ListConfig
def get_path_by_cluster_name_and_hierarchy(param, cluster_hierarchy : Dict, cluster_name : str):
    """
    Hierarchy is something like:
    A:
        B:
          - b1
          - b2
          - b3
        C:
         -  c1

    This function will first find the path to the cluster_name, e.g. A,C,c1, if cluster_name is c1 and then try the get_path_by_clusternamme for each entry starting with the most specific one.
    """
    # If param is already a plain resolved value (string, list, None) - not a
    # cluster-keyed dict - return it directly without resolution.
    if param is None:
        return None
    if not isinstance(param, (dict, DictConfig)):
        # Already a concrete path (e.g. injected by a preprocessing task).
        if isinstance(param, ListConfig):
            return list(param)
        return param

    cluster_name_env = os.environ.get("CLUSTER_NAME", cluster_name)

    # first find the hierarchy path to the cluster_name_env
    def find_hierarchy_path(hierarchy: Dict, target: str, current_path: list) -> list | None:
        for key, value in hierarchy.items():
            new_path = current_path + [key]
            if key == target:
                return new_path
            if isinstance(value, dict) or isinstance(value, DictConfig):
                result = find_hierarchy_path(value, target, new_path)
                if result is not None:
                    return result
            elif isinstance(value, list) or isinstance(value, ListConfig):
                if target in value:
                    return new_path + [target]
        return None

    hierarchy_path = find_hierarchy_path(cluster_hierarchy, cluster_name_env, [])
    if hierarchy_path is None:
        raise ValueError(f"Cluster name {cluster_name_env} not found in the provided hierarchy.")

    # now try to get the path by progressively less specific names
    for specific_name in reversed(hierarchy_path):
        if specific_name in param:
            return param[specific_name]

    if "default" in param:
        return param["default"]
    else:
        raise ValueError(f"Cluster name {cluster_name_env} not found in {param}.")

def get_path_by_cluster_name(param, cluster_name):
    cluster_name = os.environ.get("CLUSTER_NAME", cluster_name)
    # Some parameters are not used and set to None
    if param is None:
        return None
    elif cluster_name in param:
        return param[cluster_name]
    elif "default" in param:
        return param["default"]
    else:
        raise ValueError(f"Cluster name {cluster_name} not found in {param}.")


def find_hierarchy_path(hierarchy: Dict, target: str, current_path: list) -> list | None:
    for key, value in hierarchy.items():
        new_path = current_path + [key]
        if key == target:
            return new_path
        if isinstance(value, dict):
            result = find_hierarchy_path(value, target, new_path)
            if result is not None:
                return result
        elif isinstance(value, list):
            if target in value:
                return new_path + [target]
    return None