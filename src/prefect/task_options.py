"""Per-task resource options forwarded to the Ray task runner."""
from prefect_ray.context import remote_options


def task_options(num_gpus: int = 0):
    return remote_options(num_gpus=num_gpus)
