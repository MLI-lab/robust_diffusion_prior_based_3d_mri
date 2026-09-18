from prefect_ray import RayTaskRunner
from prefect.cache_policies import TASK_SOURCE, INPUTS, NONE
import os
import time
from typing import Optional
import httpx
from prefect.filesystems import LocalFileSystem

def _retry_transient(fn, *, exceptions, attempts: int = 6, delay_seconds: float = 3.0):
    """Retries fn() against a given family of transient-connectivity exceptions."""
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except exceptions:
            if attempt == attempts:
                raise
            time.sleep(delay_seconds)


def _load_block_with_retry(load_fn, *, attempts: int = 6, delay_seconds: float = 3.0):
    return _retry_transient(
        load_fn,
        exceptions=(httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout),
        attempts=attempts,
        delay_seconds=delay_seconds,
    )


def retry_s3_transfer(fn, *, attempts: int = 6, delay_seconds: float = 3.0):
    import botocore.exceptions

    return _retry_transient(
        fn,
        exceptions=(botocore.exceptions.ConnectionError,),
        attempts=attempts,
        delay_seconds=delay_seconds,
    )


def load_local_storage(storage_name : str, storage_path : str) -> LocalFileSystem:
    try:
        local_storage = _load_block_with_retry(lambda: LocalFileSystem.load(storage_name))
    except ValueError:
        print(f"Creating new storage {storage_name}, at path: {storage_path}")
        local_storage = LocalFileSystem(basepath=storage_path)
        local_storage.save(storage_name)
    return local_storage

from prefect.filesystems import RemoteFileSystem

def get_minio_endpoint_url() -> str:
    return os.environ.get("RESROB_MINIO_ENDPOINT_URL", "http://127.0.0.1:9000")


def _apply_endpoint_override(remote_storage: RemoteFileSystem) -> RemoteFileSystem:
    """Points an already-saved block at this process's real MinIO endpoint."""
    override = os.environ.get("RESROB_MINIO_ENDPOINT_URL")
    if not override:
        return remote_storage
    settings = dict(remote_storage.settings or {})
    client_kwargs = dict(settings.get("client_kwargs") or {})
    if client_kwargs.get("endpoint_url") == override:
        return remote_storage
    client_kwargs["endpoint_url"] = override
    settings["client_kwargs"] = client_kwargs
    remote_storage.settings = settings
    # settings is only read when the underlying filesystem is first built,
    # so drop any instance cached from before the override.
    remote_storage._filesystem = None
    return remote_storage


def load_remote_storage(storage_name: str, bucket_name : str) -> RemoteFileSystem:
    try:
        remote_storage = _apply_endpoint_override(
            _load_block_with_retry(lambda: RemoteFileSystem.load(storage_name))
        )
    except ValueError:
        remote_storage = RemoteFileSystem(
            basepath=f"s3://{bucket_name}/", # currently this assume that the bucket is already created
            settings={
                "key": os.environ["RESROB_MINIO_ACCESS_KEY"],
                "secret": os.environ["RESROB_MINIO_SECRET_KEY"],
                "client_kwargs": {"endpoint_url": get_minio_endpoint_url(), "use_ssl": False},
                "use_ssl": False
            })
        remote_storage.save(storage_name)
    return remote_storage

def get_caching_storage():
    return load_remote_storage("resrob-caching", "resrob-caching")

def get_ray_address() -> str:
    return os.environ.get("PREFECT_RAY_ADDRESS", os.environ.get("RAY_ADDRESS", "ray://localhost:10001"))

def get_ray_password() -> Optional[str]:
    return os.environ.get("RAY_REDIS_PASSWORD")


def _forwarded_env_vars(**fixed: str) -> dict:
    """Environment forwarded to Ray workers: fixed values plus WANDB_API_KEY / HF_HOME from the driver, when set."""
    env = dict(fixed)
    for key in ("WANDB_API_KEY", "HF_HOME"):
        if os.environ.get(key):
            env[key] = os.environ[key]
    return env

def get_and_setup_task_runner(name):
    if name == "ray_local":
        return RayTaskRunner(
            address=get_ray_address(),
            init_kwargs={
                "_redis_password": get_ray_password(),
                "log_to_driver": True,
                "runtime_env" : {
                    "working_dir": "./",
                    "excludes": ["*.pyc", "*.pyo", "__pycache__", ".git", "*.ipynb", "*.safetensors", "models-*/**"],
                    "pip": ["dict-hash", "transformers==4.47.0", "sentencepiece", "peft==0.17.0", "seaborn", "pandas", "cd-fvd", "xformers==0.0.22.post7", "torch==2.1.0", "linear_operator", "lpips", "DISTS-pytorch", "mri-nufft[cufinufft,autodiff]", "numpy==1.26.0", "torchkbnufft", "git+https://github.com/ismrmrd/ismrmrd-python", "git+https://github.com/ismrmrd/ismrmrd-python-tools", "datalad", "git-annex"],
                    "env_vars": _forwarded_env_vars(
                        NCCL_P2P_LEVEL="PXB", NCCL_DEBUG="INFO",
                        PREFECT_API_URL="http://localhost:4200/api",
                    ),
                },
                "object_store_memory": None
            }
        )
    elif name == "local":
        return None
    else:
        raise ValueError(f"Task runner {name} is not defined or not implemented.")


from pydantic import BaseModel, model_serializer
import hashlib
class StorageSettings(BaseModel):
    storage_name: str
    bucket_or_base_path: str
    is_remote: bool


def load_file_system(settings: StorageSettings):
    if settings.is_remote:
        return load_remote_storage(settings.storage_name, bucket_name=settings.bucket_or_base_path)
    else:
        return load_local_storage(settings.storage_name, storage_path=settings.bucket_or_base_path)

class StoragePath(BaseModel):
    storage_path: str
    storage_settings : StorageSettings

from contextlib import contextmanager

@contextmanager
def switch_dir_and_upload_directory_on_exit(filesystem: LocalFileSystem | RemoteFileSystem,
        temp_folder: str, fs_folder: str, new_subfolder : str, upload_empty : bool = False):

    temp_subfolder = os.path.join(temp_folder, new_subfolder)
    fs_subfolder = os.path.join(fs_folder, new_subfolder)
    os.makedirs(temp_subfolder, exist_ok=False)

    cwd_saved = os.getcwd()
    os.chdir(temp_subfolder)
    try:
        yield
    finally:
        os.chdir(cwd_saved)
        temp_subfolder_is_empty = (len(os.listdir(temp_subfolder)) == 0)
        if not temp_subfolder_is_empty or upload_empty:
            retry_s3_transfer(lambda: filesystem.put_directory(temp_subfolder, fs_subfolder, overwrite=True))

@contextmanager
def download_directory_to_temp_on_enter(filesystem: LocalFileSystem | RemoteFileSystem,
        temp_folder: str, fs_folder: str, ex_subfolder : str):

    temp_subfolder = os.path.join(temp_folder, ex_subfolder)
    fs_subfolder = os.path.join(fs_folder, ex_subfolder)
    os.makedirs(temp_subfolder, exist_ok=True)

    retry_s3_transfer(lambda: filesystem.get_directory(fs_subfolder, temp_subfolder))

    cwd_saved = os.getcwd()
    os.chdir(temp_subfolder)
    try:
        yield
    finally:
        os.chdir(cwd_saved)