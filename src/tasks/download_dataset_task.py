from __future__ import annotations

import logging
import os
import shutil
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from zipfile import BadZipFile, ZipFile, is_zipfile

from prefect import task
from prefect.cache_policies import INPUTS, TASK_SOURCE

from src.prefect.caching import CacheableDictConfig
from src.tasks.dataset_pipeline_utils import cache_base_from_subfolder, concrete_path_resolver, ensure_dir, resolve_path, write_json


def _normalize_download_url(url: str) -> str:
    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    file_id = query.get("fileId")
    if parsed.path.endswith("/file.xhtml") and file_id:
        api_query = {}
        if "version" in query:
            api_query["version"] = query["version"]
        return urlunparse(
            (
                parsed.scheme,
                parsed.netloc,
                f"/api/access/datafile/{file_id}",
                "",
                urlencode(api_query),
                "",
            )
        )
    return url


def _has_hdf5_signature(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(8) == b"\x89HDF\r\n\x1a\n"
    except OSError:
        return False


@contextmanager
def _directory_lock(lock_path: Path, timeout: int = 7200, poll_seconds: float = 5.0):
    start = time.monotonic()
    while True:
        try:
            lock_path.mkdir(parents=True)
            break
        except FileExistsError:
            if time.monotonic() - start > timeout:
                raise TimeoutError(f"Timed out waiting for download lock {lock_path}.")
            time.sleep(poll_seconds)
    try:
        yield
    finally:
        try:
            lock_path.rmdir()
        except OSError:
            logging.warning("Could not remove download lock %s", lock_path)


def _is_valid_zip(path: Path) -> bool:
    try:
        return path.exists() and is_zipfile(path)
    except OSError:
        return False


def _download_url_csv_entry(
    args: tuple[
        tuple[str, Optional[str]],
        str,
        bool,
        bool,
        int,
    ]
) -> bool:
    import requests

    entry, target_dir_raw, exists_ok, validate_hdf5, timeout = args
    target_dir = Path(target_dir_raw)
    url, filename = entry
    fname = filename or url.rsplit("/", 1)[-1].split("?", 1)[0]
    suffix = "" if fname.endswith(".h5") else ".h5"
    out_file = target_dir / f"{fname}{suffix}"
    if exists_ok and out_file.exists():
        if validate_hdf5 and not _has_hdf5_signature(out_file):
            logging.warning("%s is not an HDF5 file; downloading it again.", out_file)
        else:
            return True

    response = requests.get(
        _normalize_download_url(url),
        allow_redirects=True,
        timeout=timeout,
    )
    response.raise_for_status()
    if validate_hdf5 and response.content[:8] != b"\x89HDF\r\n\x1a\n":
        content_type = response.headers.get("content-type", "unknown")
        raise RuntimeError(
            f"Downloaded {url} for {out_file} but response is not HDF5 "
            f"(content-type={content_type!r})."
        )
    tmp = out_file.with_name(
        out_file.stem + f".{os.getpid()}.{uuid.uuid4().hex}.tmp{out_file.suffix}"
    )
    with open(tmp, "wb") as f:
        f.write(response.content)
    os.replace(tmp, out_file)
    return True


@task(
    cache_policy=TASK_SOURCE + INPUTS,
    name="Download Dataset Task",
    tags=["dataset-download", "dataset-preparation"],
    version="1.0",
    retries=0,
)
def download_dataset_task(
    dataset_name: str,
    local_cache_path: str,
    raw_cache_subfolder: str = "raw",
    raw_subfolder: Optional[str] = None,
    raw_path: Optional[str] = None,
    download_cfg: Optional[CacheableDictConfig] = None,
) -> Dict[str, Any]:
    """Download or validate a raw dataset folder."""
    logging.getLogger().setLevel(logging.INFO)
    path_resolver = concrete_path_resolver()

    if raw_path is not None:
        raw_path = Path(resolve_path(raw_path, path_resolver))
        raw_base_path = raw_path.parent
    else:
        raw_base_path = cache_base_from_subfolder(local_cache_path, raw_cache_subfolder, "raw")
        raw_subfolder_name = dataset_name if raw_subfolder is None else str(raw_subfolder)
        raw_path = raw_base_path / raw_subfolder_name

    mode = str(getattr(download_cfg.cfg if download_cfg else {}, "mode", "manual"))
    exists_ok = bool(getattr(download_cfg.cfg if download_cfg else {}, "exists_ok", True))
    marker_path = raw_path / "_download_meta.json"

    if mode in ("manual", "validate", "none"):
        if not raw_path.exists() and not exists_ok:
            raise FileNotFoundError(
                f"{dataset_name}: expected raw dataset at {raw_path}. "
                "Download is configured as manual/validate."
            )
        ensure_dir(raw_path)
        status = "validated" if raw_path.exists() else "created_empty"

    elif mode == "datalad_zip":
        try:
            from datalad.api import clone, get
        except Exception as exc:
            raise RuntimeError("download.mode=datalad_zip requires datalad.") from exc

        source = str(download_cfg.cfg.source)
        if "dataset_dir" in download_cfg.cfg:
            dataset_dir = Path(resolve_path(download_cfg.cfg.dataset_dir, path_resolver))
        else:
            dataset_dir = raw_base_path / str(getattr(download_cfg.cfg, "dataset_subfolder", dataset_name)) / "_datalad"
        zip_relpath = Path(str(download_cfg.cfg.zip_relpath))
        zip_path = dataset_dir / zip_relpath
        extract_dir = raw_path
        lock_timeout = int(getattr(download_cfg.cfg, "lock_timeout", 7200))
        lock_path = raw_base_path / f".{dataset_name}_datalad_zip.lock"

        def _download_single_stream(url: str, dest: Path, timeout: int) -> None:
            import requests

            with requests.get(url, stream=True, timeout=timeout) as response:
                response.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)

        def _download_parallel_ranges(url: str, dest: Path, timeout: int, num_connections: int) -> None:
            import requests

            head = requests.head(url, timeout=timeout, allow_redirects=True)
            head.raise_for_status()
            total_size = int(head.headers.get("content-length", 0) or 0)
            accepts_ranges = head.headers.get("accept-ranges", "").lower() == "bytes"
            if not total_size or not accepts_ranges:
                _download_single_stream(url, dest, timeout)
                return

            with open(dest, "wb") as f:
                f.truncate(total_size)

            part_size = -(-total_size // num_connections)  # ceil div
            byte_ranges = [
                (start, min(start + part_size, total_size) - 1)
                for start in range(0, total_size, part_size)
            ]

            def _fetch_range(start: int, end: int) -> None:
                headers = {"Range": f"bytes={start}-{end}"}
                with requests.get(url, headers=headers, stream=True, timeout=timeout) as response:
                    response.raise_for_status()
                    with open(dest, "r+b") as f:
                        f.seek(start)
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                f.write(chunk)

            with ThreadPoolExecutor(max_workers=num_connections) as executor:
                futures = [executor.submit(_fetch_range, start, end) for start, end in byte_ranges]
                for future in as_completed(futures):
                    future.result()

        def _download_fallback_zip(reason: Exception | None = None) -> None:
            if not getattr(download_cfg.cfg, "fallback_url", None):
                if reason is not None:
                    raise reason
                raise BadZipFile(f"Downloaded archive at {zip_path} is not a valid zip file.")

            fallback_url = str(download_cfg.cfg.fallback_url)
            tmp = zip_path.with_name(zip_path.name + f".{os.getpid()}.tmp")
            ensure_dir(zip_path.parent)
            timeout = int(getattr(download_cfg.cfg, "timeout", 120))
            num_connections = int(getattr(download_cfg.cfg, "fallback_parallel_connections", 1))
            try:
                if num_connections > 1:
                    _download_parallel_ranges(fallback_url, tmp, timeout, num_connections)
                else:
                    _download_single_stream(fallback_url, tmp, timeout)
                os.replace(tmp, zip_path)
            except Exception as fallback_exc:
                tmp.unlink(missing_ok=True)
                message = f"Fallback download from {fallback_url} failed."
                if reason is not None:
                    message = f"Datalad get failed for {zip_path}; " + message
                raise RuntimeError(message) from fallback_exc

        with _directory_lock(lock_path, timeout=lock_timeout):
            if exists_ok and raw_path.exists() and any(raw_path.rglob("*.h5")):
                status = "validated"
            else:
                if not dataset_dir.exists():
                    clone(source=source, path=str(dataset_dir))
                if bool(getattr(download_cfg.cfg, "skip_datalad_get", False)):
                    _download_fallback_zip()
                else:
                    try:
                        get(path=str(zip_path), dataset=str(dataset_dir))
                    except Exception as datalad_exc:
                        _download_fallback_zip(datalad_exc)

                if not _is_valid_zip(zip_path):
                    logging.warning("%s is not a valid zip file; trying fallback download.", zip_path)
                    _download_fallback_zip()
                if not _is_valid_zip(zip_path):
                    raise BadZipFile(f"Downloaded archive at {zip_path} is not a valid zip file.")

                ensure_dir(extract_dir)
                with ZipFile(zip_path, "r") as zf:
                    zf.extractall(extract_dir)
                if bool(getattr(download_cfg.cfg, "remove_archive", False)):
                    zip_path.unlink(missing_ok=True)
                status = "downloaded"

    elif mode == "datalad":
        try:
            from datalad.api import clone, get
        except Exception as exc:
            raise RuntimeError("download.mode=datalad requires datalad.") from exc

        source = str(download_cfg.cfg.source)
        get_paths_cfg = getattr(download_cfg.cfg, "get_paths", None)
        if get_paths_cfg is None:
            get_relpaths = [Path(".")]
        else:
            get_relpaths = [Path(str(path)) for path in list(get_paths_cfg)]
        recursive = bool(getattr(download_cfg.cfg, "recursive", True))

        def _requested_paths_present() -> bool:
            return all((raw_path / relpath).exists() for relpath in get_relpaths)

        if raw_path.exists() and not (raw_path / ".git").exists():
            if any(raw_path.iterdir()):
                if exists_ok and _requested_paths_present():
                    status = "validated"
                    meta = {
                        "dataset_name": dataset_name,
                        "raw_path": str(raw_path),
                        "raw_base_path": str(raw_base_path),
                        "download_mode": mode,
                        "status": status,
                        "counts": None,
                    }
                    write_json(marker_path, meta)
                    return {"dataset_name": dataset_name, "raw": str(raw_path), "meta": str(marker_path), "status": status, "counts": None}
                raise FileExistsError(
                    f"{raw_path} exists and is not a DataLad dataset; remove it or set raw_path/raw_subfolder to an empty target."
                )
            raw_path.rmdir()

        if not raw_path.exists():
            ensure_dir(raw_path.parent)
            clone(source=source, path=str(raw_path))

        get_targets = [raw_path / relpath for relpath in get_relpaths]
        get(path=[str(path) for path in get_targets], dataset=str(raw_path), recursive=recursive)
        status = "downloaded"

    elif mode == "url_csv":
        import csv
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        from tqdm import tqdm

        def _read_url_entries(csv_path: Path) -> list[tuple[str, Optional[str]]]:
            with open(csv_path, newline="") as csvfile:
                sample = csvfile.read(2048)
                csvfile.seek(0)
                try:
                    dialect = csv.Sniffer().sniff(sample, delimiters=",; \t")
                except csv.Error:
                    dialect = csv.excel
                reader = csv.DictReader(csvfile, dialect=dialect)
                if reader.fieldnames and any(name and name.strip().lower() == "url" for name in reader.fieldnames):
                    url_field = next(name for name in reader.fieldnames if name and name.strip().lower() == "url")
                    filename_field = next(
                        (
                            name
                            for name in reader.fieldnames
                            if name and name.strip().lower() in ("filename", "file_name", "name")
                        ),
                        None,
                    )
                    return [
                        (
                            row[url_field].strip(),
                            row.get(filename_field, "").strip() if filename_field is not None else None,
                        )
                        for row in reader
                        if row.get(url_field, "").strip()
                    ]
                csvfile.seek(0)
                rows = list(csv.reader(csvfile, dialect=dialect))
            entries = []
            for row in rows:
                if not row:
                    continue
                first = str(row[0]).strip()
                if not first or first.lower() == "url":
                    continue
                entries.append((first, None))
            return entries

        def _download_url_entries(
            entries: list[tuple[str, Optional[str]]],
            target_dir: Path,
            label: str,
            max_files: Optional[int],
        ) -> int:
            ensure_dir(target_dir)
            selected = entries[: int(max_files)] if max_files is not None and int(max_files) > 0 else entries
            num_workers = max(1, int(getattr(download_cfg.cfg, "num_workers", 1)))
            validate_hdf5 = bool(getattr(download_cfg.cfg, "validate_hdf5", False))
            timeout = int(getattr(download_cfg.cfg, "timeout", 120))
            worker_backend = str(
                getattr(
                    download_cfg.cfg,
                    "worker_backend",
                    "mp" if bool(getattr(download_cfg.cfg, "mp", False)) else "thread",
                )
            ).lower()
            process_backends = {"mp", "process", "processes", "multiprocessing"}
            thread_backends = {"thread", "threads", "threading"}
            if worker_backend not in process_backends | thread_backends:
                raise ValueError(
                    "download.worker_backend must be one of "
                    f"{sorted(process_backends | thread_backends)}, got {worker_backend!r}."
                )

            tasks = [
                (entry, str(target_dir), exists_ok, validate_hdf5, timeout)
                for entry in selected
            ]
            if num_workers == 1 or len(tasks) <= 1:
                return sum(
                    1
                    for task_args in tqdm(tasks, desc=f"Downloading {dataset_name}/{label}")
                    if _download_url_csv_entry(task_args)
                )

            downloaded = 0
            max_workers = min(num_workers, len(tasks))
            if worker_backend in process_backends:
                executor_ctx = ProcessPoolExecutor(
                    max_workers=max_workers,
                    mp_context=mp.get_context("spawn"),
                )
            else:
                executor_ctx = ThreadPoolExecutor(max_workers=max_workers)

            with executor_ctx as executor:
                futures = [
                    executor.submit(_download_url_csv_entry, task_args)
                    for task_args in tasks
                ]
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"Downloading {dataset_name}/{label}",
                ):
                    if future.result():
                        downloaded += 1
            return downloaded

        ensure_dir(raw_path)
        max_files = getattr(download_cfg.cfg, "max_files", None)
        counts = {}
        url_csv_paths = getattr(download_cfg.cfg, "url_csv_paths", None)
        if url_csv_paths is not None:
            per_fold_max = getattr(download_cfg.cfg, "max_files_per_fold", None)
            for fold, csv_path_raw in dict(url_csv_paths).items():
                csv_path = Path(resolve_path(csv_path_raw, path_resolver))
                counts[str(fold)] = _download_url_entries(
                    _read_url_entries(csv_path),
                    raw_path / str(fold),
                    str(fold),
                    per_fold_max if per_fold_max is not None else max_files,
                )
        else:
            csv_path = Path(resolve_path(download_cfg.cfg.url_csv_path, path_resolver))
            counts["root"] = _download_url_entries(_read_url_entries(csv_path), raw_path, "root", max_files)
        status = "downloaded"

    elif mode == "copy":
        source_path = Path(resolve_path(download_cfg.cfg.source_path, path_resolver))
        if raw_path.exists() and exists_ok:
            status = "validated"
        else:
            if raw_path.exists():
                raise FileExistsError(f"{raw_path} exists and exists_ok=False.")
            shutil.copytree(source_path, raw_path)
            status = "copied"

    else:
        raise ValueError(f"Unknown download.mode={mode!r} for {dataset_name}.")

    meta = {
        "dataset_name": dataset_name,
        "raw_path": str(raw_path),
        "raw_base_path": str(raw_base_path),
        "download_mode": mode,
        "status": status,
        "counts": locals().get("counts", None),
    }
    write_json(marker_path, meta)
    return {"dataset_name": dataset_name, "raw": str(raw_path), "meta": str(marker_path), "status": status, "counts": locals().get("counts", None)}
