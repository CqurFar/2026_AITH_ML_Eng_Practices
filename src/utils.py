import hashlib
import json
import logging
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def fsync_file(path: Path) -> None:
    with path.open("rb+") as file:
        file.flush()
        os.fsync(file.fileno())


def write_json_atomic(
    path: Path, value: object, *, indent: int = 2, sort_keys: bool = False
) -> None:
    text = (
        json.dumps(value, ensure_ascii=False, indent=indent, sort_keys=sort_keys, allow_nan=False)
        + "\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as file:
        temporary = Path(file.name)
    try:
        temporary.write_text(text, encoding="utf-8", newline="\n")
        fsync_file(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")


def file_logger(path: Path, name: str) -> logging.Logger:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    for handler in log.handlers:
        if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == path:
            return log
    handler = logging.FileHandler(path)
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    log.addHandler(handler)
    return log


def run_module(module: str, args: list[str] | None = None) -> None:
    subprocess.run([sys.executable, "-m", module, *(args or [])], check=True)
