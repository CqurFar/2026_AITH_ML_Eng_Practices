from __future__ import annotations

import argparse
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

from src.config import COMPETITION, RAW_DIR
from src.data.data import audit_data

DEFAULT_DIR = RAW_DIR


def _extract(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            name = PurePosixPath(member.filename.replace("\\", "/"))
            if (
                name.is_absolute()
                or ".." in name.parts
                or ":" in member.filename
                or stat.S_ISLNK(member.external_attr >> 16)
            ):
                raise ValueError(f"Unsafe archive member: {member.filename}")
            target = destination.joinpath(*name.parts)
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open(member) as src, target.open("xb") as dst:
                shutil.copyfileobj(src, dst)


def get_data(destination: str | Path = DEFAULT_DIR, timeout: float = 3600) -> Path:
    destination = Path(destination)
    if destination.exists():
        if not destination.is_dir():
            raise FileExistsError(f"Destination is not a directory: {destination}")
        report = audit_data(destination)
        if report["issues"]:
            raise ValueError(
                f"Existing dataset has {len(report['issues'])} audit issues; files were not changed"
            )
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="whale-download-", dir=destination.parent) as temporary:
        staging = Path(temporary)
        try:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "kaggle",
                    "competitions",
                    "download",
                    COMPETITION,
                    "-p",
                    str(staging),
                ],
                check=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"Kaggle download exceeded {timeout:g} seconds") from exc
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Kaggle CLI not found; install it and configure credentials"
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "Kaggle download failed; check credentials, network, "
                "competition access and accepted rules"
            ) from exc
        bundle = staging / f"{COMPETITION}.zip"
        if bundle.exists():
            if not bundle.is_file() or bundle.stat().st_size == 0:
                raise RuntimeError("Kaggle returned an empty competition archive")
            _extract(bundle, staging)
        for name in ("train.csv", "sample_submission.csv"):
            if not (staging / name).is_file():
                raise FileNotFoundError(f"Downloaded dataset is missing {name}")
        for name in ("train", "test"):
            target = staging / name
            archive = staging / f"{name}.zip"
            if target.is_dir():
                continue
            if not archive.is_file():
                raise FileNotFoundError(f"Downloaded dataset is missing {name} or {name}.zip")
            unpacked = staging / f"unpacked-{name}"
            unpacked.mkdir()
            _extract(archive, unpacked)
            target.mkdir()
            for image in sorted(unpacked.rglob("*")):
                if image.is_file():
                    output = target / image.name
                    if output.exists():
                        raise ValueError(f"Duplicate image name in {name}.zip: {image.name}")
                    shutil.move(str(image), output)
        report = audit_data(staging)
        if report["issues"]:
            raise ValueError(
                f"Downloaded dataset has {len(report['issues'])} audit issues: "
                f"{report['issues'][:5]}"
            )
        if bundle.exists():
            bundle.unlink()
        for name in ("train", "test"):
            unpacked = staging / f"unpacked-{name}"
            if unpacked.exists():
                shutil.rmtree(unpacked)
        staging.rename(destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--timeout", type=float, default=3600)
    args = parser.parse_args()
    print(get_data(args.destination, args.timeout))


if __name__ == "__main__":
    main()
