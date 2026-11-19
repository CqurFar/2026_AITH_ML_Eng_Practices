from __future__ import annotations

import csv
import hashlib
import json
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError


@dataclass(frozen=True, slots=True)
class Record:
    image: str
    identity: str
    path: Path
    digest: str


@dataclass(frozen=True, slots=True)
class Split:
    train: list[Record]
    known: list[Record]
    unseen: list[Record]
    excluded: list[Record]


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != ["Image", "Id"]:
            raise ValueError(f"Invalid CSV columns in {path}: {reader.fieldnames}")
        rows = list(reader)
    names = set()
    for row in rows:
        name, identity = row.get("Image"), row.get("Id")
        if (
            None in row
            or not name
            or not identity
            or not identity.strip()
            or identity != identity.strip()
            or Path(name).name != name
            or name in {".", ".."}
            or "\\" in name
        ):
            raise ValueError(f"Invalid CSV row in {path}: {row}")
        if name in names:
            raise ValueError(f"Duplicate image name in {path}: {name}")
        names.add(name)
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    return rows


def audit_data(data_dir: str | Path, report_path: str | Path | None = None) -> dict:
    root = Path(data_dir)
    train_rows = _rows(root / "train.csv")
    test_rows = _rows(root / "sample_submission.csv")
    issues = []
    hashes = {}
    formats = {
        ".jpg": "JPEG",
        ".jpeg": "JPEG",
        ".png": "PNG",
        ".webp": "WEBP",
        ".bmp": "BMP",
        ".tif": "TIFF",
        ".tiff": "TIFF",
    }
    for folder, rows in (("train", train_rows), ("test", test_rows)):
        directory = root / folder
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing image directory: {directory}")
        expected = {row["Image"] for row in rows}
        actual = {p.name for p in directory.iterdir() if p.is_file()}
        for name in sorted(expected - actual):
            issues.append(f"missing: {folder}/{name}")
        for name in sorted(actual - expected):
            issues.append(f"extra: {folder}/{name}")
        for name in sorted(expected & actual):
            path = directory / name
            try:
                digest = hashlib.sha256()
                with path.open("rb") as file:
                    for chunk in iter(lambda: file.read(1024 * 1024), b""):
                        digest.update(chunk)
                hashes[f"{folder}/{name}"] = digest.hexdigest()
                with Image.open(path) as image:
                    actual_format = image.format
                    image.load()
                if formats.get(path.suffix.lower()) != actual_format:
                    issues.append(f"format mismatch: {folder}/{name} ({actual_format})")
            except (OSError, ValueError, UnidentifiedImageError) as exc:
                issues.append(f"unreadable: {folder}/{name} ({type(exc).__name__})")
    train_digests = {digest for name, digest in hashes.items() if name.startswith("train/")}
    for row in test_rows:
        name = row["Image"]
        if hashes.get(f"test/{name}") in train_digests:
            issues.append(f"train/test duplicate: test/{name}")
    counts = Counter(row["Id"] for row in train_rows)
    known_counts = [count for identity, count in counts.items() if identity != "new_whale"]
    report = {
        "train_images": len(train_rows),
        "test_images": len(test_rows),
        "unique_ids": len(counts),
        "id_counts": dict(sorted(counts.items())),
        "new_whale_fraction": counts.get("new_whale", 0) / len(train_rows),
        "known_singleton_fraction": sum(count == 1 for count in known_counts) / len(known_counts)
        if known_counts
        else 0,
        "known_at_most_5_fraction": sum(count <= 5 for count in known_counts) / len(known_counts)
        if known_counts
        else 0,
        "known_count_min": min(known_counts, default=0),
        "known_count_median": statistics.median(known_counts) if known_counts else 0,
        "known_count_mean": statistics.mean(known_counts) if known_counts else 0,
        "known_count_max": max(known_counts, default=0),
        "sha256": hashes,
        "issues": issues,
    }
    if report_path is not None:
        output = Path(report_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def load_records(data_dir: str | Path, report: dict | None = None) -> list[Record]:
    root = Path(data_dir)
    report = audit_data(root) if report is None else report
    if report["issues"]:
        raise ValueError(f"Data audit found {len(report['issues'])} issues; inspect audit report")
    records = []
    for row in _rows(root / "train.csv"):
        name = row["Image"]
        records.append(
            Record(name, row["Id"], root / "train" / name, report["sha256"][f"train/{name}"])
        )
    return records


def build_folds(
    records: list[Record], seed: int, n_folds: int
) -> tuple[list[Split], dict[str, list[Record]], list[Record]]:
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    ordered = sorted(records, key=lambda item: item.image)
    by_digest: dict[str, list[Record]] = defaultdict(list)
    for record in ordered:
        by_digest[record.digest].append(record)

    identities: dict[str, list[list[Record]]] = defaultdict(list)
    conflicts: dict[str, list[Record]] = {}
    unknown: list[Record] = []
    unknown_duplicates: list[Record] = []
    for digest, group in sorted(by_digest.items()):
        identity = group[0].identity
        if any(record.identity != identity for record in group):
            conflicts[digest] = group
        elif identity == "new_whale":
            unknown.append(group[0])
            unknown_duplicates.extend(group[1:])
        else:
            identities[identity].append(group)

    rng = random.Random(seed)
    eligible = [groups for groups in identities.values() if len(groups) > 1]
    rng.shuffle(eligible)
    eligible.sort(key=lambda groups: sum(map(len, groups)), reverse=True)
    fold_order = list(range(n_folds))
    rng.shuffle(fold_order)
    unseen_folds: dict[str, int] = {}
    unseen_counts = [0] * n_folds
    unseen_sizes = [0] * n_folds
    for groups in eligible:
        fold = min(fold_order, key=lambda index: (unseen_counts[index], unseen_sizes[index]))
        unseen_folds[groups[0][0].identity] = fold
        unseen_counts[fold] += 1
        unseen_sizes[fold] += sum(map(len, groups))

    known_folds: dict[str, int] = {}
    known_sizes = [0] * n_folds
    for groups in eligible:
        candidates = [fold for fold in fold_order if fold != unseen_folds[groups[0][0].identity]]
        rng.shuffle(groups)
        groups.sort(key=len, reverse=True)
        identity_sizes = [0] * n_folds
        validation_groups = groups if len(candidates) > 1 else groups[:-1]
        for group in validation_groups:
            fold = min(candidates, key=lambda index: (identity_sizes[index], known_sizes[index]))
            known_folds[group[0].digest] = fold
            identity_sizes[fold] += 1
            known_sizes[fold] += len(group)

    rng.shuffle(unknown)
    unknown_folds = {
        record.image: fold_order[index % n_folds] for index, record in enumerate(unknown)
    }
    splits = [Split([], [], [], []) for _ in range(n_folds)]
    for record in ordered:
        for fold, split in enumerate(splits):
            if record.digest in conflicts:
                split.excluded.append(record)
            elif record.identity == "new_whale":
                cohort = split.unseen if unknown_folds.get(record.image) == fold else split.excluded
                cohort.append(record)
            elif unseen_folds.get(record.identity) == fold:
                split.unseen.append(record)
            elif known_folds.get(record.digest) == fold:
                split.known.append(record)
            else:
                split.train.append(record)
    return splits, conflicts, unknown_duplicates
