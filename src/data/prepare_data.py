import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

from src.config import AuditConfig
from src.data import Split, audit_data, build_folds, load_records
from src.steps import step


def _persist_manifest(path: Path, manifest: dict[str, object]) -> None:
    text = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    try:
        if path.read_text(encoding="utf-8") == text:
            return
    except (FileNotFoundError, UnicodeDecodeError):
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
    try:
        temporary.write_text(text, encoding="utf-8", newline="\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare(data_dir: Path, processed_dir: Path, seed: int, n_folds: int) -> list[Split]:
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    report = audit_data(data_dir, processed_dir / "audit_report.json")
    records = load_records(data_dir, report)
    fingerprint_data = {
        "labels": sorted((record.image, record.identity) for record in records),
        "sha256": report["sha256"],
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    splits, conflicts, unknown_duplicates = build_folds(records, seed, n_folds)
    manifest = {
        "version": 1,
        "fingerprint": fingerprint,
        "seed": seed,
        "n_folds": n_folds,
        "conflicting_digests": {
            digest: {record.image: record.identity for record in group}
            for digest, group in conflicts.items()
        },
        "duplicate_new_whale": [record.image for record in unknown_duplicates],
        "folds": [
            {
                "train": [record.image for record in split.train],
                "known": [record.image for record in split.known],
                "unseen": [
                    record.image for record in split.unseen if record.identity != "new_whale"
                ],
                "new_whale": [
                    record.image for record in split.unseen if record.identity == "new_whale"
                ],
                "excluded": [record.image for record in split.excluded],
            }
            for split in splits
        ],
    }
    _persist_manifest(processed_dir / "splits_manifest.json", manifest)
    return splits


@step("folds manifest")
def main() -> None:
    config = AuditConfig()
    splits = prepare(config.data_dir, config.processed_dir, config.seed, config.n_folds)
    for fold, split in enumerate(splits, start=1):
        new_whales = sum(record.identity == "new_whale" for record in split.unseen)
        print(
            f"Fold {fold}: train={len(split.train)}, known={len(split.known)}, "
            f"unseen_identity={len(split.unseen) - new_whales}, new_whale={new_whales}, "
            f"excluded={len(split.excluded)}"
        )


if __name__ == "__main__":
    main()
