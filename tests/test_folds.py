import csv
import hashlib
import json
from pathlib import Path

from src.data import Record, audit_data, build_folds, prepare

from tests.helpers.synthetic_data import make_synthetic_dataset


def _record(root: Path, image: str, identity: str, digest: str) -> Record:
    return Record(image, identity, root / "train" / image, digest)


def test_folds_keep_duplicate_digests_and_identities_out_of_training(tmp_path: Path) -> None:
    records = [
        _record(tmp_path, "a_1.png", "whale_a", "digest-a1"),
        _record(tmp_path, "a_1_copy.png", "whale_a", "digest-a1"),
        _record(tmp_path, "a_2.png", "whale_a", "digest-a2"),
        _record(tmp_path, "a_3.png", "whale_a", "digest-a3"),
        _record(tmp_path, "b_1.png", "whale_b", "digest-b1"),
        _record(tmp_path, "b_2.png", "whale_b", "digest-b2"),
        _record(tmp_path, "c_1.png", "whale_c", "digest-c1"),
        _record(tmp_path, "c_2.png", "whale_c", "digest-c2"),
    ]

    splits, conflicts, unknown_duplicates = build_folds(records, seed=42, n_folds=2)

    assert conflicts == {}
    assert unknown_duplicates == []
    for split in splits:
        cohorts = (split.train, split.known, split.unseen, split.excluded)
        names = [record.image for cohort in cohorts for record in cohort]
        assert sorted(names) == sorted(record.image for record in records)
        train_digests = {record.digest for record in split.train}
        train_identities = {record.identity for record in split.train}
        for record in split.known + split.unseen:
            assert record.digest not in train_digests
        for record in split.unseen:
            assert record.identity not in train_identities
        unseen_identities = {record.identity for record in split.unseen}
        assert not unseen_identities & {record.identity for record in split.known}
        for identity in unseen_identities:
            assert {record.image for record in records if record.identity == identity} == {
                record.image for record in split.unseen if record.identity == identity
            }
        duplicate_cohorts = [
            index for index, cohort in enumerate(cohorts) if "a_1.png" in {r.image for r in cohort}
        ]
        copy_cohorts = [
            index
            for index, cohort in enumerate(cohorts)
            if "a_1_copy.png" in {r.image for r in cohort}
        ]
        assert duplicate_cohorts == copy_cohorts


def test_singleton_identity_is_train_only_in_every_fold(tmp_path: Path) -> None:
    records = [
        _record(tmp_path, "single.png", "singleton", "digest-single"),
        _record(tmp_path, "a_1.png", "whale_a", "digest-a1"),
        _record(tmp_path, "a_2.png", "whale_a", "digest-a2"),
        _record(tmp_path, "b_1.png", "whale_b", "digest-b1"),
        _record(tmp_path, "b_2.png", "whale_b", "digest-b2"),
    ]

    splits, _, _ = build_folds(records, seed=7, n_folds=2)

    for split in splits:
        assert [record.image for record in split.train].count("single.png") == 1
        assert all(record.identity != "singleton" for record in split.known + split.unseen)


def test_conflicting_labels_are_excluded_from_every_fold(tmp_path: Path) -> None:
    records = [
        _record(tmp_path, "conflict.png", "whale_a", "digest-conflict"),
        _record(tmp_path, "conflict_copy.png", "whale_b", "digest-conflict"),
        _record(tmp_path, "a_1.png", "whale_a", "digest-a1"),
        _record(tmp_path, "a_2.png", "whale_a", "digest-a2"),
        _record(tmp_path, "b_1.png", "whale_b", "digest-b1"),
        _record(tmp_path, "b_2.png", "whale_b", "digest-b2"),
    ]

    splits, conflicts, _ = build_folds(records, seed=42, n_folds=2)

    assert {record.image for record in conflicts["digest-conflict"]} == {
        "conflict.png",
        "conflict_copy.png",
    }
    for split in splits:
        assert {record.image for record in split.excluded} == {"conflict.png", "conflict_copy.png"}
        assert all(
            record.image not in {"conflict.png", "conflict_copy.png"}
            for record in split.train + split.known + split.unseen
        )


def test_duplicate_new_whale_images_keep_one_representative(tmp_path: Path) -> None:
    records = [
        _record(tmp_path, "unknown.png", "new_whale", "digest-unknown"),
        _record(tmp_path, "unknown_copy.png", "new_whale", "digest-unknown"),
        _record(tmp_path, "other_unknown.png", "new_whale", "digest-other"),
        _record(tmp_path, "a_1.png", "whale_a", "digest-a1"),
        _record(tmp_path, "a_2.png", "whale_a", "digest-a2"),
        _record(tmp_path, "b_1.png", "whale_b", "digest-b1"),
        _record(tmp_path, "b_2.png", "whale_b", "digest-b2"),
    ]

    splits, _, unknown_duplicates = build_folds(records, seed=42, n_folds=2)

    assert [record.image for record in unknown_duplicates] == ["unknown_copy.png"]
    assert sorted(
        record.image
        for split in splits
        for record in split.unseen
        if record.identity == "new_whale"
    ) == ["other_unknown.png", "unknown.png"]
    for split in splits:
        assert "unknown_copy.png" in {record.image for record in split.excluded}


def test_prepare_manifest_fingerprint_covers_exact_csv_labels_and_image_digests(
    tmp_path: Path,
) -> None:
    root = make_synthetic_dataset(tmp_path / "raw", n_identities=2, images_per_identity=2)
    processed = tmp_path / "processed"
    report = audit_data(root)
    assert report["issues"] == []

    prepare(root, processed, seed=17, n_folds=2)

    manifest = json.loads((processed / "splits_manifest.json").read_text(encoding="utf-8"))
    with (root / "train.csv").open(encoding="utf-8", newline="") as file:
        labels = sorted((row["Image"], row["Id"]) for row in csv.DictReader(file))
    digests = {
        f"{folder}/{path.name}": hashlib.sha256(path.read_bytes()).hexdigest()
        for folder in ("train", "test")
        for path in sorted((root / folder).iterdir())
    }
    canonical = json.dumps(
        {"labels": labels, "sha256": digests}, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    assert manifest["fingerprint"] == hashlib.sha256(canonical).hexdigest()
    assert manifest["seed"] == 17
    assert manifest["n_folds"] == 2
    assert len(manifest["folds"]) == 2
