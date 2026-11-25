import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from src.config import training_config
from src.data import Record, Split
from src.models import model_building
from torch import nn


def _stub_training(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "train.csv").write_text("Image,Id\ntrain.png,whale_a\n", encoding="utf-8")
    config = training_config(data_dir=raw_dir, artifacts_dir=tmp_path / "artifacts")
    record = Record("train.png", "whale_a", raw_dir / "train" / "train.png", "digest")
    split = Split(train=[record], known=[], unseen=[], excluded=[])
    prediction = {
        "image": record.image,
        "cohort": "known",
        "target": record.identity,
        "candidates": [record.identity],
    }
    monkeypatch.setattr(model_building, "prepare", lambda *_args: [split])
    monkeypatch.setattr(model_building, "CUDAConfig", lambda: SimpleNamespace(device="cpu"))
    monkeypatch.setattr(model_building, "_file_logger", lambda *_args: logging.getLogger(__name__))
    monkeypatch.setattr(
        model_building,
        "evaluate_fold",
        lambda *_args: ([prediction], {"map_at_5": 1.0}, 0.5),
    )
    monkeypatch.setattr(
        model_building, "fit_model", lambda *_args, **_kwargs: (nn.Linear(1, 1), [])
    )
    return config


def _files(directory: Path) -> dict[Path, bytes]:
    return {
        path.relative_to(directory): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file()
    }


def test_train_publishes_hashed_run_and_updates_latest_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _stub_training(tmp_path, monkeypatch)
    real_replace = Path.replace
    replacements = []

    def tracked_replace(source, target):
        target = Path(target)
        if target.name == "latest.json":
            run_dir = next((config.artifacts_dir / "runs").iterdir())
            assert (run_dir / "manifest.json").is_file()
            for filename in ("model.pkl", "metrics.json", "oof.json"):
                assert (config.artifacts_dir / filename).read_bytes() == (
                    run_dir / filename
                ).read_bytes()
        replacements.append(target)
        return real_replace(source, target)

    monkeypatch.setattr(Path, "replace", tracked_replace)
    metrics = model_building.train(config)
    run_dir = config.artifacts_dir / "runs" / metrics["run_id"]

    assert replacements[-1] == config.artifacts_dir / "latest.json"
    assert {path.name for path in run_dir.iterdir()} == {
        "model.pkl",
        "metrics.json",
        "oof.json",
        "environment.json",
        "manifest.json",
    }
    manifest_bytes = (run_dir / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    assert manifest["run_id"] == metrics["run_id"]
    for filename in ("model.pkl", "metrics.json", "oof.json"):
        assert (
            manifest["files"][filename]
            == hashlib.sha256((run_dir / filename).read_bytes()).hexdigest()
        )
    artifact = torch.load(run_dir / "model.pkl", map_location="cpu", weights_only=True)
    assert (
        json.loads((run_dir / "environment.json").read_text(encoding="utf-8"))
        == artifact["environment"]
    )
    latest = json.loads((config.artifacts_dir / "latest.json").read_text(encoding="utf-8"))
    assert latest["run_id"] == metrics["run_id"]
    assert latest["manifest_sha256"] == hashlib.sha256(manifest_bytes).hexdigest()
    for filename in ("model.pkl", "metrics.json", "oof.json"):
        assert (config.artifacts_dir / filename).read_bytes() == (run_dir / filename).read_bytes()


def test_train_keeps_previous_runs_immutable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _stub_training(tmp_path, monkeypatch)
    first = model_building.train(config)
    first_dir = config.artifacts_dir / "runs" / first["run_id"]
    first_files = _files(first_dir)

    second = model_building.train(config)
    second_dir = config.artifacts_dir / "runs" / second["run_id"]

    assert second_dir.is_dir()
    assert first["run_id"] != second["run_id"]
    assert _files(first_dir) == first_files
    latest = json.loads((config.artifacts_dir / "latest.json").read_text(encoding="utf-8"))
    assert latest["run_id"] == second["run_id"]


def test_failed_run_publication_preserves_latest_and_compatibility_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _stub_training(tmp_path, monkeypatch)
    model_building.train(config)
    before = _files(config.artifacts_dir)
    real_replace = Path.replace

    def failing_replace(source, target):
        if Path(target).parent.name == "runs":
            raise OSError("publication failed")
        return real_replace(source, target)

    monkeypatch.setattr(Path, "replace", failing_replace)
    with pytest.raises(OSError, match="publication failed"):
        model_building.train(config)

    assert _files(config.artifacts_dir) == before


def test_existing_run_is_never_overwritten(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _stub_training(tmp_path, monkeypatch)

    class FixedDateTime:
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 11, 25, 12, 0, 0, tzinfo=UTC)

    monkeypatch.setattr(model_building, "datetime", FixedDateTime)
    first = model_building.train(config)
    run_dir = config.artifacts_dir / "runs" / first["run_id"]
    sentinel = run_dir / "foreign.txt"
    sentinel.write_text("keep me", encoding="utf-8")
    before = _files(config.artifacts_dir)

    with pytest.raises(FileExistsError):
        model_building.train(config)

    assert _files(config.artifacts_dir) == before
    assert sentinel.read_text(encoding="utf-8") == "keep me"
