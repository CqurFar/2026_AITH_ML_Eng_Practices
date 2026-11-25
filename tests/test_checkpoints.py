import hashlib
import json
from pathlib import Path

import pytest
import torch
from src.config import PreprocessingConfig, training_config
from src.data import Record, Split
from src.models import model_building
from torch import nn


def _fit_context(config, *, role="fold", fold=1, rows=None, classes=None, preprocessing=None):
    return model_building.fit_context(
        role=role,
        fold=fold,
        seed=43,
        rows=rows or [("first.png", "whale_a"), ("second.png", "whale_b")],
        classes=classes or ["whale_a", "whale_b"],
        config=config,
        preprocessing=preprocessing or PreprocessingConfig(),
    )


def _training_state():
    model = nn.Linear(1, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    return model, optimizer, scheduler


def _history(epoch: int):
    return [{"epoch": float(index + 1), "loss": 1.0} for index in range(epoch + 1)]


def _config(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "train.csv").write_text(
        "Image,Id\nfirst.png,whale_a\nsecond.png,whale_b\n", encoding="utf-8"
    )
    return training_config(
        data_dir=raw_dir,
        processed_dir=tmp_path / "processed",
        artifacts_dir=tmp_path / "artifacts",
    )


def _write_splits_manifest(config, fingerprint: str, train: list[str]) -> None:
    config.processed_dir.mkdir(parents=True, exist_ok=True)
    (config.processed_dir / "splits_manifest.json").write_text(
        json.dumps({"fingerprint": fingerprint, "folds": [{"train": train}]}), encoding="utf-8"
    )


def test_fit_context_records_training_identity(tmp_path: Path) -> None:
    config = _config(tmp_path)
    preprocessing = PreprocessingConfig()
    context = _fit_context(config, preprocessing=preprocessing)

    assert context["role"] == "fold"
    assert context["fold"] == 1
    assert context["seed"] == 43
    assert context["epochs"] == config.epochs
    assert context["classes"] == ["whale_a", "whale_b"]
    assert (
        context["data_fingerprint"]
        == hashlib.sha256((config.data_dir / "train.csv").read_bytes()).hexdigest()
    )
    assert context["preprocessing_fingerprint"]
    assert context["split_fingerprint"]
    assert context["architecture"] == {
        "name": "mobilenet_v3_small",
        "num_classes": 2,
        "pretrained": config.pretrained,
        "dropout": config.dropout,
        "unfrozen_blocks": config.unfrozen_blocks,
    }
    assert context["optimizer"] == {
        "name": "AdamW",
        "head_lr": config.head_lr,
        "backbone_lr": config.backbone_lr,
        "weight_decay": config.weight_decay,
    }
    assert context["scheduler"] == {"name": "CosineAnnealingLR", "T_max": config.epochs}


def test_fit_context_split_fingerprint_tracks_audited_image_data(tmp_path: Path) -> None:
    config = _config(tmp_path)
    rows = [("first.png", "whale_a"), ("second.png", "whale_b")]
    _write_splits_manifest(config, "first-image-digest", [row[0] for row in rows])
    first = _fit_context(config, rows=rows)
    _write_splits_manifest(config, "changed-image-digest", [row[0] for row in rows])
    second = _fit_context(config, rows=rows)

    assert first["split_fingerprint"] != second["split_fingerprint"]


def test_fit_context_split_fingerprint_tracks_fold_membership(tmp_path: Path) -> None:
    config = _config(tmp_path)
    rows = [("first.png", "whale_a"), ("second.png", "whale_b")]
    _write_splits_manifest(config, "image-digest", ["first.png"])
    first = _fit_context(config, fold=1, rows=rows)
    _write_splits_manifest(config, "image-digest", ["second.png"])
    second = _fit_context(config, fold=1, rows=rows)

    assert first["split_fingerprint"] != second["split_fingerprint"]


def test_fold_and_final_checkpoint_paths_differ(tmp_path: Path) -> None:
    config = _config(tmp_path)
    fold = _fit_context(config)
    final = _fit_context(config, role="final", fold=None)

    assert model_building.checkpoint_path(config, fold) != model_building.checkpoint_path(
        config, final
    )


@pytest.mark.parametrize(
    "field", ["fold", "data_fingerprint", "split_fingerprint", "config_fingerprint"]
)
def test_checkpoint_rejects_same_shape_foreign_context(tmp_path: Path, field: str) -> None:
    config = _config(tmp_path)
    context = _fit_context(config)
    foreign = {**context, field: 2 if field == "fold" else "foreign"}
    path = model_building.checkpoint_path(config, context)
    model, optimizer, scheduler = _training_state()
    model_building._save_checkpoint(path, context, 0, model, optimizer, scheduler, _history(0))
    original = path.read_bytes()

    with pytest.raises(model_building.CheckpointMismatchError, match=path.name):
        model_building._load_checkpoint(path, foreign, model, optimizer, scheduler, [])

    assert path.read_bytes() == original


def test_checkpoint_with_non_text_context_key_fails_closed(tmp_path: Path) -> None:
    config = _config(tmp_path)
    context = _fit_context(config)
    foreign = {**context, 1: "foreign"}
    path = model_building.checkpoint_path(config, context)
    model, optimizer, scheduler = _training_state()
    model_building._save_checkpoint(path, foreign, 0, model, optimizer, scheduler, _history(0))

    with pytest.raises(model_building.CheckpointMismatchError, match="context mismatch"):
        model_building._load_checkpoint(path, context, model, optimizer, scheduler, [])


def test_checkpoint_rejects_inconsistent_history_without_deletion(tmp_path: Path) -> None:
    config = _config(tmp_path)
    context = _fit_context(config)
    path = model_building.checkpoint_path(config, context)
    model, optimizer, scheduler = _training_state()
    model_building._save_checkpoint(path, context, 1, model, optimizer, scheduler, [])
    original = path.read_bytes()

    with pytest.raises(model_building.CheckpointCorruptError, match="history"):
        model_building._load_checkpoint(path, context, model, optimizer, scheduler, [])

    assert path.read_bytes() == original


def test_checkpoint_rejects_epoch_beyond_configured_training(tmp_path: Path) -> None:
    config = _config(tmp_path)
    context = _fit_context(config)
    path = model_building.checkpoint_path(config, context)
    model, optimizer, scheduler = _training_state()
    history = [{"epoch": float(index + 1), "loss": 1.0} for index in range(config.epochs + 1)]
    model_building._save_checkpoint(
        path, context, config.epochs, model, optimizer, scheduler, history
    )

    with pytest.raises(model_building.CheckpointCorruptError, match="epoch"):
        model_building._load_checkpoint(path, context, model, optimizer, scheduler, [])


def test_corrupt_checkpoint_fails_closed_without_deletion(tmp_path: Path) -> None:
    config = _config(tmp_path)
    context = _fit_context(config)
    path = model_building.checkpoint_path(config, context)
    path.parent.mkdir(parents=True)
    corrupt = b"not a torch checkpoint"
    path.write_bytes(corrupt)
    model, optimizer, scheduler = _training_state()

    with pytest.raises(model_building.CheckpointCorruptError, match="Corrupt checkpoint"):
        model_building._load_checkpoint(path, context, model, optimizer, scheduler, [])

    assert path.read_bytes() == corrupt


def test_checkpoint_save_failure_preserves_existing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    context = _fit_context(config)
    path = model_building.checkpoint_path(config, context)
    model, optimizer, scheduler = _training_state()
    model_building._save_checkpoint(path, context, 0, model, optimizer, scheduler, _history(0))
    original = path.read_bytes()
    real_save = torch.save

    def failing_save(payload, destination):
        Path(destination).write_bytes(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(model_building.torch, "save", failing_save)
    with pytest.raises(OSError, match="disk full"):
        model_building._save_checkpoint(path, context, 1, model, optimizer, scheduler, _history(1))

    assert path.read_bytes() == original
    assert list(path.parent.glob("*.tmp")) == []
    monkeypatch.setattr(model_building.torch, "save", real_save)


def test_checkpoint_load_uses_weights_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(tmp_path)
    context = _fit_context(config)
    path = model_building.checkpoint_path(config, context)
    model, optimizer, scheduler = _training_state()
    model_building._save_checkpoint(path, context, 0, model, optimizer, scheduler, _history(0))
    real_load = torch.load
    seen = {}

    def tracked_load(*args, **kwargs):
        seen.update(kwargs)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(model_building.torch, "load", tracked_load)
    model_building._load_checkpoint(path, context, model, optimizer, scheduler, [])

    assert seen["weights_only"] is True


def test_matching_checkpoint_resumes_model_optimizer_scheduler_and_history(tmp_path: Path) -> None:
    config = _config(tmp_path)
    context = _fit_context(config)
    path = model_building.checkpoint_path(config, context)
    model, optimizer, scheduler = _training_state()
    history = [{"epoch": 1.0, "loss": 1.5}]
    model_building._save_checkpoint(path, context, 0, model, optimizer, scheduler, history)
    saved_model = {key: value.clone() for key, value in model.state_dict().items()}
    saved_optimizer = optimizer.state_dict()
    saved_scheduler = scheduler.state_dict()
    restored_model, restored_optimizer, restored_scheduler = _training_state()
    restored_history = []

    start_epoch = model_building._load_checkpoint(
        path, context, restored_model, restored_optimizer, restored_scheduler, restored_history
    )

    assert start_epoch == 1
    assert restored_history == history
    assert all(
        torch.equal(value, saved_model[key]) for key, value in restored_model.state_dict().items()
    )
    assert restored_optimizer.state_dict()["param_groups"] == saved_optimizer["param_groups"]
    assert restored_scheduler.state_dict() == saved_scheduler


def test_evaluate_fold_passes_fold_identity_to_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = training_config(data_dir=tmp_path / "raw", artifacts_dir=tmp_path / "artifacts")
    train_record = Record("train.png", "whale_a", tmp_path / "train.png", "digest")
    known_record = Record("known.png", "whale_a", tmp_path / "known.png", "digest")
    seen = {}

    def fake_fit_model(*_args, **kwargs):
        seen.update(kwargs)
        return nn.Linear(1, 1), []

    monkeypatch.setattr(model_building, "calibration_split", lambda *_args: ([], [known_record]))
    monkeypatch.setattr(model_building, "fit_model", fake_fit_model)
    monkeypatch.setattr(
        model_building,
        "probabilities",
        lambda *_args: (torch.tensor([[0.9]]), [known_record.identity]),
    )
    monkeypatch.setattr(model_building, "select_threshold", lambda *_args: 0.5)

    model_building.evaluate_fold(
        Split(train=[train_record], known=[known_record], unseen=[], excluded=[]),
        2,
        config,
        torch.device("cpu"),
        PreprocessingConfig(),
    )

    assert seen == {"role": "fold", "fold": 2}


def test_legacy_shared_checkpoint_is_not_deleted(tmp_path: Path) -> None:
    config = _config(tmp_path)
    context = _fit_context(config)
    legacy = config.checkpoints_dir / "fit_model.pt"
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b"foreign legacy state")
    model, optimizer, scheduler = _training_state()
    history = []

    start_epoch = model_building._load_checkpoint(
        model_building.checkpoint_path(config, context),
        context,
        model,
        optimizer,
        scheduler,
        history,
    )

    assert start_epoch == 0
    assert history == []
    assert legacy.read_bytes() == b"foreign legacy state"
