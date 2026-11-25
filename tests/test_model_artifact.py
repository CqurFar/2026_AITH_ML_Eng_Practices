import hashlib
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import src.config as config_module
import torch
from PIL import Image
from src.config import PreprocessingConfig, training_config
from src.data import Record, Split
from src.models import model_building
from src.preprocessing import preprocess_image
from torch import nn

EXPECTED_PREPROCESSING = {
    "target_height": 256,
    "target_width": 512,
    "interpolation": "bilinear",
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}


def test_preprocessing_settings_round_trip() -> None:
    preprocessing = PreprocessingConfig()
    payload = preprocessing.to_dict()
    assert payload == EXPECTED_PREPROCESSING

    restored = PreprocessingConfig.from_dict(payload)
    assert restored.to_dict() == payload


def test_model_artifact_round_trip_preserves_preprocessing_and_class_order(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    train_csv = b"Image,Id\nfirst.jpg,whale_b\nsecond.jpg,whale_a\n"
    (raw_dir / "train.csv").write_bytes(train_csv)
    config = training_config(data_dir=raw_dir, artifacts_dir=tmp_path / "artifacts")
    preprocessing = PreprocessingConfig()
    classes = ["whale_b", "whale_a"]
    artifact = model_building.model_artifact(
        nn.Linear(1, len(classes)),
        classes=classes,
        threshold=0.73,
        preprocessing=preprocessing,
        config=config,
        device=torch.device("cpu"),
        run_id="test-run",
    )

    required = {
        "preprocessing",
        "classes",
        "threshold",
        "data_fingerprint",
        "config_fingerprint",
        "environment",
    }
    assert required <= artifact.keys()
    assert "image_size" not in artifact
    assert artifact["data_fingerprint"] == hashlib.sha256(train_csv).hexdigest()
    canonical_config = json.dumps(artifact["config"], sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    assert artifact["config_fingerprint"] == hashlib.sha256(canonical_config).hexdigest()
    assert artifact["environment"]["device"] == "cpu"

    path = tmp_path / "model.pkl"
    torch.save(artifact, path)
    restored_artifact = torch.load(path, map_location="cpu", weights_only=True)
    restored_preprocessing = PreprocessingConfig.from_dict(restored_artifact["preprocessing"])
    assert restored_artifact["classes"] == classes
    assert restored_artifact["threshold"] == 0.73
    assert restored_preprocessing.to_dict() == EXPECTED_PREPROCESSING

    image = Image.new("RGB", (140, 90), color=(120, 60, 30))
    assert torch.equal(
        preprocess_image(image, restored_preprocessing),
        preprocess_image(image, preprocessing),
    )


def test_train_reuses_one_preprocessing_instance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
    seen = {}
    config_module.load_config.cache_clear()
    expected_preprocessing = PreprocessingConfig().to_dict()

    def fake_evaluate_fold(_split, _fold, _config, _device, preprocessing):
        seen["fold"] = preprocessing
        config_module.load_config()["preprocessing"]["target_height"] = 1
        return [prediction], {"map_at_5": 1.0, "calibration_samples": 0}, 0.5

    def fake_fit_model(_rows, _classes, _config, _device, _seed, preprocessing, **_kwargs):
        seen["fit"] = preprocessing
        return nn.Linear(1, 1), []

    real_model_artifact = model_building.model_artifact

    def capturing_model_artifact(*args, **kwargs):
        seen["artifact"] = kwargs["preprocessing"]
        return real_model_artifact(*args, **kwargs)

    monkeypatch.setattr(model_building, "prepare", lambda *_args: [split])
    monkeypatch.setattr(model_building, "CUDAConfig", lambda: SimpleNamespace(device="cpu"))
    monkeypatch.setattr(model_building, "_file_logger", lambda *_args: logging.getLogger(__name__))
    monkeypatch.setattr(model_building, "evaluate_fold", fake_evaluate_fold)
    monkeypatch.setattr(model_building, "fit_model", fake_fit_model)
    monkeypatch.setattr(model_building, "model_artifact", capturing_model_artifact)
    try:
        model_building.train(config)
    finally:
        config_module.load_config.cache_clear()

    assert seen["fold"] is seen["fit"] is seen["artifact"]
    artifact = torch.load(config.artifacts_dir / "model.pkl", weights_only=True)
    assert artifact["preprocessing"] == expected_preprocessing


def test_evaluate_fold_shares_preprocessing_with_fit_and_oof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = training_config(data_dir=tmp_path / "raw", artifacts_dir=tmp_path / "artifacts")
    train_record = Record("train.png", "whale_a", tmp_path / "train.png", "train-digest")
    known_record = Record("known.png", "whale_a", tmp_path / "known.png", "known-digest")
    preprocessing = PreprocessingConfig()
    seen = []

    def fake_fit_model(_rows, _classes, _config, _device, _seed, preprocessing, **_kwargs):
        seen.append(preprocessing)
        return nn.Linear(1, 1), []

    def fake_probabilities(_model, _rows, _directory, _config, _device, preprocessing):
        seen.append(preprocessing)
        return torch.tensor([[0.9]]), [known_record.identity]

    monkeypatch.setattr(model_building, "calibration_split", lambda *_args: ([], [known_record]))
    monkeypatch.setattr(model_building, "fit_model", fake_fit_model)
    monkeypatch.setattr(model_building, "probabilities", fake_probabilities)
    monkeypatch.setattr(model_building, "select_threshold", lambda *_args: 0.5)

    model_building.evaluate_fold(
        Split(train=[train_record], known=[known_record], unseen=[], excluded=[]),
        1,
        config,
        torch.device("cpu"),
        preprocessing,
    )

    assert len(seen) == 2
    assert seen[0] is preprocessing
    assert seen[1] is preprocessing
