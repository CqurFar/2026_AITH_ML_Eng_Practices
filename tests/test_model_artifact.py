import hashlib
import json
from pathlib import Path

import torch
from PIL import Image
from src.config import PreprocessingConfig, training_config
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
