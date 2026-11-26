import json
from dataclasses import replace

import numpy as np
import pytest
import torch
from PIL import Image
from src.config import PreprocessingConfig, training_config
from src.data import Record
from src.models import model_prediction
from src.preprocessing import (
    WhaleDataset,
    bake,
    load_image,
    load_prepared,
    prepared_file,
    preprocess_image,
    save_prepared,
)

PREPROCESSING = {
    "target_height": 256,
    "target_width": 512,
    "interpolation": "bilinear",
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}


def test_direct_baked_and_prediction_tensors_match(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows, columns = np.indices((90, 140))
    pixels = np.stack((rows * 2 % 256, columns * 3 % 256, (rows + columns) % 256), axis=-1).astype(
        np.uint8
    )
    image_path = tmp_path / "raw" / "test" / "whale.png"
    image_path.parent.mkdir(parents=True)
    Image.fromarray(pixels).save(image_path)

    preprocessing = PreprocessingConfig()
    record = Record(image_path.name, "whale_a", image_path, "synthetic-digest")
    direct = preprocess_image(load_image(image_path), preprocessing)
    prepared_dir = tmp_path / "prepared"
    assert bake([record], preprocessing, train_dir=prepared_dir) == 1
    baked = WhaleDataset([record], preprocessing, train_dir=prepared_dir)[0][0]

    classes = ["whale_a", "whale_b"]
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    torch.save(
        {
            "state_dict": model_prediction.make_model(len(classes)).state_dict(),
            "classes": classes,
            "threshold": 0.0,
            "image_size": 128,
            "preprocessing": PREPROCESSING,
        },
        artifacts_dir / "model.pkl",
    )
    (image_path.parent.parent / "sample_submission.csv").write_text(
        "Image,Id\nwhale.png,\n", encoding="utf-8"
    )
    config = replace(
        training_config(data_dir=image_path.parent.parent, artifacts_dir=artifacts_dir),
        batch_size=1,
        num_workers=0,
        pin_memory=False,
    )

    captured: list[torch.Tensor] = []
    real_make_model = model_prediction.make_model

    def capturing_make_model(*args, **kwargs):
        model = real_make_model(*args, **kwargs)
        model.register_forward_pre_hook(lambda _module, inputs: captured.append(inputs[0].clone()))
        return model

    monkeypatch.setattr(model_prediction, "make_model", capturing_make_model)
    monkeypatch.setattr(model_prediction.torch.cuda, "is_available", lambda: False)
    model_prediction.predict(config, tmp_path / "submission.csv")

    assert len(captured) == 1
    prediction = captured[0][0].cpu()
    for tensor in (direct, baked, prediction):
        assert tensor.shape == (3, 256, 512)
        assert tensor.dtype == torch.float32
        assert torch.equal(tensor, direct)


def _baked_record(tmp_path):
    image_path = tmp_path / "whale.png"
    Image.new("RGB", (140, 90), color=(120, 60, 30)).save(image_path)
    return PreprocessingConfig(), Record(image_path.name, "whale_a", image_path, "digest")


def test_baked_file_is_hwc_uint8_and_loads_as_chw(tmp_path) -> None:
    preprocessing, record = _baked_record(tmp_path)
    prepared_dir = tmp_path / "prepared"
    assert bake([record], preprocessing, train_dir=prepared_dir) == 1

    stored = np.load(prepared_file(prepared_dir, record.image), allow_pickle=False)
    assert stored.shape == (256, 512, 3)
    assert stored.dtype == np.uint8
    loaded = load_prepared(prepared_file(prepared_dir, record.image))
    assert loaded.shape == (3, 256, 512)
    assert torch.equal(
        WhaleDataset([record], preprocessing, train_dir=prepared_dir)[0][0],
        preprocess_image(load_image(record.path), preprocessing),
    )


def test_bake_records_preprocessing_identity(tmp_path) -> None:
    preprocessing, record = _baked_record(tmp_path)
    prepared_dir = tmp_path / "prepared"

    assert bake([record], preprocessing, train_dir=prepared_dir) == 1

    manifest = json.loads((prepared_dir / "preprocessing.json").read_text(encoding="utf-8"))
    assert manifest == PREPROCESSING


@pytest.mark.parametrize("with_manifest", [False, True])
def test_bake_rejects_stale_prepared_data(tmp_path, with_manifest: bool) -> None:
    preprocessing, record = _baked_record(tmp_path)
    prepared_dir = tmp_path / "prepared"
    prepared_file_path = prepared_file(prepared_dir, record.image)
    if with_manifest:
        assert bake([record], preprocessing, train_dir=prepared_dir) == 1
    else:
        save_prepared(torch.zeros((3, 256, 512), dtype=torch.uint8), prepared_file_path)
    original = prepared_file_path.read_bytes()
    changed = PreprocessingConfig.from_dict({**PREPROCESSING, "target_width": 256})

    with pytest.raises(ValueError, match="(?i)preprocessing"):
        bake([record], changed, train_dir=prepared_dir)

    assert prepared_file_path.read_bytes() == original


@pytest.mark.parametrize("with_manifest", [False, True])
def test_dataset_rejects_unverified_prepared_data(tmp_path, with_manifest: bool) -> None:
    preprocessing, record = _baked_record(tmp_path)
    prepared_dir = tmp_path / "prepared"
    if with_manifest:
        assert bake([record], preprocessing, train_dir=prepared_dir) == 1
    else:
        save_prepared(
            torch.zeros((3, 256, 512), dtype=torch.uint8),
            prepared_file(prepared_dir, record.image),
        )
    changed = PreprocessingConfig.from_dict({**PREPROCESSING, "target_width": 256})

    with pytest.raises(ValueError, match="(?i)preprocessing"):
        WhaleDataset([record], changed, train_dir=prepared_dir)
