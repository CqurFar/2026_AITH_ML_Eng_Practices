from dataclasses import replace

import numpy as np
import pytest
import torch
from PIL import Image
from src.config import PreprocessingConfig, training_config
from src.data import Record
from src.models import model_prediction
from src.preprocessing import WhaleDataset, bake, load_image, preprocess_image

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
