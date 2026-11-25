import csv
import json
import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from src.config import PreprocessingConfig, training_config
from src.data import Record, load_records
from src.models import model_building, model_prediction
from src.models.model_evaluation import ranked_predictions
from src.preprocessing import WhaleDataset, bake
from torch.utils.data import DataLoader

from tests.helpers.synthetic_data import make_synthetic_dataset


def test_tiny_cpu_training_save_load_predict_parity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = make_synthetic_dataset(tmp_path / "raw", n_identities=2, images_per_identity=2)
    preprocessing = PreprocessingConfig.from_dict(
        {
            "target_height": 32,
            "target_width": 48,
            "interpolation": "bilinear",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        }
    )
    config = replace(
        training_config(
            data_dir=root,
            processed_dir=tmp_path / "processed",
            artifacts_dir=tmp_path / "artifacts",
        ),
        epochs=1,
        batch_size=4,
        num_workers=0,
        pretrained=False,
        pin_memory=False,
        unfrozen_blocks=0,
        dropout=0.0,
        label_smoothing=0.0,
    )
    records = load_records(root)
    rows = [(record.image, record.identity) for record in records]
    classes = sorted({identity for _, identity in rows})
    assert len(rows) == 4
    assert classes == ["whale_0", "whale_1"]
    assert bake(records, preprocessing, train_dir=config.train_dir) == 4
    assert json.loads((config.train_dir / "preprocessing.json").read_text(encoding="utf-8")) == (
        preprocessing.to_dict()
    )

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(config.seed)
        initial_head = (
            model_building.make_model(len(classes), pretrained=False, dropout=config.dropout)
            .classifier[-1]
            .weight.detach()
            .clone()
        )
        model, history = model_building.fit_model(
            rows,
            classes,
            config,
            torch.device("cpu"),
            config.seed,
            preprocessing,
        )
        assert len(history) == 1
        assert math.isfinite(history[0]["loss"])
        assert not torch.equal(model.classifier[-1].weight.detach().cpu(), initial_head)

        artifact = model_building.model_artifact(
            model,
            classes=classes,
            threshold=0.5,
            preprocessing=preprocessing,
            config=config,
            device=torch.device("cpu"),
            run_id="smoke-run",
        )
        model_path = config.artifacts_dir / "model.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(artifact, model_path)

        with (root / "sample_submission.csv").open(encoding="utf-8", newline="") as file:
            test_names = [row["Image"] for row in csv.DictReader(file)]
        test_records = [Record(name, "", root / "test" / name, "") for name in test_names]
        loader = DataLoader(
            WhaleDataset(test_records, preprocessing),
            batch_size=config.batch_size,
            num_workers=0,
        )
        model.eval()
        with torch.inference_mode():
            direct_probabilities = torch.cat([model(images).softmax(dim=1) for images, _ in loader])
        expected_ids = [
            " ".join(candidates)
            for candidates in ranked_predictions(direct_probabilities, classes, 0.5)
        ]

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        output = model_prediction.predict(config, tmp_path / "submission.csv")
        with output.open(encoding="utf-8", newline="") as file:
            submission = list(csv.DictReader(file))

        assert artifact["classes"] == classes
        assert artifact["preprocessing"] == preprocessing.to_dict()
        assert submission == [
            {"Image": name, "Id": candidates} for name, candidates in zip(test_names, expected_ids)
        ]
        model_prediction.validate_submission(submission, test_names, classes)
    finally:
        torch.set_num_threads(previous_threads)
