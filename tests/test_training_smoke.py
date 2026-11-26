import csv
import json
import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from src.config import PreprocessingConfig, training_config
from src.data import Record, load_records
from src.data.prepare_data import prepare
from src.models import model_building, model_prediction
from src.models.model_evaluation import ranked_predictions
from src.preprocessing import WhaleDataset, bake
from src.preprocessing.augmentation import train_augment
from torch.utils.data import DataLoader

from tests.helpers.synthetic_data import make_synthetic_dataset


@pytest.mark.parametrize("augment_on_device", [False, True])
def test_tiny_cpu_training_save_load_predict_parity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    augment_on_device: bool,
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
        prefetch_factor=2,
        unfrozen_blocks=0,
        dropout=0.0,
        label_smoothing=0.0,
        augment_on_device=augment_on_device,
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
    prepare(root, config.processed_dir, config.seed, config.n_folds)

    building_datasets = []
    building_loaders = []
    real_dataset = model_building.WhaleDataset
    real_loader = model_building.DataLoader

    def capture_dataset(*args, **kwargs):
        dataset = real_dataset(*args, **kwargs)
        building_datasets.append(
            (dataset.augmentation is not None, getattr(dataset, "normalize_on_device", False))
        )
        return dataset

    def capture_loader(dataset, *args, **kwargs):
        building_loaders.append(kwargs)
        return real_loader(dataset, *args, **kwargs)

    monkeypatch.setattr(model_building, "WhaleDataset", capture_dataset)
    monkeypatch.setattr(model_building, "DataLoader", capture_loader)

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

        scores, targets = model_building.probabilities(
            model, rows, root / "train", config, torch.device("cpu"), preprocessing
        )
        assert scores.shape == (len(rows), len(classes))
        assert targets == [identity for _, identity in rows]
        assert building_datasets == [(True, True), (False, True)]
        for options in building_loaders:
            assert options["num_workers"] == config.num_workers
            assert options["pin_memory"] is config.pin_memory
            assert options["persistent_workers"] is (config.num_workers > 0)
            assert options["prefetch_factor"] == (
                config.prefetch_factor if config.num_workers > 0 else None
            )

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

        prediction_datasets = []
        prediction_loaders = []
        real_prediction_dataset = model_prediction.WhaleDataset
        real_prediction_loader = model_prediction.DataLoader

        def capture_prediction_dataset(*args, **kwargs):
            dataset = real_prediction_dataset(*args, **kwargs)
            prediction_datasets.append(
                (dataset.augmentation is not None, getattr(dataset, "normalize_on_device", False))
            )
            return dataset

        def capture_prediction_loader(dataset, *args, **kwargs):
            prediction_loaders.append(kwargs)
            return real_prediction_loader(dataset, *args, **kwargs)

        monkeypatch.setattr(model_prediction, "WhaleDataset", capture_prediction_dataset)
        monkeypatch.setattr(model_prediction, "DataLoader", capture_prediction_loader)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        output = model_prediction.predict(config, tmp_path / "submission.csv")
        with output.open(encoding="utf-8", newline="") as file:
            submission = list(csv.DictReader(file))

        assert artifact["classes"] == classes
        assert artifact["preprocessing"] == preprocessing.to_dict()
        assert submission == [
            {"Image": name, "Id": candidates} for name, candidates in zip(test_names, expected_ids)
        ]
        assert prediction_datasets == [(False, True)]
        for options in prediction_loaders:
            assert options["num_workers"] == config.num_workers
            assert options["pin_memory"] is config.pin_memory
            assert options["persistent_workers"] is (config.num_workers > 0)
            assert options["prefetch_factor"] == (
                config.prefetch_factor if config.num_workers > 0 else None
            )
        model_prediction.validate_submission(submission, test_names, classes)
    finally:
        torch.set_num_threads(previous_threads)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_training_augments_uint8_batch_before_normalization(
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
        prefetch_factor=2,
        unfrozen_blocks=0,
        dropout=0.0,
        label_smoothing=0.0,
        augment_on_device=True,
    )
    records = load_records(root)
    rows = [(record.image, record.identity) for record in records]
    classes = sorted({identity for _, identity in rows})
    assert bake(records, preprocessing, train_dir=config.train_dir) == 4

    datasets = []
    real_dataset = model_building.WhaleDataset

    def capture_dataset(*args, **kwargs):
        dataset = real_dataset(*args, **kwargs)
        datasets.append(dataset)
        return dataset

    events = []
    real_augment = train_augment

    def tracked_augment():
        augmentation = real_augment()

        def apply(images):
            events.append(("augment", images.dtype, tuple(images.shape), images.device.type))
            return augmentation(images)

        return apply

    real_normalize = model_building.normalize_batch

    def tracked_normalize(images, preprocessing, device):
        events.append(("normalize", images.dtype, tuple(images.shape), images.device.type))
        return real_normalize(images, preprocessing, device)

    monkeypatch.setattr(model_building, "WhaleDataset", capture_dataset)
    monkeypatch.setattr(model_building, "train_augment", tracked_augment, raising=False)
    monkeypatch.setattr(model_building, "normalize_batch", tracked_normalize)

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(config.seed)
        _, history = model_building.fit_model(
            rows,
            classes,
            config,
            torch.device("cuda"),
            config.seed,
            preprocessing,
        )
    finally:
        torch.set_num_threads(previous_threads)

    assert len(datasets) == 1
    assert datasets[0].augmentation is None
    assert datasets[0].normalize_on_device is True
    assert len(history) == 1
    assert math.isfinite(history[0]["loss"])
    assert [event[0] for event in events] == ["augment", "normalize"]
    for _, dtype, shape, device_type in events:
        assert dtype == torch.uint8
        assert shape == (4, 3, 32, 48)
        assert device_type == "cuda"
