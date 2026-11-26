import polars as pl
import pytest
import torch
from src.config import PreprocessingConfig
from src.data import Record
from src.preprocessing import WhaleDataset, bake, load_image, normalize_batch, preprocess_image
from torch.utils.data import DataLoader

from tests.helpers.synthetic_data import make_synthetic_dataset


@pytest.fixture
def dataset_root(tmp_path):
    return make_synthetic_dataset(tmp_path / "dataset")


def _records(root, limit: int = 5) -> list[Record]:
    frame = pl.read_csv(root / "train.csv")
    return [
        Record(row["Image"], row["Id"], root / "train" / row["Image"], "")
        for row in frame.head(limit).to_dicts()
    ]


class TestWhaleDataset:
    def test_returns_tensor_and_label(self, dataset_root):
        config = PreprocessingConfig()
        records = _records(dataset_root)
        dataset = WhaleDataset(records, config)
        tensor, label = dataset[0]
        assert tensor.shape == (3, config.target_height, config.target_width)
        assert tensor.dtype == torch.float32
        assert label == records[0].identity

    def test_augment_mode(self, dataset_root):
        config = PreprocessingConfig()
        records = _records(dataset_root, limit=2)
        dataset = WhaleDataset(records, config, augment=True)
        tensor, _ = dataset[0]
        assert tensor.shape == (3, config.target_height, config.target_width)

    def test_train_mode(self, tmp_path, dataset_root):
        config = PreprocessingConfig()
        records = _records(dataset_root, limit=2)
        prepared = tmp_path / "train_baked"
        assert bake(records, config, train_dir=prepared) == 2
        dataset = WhaleDataset(records, config, train_dir=prepared)
        tensor, label = dataset[0]
        assert tensor.shape == (3, config.target_height, config.target_width)
        assert label == records[0].identity
        direct = preprocess_image(load_image(records[0].path), config)
        assert torch.equal(tensor, direct)
        dataset = WhaleDataset(records, config, augment=True, train_dir=prepared)
        assert dataset[0][0].shape == (3, config.target_height, config.target_width)

    @pytest.mark.parametrize("baked", [False, True])
    def test_device_normalization_returns_uint8_chw(self, tmp_path, dataset_root, baked: bool):
        config = PreprocessingConfig()
        records = _records(dataset_root, limit=2)
        prepared = tmp_path / "train_baked"
        if baked:
            assert bake(records, config, train_dir=prepared) == 2
        train_dir = prepared if baked else None
        reference = WhaleDataset(records, config, train_dir=train_dir)[0][0]
        dataset = WhaleDataset(records, config, train_dir=train_dir, normalize_on_device=True)

        tensor, label = dataset[0]

        assert tensor.dtype == torch.uint8
        assert tensor.shape == (3, config.target_height, config.target_width)
        assert label == records[0].identity
        assert torch.equal(normalize_batch(tensor, config, torch.device("cpu")), reference)

    @pytest.mark.parametrize("baked", [False, True])
    def test_device_normalization_augmentation_preserves_uint8(
        self, tmp_path, dataset_root, baked: bool
    ):
        config = PreprocessingConfig()
        records = _records(dataset_root, limit=2)
        prepared = tmp_path / "train_baked"
        if baked:
            assert bake(records, config, train_dir=prepared) == 2
        dataset = WhaleDataset(
            records,
            config,
            augment=True,
            train_dir=prepared if baked else None,
            normalize_on_device=True,
        )

        tensor, _ = dataset[0]

        assert tensor.dtype == torch.uint8
        assert tensor.shape == (3, config.target_height, config.target_width)

    def test_dataloader_batches(self, dataset_root):
        config = PreprocessingConfig()
        records = _records(dataset_root, limit=4)
        dataset = WhaleDataset(records, config)
        images, labels = next(iter(DataLoader(dataset, batch_size=2)))
        assert images.shape == (2, 3, config.target_height, config.target_width)
        assert len(labels) == 2
