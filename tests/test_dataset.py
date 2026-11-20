import polars as pl
import torch
from src.config import PreprocessingConfig
from src.data import Record
from src.preprocessing import WhaleDataset, bake, load_image, preprocess_image
from torch.utils.data import DataLoader


def _records(config: PreprocessingConfig, limit: int = 5) -> list[Record]:
    frame = pl.read_csv(config.data_dir / "train.csv")
    return [
        Record(row["Image"], row["Id"], config.data_dir / "train" / row["Image"], "")
        for row in frame.head(limit).to_dicts()
    ]


class TestWhaleDataset:
    def test_returns_tensor_and_label(self):
        config = PreprocessingConfig()
        records = _records(config)
        dataset = WhaleDataset(records, config)
        tensor, label = dataset[0]
        assert tensor.shape == (3, config.target_height, config.target_width)
        assert label == records[0].identity

    def test_augment_mode(self):
        config = PreprocessingConfig()
        records = _records(config, limit=2)
        dataset = WhaleDataset(records, config, augment=True)
        tensor, _ = dataset[0]
        assert tensor.shape == (3, config.target_height, config.target_width)

    def test_train_mode(self, tmp_path):
        config = PreprocessingConfig()
        records = _records(config, limit=2)
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

    def test_dataloader_batches(self):
        config = PreprocessingConfig()
        records = _records(config, limit=4)
        dataset = WhaleDataset(records, config)
        images, labels = next(iter(DataLoader(dataset, batch_size=2)))
        assert images.shape == (2, 3, config.target_height, config.target_width)
        assert len(labels) == 2
