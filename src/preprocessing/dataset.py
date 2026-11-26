from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.config import PreprocessingConfig
from src.data import Record
from src.preprocessing.augmentation import train_augment
from src.preprocessing.transforms import (
    load_image,
    load_prepared,
    prepared_file,
    prepared_tensor,
    to_model_tensor,
    validate_preprocessing_manifest,
)


class WhaleDataset(Dataset):
    def __init__(
        self,
        records: list[Record],
        config: PreprocessingConfig,
        augment: bool = False,
        train_dir: Path | None = None,
        *,
        normalize_on_device: bool = False,
    ):
        self.records = records
        self.config = config
        self.augmentation = train_augment() if augment else None
        self.train_dir = train_dir
        self.normalize_on_device = normalize_on_device
        if train_dir is not None:
            validate_preprocessing_manifest(train_dir, config)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        record = self.records[index]
        if self.train_dir is not None:
            tensor = load_prepared(prepared_file(self.train_dir, record.image))
            if self.augmentation is not None:
                tensor = self.augmentation(tensor)
        else:
            image = load_image(record.path)
            if self.augmentation is not None:
                image = self.augmentation(image)
            tensor = prepared_tensor(image, self.config)
        if not self.normalize_on_device:
            tensor = to_model_tensor(tensor, self.config)
        return tensor, record.identity
