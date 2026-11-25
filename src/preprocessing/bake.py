import argparse
from pathlib import Path

import polars as pl
from tqdm import tqdm

from src.config import PreprocessingConfig
from src.data import Record
from src.preprocessing.transforms import (
    PREPROCESSING_MANIFEST,
    load_image,
    prepared_file,
    prepared_tensor,
    save_prepared,
    save_preprocessing_manifest,
    validate_preprocessing_manifest,
)
from src.steps import step


def bake(records: list[Record], config: PreprocessingConfig, train_dir: Path | None = None) -> int:
    directory = Path(train_dir) if train_dir is not None else config.train_dir
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / PREPROCESSING_MANIFEST
    if manifest.exists():
        validate_preprocessing_manifest(directory, config)
    elif any(directory.glob("*.npy")):
        raise ValueError(
            f"Missing preprocessing metadata in {directory}; clear prepared data and re-bake"
        )
    count = 0
    for record in tqdm(records, desc="bake", unit="img"):
        tensor = prepared_tensor(load_image(record.path), config)
        save_prepared(tensor, prepared_file(directory, record.image))
        count += 1
    save_preprocessing_manifest(directory, config)
    return count


def _train_records(config: PreprocessingConfig) -> list[Record]:
    frame = pl.read_csv(config.data_dir / "train.csv")
    return [
        Record(row["Image"], row["Id"], config.data_dir / "train" / row["Image"], "")
        for row in frame.to_dicts()
    ]


@step("preprocess")
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--train-dir", type=Path)
    args = parser.parse_args()
    config = PreprocessingConfig(data_dir=args.data_dir, train_dir=args.train_dir)
    count = bake(_train_records(config), config)
    print(f"baked {count} images to {config.train_dir}")


if __name__ == "__main__":
    main()
