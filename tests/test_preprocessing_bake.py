import sys
from pathlib import Path

import pytest
from src.preprocessing.bake import main as bake_main

from tests.helpers.synthetic_data import make_synthetic_dataset


def test_bake_cli_uses_overridden_data_and_train_dirs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = make_synthetic_dataset(tmp_path / "raw", n_identities=1, images_per_identity=1)
    train_dir = tmp_path / "isolated" / "train"
    monkeypatch.setattr(
        sys,
        "argv",
        ["bake", "--data-dir", str(data_dir), "--train-dir", str(train_dir)],
    )

    bake_main()

    assert (train_dir / "whale_0_0.npy").is_file()
    assert (train_dir / "preprocessing.json").is_file()
