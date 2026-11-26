import importlib
from pathlib import Path

import pytest
import src.config as config_module

MINIMAL_CONFIG = """\
audit:
  data_dir: data/raw/example
training:
  seed: 42
"""


def test_one_yaml_read_per_explicit_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "custom.yaml"
    config_path.write_text(MINIMAL_CONFIG, encoding="utf-8")
    real_read_text = Path.read_text
    reads: list[Path] = []

    def counting_read_text(self: Path, *args: object, **kwargs: object) -> str:
        reads.append(self)
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", counting_read_text)
    config_module.load_config.cache_clear()
    first = config_module.load_config(config_path)
    second = config_module.load_config(config_path)
    assert first == second
    assert reads == [config_path]


def test_config_path_aliases_share_cache_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "custom.yaml"
    config_path.write_text(MINIMAL_CONFIG, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config_module, "DEFAULT_CONFIG_PATH", config_path)
    real_read_text = Path.read_text
    reads: list[Path] = []

    def counting_read_text(self: Path, *args: object, **kwargs: object) -> str:
        reads.append(self)
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", counting_read_text)
    config_module.load_config.cache_clear()
    default = config_module.load_config()
    explicit = config_module.load_config(config_path)
    relative = config_module.load_config(Path("custom.yaml"))
    config_module.load_config.cache_clear()

    assert default == explicit == relative
    assert reads == [config_path]


def test_relative_custom_roots_resolve_from_repo_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    config = config_module.training_config(
        data_dir=Path("data/raw/custom"),
        processed_dir=Path("data/processed/custom"),
        artifacts_dir=Path("artifacts/custom"),
    )
    root = config_module.ROOT
    assert config.data_dir == root / "data/raw/custom"
    assert config.processed_dir == root / "data/processed/custom"
    assert config.artifacts_dir == root / "artifacts/custom"


def test_child_paths_follow_custom_roots(tmp_path: Path) -> None:
    config = config_module.training_config(
        data_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        artifacts_dir=tmp_path / "artifacts",
    )
    assert config.train_dir == tmp_path / "processed" / "train"
    assert config.checkpoints_dir == tmp_path / "artifacts" / "checkpoints"
    assert config.logs_dir == tmp_path / "artifacts" / "logs"


def test_training_loader_defaults() -> None:
    config = config_module.TrainingConfig()

    assert config.num_workers == 8
    assert config.pin_memory is True
    assert config.prefetch_factor == 4
    assert config.augment_on_device is True


@pytest.mark.parametrize("augment_on_device", [0, 1, "true", None])
def test_training_config_rejects_non_boolean_augment_on_device(augment_on_device: object) -> None:
    with pytest.raises(ValueError, match="augment_on_device"):
        config_module.TrainingConfig(augment_on_device=augment_on_device)


def test_training_config_rejects_non_boolean_yaml_augment_on_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "custom.yaml"
    config_path.write_text(
        config_module.DEFAULT_CONFIG_PATH.read_text(encoding="utf-8").replace(
            "augment_on_device: true", 'augment_on_device: "false"'
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(config_module, "DEFAULT_CONFIG_PATH", config_path)
    config_module.load_config.cache_clear()
    try:
        with pytest.raises(ValueError, match="augment_on_device"):
            config_module.TrainingConfig()
    finally:
        config_module.load_config.cache_clear()


@pytest.mark.parametrize("prefetch_factor", [0, -1])
def test_training_config_rejects_nonpositive_prefetch_factor(prefetch_factor: int) -> None:
    with pytest.raises(ValueError, match="prefetch_factor"):
        config_module.TrainingConfig(prefetch_factor=prefetch_factor)


@pytest.mark.parametrize(
    ("raw", "processed", "artifacts"),
    [
        (Path("data/raw/x"), Path("data/raw/x"), Path("artifacts/x")),
        (Path("data/raw"), Path("data/raw/x"), Path("artifacts/x")),
        (Path("data/processed/x/train"), Path("data/processed/x"), Path("artifacts/x")),
        (Path("artifacts/x/model"), Path("data/processed/x"), Path("artifacts/x")),
    ],
)
def test_raw_output_overlap_raises(raw: Path, processed: Path, artifacts: Path) -> None:
    with pytest.raises(ValueError, match="overlap"):
        config_module.training_config(
            data_dir=raw, processed_dir=processed, artifacts_dir=artifacts
        )


@pytest.mark.parametrize(
    ("raw", "train"),
    [
        (Path("data/raw/x"), Path("data/raw/x")),
        (Path("data/raw/x"), Path("data/raw/x/train")),
        (Path("data/processed/x/train"), Path("data/processed/x")),
    ],
)
def test_preprocessing_raw_output_overlap_raises(raw: Path, train: Path) -> None:
    with pytest.raises(ValueError, match="overlap"):
        config_module.PreprocessingConfig(data_dir=raw, train_dir=train)


def test_preprocessing_no_argument_defaults_match_config() -> None:
    config = config_module.PreprocessingConfig()
    section = config_module.load_config()["preprocessing"]
    assert config.data_dir == config_module.ROOT / str(section["data_dir"])
    assert config.train_dir == config_module.ROOT / str(section["train_dir"])
    assert config.to_dict() == {
        "target_height": 256,
        "target_width": 512,
        "interpolation": "bilinear",
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }


def test_preprocessing_relative_overrides_resolve_from_repo_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    config = config_module.PreprocessingConfig(
        data_dir=Path("data/raw/custom"),
        train_dir=Path("data/processed/custom/train"),
    )
    assert config.data_dir == config_module.ROOT / "data/raw/custom"
    assert config.train_dir == config_module.ROOT / "data/processed/custom/train"


def test_preprocessing_absolute_overrides_are_preserved(tmp_path: Path) -> None:
    data_dir = tmp_path / "raw"
    train_dir = tmp_path / "processed" / "train"
    config = config_module.PreprocessingConfig(data_dir=data_dir, train_dir=train_dir)
    assert config.data_dir == data_dir
    assert config.train_dir == train_dir


def test_preprocessing_data_override_keeps_default_train_dir(tmp_path: Path) -> None:
    data_dir = tmp_path / "raw"
    section = config_module.load_config()["preprocessing"]
    config = config_module.PreprocessingConfig(data_dir=data_dir)
    assert config.data_dir == data_dir
    assert config.train_dir == config_module.ROOT / str(section["train_dir"])


def test_preprocessing_train_override_keeps_default_data_dir(tmp_path: Path) -> None:
    train_dir = tmp_path / "processed" / "train"
    section = config_module.load_config()["preprocessing"]
    config = config_module.PreprocessingConfig(train_dir=train_dir)
    assert config.data_dir == config_module.ROOT / str(section["data_dir"])
    assert config.train_dir == train_dir


def test_preprocessing_path_overrides_are_keyword_only(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        config_module.PreprocessingConfig(tmp_path, tmp_path)


def test_import_does_not_read_config_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    real_read_text = Path.read_text
    reads: list[Path] = []

    def counting_read_text(self: Path, *args: object, **kwargs: object) -> str:
        reads.append(self)
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", counting_read_text)
    importlib.reload(config_module)
    assert not any(path.name == "config.yaml" for path in reads)
