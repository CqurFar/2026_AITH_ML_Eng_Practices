import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import yaml

try:
    import torch
except ImportError:
    torch = None

logger = logging.getLogger(__name__)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(ch)

COMPETITION = "humpback-whale-identification"
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / COMPETITION
DEFAULT_CONFIG_PATH = ROOT / "config.yaml"


@lru_cache(maxsize=32)
def _load_config(path: Path) -> dict[str, object]:
    return cast("dict[str, object]", yaml.safe_load(path.read_text(encoding="utf-8")) or {})


def load_config(path: Path | None = None) -> dict[str, object]:
    """Read YAML once per resolved path; the default path stays cached for the process."""
    target = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    return _load_config(target.resolve())


load_config.cache_clear = _load_config.cache_clear


def _section(name: str) -> dict[str, Any]:
    return cast("dict[str, Any]", load_config()[name])


def _p(section: dict[str, Any], key: str) -> Path:
    return Path(ROOT, section[key])


def _resolve(path: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _reject_overlap(raw: Path, outputs: list[Path]) -> None:
    source = raw.resolve()
    for output in outputs:
        target = output.resolve()
        if source == target or target.is_relative_to(source) or source.is_relative_to(target):
            raise ValueError(f"raw data directory overlaps output directory: {raw} vs {output}")


class CUDAConfig:
    def __init__(self):
        self.cuda_settings = {
            "CUDA_AUTO_BOOST": "1",
            "CUDA_MODULE_LOADING": "LAZY",
            "CUDA_FORCE_PRELOAD_LIBRARIES": "1",
            "CUDA_DEVICE_MAX_CONNECTIONS": "32",
            "CUDA_CACHE_MAXSIZE": "12884901888",
            "PYTORCH_CUDA_ALLOC_CONF": "",
        }
        self.apply_cuda_settings()
        self.device = self.determine_device()

    def apply_cuda_settings(self):
        for key, value in self.cuda_settings.items():
            os.environ[key] = value

    @staticmethod
    def determine_device():
        if torch is None:
            return "cpu"

        if torch.cuda.is_available():
            try:
                torch.cuda.init()
                return "cuda:0"
            except Exception as e:
                logger.warning(f"CUDA initialization failed: {str(e)}. Falling back to CPU.")
        return "cpu"


class AuditConfig:
    def __init__(self) -> None:
        section = _section("audit")
        self.data_dir = _p(section, "data_dir")
        self.processed_dir = _p(section, "processed_dir")
        self.seed = int(section["seed"])
        self.n_folds = int(section["n_folds"])


class EDAConfig:
    def __init__(self) -> None:
        section = _section("eda")
        self.data_dir = _p(section, "data_dir")
        self.eda_dir = _p(section, "eda_dir")
        self.figures_dir = _p(section, "figures_dir")
        self.duplicate_distance = int(section["duplicate_distance"])
        self.seed = int(section["seed"])


class PreprocessingConfig:
    def __init__(
        self,
        *,
        data_dir: Path | None = None,
        train_dir: Path | None = None,
    ) -> None:
        section = _section("preprocessing")
        self.data_dir = _resolve(data_dir) if data_dir is not None else _p(section, "data_dir")
        self.train_dir = _resolve(train_dir) if train_dir is not None else _p(section, "train_dir")
        _reject_overlap(self.data_dir, [self.train_dir])
        self.target_height = int(section["target_height"])
        self.target_width = int(section["target_width"])
        self.interpolation = str(section["interpolation"])
        self.mean = tuple(section["mean"])
        self.std = tuple(section["std"])
        self.seed = int(section["seed"])

    def to_dict(self) -> dict[str, object]:
        return {
            "target_height": self.target_height,
            "target_width": self.target_width,
            "interpolation": self.interpolation,
            "mean": list(self.mean),
            "std": list(self.std),
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> "PreprocessingConfig":
        config = cls.__new__(cls)
        config.target_height = int(values["target_height"])
        config.target_width = int(values["target_width"])
        config.interpolation = str(values["interpolation"])
        config.mean = tuple(values["mean"])
        config.std = tuple(values["std"])
        return config


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    data_dir: Path = field(default_factory=lambda: _p(_section("training"), "data_dir"))
    processed_dir: Path = field(default_factory=lambda: _p(_section("training"), "processed_dir"))
    artifacts_dir: Path = field(default_factory=lambda: _p(_section("training"), "artifacts_dir"))
    checkpoints_dir: Path = field(
        default_factory=lambda: _p(_section("training"), "artifacts_dir") / "checkpoints"
    )
    logs_dir: Path = field(
        default_factory=lambda: _p(_section("training"), "artifacts_dir") / "logs"
    )
    train_dir: Path = field(
        default_factory=lambda: _p(_section("training"), "processed_dir") / "train"
    )
    seed: int = field(default_factory=lambda: int(_section("training")["seed"]))
    n_folds: int = field(default_factory=lambda: int(_section("training")["n_folds"]))
    epochs: int = field(default_factory=lambda: int(_section("training")["epochs"]))
    batch_size: int = field(default_factory=lambda: int(_section("training")["batch_size"]))
    num_workers: int = field(default_factory=lambda: int(_section("training")["num_workers"]))
    head_lr: float = field(default_factory=lambda: float(_section("training")["head_lr"]))
    backbone_lr: float = field(default_factory=lambda: float(_section("training")["backbone_lr"]))
    weight_decay: float = field(default_factory=lambda: float(_section("training")["weight_decay"]))
    label_smoothing: float = field(
        default_factory=lambda: float(_section("training")["label_smoothing"])
    )
    grad_clip: float = field(default_factory=lambda: float(_section("training")["grad_clip"]))
    dropout: float = field(default_factory=lambda: float(_section("training")["dropout"]))
    unfrozen_blocks: int = field(
        default_factory=lambda: int(_section("training")["unfrozen_blocks"])
    )
    pretrained: bool = field(default_factory=lambda: bool(_section("training")["pretrained"]))
    pin_memory: bool = field(default_factory=lambda: bool(_section("training")["pin_memory"]))
    prefetch_factor: int = field(
        default_factory=lambda: int(_section("training")["prefetch_factor"])
    )

    def __post_init__(self) -> None:
        if type(self.prefetch_factor) is not int or self.prefetch_factor < 1:
            raise ValueError("prefetch_factor must be a positive integer")


def training_config(
    *,
    data_dir: Path | None = None,
    processed_dir: Path | None = None,
    artifacts_dir: Path | None = None,
) -> TrainingConfig:
    """Build a resolved config; child paths follow the chosen roots."""
    defaults = TrainingConfig()
    roots = {
        "data_dir": _resolve(data_dir) if data_dir is not None else defaults.data_dir,
        "processed_dir": (
            _resolve(processed_dir) if processed_dir is not None else defaults.processed_dir
        ),
        "artifacts_dir": (
            _resolve(artifacts_dir) if artifacts_dir is not None else defaults.artifacts_dir
        ),
    }
    children = {
        "train_dir": roots["processed_dir"] / "train",
        "checkpoints_dir": roots["artifacts_dir"] / "checkpoints",
        "logs_dir": roots["artifacts_dir"] / "logs",
    }
    _reject_overlap(
        roots["data_dir"],
        [roots["processed_dir"], roots["artifacts_dir"], *children.values()],
    )
    return replace(defaults, **roots, **children)
