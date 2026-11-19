import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


def _data() -> dict[str, Any]:
    return yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8")) or {}


def _p(section: dict[str, Any], key: str) -> Path:
    return Path(ROOT, section[key])


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
        section = _data()["audit"]
        self.data_dir = _p(section, "data_dir")
        self.processed_dir = _p(section, "processed_dir")
        self.seed = int(section["seed"])
        self.n_folds = int(section["n_folds"])


class EDAConfig:
    def __init__(self) -> None:
        section = _data()["eda"]
        self.data_dir = _p(section, "data_dir")
        self.eda_dir = _p(section, "eda_dir")
        self.figures_dir = _p(section, "figures_dir")
        self.duplicate_distance = int(section["duplicate_distance"])
        self.seed = int(section["seed"])


class PreprocessingConfig:
    def __init__(self) -> None:
        section = _data()["preprocessing"]
        self.data_dir = _p(section, "data_dir")
        self.train_dir = _p(section, "train_dir")
        self.target_height = int(section["target_height"])
        self.target_width = int(section["target_width"])
        self.interpolation = str(section["interpolation"])
        self.mean = tuple(section["mean"])
        self.std = tuple(section["std"])
        self.seed = int(section["seed"])


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    data_dir: Path = _p(_data()["training"], "data_dir")
    processed_dir: Path = _p(_data()["training"], "processed_dir")
    artifacts_dir: Path = _p(_data()["training"], "artifacts_dir")
    checkpoints_dir: Path = _p(_data()["training"], "checkpoints_dir")
    logs_dir: Path = _p(_data()["training"], "logs_dir")
    train_dir: Path = _p(_data()["training"], "train_dir")
    seed: int = int(_data()["training"]["seed"])
    n_folds: int = int(_data()["training"]["n_folds"])
    epochs: int = int(_data()["training"]["epochs"])
    batch_size: int = int(_data()["training"]["batch_size"])
    num_workers: int = int(_data()["training"]["num_workers"])
    image_size: int = int(_data()["training"]["image_size"])
    head_lr: float = float(_data()["training"]["head_lr"])
    backbone_lr: float = float(_data()["training"]["backbone_lr"])
    weight_decay: float = float(_data()["training"]["weight_decay"])
    label_smoothing: float = float(_data()["training"]["label_smoothing"])
    grad_clip: float = float(_data()["training"]["grad_clip"])
    dropout: float = float(_data()["training"]["dropout"])
    unfrozen_blocks: int = int(_data()["training"]["unfrozen_blocks"])
    pretrained: bool = bool(_data()["training"]["pretrained"])
    pin_memory: bool = bool(_data()["training"]["pin_memory"])
