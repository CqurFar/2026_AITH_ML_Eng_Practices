import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional as F

from src.config import PreprocessingConfig

INTERPOLATIONS = {
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic": InterpolationMode.BICUBIC,
    "nearest": InterpolationMode.NEAREST,
}
PREPROCESSING_MANIFEST = "preprocessing.json"


def load_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def resize_fit(
    image: Image.Image, height: int, width: int, interpolation: InterpolationMode
) -> Image.Image:
    image_width, image_height = image.size
    scale = min(height / image_height, width / image_width)
    new_height = max(1, int(image_height * scale))
    new_width = max(1, int(image_width * scale))
    return F.resize(image, [new_height, new_width], interpolation=interpolation, antialias=True)


def pad_to_size(image: Image.Image, height: int, width: int) -> Image.Image:
    image_width, image_height = image.size
    pad_h = max(0, height - image_height)
    pad_w = max(0, width - image_width)
    if pad_h == 0 and pad_w == 0:
        return image
    top = pad_h // 2
    left = pad_w // 2
    return F.pad(image, [left, top, pad_w - left, pad_h - top], fill=0, padding_mode="constant")


def normalize(
    tensor: torch.Tensor, mean: tuple[float, ...], std: tuple[float, ...]
) -> torch.Tensor:
    if any(value <= 0 for value in std):
        raise ValueError("normalization std must be positive")
    mean_t = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    std_t = torch.tensor(std, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    return (tensor - mean_t) / std_t


def prepared_image(image: Image.Image, config: PreprocessingConfig) -> Image.Image:
    interpolation = INTERPOLATIONS[config.interpolation]
    resized = resize_fit(image, config.target_height, config.target_width, interpolation)
    return pad_to_size(resized, config.target_height, config.target_width)


def prepared_tensor(image: Image.Image, config: PreprocessingConfig) -> torch.Tensor:
    return F.pil_to_tensor(prepared_image(image, config))


def to_model_tensor(tensor: torch.Tensor, config: PreprocessingConfig) -> torch.Tensor:
    return normalize(tensor.to(torch.float32) / 255.0, config.mean, config.std)


def normalize_batch(
    tensor: torch.Tensor, config: PreprocessingConfig, device: torch.device
) -> torch.Tensor:
    tensor = tensor.to(device=device)
    if tensor.dtype == torch.uint8:
        tensor = tensor.to(torch.float32) / 255.0
    else:
        tensor = tensor.to(torch.float32)
    return normalize(tensor, config.mean, config.std)


def preprocess_image(image: Image.Image, config: PreprocessingConfig) -> torch.Tensor:
    return to_model_tensor(prepared_tensor(image, config), config)


def prepared_file(prepared_dir: Path, image_name: str) -> Path:
    return Path(prepared_dir) / f"{Path(image_name).stem}.npy"


def save_prepared(tensor: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, tensor.permute(1, 2, 0).numpy())


def load_prepared(path: Path) -> torch.Tensor:
    return torch.from_numpy(np.load(path)).permute(2, 0, 1)


def save_preprocessing_manifest(directory: Path, config: PreprocessingConfig) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / PREPROCESSING_MANIFEST).write_text(
        json.dumps(config.to_dict(), indent=2) + "\n", encoding="utf-8"
    )


def validate_preprocessing_manifest(directory: Path, config: PreprocessingConfig) -> None:
    path = directory / PREPROCESSING_MANIFEST
    if not path.exists():
        raise ValueError(
            f"Missing preprocessing metadata in {directory}; clear prepared data and re-bake"
        )
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Invalid preprocessing metadata in {directory}; clear prepared data and re-bake"
        ) from exc
    if stored != config.to_dict():
        raise ValueError(
            f"Preprocessing metadata in {directory} does not match; clear prepared data and re-bake"
        )
