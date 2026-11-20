import numpy as np
import pytest
import torch
from PIL import Image
from src.config import PreprocessingConfig
from src.preprocessing.transforms import (
    load_image,
    normalize,
    pad_to_size,
    preprocess_image,
    resize_fit,
)
from torchvision.transforms import InterpolationMode

INTERPOLATION = InterpolationMode.BILINEAR


def _make_image(size, color=(120, 60, 30), mode="RGB"):
    return Image.new(mode, size, color)


class TestLoadImage:
    def test_rgb(self, tmp_path):
        path = tmp_path / "rgb.png"
        _make_image((10, 8)).save(path)
        image = load_image(path)
        assert image.mode == "RGB"
        assert image.size == (10, 8)

    def test_grayscale_to_rgb(self, tmp_path):
        path = tmp_path / "gray.png"
        Image.new("L", (10, 8), 128).save(path)
        image = load_image(path)
        array = np.asarray(image)
        assert image.mode == "RGB"
        assert array.shape == (8, 10, 3)
        assert (array[..., 0] == array[..., 1]).all()
        assert (array[..., 1] == array[..., 2]).all()

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_image(tmp_path / "missing.png")

    def test_corrupt_file(self, tmp_path):
        path = tmp_path / "broken.png"
        path.write_bytes(b"not an image")
        with pytest.raises(OSError):
            load_image(path)


class TestResizeFit:
    def test_exact_size(self):
        image = resize_fit(_make_image((100, 50)), 256, 512, INTERPOLATION)
        assert image.size == (512, 256)

    def test_portrait_fits_height(self):
        image = resize_fit(_make_image((50, 100)), 256, 512, INTERPOLATION)
        assert image.size == (128, 256)

    def test_downscale_fits(self):
        image = resize_fit(_make_image((2000, 3000)), 256, 512, INTERPOLATION)
        assert image.size[0] <= 512
        assert image.size[1] <= 256

    def test_identity_size(self):
        image = resize_fit(_make_image((512, 256)), 256, 512, INTERPOLATION)
        assert image.size == (512, 256)


class TestPadToSize:
    def test_no_padding_when_exact(self):
        image = _make_image((512, 256))
        assert pad_to_size(image, 256, 512) is image

    def test_exact_size(self):
        padded = pad_to_size(_make_image((100, 50)), 256, 512)
        assert padded.size == (512, 256)

    def test_centered_black(self):
        image = _make_image((2, 2), color=(255, 255, 255))
        padded = pad_to_size(image, 6, 6)
        array = np.asarray(padded)
        assert padded.size == (6, 6)
        assert (array[2, 2] == 255).all()
        assert (array[0, 0] == 0).all()


class TestNormalize:
    def test_formula(self):
        tensor = torch.tensor([[[0.0, 0.5]], [[0.5, 1.0]], [[1.0, 0.0]]])
        mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        out = normalize(tensor, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        assert torch.allclose(out, (tensor - mean) / std)

    def test_zero_std_rejected(self):
        with pytest.raises(ValueError):
            normalize(torch.zeros(3, 2, 2), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))


class TestPreprocessImage:
    def test_output_shape_and_dtype(self):
        config = PreprocessingConfig()
        tensor = preprocess_image(_make_image((1000, 600)), config)
        assert tensor.shape == (3, config.target_height, config.target_width)
        assert tensor.dtype == torch.float32
        assert torch.isfinite(tensor).all()
        assert not torch.isnan(tensor).any()

    def test_deterministic(self):
        config = PreprocessingConfig()
        image = _make_image((300, 200))
        assert torch.equal(preprocess_image(image, config), preprocess_image(image, config))
