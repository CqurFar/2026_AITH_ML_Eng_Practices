import pytest
import torch
from PIL import Image
from src.preprocessing.augmentation import train_augment


def test_train_augment_preserves_size_and_mode():
    image = Image.new("RGB", (100, 60), (120, 60, 30))
    out = train_augment()(image)
    assert isinstance(out, Image.Image)
    assert out.mode == "RGB"
    assert out.size == (100, 60)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_train_augment_cuda_batch_preserves_uint8_nchw_and_is_seeded():
    images = torch.arange(2 * 3 * 32 * 48, dtype=torch.uint8).reshape(2, 3, 32, 48).cuda()

    torch.manual_seed(42)
    augmented = train_augment()(images)
    torch.manual_seed(42)
    repeated = train_augment()(images)

    assert augmented.shape == images.shape
    assert augmented.dtype == torch.uint8
    assert augmented.device == images.device
    assert not torch.equal(augmented, images)
    assert torch.equal(augmented, repeated)
