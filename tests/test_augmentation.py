from PIL import Image
from src.preprocessing.augmentation import train_augment


def test_train_augment_preserves_size_and_mode():
    image = Image.new("RGB", (100, 60), (120, 60, 30))
    out = train_augment()(image)
    assert isinstance(out, Image.Image)
    assert out.mode == "RGB"
    assert out.size == (100, 60)
