import torchvision.transforms.v2 as v2


def train_augment() -> v2.Compose:
    return v2.Compose(
        [
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomRotation(degrees=10),
            v2.ColorJitter(brightness=0.15, contrast=0.15),
            v2.RandomApply([v2.GaussianBlur(kernel_size=5, sigma=1.0)], p=0.05),
        ]
    )
