import csv
from pathlib import Path

from PIL import Image


def make_synthetic_dataset(root: Path, n_identities: int = 3, images_per_identity: int = 2) -> Path:
    root = Path(root)
    for folder in ("train", "test"):
        (root / folder).mkdir(parents=True, exist_ok=True)

    with (root / "train.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Image", "Id"])
        for identity_index in range(n_identities):
            for image_index in range(images_per_identity):
                name = f"whale_{identity_index}_{image_index}.png"
                color = (
                    (identity_index * 53 + image_index * 17) % 256,
                    (identity_index * 97 + image_index * 29) % 256,
                    (identity_index * 31 + image_index * 71) % 256,
                )
                Image.new("RGB", (24, 16), color).save(root / "train" / name)
                writer.writerow([name, f"whale_{identity_index}"])

    test_names = ["test_0.png", "test_1.png"]
    for index, name in enumerate(test_names):
        Image.new("RGB", (20, 14), (40 + index * 80, 60, 90)).save(root / "test" / name)
    with (root / "sample_submission.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Image", "Id"])
        writer.writerows([name, "new_whale"] for name in test_names)
    return root
