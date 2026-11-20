import argparse
import csv
from dataclasses import replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.config import TrainingConfig
from src.models.model_building import WhaleDataset, make_model
from src.models.model_evaluation import ranked_predictions
from src.steps import step


def validate_submission(
    rows: list[dict[str, str]], expected_images: list[str], classes: list[str]
) -> None:
    names = [row["Image"] for row in rows]
    if len(names) != len(set(names)) or set(names) != set(expected_images):
        raise ValueError("Submission images do not match the test set")
    allowed = set(classes) | {"new_whale"}
    for row in rows:
        candidates = row["Id"].split()
        if not row["Image"] or not candidates or len(candidates) > 5:
            raise ValueError("Each image needs between one and five candidates")
        if len(candidates) != len(set(candidates)) or not set(candidates) <= allowed:
            raise ValueError("Submission has duplicate or unknown candidates")
        if row["Id"] != " ".join(candidates):
            raise ValueError("Candidates must be separated by single spaces")


def predict(config: TrainingConfig | None = None, output_path: Path | None = None) -> Path:
    config = config or TrainingConfig()
    with (config.data_dir / "sample_submission.csv").open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != ["Image", "Id"]:
            raise ValueError("sample_submission.csv must have Image,Id columns")
        names = [row["Image"] for row in reader]
    if not names or any(not name for name in names) or len(names) != len(set(names)):
        raise ValueError("Sample submission contains empty or duplicate image names")
    test_dir = config.data_dir / "test"
    test_names = {path.name for path in test_dir.iterdir() if path.is_file()}
    if set(names) != test_names:
        raise ValueError("Sample submission images do not match test directory")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(
        config.artifacts_dir / "model.pkl", map_location=device, weights_only=True
    )
    classes = checkpoint["classes"]
    model = make_model(len(classes), pretrained=False).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    dataset = WhaleDataset([(name, "") for name in names], test_dir, checkpoint["image_size"])
    loader = DataLoader(dataset, batch_size=config.batch_size, num_workers=config.num_workers)
    predictions = []
    with torch.inference_mode():
        for images, _ in loader:
            probabilities = model(images.to(device)).softmax(dim=1).cpu()
            predictions.extend(ranked_predictions(probabilities, classes, checkpoint["threshold"]))
    rows = [{"Image": name, "Id": " ".join(labels)} for name, labels in zip(names, predictions)]
    validate_submission(rows, names, classes)
    output_path = Path(output_path) if output_path else config.artifacts_dir / "submission.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["Image", "Id"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Submission saved to {output_path}")
    return output_path


@step("predict")
def main() -> None:
    defaults = TrainingConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=defaults.data_dir)
    parser.add_argument("--artifacts-dir", type=Path, default=defaults.artifacts_dir)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    predict(
        replace(defaults, data_dir=args.data_dir, artifacts_dir=args.artifacts_dir), args.output
    )


if __name__ == "__main__":
    main()
