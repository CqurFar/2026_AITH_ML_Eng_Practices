import csv
import json

from src.config import EDAConfig
from src.data.data import audit_data


def main() -> None:
    config = EDAConfig()
    report = audit_data(config.data_dir)
    out_dir = EDAConfig().eda_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dataset_integrity_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (out_dir / "dataset_integrity_problems.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["issue"])
        for issue in report["issues"]:
            writer.writerow([issue])
    summary = {
        "train_images": report["train_images"],
        "test_images": report["test_images"],
        "unique_ids": report["unique_ids"],
        "issues": len(report["issues"]),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
