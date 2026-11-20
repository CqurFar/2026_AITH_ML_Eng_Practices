import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import TypedDict

import torch

from src.config import TrainingConfig
from src.data import Record
from src.steps import step

UNKNOWN = "new_whale"


class Prediction(TypedDict):
    image: str
    cohort: str
    target: str
    candidates: list[str]


def calibration_split(
    known: list[Record], unseen: list[Record], seed: int, fraction: float
) -> tuple[list[Record], list[Record]]:
    if not 0 < fraction < 1:
        raise ValueError("Calibration fraction must be between zero and one")
    rng = random.Random(seed)
    calibration, evaluation = [], []
    for records, whole_identity in ((known, False), (unseen, True)):
        groups = defaultdict(list)
        for record in records:
            key = (
                record.identity if whole_identity and record.identity != UNKNOWN else record.digest
            )
            groups[key].append(record)
        keys = sorted(groups)
        if len(keys) < 2:
            raise ValueError("Each validation cohort needs two independent calibration groups")
        rng.shuffle(keys)
        cut = min(len(keys) - 1, max(1, round(len(keys) * fraction)))
        calibration.extend(record for key in keys[:cut] for record in groups[key])
        evaluation.extend(record for key in keys[cut:] for record in groups[key])
    return calibration, evaluation


def rank_scores(probabilities: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    values, order = probabilities.topk(min(4, probabilities.shape[1]), dim=1)
    return values[:, 0], order


def ranked_predictions(
    probabilities: torch.Tensor, classes: list[str], threshold: float
) -> list[list[str]]:
    confidence, order = rank_scores(probabilities)
    predictions = []
    for score, indices in zip(confidence.tolist(), order.tolist(), strict=True):
        known = [classes[index] for index in indices]
        predictions.append([UNKNOWN, *known] if score < threshold else [*known, UNKNOWN])
    return predictions


def map_at_5(targets: list[str], predictions: list[list[str]]) -> float:
    if not targets or len(targets) != len(predictions):
        raise ValueError("Targets and predictions must have the same non-zero length")
    ranks = (
        next((1 / rank for rank, item in enumerate(row[:5], start=1) if target == item), 0.0)
        for target, row in zip(targets, predictions, strict=True)
    )
    return sum(ranks) / len(targets)


def select_threshold(probabilities: torch.Tensor, targets: list[str]) -> float:
    n_unknown = targets.count(UNKNOWN)
    n_known = len(targets) - n_unknown
    if not n_unknown or not n_known or len(probabilities) != len(targets):
        raise ValueError("Calibration requires probabilities for both cohorts")
    confidence, _ = rank_scores(probabilities)
    changes = defaultdict(int)
    for score, target in zip(confidence.tolist(), targets, strict=True):
        changes[score] += n_known if target == UNKNOWN else -n_unknown
    scores = sorted(changes)
    best, current, threshold = 0, 0, 0.0
    for index, score in enumerate(scores):
        current += changes[score]
        if current > best:
            best = current
            threshold = (score + scores[index + 1]) / 2 if index + 1 < len(scores) else 1.000001
    return threshold


def summarize(rows: list[Prediction]) -> dict[str, float | int]:
    targets = [row["target"] for row in rows]
    predictions = [row["candidates"] for row in rows]
    metrics = {
        "samples": len(rows),
        "map_at_5": map_at_5(targets, predictions),
        "top_1_accuracy": sum(row[0] == t for t, row in zip(targets, predictions, strict=True))
        / len(rows),
        "top_5_recall": sum(t in row[:5] for t, row in zip(targets, predictions, strict=True))
        / len(rows),
        "unknown_first_rate": sum(row[0] == UNKNOWN for row in predictions) / len(rows),
    }
    for cohort in ("known", "unseen_identity", "new_whale"):
        members = [row for row in rows if row["cohort"] == cohort]
        if members:
            metrics[f"{cohort}_samples"] = len(members)
            metrics[f"{cohort}_map_at_5"] = map_at_5(
                [row["target"] for row in members], [row["candidates"] for row in members]
            )
    return metrics


@step("validate")
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions", type=Path, default=TrainingConfig().artifacts_dir / "oof.json"
    )
    args = parser.parse_args()
    rows = json.loads(args.predictions.read_text(encoding="utf-8"))
    print(json.dumps(summarize(rows), indent=2))


if __name__ == "__main__":
    main()
