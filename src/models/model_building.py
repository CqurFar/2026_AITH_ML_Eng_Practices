import argparse
import hashlib
import json
import logging
import statistics
import subprocess
from dataclasses import asdict, replace
from datetime import UTC, datetime
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import cast

import torch
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models.mobilenetv3 import MobileNetV3
from tqdm import tqdm

from src.config import COMPETITION, CUDAConfig, PreprocessingConfig, TrainingConfig
from src.data import Record, Split
from src.data.prepare_data import prepare
from src.models.model_evaluation import (
    UNKNOWN,
    Prediction,
    calibration_split,
    ranked_predictions,
    select_threshold,
    summarize,
)
from src.preprocessing.dataset import WhaleDataset as PreparedWhaleDataset
from src.steps import step

logger = logging.getLogger(__name__)


def _file_logger(config: TrainingConfig, name: str) -> logging.Logger:
    path = config.logs_dir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    for handler in log.handlers:
        if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == path:
            return log
    handler = logging.FileHandler(path)
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    log.addHandler(handler)
    return log


class WhaleDataset(Dataset[tuple[torch.Tensor, int | str]]):
    def __init__(
        self,
        rows: list[tuple[str, str]],
        directory: Path,
        image_size: int,
        class_index: dict[str, int] | None = None,
        augment: bool = False,
    ) -> None:
        if image_size < 1:
            raise ValueError("image_size must be positive")
        self.rows = rows
        self.directory = directory
        self.class_index = class_index
        augmentation = (
            [
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            ]
            if augment
            else []
        )
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                *augmentation,
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int | str]:
        name, label = self.rows[index]
        with Image.open(self.directory / name) as image:
            image = ImageOps.exif_transpose(image)
            tensor = self.transform(image.convert("RGB"))
        return tensor, self.class_index[label] if self.class_index is not None else label


def make_model(num_classes: int, pretrained: bool = False, dropout: float = 0.2) -> nn.Module:
    if num_classes < 1:
        raise ValueError("num_classes must be positive")
    if not 0 <= dropout < 1:
        raise ValueError("dropout must be in [0, 1)")
    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_small(weights=weights, dropout=dropout)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


def _save_checkpoint(
    config: TrainingConfig,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.AdamW,
    scheduler: torch.optim.lr_scheduler.CosineAnnealingLR,
    history: list[dict[str, float]],
) -> None:
    path = config.checkpoints_dir / "fit_model.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "history": history,
        },
        path,
    )


def _load_checkpoint(
    config: TrainingConfig,
    model: nn.Module,
    optimizer: torch.optim.AdamW,
    scheduler: torch.optim.lr_scheduler.CosineAnnealingLR,
    history: list[dict[str, float]],
) -> int:
    path = config.checkpoints_dir / "fit_model.pt"
    if not path.exists():
        return 0
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError:
        path.unlink(missing_ok=True)
        return 0
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    history.extend(checkpoint["history"])
    return int(checkpoint["epoch"]) + 1


def fit_model(
    rows: list[tuple[str, str]],
    classes: list[str],
    config: TrainingConfig,
    device: torch.device,
    seed: int,
) -> tuple[nn.Module, list[dict[str, float]]]:
    class_index = {label: index for index, label in enumerate(classes)}
    if not rows or not classes or len(class_index) != len(classes):
        raise ValueError("Training requires rows and unique classes")
    missing = {label for _, label in rows} - class_index.keys()
    if missing:
        raise ValueError(f"Training label absent from classes: {min(missing)}")
    if config.epochs < 1:
        raise ValueError("epochs must be positive")
    for name, value in (
        ("head_lr", config.head_lr),
        ("backbone_lr", config.backbone_lr),
        ("grad_clip", config.grad_clip),
    ):
        if not isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not isfinite(config.weight_decay) or config.weight_decay < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    if not 0 <= config.label_smoothing <= 1:
        raise ValueError("label_smoothing must be in [0, 1]")

    torch.manual_seed(seed)
    use_cuda = device.type == "cuda"
    records = [
        Record(name, identity, config.data_dir / "train" / name, "") for name, identity in rows
    ]
    dataset = PreparedWhaleDataset(
        records, PreprocessingConfig(), augment=True, train_dir=config.train_dir
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.num_workers > 0,
    )
    model = cast(MobileNetV3, make_model(len(classes), config.pretrained, config.dropout)).to(
        device
    )
    blocks = list(model.features.children())
    if not 0 <= config.unfrozen_blocks <= len(blocks):
        raise ValueError(f"unfrozen_blocks must be in [0, {len(blocks)}]")
    split = len(blocks) - config.unfrozen_blocks
    frozen, unfrozen = blocks[:split], blocks[split:]
    for block in frozen:
        block.requires_grad_(False)
    groups = [{"params": model.classifier.parameters(), "lr": config.head_lr}]
    if unfrozen:
        groups.append(
            {
                "params": (parameter for block in unfrozen for parameter in block.parameters()),
                "lr": config.backbone_lr,
            }
        )
    optimizer = torch.optim.AdamW(groups, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    history: list[dict[str, float]] = []
    start_epoch = _load_checkpoint(config, model, optimizer, scheduler, history)
    log = _file_logger(config, "fit_model.log")
    log.info(
        "start_epoch=%d epochs=%d rows=%d device=%s",
        start_epoch + 1,
        config.epochs,
        len(dataset),
        device,
    )

    for epoch in range(start_epoch, config.epochs):
        model.train()
        for block in frozen:
            block.eval()
        loss_sum = torch.zeros((), device=device)
        grad_sum = torch.zeros((), device=device)
        for images, labels in tqdm(loader, desc=f"epoch {epoch + 1}/{config.epochs}"):
            singleton = len(labels) == 1
            if singleton:
                for block in unfrozen:
                    block.eval()
            images = images.to(device, non_blocking=use_cuda)
            label_ids = torch.tensor(
                [class_index[label] for label in labels], dtype=torch.long, device=device
            )
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), label_ids)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(trainable, config.grad_clip)
            optimizer.step()
            if singleton:
                for block in unfrozen:
                    block.train()
            loss_sum += loss.detach() * len(labels)
            grad_sum += grad_norm
        learning_rates = scheduler.get_last_lr()
        history.append(
            {
                "epoch": float(epoch + 1),
                "loss": loss_sum.item() / len(dataset),
                "grad_norm": grad_sum.item() / len(loader),
                "head_lr": learning_rates[0],
                "backbone_lr": learning_rates[1] if unfrozen else 0.0,
            }
        )
        entry = history[-1]
        log.info(
            "epoch=%d loss=%.4f grad_norm=%.4f head_lr=%.5f backbone_lr=%.5f",
            int(entry["epoch"]),
            entry["loss"],
            entry["grad_norm"],
            entry["head_lr"],
            entry["backbone_lr"],
        )
        scheduler.step()
        _save_checkpoint(config, epoch, model, optimizer, scheduler, history)
    return model, history


def probabilities(
    model: nn.Module,
    rows: list[tuple[str, str]],
    directory: Path,
    config: TrainingConfig,
    device: torch.device,
) -> tuple[torch.Tensor, list[str]]:
    if not rows:
        raise ValueError("Inference requires at least one row")
    use_cuda = device.type == "cuda"
    records = [Record(name, identity, directory / name, "") for name, identity in rows]
    dataset = PreparedWhaleDataset(records, PreprocessingConfig(), train_dir=config.train_dir)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    model.eval()
    result = None
    targets: list[str] = []
    for images, labels in loader:
        with torch.inference_mode():
            batch = model(images.to(device, non_blocking=use_cuda)).softmax(dim=1)
        if result is None:
            result = torch.empty((len(dataset), batch.shape[1]), dtype=batch.dtype, device="cpu")
        offset = len(targets)
        result[offset : offset + len(labels)].copy_(batch)
        targets.extend(labels)
    assert result is not None
    return result, targets


def evaluate_fold(
    split: Split, fold: int, config: TrainingConfig, device: torch.device
) -> tuple[list[Prediction], dict[str, object], float]:
    train_rows = [(record.image, record.identity) for record in split.train]
    classes = sorted({label for _, label in train_rows})
    seed = config.seed + fold
    calibration, evaluation = calibration_split(split.known, split.unseen, seed, 0.4)
    model, history = fit_model(train_rows, classes, config, device, seed)
    known_names = {record.image for record in split.known}

    def target(record: Record) -> str:
        return record.identity if record.image in known_names else UNKNOWN

    scores, targets = probabilities(
        model,
        [(record.image, target(record)) for record in calibration + evaluation],
        config.data_dir / "train",
        config,
        device,
    )
    cut = len(calibration)
    threshold = select_threshold(scores[:cut], targets[:cut])
    ranked = ranked_predictions(scores[cut:], classes, threshold)
    predictions: list[Prediction] = [
        {
            "image": record.image,
            "cohort": "known"
            if label != UNKNOWN
            else ("new_whale" if record.identity == UNKNOWN else "unseen_identity"),
            "target": label,
            "candidates": candidates,
        }
        for record, label, candidates in zip(evaluation, targets[cut:], ranked, strict=True)
    ]
    metrics: dict[str, object] = {
        **summarize(predictions),
        "fold": fold,
        "train_samples": len(train_rows),
        "calibration_samples": cut,
        "threshold": threshold,
        "history": history,
    }
    return predictions, metrics, threshold


def code_version() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        )
    except FileNotFoundError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def train(config: TrainingConfig | None = None) -> dict[str, object]:
    config = config or TrainingConfig()
    start = perf_counter()
    device = torch.device(CUDAConfig().device)
    print(f"Device: {device}", flush=True)
    splits = prepare(config.data_dir, config.processed_dir, config.seed, config.n_folds)
    all_predictions: list[Prediction] = []
    fold_metrics = []
    thresholds = []
    for fold, split in enumerate(splits, start=1):
        print(f"Fold {fold}/{len(splits)}", flush=True)
        log = _file_logger(config, "train.log")
        log.info(
            "fold=%d train=%d known=%d unseen=%d excluded=%d",
            fold,
            len(split.train),
            len(split.known),
            len(split.unseen),
            len(split.excluded),
        )
        predictions, results, threshold = evaluate_fold(split, fold, config, device)
        log.info("fold=%d map_at_5=%.4f threshold=%.4f", fold, results["map_at_5"], threshold)
        all_predictions.extend(predictions)
        fold_metrics.append(results)
        thresholds.append(threshold)
        print(f"Fold {fold} MAP@5: {results['map_at_5']:.4f}", flush=True)

    records = {
        record.image: record.identity
        for split in splits
        for record in split.train + split.known + split.unseen
        if record.identity != UNKNOWN
    }
    classes = sorted(set(records.values()))
    print(f"Final refit: {len(records)} images, {len(classes)} classes", flush=True)
    model, history = fit_model(sorted(records.items()), classes, config, device, config.seed)
    run_id = datetime.now(UTC).isoformat()
    settings = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(config).items()
    }
    threshold = statistics.median(thresholds)
    metrics: dict[str, object] = {
        **summarize(all_predictions),
        "run_id": run_id,
        "git_commit": code_version(),
        "dataset": COMPETITION,
        "train_csv_sha256": hashlib.sha256(
            (config.data_dir / "train.csv").read_bytes()
        ).hexdigest(),
        "architecture": "mobilenet_v3_small",
        "protocol": "stratified grouped CV, held-out threshold calibration",
        "config": settings,
        "device": str(device),
        "runtime_seconds": round(perf_counter() - start, 2),
        "folds": fold_metrics,
        "final_train_samples": len(records),
        "final_classes": len(classes),
        "final_history": history,
        "threshold": threshold,
    }
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=config.artifacts_dir) as stage:
        staged = Path(stage)
        torch.save(
            {
                "run_id": run_id,
                "state_dict": model.cpu().state_dict(),
                "classes": classes,
                "threshold": threshold,
                "image_size": config.image_size,
                "config": settings,
            },
            staged / "model.pkl",
        )
        for filename, content in (("metrics.json", metrics), ("oof.json", all_predictions)):
            (staged / filename).write_text(
                json.dumps(content, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
        (config.artifacts_dir / "metrics.json").unlink(missing_ok=True)
        for filename in ("model.pkl", "oof.json", "metrics.json"):
            (staged / filename).replace(config.artifacts_dir / filename)
    print(json.dumps(summarize(all_predictions), indent=2), flush=True)
    print(f"Artifacts saved to {config.artifacts_dir}", flush=True)
    return metrics


@step("train")
def main() -> None:
    defaults = TrainingConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=defaults.data_dir)
    parser.add_argument("--processed-dir", type=Path, default=defaults.processed_dir)
    parser.add_argument("--artifacts-dir", type=Path, default=defaults.artifacts_dir)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--folds", type=int, default=defaults.n_folds)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()
    train(
        replace(
            defaults,
            data_dir=args.data_dir,
            processed_dir=args.processed_dir,
            artifacts_dir=args.artifacts_dir,
            epochs=args.epochs,
            n_folds=args.folds,
            pretrained=not args.no_pretrained,
        )
    )


if __name__ == "__main__":
    main()
