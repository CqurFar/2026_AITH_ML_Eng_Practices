import argparse
import json
import logging
import platform
import shutil
import statistics
import subprocess
from collections import defaultdict
from dataclasses import asdict, replace
from math import isfinite
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from time import perf_counter
from typing import cast

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import __version__ as torchvision_version
from torchvision import models
from torchvision.models.mobilenetv3 import MobileNetV3
from tqdm import tqdm

from src.config import COMPETITION, CUDAConfig, PreprocessingConfig, TrainingConfig, training_config
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
from src.preprocessing import WhaleDataset, normalize_batch
from src.preprocessing.augmentation import train_augment
from src.steps import step
from src.utils import (
    file_logger,
    fingerprint,
    fsync_file,
    sha256_file,
    utc_run_id,
    write_json_atomic,
)

logger = logging.getLogger(__name__)
_REPO_ROOT = Path(__file__).resolve().parents[2]


class CheckpointError(RuntimeError):
    pass


class CheckpointMismatchError(CheckpointError):
    pass


class CheckpointCorruptError(CheckpointError):
    pass


def _config_settings(config: TrainingConfig) -> dict[str, object]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(config).items()
    }


def _dataset_fingerprint(config: TrainingConfig) -> str:
    path = config.processed_dir / "splits_manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"Missing splits manifest: {path}") from None
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid splits manifest {path}") from exc
    if not isinstance(manifest, dict) or not isinstance(manifest.get("fingerprint"), str):
        raise ValueError(f"Invalid splits manifest {path}: missing data fingerprint")
    return manifest["fingerprint"]


def _split_fingerprint(
    config: TrainingConfig,
    role: str,
    fold: int | None,
    rows: list[tuple[str, str]],
) -> str:
    path = config.processed_dir / "splits_manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return fingerprint([list(row) for row in rows])
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid splits manifest {path}") from exc
    if not isinstance(manifest, dict) or not isinstance(manifest.get("fingerprint"), str):
        raise ValueError(f"Invalid splits manifest {path}: missing data fingerprint")
    membership = None
    if role == "fold":
        folds = manifest.get("folds")
        if not isinstance(folds, list) or fold is None or not 1 <= fold <= len(folds):
            raise ValueError(f"Invalid splits manifest {path}: missing fold {fold}")
        membership = folds[fold - 1]
    return fingerprint(
        {
            "rows": [list(row) for row in rows],
            "data": manifest["fingerprint"],
            "fold": membership,
        }
    )


def fit_context(
    *,
    role: str,
    fold: int | None,
    seed: int,
    rows: list[tuple[str, str]],
    classes: list[str],
    config: TrainingConfig,
    preprocessing: PreprocessingConfig,
) -> dict[str, object]:
    if role not in {"fold", "final"}:
        raise ValueError("role must be 'fold' or 'final'")
    if (role == "fold") != (fold is not None):
        raise ValueError("fold checkpoints require a fold; final checkpoints do not")
    return {
        "role": role,
        "fold": fold,
        "seed": seed,
        "epochs": config.epochs,
        "data_fingerprint": sha256_file(config.data_dir / "train.csv"),
        "split_fingerprint": _split_fingerprint(config, role, fold, rows),
        "preprocessing_fingerprint": fingerprint(preprocessing.to_dict()),
        "classes": list(classes),
        "architecture": {
            "name": "mobilenet_v3_small",
            "num_classes": len(classes),
            "pretrained": config.pretrained,
            "dropout": config.dropout,
            "unfrozen_blocks": config.unfrozen_blocks,
        },
        "optimizer": {
            "name": "AdamW",
            "head_lr": config.head_lr,
            "backbone_lr": config.backbone_lr,
            "weight_decay": config.weight_decay,
        },
        "scheduler": {"name": "CosineAnnealingLR", "T_max": config.epochs},
        "config_fingerprint": fingerprint(_config_settings(config)),
    }


def checkpoint_path(config: TrainingConfig, context: dict[str, object]) -> Path:
    fold = context["fold"]
    label = "final" if context["role"] == "final" else str(fold)
    return config.checkpoints_dir / f"fit_{context['role']}_{label}_{fingerprint(context)}.pt"


def _file_logger(config: TrainingConfig, name: str) -> logging.Logger:
    return file_logger(config.logs_dir / name, name)


def make_model(num_classes: int, pretrained: bool = False, dropout: float = 0.2) -> nn.Module:
    if num_classes < 1:
        raise ValueError("num_classes must be positive")
    if not 0 <= dropout < 1:
        raise ValueError("dropout must be in [0, 1)")
    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_small(weights=weights, dropout=dropout)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


def runtime_environment(device: torch.device) -> dict[str, object]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torchvision": str(torchvision_version),
        "cuda": torch.version.cuda,
        "device": str(device),
        "git_commit": code_version(),
        "git_dirty": git_dirty(),
    }


def model_artifact(
    model: nn.Module,
    *,
    classes: list[str],
    threshold: float,
    preprocessing: PreprocessingConfig,
    config: TrainingConfig,
    device: torch.device,
    run_id: str,
) -> dict[str, object]:
    settings = _config_settings(config)
    return {
        "run_id": run_id,
        "state_dict": model.cpu().state_dict(),
        "preprocessing": preprocessing.to_dict(),
        "classes": list(classes),
        "threshold": threshold,
        "data_fingerprint": sha256_file(config.data_dir / "train.csv"),
        "dataset_fingerprint": _dataset_fingerprint(config),
        "config_fingerprint": fingerprint(settings),
        "environment": runtime_environment(device),
        "config": settings,
    }


def _save_checkpoint(
    path: Path,
    context: dict[str, object],
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.AdamW,
    scheduler: torch.optim.lr_scheduler.CosineAnnealingLR,
    history: list[dict[str, float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as file:
        temporary = Path(file.name)
    try:
        torch.save(
            {
                "context": context,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "history": history,
            },
            temporary,
        )
        fsync_file(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_checkpoint(
    path: Path,
    context: dict[str, object],
    model: nn.Module,
    optimizer: torch.optim.AdamW,
    scheduler: torch.optim.lr_scheduler.CosineAnnealingLR,
    history: list[dict[str, float]],
) -> int:
    if not path.exists():
        return 0
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise CheckpointCorruptError(f"Corrupt checkpoint {path}: {exc}") from exc
    if not isinstance(checkpoint, dict) or "context" not in checkpoint:
        raise CheckpointCorruptError(f"Corrupt checkpoint {path}: missing fit context")
    stored_context = checkpoint["context"]
    if not isinstance(stored_context, dict):
        raise CheckpointCorruptError(f"Corrupt checkpoint {path}: invalid fit context")
    if stored_context != context:
        different = sorted(
            (
                key
                for key in stored_context.keys() | context.keys()
                if stored_context.get(key) != context.get(key)
            ),
            key=str,
        )
        raise CheckpointMismatchError(
            f"Checkpoint context mismatch for {path}: {', '.join(map(str, different))}"
        )
    try:
        epoch = checkpoint["epoch"]
        if type(epoch) is not int or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")
        epochs = context.get("epochs")
        if type(epochs) is not int or epoch >= epochs:
            raise ValueError("epoch is outside the configured training run")
        stored_history = checkpoint["history"]
        if not isinstance(stored_history, list) or len(stored_history) != epoch + 1:
            raise ValueError("history does not match the checkpoint epoch")
        for expected, entry in enumerate(stored_history, start=1):
            if not isinstance(entry, dict) or entry.get("epoch") != float(expected):
                raise ValueError("history epochs are not sequential")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    except Exception as exc:
        raise CheckpointCorruptError(f"Corrupt checkpoint {path}: {exc}") from exc
    history.extend(checkpoint["history"])
    return epoch + 1


def fit_model(
    rows: list[tuple[str, str]],
    classes: list[str],
    config: TrainingConfig,
    device: torch.device,
    seed: int,
    preprocessing: PreprocessingConfig,
    *,
    role: str = "final",
    fold: int | None = None,
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

    context = fit_context(
        role=role,
        fold=fold,
        seed=seed,
        rows=rows,
        classes=classes,
        config=config,
        preprocessing=preprocessing,
    )
    path = checkpoint_path(config, context)
    torch.manual_seed(seed)
    use_cuda = device.type == "cuda"
    augment_on_device = config.augment_on_device and use_cuda
    augmentation = train_augment() if augment_on_device else None
    records = [
        Record(name, identity, config.data_dir / "train" / name, "") for name, identity in rows
    ]
    dataset = WhaleDataset(
        records,
        preprocessing,
        augment=not augment_on_device,
        train_dir=config.train_dir,
        normalize_on_device=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.num_workers > 0,
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
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
    start_epoch = _load_checkpoint(path, context, model, optimizer, scheduler, history)
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
            if augmentation is not None:
                images = augmentation(images)
            images = normalize_batch(images, preprocessing, device)
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
        _save_checkpoint(path, context, epoch, model, optimizer, scheduler, history)
    return model, history


def probabilities(
    model: nn.Module,
    rows: list[tuple[str, str]],
    directory: Path,
    config: TrainingConfig,
    device: torch.device,
    preprocessing: PreprocessingConfig,
) -> tuple[torch.Tensor, list[str]]:
    if not rows:
        raise ValueError("Inference requires at least one row")
    use_cuda = device.type == "cuda"
    records = [Record(name, identity, directory / name, "") for name, identity in rows]
    dataset = WhaleDataset(
        records, preprocessing, train_dir=config.train_dir, normalize_on_device=True
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.num_workers > 0,
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
    )
    model.eval()
    result = None
    targets: list[str] = []
    for images, labels in loader:
        with torch.inference_mode():
            images = images.to(device, non_blocking=use_cuda)
            images = normalize_batch(images, preprocessing, device)
            batch = model(images).softmax(dim=1)
        if result is None:
            result = torch.empty((len(dataset), batch.shape[1]), dtype=batch.dtype, device="cpu")
        offset = len(targets)
        result[offset : offset + len(labels)].copy_(batch)
        targets.extend(labels)
    assert result is not None
    return result, targets


def evaluate_fold(
    split: Split,
    fold: int,
    config: TrainingConfig,
    device: torch.device,
    preprocessing: PreprocessingConfig,
) -> tuple[list[Prediction], dict[str, object], float]:
    train_rows = [(record.image, record.identity) for record in split.train]
    classes = sorted({label for _, label in train_rows})
    seed = config.seed + fold
    calibration, evaluation = calibration_split(split.known, split.unseen, seed, 0.4)
    model, history = fit_model(
        train_rows,
        classes,
        config,
        device,
        seed,
        preprocessing,
        role="fold",
        fold=fold,
    )
    known_names = {record.image for record in split.known}

    def target(record: Record) -> str:
        return record.identity if record.image in known_names else UNKNOWN

    scores, targets = probabilities(
        model,
        [(record.image, target(record)) for record in calibration + evaluation],
        config.data_dir / "train",
        config,
        device,
        preprocessing,
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
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def git_dirty() -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return None
    return bool(result.stdout.strip()) if result.returncode == 0 else None


def _publish_run(
    config: TrainingConfig,
    run_id: str,
    artifact: dict[str, object],
    metrics: dict[str, object],
    oof: list[Prediction],
) -> tuple[Path, str]:
    runs_dir = config.artifacts_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_dir = runs_dir / run_id
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    with TemporaryDirectory(prefix=".run-", dir=runs_dir) as stage:
        staged = Path(stage)
        torch.save(artifact, staged / "model.pkl")
        fsync_file(staged / "model.pkl")
        shutil.copyfile(
            config.processed_dir / "splits_manifest.json", staged / "splits_manifest.json"
        )
        fsync_file(staged / "splits_manifest.json")
        for filename, content in (("metrics.json", metrics), ("oof.json", oof)):
            (staged / filename).write_text(
                json.dumps(content, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
            fsync_file(staged / filename)
        write_json_atomic(staged / "environment.json", artifact["environment"])
        filenames = (
            "model.pkl",
            "metrics.json",
            "oof.json",
            "environment.json",
            "splits_manifest.json",
        )
        manifest = {
            "version": 1,
            "run_id": run_id,
            "files": {filename: sha256_file(staged / filename) for filename in filenames},
        }
        write_json_atomic(staged / "manifest.json", manifest)
        if run_dir.exists():
            raise FileExistsError(f"Run directory already exists: {run_dir}")
        # shortcut: single-process publication; concurrent publishers need a no-replace rename.
        staged.replace(run_dir)
    return run_dir, sha256_file(run_dir / "manifest.json")


def _publish_compatibility(config: TrainingConfig, run_dir: Path) -> None:
    filenames = ("model.pkl", "metrics.json", "oof.json")
    with TemporaryDirectory(prefix=".compat-", dir=config.artifacts_dir) as stage:
        staged = Path(stage)
        for filename in filenames:
            shutil.copyfile(run_dir / filename, staged / filename)
        for filename in filenames:
            fsync_file(staged / filename)
            (staged / filename).replace(config.artifacts_dir / filename)


def train(config: TrainingConfig | None = None) -> dict[str, object]:
    config = config or training_config()
    preprocessing = PreprocessingConfig()
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
        predictions, results, threshold = evaluate_fold(split, fold, config, device, preprocessing)
        log.info("fold=%d map_at_5=%.4f threshold=%.4f", fold, results["map_at_5"], threshold)
        all_predictions.extend(predictions)
        fold_metrics.append(results)
        thresholds.append(threshold)
        print(f"Fold {fold} MAP@5: {results['map_at_5']:.4f}", flush=True)

    identity_digests: dict[str, set[str]] = defaultdict(set)
    for split in splits:
        for record in split.train + split.known + split.unseen + split.excluded:
            if record.identity != UNKNOWN:
                identity_digests[record.identity].add(record.digest)
    records = {
        record.image: record.identity
        for split in splits
        for record in split.train + split.known + split.unseen
        if record.identity != UNKNOWN
    }
    classes = sorted(set(records.values()))
    print(f"Final refit: {len(records)} images, {len(classes)} classes", flush=True)
    model, history = fit_model(
        sorted(records.items()), classes, config, device, config.seed, preprocessing
    )
    run_id = utc_run_id()
    threshold = statistics.median(thresholds)
    artifact = model_artifact(
        model,
        classes=classes,
        threshold=threshold,
        preprocessing=preprocessing,
        config=config,
        device=device,
        run_id=run_id,
    )
    settings = cast("dict[str, object]", artifact["config"])
    environment = cast("dict[str, object]", artifact["environment"])
    excluded_images = {record.image for split in splits for record in split.excluded}
    metrics: dict[str, object] = {
        **summarize(all_predictions),
        "run_id": run_id,
        "evaluated_samples": len(all_predictions),
        "calibration_samples": sum(fold["calibration_samples"] for fold in fold_metrics),
        "excluded_samples": len(excluded_images),
        "singleton_train_only": sum(len(digests) == 1 for digests in identity_digests.values()),
        "git_commit": environment["git_commit"],
        "git_dirty": environment["git_dirty"],
        "environment": environment,
        "dataset": COMPETITION,
        "train_csv_sha256": artifact["data_fingerprint"],
        "dataset_fingerprint": artifact["dataset_fingerprint"],
        "architecture": "mobilenet_v3_small",
        "protocol": (
            "custom open-set assignment: digest-grouped CV folds, identity-grouped calibration, "
            "digest-grouped new_whale"
        ),
        "config": settings,
        "device": str(device),
        "runtime_seconds": round(perf_counter() - start, 2),
        "folds": fold_metrics,
        "final_train_samples": len(records),
        "final_classes": len(classes),
        "final_history": history,
        "threshold": threshold,
    }
    run_dir, manifest_sha256 = _publish_run(config, run_id, artifact, metrics, all_predictions)
    _publish_compatibility(config, run_dir)
    write_json_atomic(
        config.artifacts_dir / "latest.json",
        {
            "run_id": run_id,
            "manifest": run_dir.relative_to(config.artifacts_dir).as_posix(),
            "manifest_sha256": manifest_sha256,
        },
    )
    print(json.dumps(summarize(all_predictions), indent=2), flush=True)
    print(f"Artifacts saved to {run_dir}", flush=True)
    return metrics


@step("train")
def main() -> None:
    defaults = training_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=defaults.data_dir)
    parser.add_argument("--processed-dir", type=Path, default=defaults.processed_dir)
    parser.add_argument("--artifacts-dir", type=Path, default=defaults.artifacts_dir)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--folds", type=int, default=defaults.n_folds)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()
    config = training_config(
        data_dir=args.data_dir,
        processed_dir=args.processed_dir,
        artifacts_dir=args.artifacts_dir,
    )
    train(
        replace(config, epochs=args.epochs, n_folds=args.folds, pretrained=not args.no_pretrained)
    )


if __name__ == "__main__":
    main()
