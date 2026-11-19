from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from functools import partial
from pathlib import Path

import numpy as np
import polars as pl
from PIL import Image, ImageOps
from tqdm import tqdm

from src.config import EDAConfig
from src.visualization.common import id_map, montage, save_bar, save_hist, save_scatter

WORKERS = 8
MAX_IN_FLIGHT = 16

_DCT_BASIS: np.ndarray | None = None


def _luma(arr: np.ndarray) -> np.ndarray:
    return 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]


def _laplacian_variance(gray: np.ndarray) -> float:
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    padded = np.pad(gray, 1, mode="edge")
    lap = (
        4 * padded[1:-1, 1:-1]
        - padded[:-2, 1:-1]
        - padded[2:, 1:-1]
        - padded[1:-1, :-2]
        - padded[1:-1, 2:]
    )
    return float(np.var(lap))


def _dct_basis() -> np.ndarray:
    global _DCT_BASIS
    if _DCT_BASIS is None:
        n = 32
        freqs = np.arange(n)
        _DCT_BASIS = np.cos(np.pi * np.outer(freqs, 2 * freqs + 1) / (2 * n))
    return _DCT_BASIS


def _phash(image: Image.Image) -> int:
    small = image.convert("L").resize((32, 32), Image.Resampling.BILINEAR)
    arr = np.asarray(small, dtype=np.float64)
    basis = _dct_basis()
    low = (basis @ arr @ basis.T)[:8, :8]
    bits = (low > np.median(low)).astype(np.uint8)
    return int.from_bytes(np.packbits(bits).tobytes(), "big")


def _scan_one(path: Path, split: str) -> dict:
    try:
        with Image.open(path) as opened:
            image_format = opened.format or ""
            mode = opened.mode
            image = ImageOps.exif_transpose(opened)
            arr_pre = np.asarray(image)
            channels = int(arr_pre.shape[2]) if arr_pre.ndim == 3 else 1
            phash = _phash(image)
            arr = np.asarray(image.convert("RGB"), dtype=np.float32)
            width, height = image.size
            size_bytes = path.stat().st_size
        gray = _luma(arr)
        means = arr.mean(axis=(0, 1))
        return {
            "image": path.name,
            "split": split,
            "width": int(width),
            "height": int(height),
            "aspect_ratio": float(width / height),
            "resolution": int(width * height),
            "format": image_format,
            "mode": mode,
            "channels": channels,
            "file_size_bytes": size_bytes,
            "mean_r": float(means[0]),
            "mean_g": float(means[1]),
            "mean_b": float(means[2]),
            "brightness_mean": float(gray.mean()),
            "brightness_std": float(gray.std()),
            "blur_score": _laplacian_variance(gray),
            "phash": phash,
        }
    except Exception as exc:
        return {"image": path.name, "split": split, "error": f"{type(exc).__name__}: {exc}"}


def _scan(
    split: str,
    names: list[str],
    directory: Path,
    out: list[dict],
    errors: list[dict],
) -> None:
    total = len(names)
    fn = partial(_scan_one, split=split)
    with tqdm(total=total, desc=f"metadata {split}", unit="img") as bar:
        with ProcessPoolExecutor(max_workers=WORKERS) as executor:
            in_flight: dict = {}
            index = 0
            while index < total or in_flight:
                while index < total and len(in_flight) < MAX_IN_FLIGHT:
                    future = executor.submit(fn, directory / names[index])
                    in_flight[future] = names[index]
                    index += 1
                done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in done:
                    del in_flight[future]
                    record = future.result()
                    if "error" in record:
                        errors.append(record)
                    else:
                        out.append(record)
                    bar.update(1)


def _extreme_montage(config, df, column, out_path, id_by, n=6):
    rows = df.to_dicts()
    arr = df[column].to_numpy()
    order = np.argsort(arr)
    picked = [(rows[order[i]], float(arr[order[i]])) for i in range(n)]
    picked += [(rows[order[-i]], float(arr[order[-i]])) for i in range(1, n + 1)]
    items = []
    for row, val in picked:
        path = config.data_dir / row["split"] / row["image"]
        image = Image.open(path).convert("RGB")
        wid = f"[{id_by[row['image']]}] " if row["image"] in id_by else ""
        items.append((image, f"{row['image']} {wid}{column}={val:.2f}".strip()))
    montage(items, out_path, ncols=len(items))


def _bar_value_counts(df, column, out_path):
    counts = df.group_by(column).agg(pl.len().alias("count")).sort("count", descending=True)
    save_bar(counts[column].to_list(), counts["count"].to_list(), out_path, column)


def main() -> None:
    config = EDAConfig()
    rows: list[dict] = []
    errors: list[dict] = []
    _scan(
        "train",
        pl.read_csv(config.data_dir / "train.csv")["Image"].to_list(),
        config.data_dir / "train",
        rows,
        errors,
    )
    _scan(
        "test",
        pl.read_csv(config.data_dir / "sample_submission.csv")["Image"].to_list(),
        config.data_dir / "test",
        rows,
        errors,
    )
    rows.sort(key=lambda row: (row["split"], row["image"]))
    df = pl.DataFrame(rows, schema_overrides={"phash": pl.UInt64})
    out_dir = EDAConfig().eda_dir
    meta_dir = EDAConfig().figures_dir / "image_metadata"
    quality_dir = EDAConfig().figures_dir / "image_quality"
    meta_dir.mkdir(parents=True, exist_ok=True)
    quality_dir.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_dir / "image_metadata.parquet")
    if errors:
        pl.DataFrame(errors).write_csv(out_dir / "scan_errors.csv")
        print(f"scan errors: {len(errors)} -> {EDAConfig().eda_dir}/scan_errors.csv")

    for column in ("width", "height", "resolution", "aspect_ratio", "file_size_bytes"):
        save_hist(df[column].to_numpy(), meta_dir / f"{column}_hist.png", column)
    save_scatter(
        df["width"].to_numpy(),
        df["height"].to_numpy(),
        meta_dir / "width_height_scatter.png",
        xlabel="width",
        ylabel="height",
    )
    _bar_value_counts(df, "format", meta_dir / "format_counts.png")
    _bar_value_counts(df, "mode", meta_dir / "mode_counts.png")
    for column in ("brightness_mean", "brightness_std", "blur_score"):
        save_hist(df[column].to_numpy(), quality_dir / f"{column}_hist.png", column)
    save_scatter(
        df["brightness_mean"].to_numpy(),
        df["brightness_std"].to_numpy(),
        quality_dir / "brightness_vs_contrast.png",
        xlabel="brightness",
        ylabel="contrast",
    )
    save_scatter(
        df["blur_score"].to_numpy(),
        df["resolution"].to_numpy(),
        quality_dir / "blur_score_vs_resolution.png",
        xlabel="blur score",
        ylabel="resolution",
    )

    id_by = id_map(config.data_dir)
    _extreme_montage(config, df, "blur_score", quality_dir / "extreme_blur.png", id_by)
    _extreme_montage(config, df, "brightness_mean", quality_dir / "extreme_brightness.png", id_by)
    _extreme_montage(config, df, "brightness_std", quality_dir / "extreme_contrast.png", id_by)

    print(
        df.group_by("split").agg(pl.len().alias("count")).sort("count", descending=True).to_dicts()
    )
    print(
        df.select(
            [
                pl.col(c).mean().alias(f"mean_{c}")
                for c in (
                    "width",
                    "height",
                    "resolution",
                    "brightness_mean",
                    "brightness_std",
                    "blur_score",
                )
            ]
        ).to_dicts()
    )


if __name__ == "__main__":
    main()
