import numpy as np
import polars as pl

from src.config import EDAConfig
from src.visualization.common import save_hist


def _wasserstein_1d(a: np.ndarray, b: np.ndarray) -> float:
    grid = np.union1d(a, b)
    ca = np.searchsorted(np.sort(a), grid, side="right") / a.size
    cb = np.searchsorted(np.sort(b), grid, side="right") / b.size
    return float(np.trapezoid(np.abs(ca - cb), grid))


def _ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    grid = np.union1d(a, b)
    ca = np.searchsorted(np.sort(a), grid, side="right") / a.size
    cb = np.searchsorted(np.sort(b), grid, side="right") / b.size
    return float(np.max(np.abs(ca - cb)))


def main() -> None:
    df = pl.read_parquet(EDAConfig().eda_dir / "image_metadata.parquet")
    features = [
        "width",
        "height",
        "aspect_ratio",
        "resolution",
        "file_size_bytes",
        "brightness_mean",
        "brightness_std",
        "blur_score",
    ]
    rng = np.random.default_rng(42)
    rows = []
    for feature in features:
        train = df.filter(pl.col("split") == "train")[feature].to_numpy()
        test = df.filter(pl.col("split") == "test")[feature].to_numpy()
        n = min(20000, len(train), len(test))
        t_sample = rng.choice(train, n, replace=False)
        s_sample = rng.choice(test, n, replace=False)
        rows.append(
            {
                "feature": feature,
                "train_mean": float(train.mean()),
                "train_std": float(train.std()),
                "test_mean": float(test.mean()),
                "test_std": float(test.std()),
                "mean_difference": float(train.mean() - test.mean()),
                "wasserstein_1": _wasserstein_1d(t_sample, s_sample),
                "ks_statistic": _ks_statistic(t_sample, s_sample),
                "train_n": int(len(train)),
                "test_n": int(len(test)),
            }
        )
    result = pl.DataFrame(rows).sort("wasserstein_1", descending=True)
    out_dir = EDAConfig().eda_dir
    figures = EDAConfig().figures_dir / "consistency"
    figures.mkdir(parents=True, exist_ok=True)
    result.write_csv(out_dir / "train_test_consistency.csv")
    for feature in ("width", "height", "brightness_mean", "blur_score", "file_size_bytes"):
        save_hist(
            df.filter(pl.col("split") == "train")[feature].to_numpy(),
            figures / f"{feature}_train_hist.png",
            title=f"{feature} — train",
        )
        save_hist(
            df.filter(pl.col("split") == "test")[feature].to_numpy(),
            figures / f"{feature}_test_hist.png",
            title=f"{feature} — test",
        )
    print(result.head(10).to_dicts())


if __name__ == "__main__":
    main()
