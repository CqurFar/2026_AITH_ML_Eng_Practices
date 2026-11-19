import polars as pl

from src.config import EDAConfig
from src.visualization.common import save_bar, save_hist


def stats_table(train: pl.DataFrame, counts: pl.DataFrame) -> pl.DataFrame:
    known = counts.filter(pl.col("Id") != "new_whale")
    new_whale = counts.filter(pl.col("Id") == "new_whale")
    new_whale_images = int(new_whale["image_count"].sum()) if len(new_whale) else 0
    return pl.DataFrame(
        [
            {
                "new_whale_images": new_whale_images,
                "new_whale_fraction": new_whale_images / len(train),
                "n_known_ids": len(known),
                "n_classes_1": len(known.filter(pl.col("image_count") == 1)),
                "n_classes_2": len(known.filter(pl.col("image_count") == 2)),
                "n_classes_3_5": len(
                    known.filter((pl.col("image_count") >= 3) & (pl.col("image_count") <= 5))
                ),
                "n_classes_6_10": len(
                    known.filter((pl.col("image_count") >= 6) & (pl.col("image_count") <= 10))
                ),
                "n_classes_10plus": len(known.filter(pl.col("image_count") > 10)),
                "known_min_per_class": known["image_count"].min(),
                "known_median_per_class": known["image_count"].median(),
                "known_mean_per_class": known["image_count"].mean(),
                "known_max_per_class": known["image_count"].max(),
            }
        ]
    )


def main() -> None:
    config = EDAConfig()
    train = pl.read_csv(config.data_dir / "train.csv")
    counts = (
        train.group_by("Id").agg(pl.len().alias("image_count")).sort("image_count", descending=True)
    )
    out_dir = EDAConfig().eda_dir
    figures = EDAConfig().figures_dir / "labels"
    figures.mkdir(parents=True, exist_ok=True)
    counts.write_csv(out_dir / "class_distribution.csv")
    stats_table(train, counts).write_csv(out_dir / "label_stats.csv")
    save_bar(
        counts["Id"].to_list(),
        counts["image_count"].to_list(),
        figures / "top_classes.png",
        "most frequent classes",
        top=25,
    )
    save_hist(
        counts["image_count"].to_list(),
        figures / "class_size_histogram.png",
        "images per class",
    )
    print(stats_table(train, counts).to_dicts()[0])


if __name__ == "__main__":
    main()
