import numpy as np
import polars as pl
from PIL import Image

from src.config import EDAConfig
from src.visualization.common import id_map, montage


def _train_items(config, names, id_by):
    items = []
    for name in names:
        image = Image.open(config.data_dir / "train" / name).convert("RGB")
        wid = f"[{id_by[name]}]" if name in id_by else ""
        items.append((image, f"{name} {wid}".strip()))
    return items


def main() -> None:
    config = EDAConfig()
    counts = pl.read_csv(EDAConfig().eda_dir / "class_distribution.csv")
    id_by = id_map(config.data_dir)
    rng = np.random.default_rng(42)
    out_dir = EDAConfig().figures_dir / "galleries"
    out_dir.mkdir(parents=True, exist_ok=True)

    train = pl.read_csv(config.data_dir / "train.csv")
    train_names = train["Image"].to_list()
    sample_idx = rng.choice(len(train_names), min(12, len(train_names)), replace=False)
    montage(
        _train_items(config, [train_names[i] for i in sample_idx], id_by),
        out_dir / "random_train.png",
        ncols=4,
    )

    top_classes = counts.filter(pl.col("Id") != "new_whale").head(8)["Id"].to_list()
    for cls_id in top_classes:
        names = train.filter(pl.col("Id") == cls_id)["Image"].to_list()[:6]
        if names:
            safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in cls_id)
            montage(
                _train_items(config, names, id_by), out_dir / f"class_{safe}.png", ncols=len(names)
            )

    singletons = counts.filter((pl.col("Id") != "new_whale") & (pl.col("image_count") == 1))[
        "Id"
    ].to_list()
    n_sample = min(6, len(singletons))
    picked = [singletons[i] for i in rng.choice(len(singletons), n_sample, replace=False)]
    names = [
        name
        for cls_id in picked
        for name in train.filter(pl.col("Id") == cls_id)["Image"].to_list()[:1]
    ]
    montage(_train_items(config, names, id_by), out_dir / "singletons.png", ncols=len(names))

    new_whale_names = train.filter(pl.col("Id") == "new_whale")["Image"].to_list()
    nw_sample = [
        new_whale_names[i]
        for i in rng.choice(len(new_whale_names), min(8, len(new_whale_names)), replace=False)
    ]
    montage(_train_items(config, nw_sample, id_by), out_dir / "new_whale_sample.png", ncols=4)
    print(f"galleries saved to {out_dir}")


if __name__ == "__main__":
    main()
