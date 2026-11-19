import csv
import hashlib
import json

import numpy as np
import polars as pl
from PIL import Image

from src.config import EDAConfig
from src.visualization.common import id_map, montage

PERCEPTUAL_THRESHOLD = 3


def _file_digest(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exact_pairs(config, train, test_names) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for row in train.to_dicts():
        path = config.data_dir / "train" / row["Image"]
        groups.setdefault(_file_digest(path), []).append(
            {"name": row["Image"], "split": "train", "id": row["Id"]}
        )
    for name in test_names:
        path = config.data_dir / "test" / name
        groups.setdefault(_file_digest(path), []).append(
            {"name": name, "split": "test", "id": None}
        )
    pairs = []
    for members in groups.values():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                first, second = members[i], members[j]
                pairs.append(
                    {
                        "match_type": "exact",
                        "first_name": first["name"],
                        "first_split": first["split"],
                        "first_id": first["id"],
                        "second_name": second["name"],
                        "second_split": second["split"],
                        "second_id": second["id"],
                        "distance": 0,
                    }
                )
    return pairs


def _perceptual_pairs(df, id_by, threshold=PERCEPTUAL_THRESHOLD) -> list[dict]:
    arr = df["phash"].to_numpy()
    names = df["image"].to_list()
    splits = df["split"].to_list()
    n = len(names)
    rows = []
    for i in range(n):
        for j in range(i + 1, min(n, i + 64)):
            distance = (int(arr[i]) ^ int(arr[j])).bit_count()
            if distance <= threshold:
                rows.append(
                    {
                        "match_type": "perceptual",
                        "first_name": names[i],
                        "first_split": splits[i],
                        "first_id": id_by.get(names[i]),
                        "second_name": names[j],
                        "second_split": splits[j],
                        "second_id": id_by.get(names[j]),
                        "distance": distance,
                    }
                )
    return rows


def _gallery(config, pairs, out_path, limit=8):
    items = []
    for pair in pairs[:limit]:
        for name_key, split_key, id_key in (
            ("first_name", "first_split", "first_id"),
            ("second_name", "second_split", "second_id"),
        ):
            name = pair[name_key]
            image = Image.open(config.data_dir / pair[split_key] / name).convert("RGB")
            wid = f"[{pair[id_key]}] " if pair[id_key] else ""
            items.append((image, f"{name} {wid}{pair['match_type']}"))
    montage(items, out_path, ncols=16)


def _pair_mae(config, first, second):
    def load(split, name):
        path = config.data_dir / split / name
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB").resize((160, 160)), dtype=np.float32)

    return float(
        np.abs(load(first["split"], first["name"]) - load(second["split"], second["name"])).mean()
    )


def main() -> None:
    config = EDAConfig()
    train = pl.read_csv(config.data_dir / "train.csv")
    test_names = pl.read_csv(config.data_dir / "sample_submission.csv")["Image"].to_list()
    id_by = id_map(config.data_dir)
    exact = _exact_pairs(config, train, test_names)
    df = pl.read_parquet(EDAConfig().eda_dir / "image_metadata.parquet")
    perceptual = sorted(_perceptual_pairs(df, id_by), key=lambda row: -row["distance"])
    pairs = perceptual + exact
    out_dir = EDAConfig().eda_dir
    with (out_dir / "duplicates.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "match_type",
                "first_name",
                "first_split",
                "first_id",
                "second_name",
                "second_split",
                "second_id",
                "distance",
            ],
        )
        writer.writeheader()
        writer.writerows(pairs)
    _gallery(config, pairs, EDAConfig().figures_dir / "duplicates" / "duplicates_top.png")
    distribution = {
        d: sum(1 for row in perceptual if row["distance"] == d)
        for d in range(PERCEPTUAL_THRESHOLD + 1)
    }
    print(
        json.dumps(
            {
                "exact": len(exact),
                "perceptual": len(perceptual),
                "total": len(pairs),
                "perceptual_distance_distribution": distribution,
                "duplicates_csv": str(out_dir / "duplicates.csv"),
            },
            indent=2,
        )
    )
    rows = []
    with (out_dir / "duplicates.csv").open(newline="", encoding="utf-8") as file:
        for pair in csv.DictReader(file):
            first = {"split": pair["first_split"], "name": pair["first_name"]}
            second = {"split": pair["second_split"], "name": pair["second_name"]}
            rows.append(
                {
                    "match_type": pair["match_type"],
                    "first_name": pair["first_name"],
                    "second_name": pair["second_name"],
                    "first_id": pair["first_id"],
                    "second_id": pair["second_id"],
                    "distance": int(pair["distance"]),
                    "mae": _pair_mae(config, first, second),
                }
            )
    with (out_dir / "duplicates_verification.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "match_type",
                "first_name",
                "second_name",
                "first_id",
                "second_id",
                "distance",
                "mae",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    buckets = {
        "mae < 1 (near-identical)": sum(1 for row in rows if row["mae"] < 1),
        "mae 1–5": sum(1 for row in rows if 1 <= row["mae"] < 5),
        "mae 5–15": sum(1 for row in rows if 5 <= row["mae"] < 15),
        "mae >= 15 (likely false positives)": sum(1 for row in rows if row["mae"] >= 15),
    }
    print(f"pairs={len(rows)}")
    for key, value in buckets.items():
        print(f"  {key}: {value}")
    print(f"top by MAE: {sorted(rows, key=lambda r: -r['mae'])[:5]}")


if __name__ == "__main__":
    main()
