from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from PIL import Image, ImageDraw


def save_hist(values, out_path, title="", bins=30):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(np.asarray(values, dtype=float), bins=bins, color="#4c72b0", edgecolor="white")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_bar(labels, values, out_path, title="", top=None):
    if top is not None:
        labels, values = list(labels)[:top], list(values)[:top]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(labels, values, color="#4c72b0")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_scatter(x, y, out_path, title="", xlabel="", ylabel=""):
    order = np.argsort(np.asarray(x, dtype=float))
    xs, ys = np.asarray(x, dtype=float)[order], np.asarray(y, dtype=float)[order]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(xs, ys, s=8, alpha=0.5, color="#4c72b0")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def montage(images_with_labels, out_path, ncols=8, thumb=(160, 160)):
    if not images_with_labels:
        return
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    rows = -(-len(images_with_labels) // ncols)
    width = ncols * thumb[0] + (ncols + 1) * 8
    line_height = thumb[1] + 22
    height = rows * line_height + 8
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (image, label) in enumerate(images_with_labels):
        column, row = index % ncols, index // ncols
        x, y = column * (thumb[0] + 8) + 8, row * line_height + 8
        box = Image.new("RGB", thumb, "white")
        image.thumbnail(thumb)
        box.paste(image, ((thumb[0] - image.width) // 2, (thumb[1] - image.height) // 2))
        canvas.paste(box, (x, y))
        draw.text((x, y + thumb[1] + 4), label, fill="black")
    canvas.save(out_path)


def id_map(data_dir):
    train = pl.read_csv(Path(data_dir) / "train.csv")
    return dict(zip(train["Image"].to_list(), train["Id"].to_list(), strict=True))
