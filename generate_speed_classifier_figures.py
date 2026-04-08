#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Визуализация датасета для обучения классификатора ограничения скорости (ImageFolder:
datasets/speed_cls_v<n>/train|val|test/<class>/*.png, labels.txt).

Аналог generate_diploma_figures.py, но под структуру make_dataset/generate_speed_classifier_dataset.py.

Запуск из корня репозитория:
  python generate_speed_classifier_figures.py
"""

from __future__ import annotations

import glob
import os
import random
import re
import sys
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)
sys.path.append(".")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from training.speed_classifier.dataset import read_labels_txt  # noqa: E402

# Префикс выходных файлов (чтобы не затирать figure8_collage.png от YOLO-датасета)
OUT_PREFIX = "speed_classifier_"


def _find_latest_speed_dataset_dir(root: str) -> str:
    """
    Ищем datasets/speed_cls_v<n> с train/ и labels.txt, берём максимальный n.
    Дополнительно: speed_cls_v<n> в корне проекта.
    """
    candidates: list[str] = []
    for pattern in (
        os.path.join(root, "datasets", "speed_cls_v*"),
        os.path.join(root, "speed_cls_v*"),
    ):
        candidates.extend(glob.glob(pattern))

    valid: list[tuple[int, str]] = []
    for p in candidates:
        if not os.path.isdir(p):
            continue
        if not os.path.isdir(os.path.join(p, "train")):
            continue
        if not os.path.isfile(os.path.join(p, "labels.txt")):
            continue
        base = os.path.basename(p)
        m = re.match(r"speed_cls_v(\d+)$", base)
        if not m:
            continue
        valid.append((int(m.group(1)), p))

    if not valid:
        raise FileNotFoundError(
            "Не найдена папка speed_cls_v<n> с подкаталогом train/ и файлом labels.txt. "
            "Ожидается путь вида datasets/speed_cls_v1 (сгенерировать: "
            "python make_dataset/generate_speed_classifier_dataset.py)."
        )
    valid.sort(key=lambda t: t[0])
    return valid[-1][1]


def _iter_image_files(folder: str) -> list[str]:
    """Рекурсивно собираем пути к изображениям."""
    exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    out: list[str] = []
    for root, _dirs, files in os.walk(folder):
        for fn in files:
            if fn.lower().endswith(exts):
                out.append(os.path.join(root, fn))
    return sorted(out)


def _pick_collage_images(data_dir: str, k: int = 6, seed: int = 801) -> list[str]:
    """По одному случайному кадру из разных классов train (если хватает), затем добор из train."""
    labels = read_labels_txt(data_dir)
    if not labels:
        raise FileNotFoundError(f"Нет или пустой labels.txt в {data_dir}")

    train_root = os.path.join(data_dir, "train")
    if not os.path.isdir(train_root):
        raise FileNotFoundError(f"Нет каталога train: {train_root}")

    rng = random.Random(seed)
    picks: list[str] = []
    class_order = list(labels)
    rng.shuffle(class_order)

    for cname in class_order:
        if len(picks) >= k:
            break
        cdir = os.path.join(train_root, cname)
        if not os.path.isdir(cdir):
            continue
        files = [f for f in os.listdir(cdir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp"))]
        if not files:
            continue
        picks.append(os.path.join(cdir, rng.choice(files)))

    if len(picks) < k:
        pool = _iter_image_files(train_root)
        pool = [p for p in pool if p not in picks]
        rng.shuffle(pool)
        while len(picks) < k and pool:
            picks.append(pool.pop())

    if len(picks) < k:
        raise FileNotFoundError(
            f"Недостаточно изображений в train для коллажа: нужно {k}, удалось набрать {len(picks)} ({train_root})."
        )
    return picks[:k]


def figure8_collage(data_dir: str, out_path: str, dpi: int = 300) -> None:
    """Коллаж 2×3: примеры из train."""
    paths = _pick_collage_images(data_dir, k=6, seed=801)
    panel_marks = ["а)", "б)", "в)", "г)", "д)", "е)"]

    images: list[np.ndarray] = []
    titles: list[str] = []
    for p, mark in zip(paths, panel_marks):
        try:
            rgb = np.array(Image.open(p).convert("RGB"))
        except Exception as e:
            raise RuntimeError(f"Не удалось прочитать изображение: {p}") from e
        images.append(rgb)
        stem = os.path.splitext(os.path.basename(p))[0]
        if len(stem) > 40:
            stem = stem[:37] + "…"
        titles.append(f"{mark} {stem}")

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle("Примеры изображений из датасета классификатора скорости (train)", fontsize=13, y=1.02)
    for ax, im, ttl in zip(np.ravel(axes), images, titles):
        ax.imshow(im)
        ax.set_title(ttl, fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def figure9_histogram(data_dir: str, out_path: str, dpi: int = 200) -> None:
    """Гистограмма: число файлов в train по каждому классу (порядок как в labels.txt)."""
    labels = read_labels_txt(data_dir)
    if not labels:
        raise FileNotFoundError(f"Нет labels.txt в {data_dir}")

    train_root = os.path.join(data_dir, "train")
    if not os.path.isdir(train_root):
        raise FileNotFoundError(f"Нет каталога train: {train_root}")

    counts: list[int] = []
    for i, cname in enumerate(labels):
        cdir = os.path.join(train_root, cname)
        if not os.path.isdir(cdir):
            counts.append(0)
            continue
        n = sum(1 for fn in os.listdir(cdir) if fn.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")))
        counts.append(n)

    counts_arr = np.array(counts, dtype=np.float64)
    num_classes = len(labels)
    x_display = np.arange(1, num_classes + 1)
    mean_val = float(np.mean(counts_arr)) if num_classes else 0.0

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x_display, counts_arr, width=0.85, color="seagreen", edgecolor="none")
    ax.axhline(mean_val, color="crimson", linestyle="--", linewidth=1.2, label=f"Среднее: {mean_val:.1f}")
    ax.set_xlabel("Индекс класса (порядок labels.txt)")
    ax.set_ylabel("Количество изображений в train")
    ax.set_title("Распределение изображений по классам (train, классификатор скорости)")
    ax.set_xlim(0.5, num_classes + 0.5)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def figure10_source_crops(out_path: str, dpi: int = 200) -> None:
    """
    Примеры исходных PNG до наложения фона в generate_speed_classifier_dataset
    (shared/signs/speed_png/pngs/<class>/...).
    """
    src_root = os.path.join("shared", "signs", "speed_png", "pngs")
    if not os.path.isdir(src_root):
        raise FileNotFoundError(
            f"Нет каталога исходных кропов: {src_root}. "
            "Он задаётся как SRC_ROOT в make_dataset/generate_speed_classifier_dataset.py."
        )

    all_imgs = _iter_image_files(src_root)
    if len(all_imgs) < 1:
        raise FileNotFoundError(f"В {src_root} не найдено изображений.")

    rng = random.Random(4242)
    k = min(4, len(all_imgs))
    picks = [all_imgs[i] for i in rng.sample(range(len(all_imgs)), k=k)]

    nrow, ncol = (1, k) if k <= 3 else (2, 2)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.5 * nrow))
    axes_flat = np.ravel(np.atleast_1d(axes))
    for ax, p in zip(axes_flat, picks):
        try:
            rgb = np.array(Image.open(p).convert("RGB"))
        except Exception as e:
            raise RuntimeError(f"Не удалось прочитать: {p}") from e
        ax.imshow(rgb)
        ax.set_title("Исходный PNG-кроп (до генерации с фоном)", fontsize=9)
        ax.axis("off")
    for j in range(len(picks), len(axes_flat)):
        axes_flat[j].axis("off")
    fig.suptitle("Исходные изображения знаков (shared/signs/speed_png/pngs)", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def figure11_pie(data_dir: str, out_path: str, dpi: int = 200) -> None:
    """
    Доля «плохих» кропов (имитация неточного ROI) vs остальных.
    В train имена вида ..._bad.png / ..._ok.png (см. generate_speed_classifier_dataset.py).
    Val/test обычно без суффикса — учитываются как обычные кадры.
    """
    all_files = []
    for split in ("train", "val", "test"):
        sp = os.path.join(data_dir, split)
        if os.path.isdir(sp):
            all_files.extend(_iter_image_files(sp))

    total = len(all_files)
    if total == 0:
        raise FileNotFoundError(f"В {data_dir} нет изображений в train/val/test.")

    # Маркер «bad» в имени файла задаётся генератором явно
    n_bad = sum(1 for p in all_files if "_bad.png" in os.path.basename(p).lower() or os.path.basename(p).lower().endswith("_bad.png"))
    n_rest = total - n_bad

    if n_bad == 0:
        # Нет bad-кропов (старая версия генератора / отключённые BAD_CROPS): показываем split train vs val+test
        def _count_split(name: str) -> int:
            p = os.path.join(data_dir, name)
            return len(_iter_image_files(p)) if os.path.isdir(p) else 0

        n_train = _count_split("train")
        n_val = _count_split("val")
        n_test = _count_split("test")
        other = n_val + n_test
        reason = (
            "Суффикс _bad в именах не встретился; показано разбиение train / (val+test). "
            "При включённых BAD_CROPS в генераторе появятся пары _ok/_bad."
        )
        sizes = [n_train, other]
        legend_labels = [
            f"Train ({n_train} шт., {100.0 * n_train / total:.1f}%)",
            f"Val + test ({other} шт., {100.0 * other / total:.1f}%)",
        ]
        colors = ["#4C72B0", "#DD8452"]
    else:
        reason = (
            "Доля «плохих» кропов по суффиксу _bad.png в имени (train). "
            "Остальное — обычные кадры и val/test."
        )
        sizes = [n_rest, n_bad]
        legend_labels = [
            f"Обычные кадры ({n_rest} шт., {100.0 * n_rest / total:.1f}%)",
            f"«Плохие» кропы, bad ({n_bad} шт., {100.0 * n_bad / total:.1f}%)",
        ]
        colors = ["#4C72B0", "#C44E52"]

    # Широкая фигура: подписи вынесены в легенду справа, подпись снизу — с переносами; savefig с bbox_inches='tight'
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    wedges, _texts, autotexts = ax.pie(
        sizes,
        labels=None,
        autopct="%1.1f%%",
        startangle=90,
        colors=colors,
        pctdistance=0.72,
        textprops={"fontsize": 11},
    )
    for t in autotexts:
        t.set_fontweight("bold")
        t.set_color("white")

    ax.legend(
        wedges,
        legend_labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=9,
        frameon=True,
        borderaxespad=0.0,
    )
    ax.set_title("Состав датасета классификатора скорости", fontsize=12, pad=12)
    ax.set_aspect("equal")

    footer = textwrap.fill(f"Всего файлов: {total}. {reason}", width=72)
    fig.text(0.5, 0.02, footer, ha="center", va="bottom", fontsize=8, transform=fig.transFigure)

    fig.subplots_adjust(left=0.06, right=0.72, bottom=0.18, top=0.90)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)


def main() -> None:
    data_dir = _find_latest_speed_dataset_dir(PROJECT_ROOT)
    print(f"[INFO] Используется датасет классификатора скорости: {data_dir}")

    out8 = os.path.join(PROJECT_ROOT, f"{OUT_PREFIX}figure8_collage.png")
    out9 = os.path.join(PROJECT_ROOT, f"{OUT_PREFIX}figure9_histogram.png")
    out10 = os.path.join(PROJECT_ROOT, f"{OUT_PREFIX}figure10_source_crops.png")
    out11 = os.path.join(PROJECT_ROOT, f"{OUT_PREFIX}figure11_pie.png")

    figure8_collage(data_dir, out8, dpi=300)
    figure9_histogram(data_dir, out9, dpi=200)
    figure10_source_crops(out10, dpi=200)
    figure11_pie(data_dir, out11, dpi=200)

    print(
        f"Графики сохранены: {OUT_PREFIX}figure8_collage.png, {OUT_PREFIX}figure9_histogram.png, "
        f"{OUT_PREFIX}figure10_source_crops.png, {OUT_PREFIX}figure11_pie.png"
    )


if __name__ == "__main__":
    main()
