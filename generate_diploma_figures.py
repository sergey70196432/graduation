#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Генерация рисунков 8–11 для раздела 2.2 ВКР из реальных данных проекта.
Запуск из корня репозитория: python generate_diploma_figures.py
"""

from __future__ import annotations

import glob
import os
import random
import re
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")  # файловый вывод без GUI (macOS / сервер)
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Корень проекта = каталог, где лежит этот скрипт
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)
sys.path.append(".")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from make_dataset import config as cfg  # noqa: E402
from make_dataset import utils  # noqa: E402


def _find_latest_dataset_dir(root: str) -> str:
    """
    Ищем datasets/dataset_<n> с подпапками images/ и labels/, берём максимальный n.
    Дополнительно проверяем dataset_<n> в корне (как в формулировке ТЗ).
    """
    candidates: list[str] = []
    for base in (
        os.path.join(root, "datasets", "dataset_*"),
        os.path.join(root, "dataset_*"),
    ):
        candidates.extend(glob.glob(base))

    valid: list[tuple[int, str]] = []
    for p in candidates:
        if not os.path.isdir(p):
            continue
        if not os.path.isdir(os.path.join(p, "images")) or not os.path.isdir(os.path.join(p, "labels")):
            continue
        base = os.path.basename(p)
        m = re.match(r"dataset_(\d+)$", base)
        if not m:
            continue
        valid.append((int(m.group(1)), p))

    if not valid:
        raise FileNotFoundError(
            "Не найдена ни одна папка dataset_<n> с подкаталогами images/ и labels/. "
            "Ожидаются пути вида datasets/dataset_1 или dataset_1 в корне проекта."
        )
    valid.sort(key=lambda t: t[0])
    return valid[-1][1]


def _pick_collage_images_from_dataset(dataset_dir: str, k: int = 6, seed: int = 801) -> list[str]:
    """
    Выбираем k путей к изображениям из dataset_dir/images/.
    Предпочитаем кадры с непустой YOLO-разметкой (на кадре есть хотя бы один знак).
    """
    images_dir = os.path.join(dataset_dir, "images")
    labels_dir = os.path.join(dataset_dir, "labels")
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Нет каталога изображений датасета: {images_dir}")
    if not os.path.isdir(labels_dir):
        raise FileNotFoundError(f"Нет каталога разметки: {labels_dir}")

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    names = sorted(n for n in os.listdir(images_dir) if os.path.splitext(n)[1].lower() in exts)
    if not names:
        raise FileNotFoundError(f"В {images_dir} нет изображений.")

    with_objects: list[str] = []
    for n in names:
        stem = os.path.splitext(n)[0]
        lp = os.path.join(labels_dir, stem + ".txt")
        if not os.path.isfile(lp):
            continue
        with open(lp, "r", encoding="utf-8") as f:
            if any(line.strip() for line in f):
                with_objects.append(os.path.join(images_dir, n))

    pool = with_objects if len(with_objects) >= k else [os.path.join(images_dir, n) for n in names]
    if len(pool) < k:
        raise FileNotFoundError(
            f"Недостаточно изображений для коллажа: нужно {k}, в пуле {len(pool)} "
            f"(каталог {images_dir})."
        )

    rng = random.Random(seed)
    # Воспроизводимый набор без повторов
    picks = rng.sample(pool, k=k)
    return picks


def figure8_collage(dataset_dir: str, out_path: str, dpi: int = 300) -> None:
    """Рисунок 8: коллаж 2×3 из шести кадров готового датасета (как сохранил генератор)."""
    paths = _pick_collage_images_from_dataset(dataset_dir, k=6, seed=801)
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
        if len(stem) > 42:
            stem = stem[:39] + "…"
        titles.append(f"{mark} {stem}")

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle("Примеры изображений из синтетического датасета", fontsize=13, y=1.02)
    for ax, im, ttl in zip(np.ravel(axes), images, titles):
        ax.imshow(im)
        ax.set_title(ttl, fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def figure9_histogram(dataset_dir: str, out_path: str, dpi: int = 200) -> None:
    """Рисунок 9: число изображений, в которых встречается класс (по label-файлам)."""
    classes_path = os.path.join(dataset_dir, "classes.txt")
    if not os.path.isfile(classes_path):
        raise FileNotFoundError(f"Нет файла classes.txt в {dataset_dir}")

    with open(classes_path, "r", encoding="utf-8") as f:
        class_lines = [ln.strip() for ln in f.readlines() if ln.strip() != ""]
    num_classes = len(class_lines)
    if num_classes == 0:
        raise ValueError(f"Файл classes.txt пуст: {classes_path}")

    labels_dir = os.path.join(dataset_dir, "labels")
    label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
    if not label_files:
        raise FileNotFoundError(f"В {labels_dir} нет .txt разметки.")

    # Для каждого класса — множество путей к label-файлам, где класс присутствует
    images_per_class: dict[int, set[str]] = defaultdict(set)
    for lp in label_files:
        with open(lp, "r", encoding="utf-8") as f:
            present: set[int] = set()
            for line in f:
                parts = line.split()
                if len(parts) < 5:
                    continue
                try:
                    cid = int(float(parts[0]))
                except ValueError:
                    continue
                if 0 <= cid < num_classes:
                    present.add(cid)
        for cid in present:
            images_per_class[cid].add(lp)

    counts = np.array([len(images_per_class.get(i, set())) for i in range(num_classes)], dtype=np.float64)
    x_display = np.arange(1, num_classes + 1)  # 1..N для оси X (как в ТЗ 1..114)
    mean_val = float(np.mean(counts)) if num_classes else 0.0

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x_display, counts, width=0.85, color="steelblue", edgecolor="none")
    ax.axhline(mean_val, color="crimson", linestyle="--", linewidth=1.2, label=f"Среднее: {mean_val:.1f}")
    ax.set_xlabel("Номер класса")
    ax.set_ylabel("Количество изображений")
    ax.set_title("Баланс классов")
    ax.set_xlim(0.5, num_classes + 0.5)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def figure10_real_examples(out_path: str, dpi: int = 200) -> None:
    """Рисунок 10: кадры видеорегистратора (real_frames или dashcam_frames)."""
    real_dir = "make_dataset/external_dataset/Road Sign.v8/train/images"
    if os.path.isdir(real_dir) and utils.list_files_by_ext_recursive(real_dir, cfg.BACKGROUND_EXTS):
        frames = utils.list_files_by_ext_recursive(real_dir, cfg.BACKGROUND_EXTS)
        caption_note = "реальный кадр"
    else:
        if not os.path.isdir(cfg.DASHCAM_FRAMES_DIR):
            raise FileNotFoundError(
                f"Нет папки и нет DASHCAM_FRAMES_DIR={cfg.DASHCAM_FRAMES_DIR!r}."
            )
        frames = utils.list_files_by_ext_recursive(cfg.DASHCAM_FRAMES_DIR, cfg.BACKGROUND_EXTS)
        if not frames:
            raise FileNotFoundError(f"В {cfg.DASHCAM_FRAMES_DIR} нет изображений для рисунка 10.")
        caption_note = "реальный кадр (видеорегистратор)"

    rng = random.Random(4242)
    picks = [frames[i] for i in rng.sample(range(len(frames)), k=min(4, len(frames)))]

    n = len(picks)
    if n <= 2:
        nrow, ncol = 1, n
    else:
        nrow, ncol = 2, 2
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.5 * nrow))
    axes_flat = np.ravel(np.atleast_1d(axes))
    for ax, p in zip(axes_flat, picks):
        try:
            rgb = np.array(Image.open(p).convert("RGB"))
        except Exception as e:
            raise RuntimeError(f"Не удалось прочитать изображение: {p}") from e
        ax.imshow(rgb)
        ax.set_title(caption_note, fontsize=10)
        ax.axis("off")
    for j in range(len(picks), len(axes_flat)):
        axes_flat[j].axis("off")
    fig.suptitle("Примеры реальных изображений, подмешанных в датасет", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def figure11_pie(dataset_dir: str, out_path: str, dpi: int = 200) -> None:
    """
    Рисунок 11: доля синтетических и реальных изображений.

    Реальные (внешний YOLO-датасет): имена с маркером __ext__ при импорте в generate_synth_yolo_dataset.
    Если таких файлов нет, используем ориентир REAL_SHARE_FALLBACK (как в ТЗ ~10–15% реальных при отсутствии явной маркировки).
    """
    images_dir = os.path.join(dataset_dir, "images")
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Нет каталога изображений: {images_dir}")

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    names = [n for n in os.listdir(images_dir) if os.path.splitext(n)[1].lower() in exts]
    total = len(names)
    if total == 0:
        raise FileNotFoundError(f"В {images_dir} нет изображений.")

    n_real_marker = sum(1 for n in names if "__ext__" in n)
    # Доля «реальных» по явному признаку импорта
    if n_real_marker > 0:
        n_real = n_real_marker
        n_synth = total - n_real
        reason = "по маркеру __ext__ в имени файла (импорт из внешнего датасета)"
    else:
        # Нет маркеров: как в ТЗ — приблизительная оценка (обоснование для текста ВКР)
        REAL_SHARE_FALLBACK = 0.10
        n_real = int(round(total * REAL_SHARE_FALLBACK))
        n_synth = total - n_real
        reason = (
            f"маркер __ext__ не найден; оценка {REAL_SHARE_FALLBACK:.0%} реальных кадров по типичному плану смешивания "
            "(см. настройки EXTERNAL_MIX_ENABLED / негативы; уточните по фактическому пайплайну генерации)"
        )

    synth_pct = 100.0 * n_synth / total
    real_pct = 100.0 * n_real / total

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(
        [n_synth, n_real],
        labels=[f"Синтетические ({synth_pct:.1f}%)", f"Реальные ({real_pct:.1f}%)"],
        autopct="%1.1f%%",
        startangle=90,
        colors=["#4C72B0", "#DD8452"],
    )
    ax.set_title("Соотношение синтетических и реальных данных в датасете")
    fig.text(0.5, 0.02, f"Итого изображений: {total}. Метод оценки: {reason}", ha="center", fontsize=8)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    dataset_dir = _find_latest_dataset_dir(PROJECT_ROOT)

    out8 = os.path.join(PROJECT_ROOT, "figure8_collage.png")
    out9 = os.path.join(PROJECT_ROOT, "figure9_histogram.png")
    out10 = os.path.join(PROJECT_ROOT, "figure10_real_examples.png")
    out11 = os.path.join(PROJECT_ROOT, "figure11_pie.png")

    print(f"[INFO] Используется датасет: {dataset_dir}")

    figure8_collage(dataset_dir, out8, dpi=300)
    figure9_histogram(dataset_dir, out9, dpi=200)
    figure10_real_examples(out10, dpi=200)
    figure11_pie(dataset_dir, out11, dpi=200)

    print(
        "Графики сохранены: figure8_collage.png, figure9_histogram.png, "
        "figure10_real_examples.png, figure11_pie.png"
    )


if __name__ == "__main__":
    main()
