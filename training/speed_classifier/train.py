from __future__ import annotations

"""
Обучение классификатора значения скорости.

Идея максимально простая:
- Датасет лежит в папке DATA_DIR и имеет структуру:
    train/<class_name>/*.png
    val/<class_name>/*.png   (опционально)
    test/<class_name>/*.png  (опционально)
- Класс = имя папки, например: "3.24_50"
- Модель: MobileNetV3-Small
- Выход: папка run_<n> с артефактами (чекпоинты, метрики, конфиг).

Запуск:
  python training/speed_classifier/train.py
"""

import json
import os
import time
from dataclasses import asdict
from contextlib import nullcontext

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Импорты сделаны с fallback, чтобы работало и так:
# - python training/speed_classifier/train.py
# - python -m training.speed_classifier.train
try:
    from training.speed_classifier.dataset import DatasetConfig, load_split_dataset, read_labels_txt
    from training.speed_classifier.model import create_model
    from training.speed_classifier.utils import pick_device
except Exception:  # pragma: no cover
    from dataset import DatasetConfig, load_split_dataset, read_labels_txt
    from model import create_model
    from utils import pick_device


# ============================================================
# ПАРАМЕТРЫ (ПРАВЬТЕ ТОЛЬКО ЭТОТ БЛОК)
# ============================================================

# Где лежит датасет (папки train/val/test внутри).
# Обычно генерируется скриптом make_dataset/generate_speed_classifier_dataset.py
DATA_DIR = "datasets/speed_cls_v1"

# Папка, куда складываются все запуски обучения.
# Каждый новый запуск автоматически создаёт подпапку run_<n> (run_1, run_2, ...).
RUNS_ROOT = "training/speed_classifier/runs"

# Если нужно переобучить "в ту же папку" или задать своё имя — укажите явно:
# - "" / None  : авто-выбор следующего номера run_<n>
# - "run_12"   : сохранит в RUNS_ROOT/run_12
# - "my_run"   : сохранит в RUNS_ROOT/my_run
RUN_NAME = ""

# Размер входа модели.
# В dataset.py все изображения приводятся к квадрату IMAGE_SIZE x IMAGE_SIZE.
IMAGE_SIZE = 128

# Параметры обучения:
# - EPOCHS: сколько эпох
# - BATCH_SIZE: размер батча
# - LEARNING_RATE: скорость обучения
# - NUM_WORKERS: сколько воркеров у DataLoader (на macOS иногда лучше 0)
# - SEED: фиксируем сид, чтобы результаты были воспроизводимее
EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
NUM_WORKERS = 4
SEED = 1337


# Какое устройство использовать (cuda / mps / cpu).
# pick_device() внутри пытается выбрать "лучшее доступное".
DEVICE = pick_device()

# AMP (mixed precision):
# - на CUDA обычно даёт ускорение
# - на MPS по-разному, но поддержка есть не во всех версиях PyTorch
# Поэтому включаем "умно": точно включаем на CUDA, на MPS пробуем, иначе выключаем.
USE_AMP = True

# Предобученные веса ImageNet:
# - True  : лучше старт, но нужно скачать веса (интернет/кеш)
# - False : без скачивания, проще повторять
PRETRAINED = False


def _set_seed(seed: int):
    """
    Фиксируем сиды. Это не гарантирует 100% детерминизм (особенно на GPU),
    но заметно снижает разброс результатов между запусками.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_autocast(device: torch.device):
    """
    Возвращает контекст autocast (mixed precision), без deprecated torch.cuda.amp.*.

    AMP полезен:
    - CUDA: обычно ускоряет и уменьшает память
    - MPS: иногда ускоряет, но зависит от версии PyTorch
    """
    amp_enabled = bool(USE_AMP) and (device.type in ("cuda", "mps"))
    if not amp_enabled:
        return nullcontext()

    # Новый API: torch.amp.autocast(device_type=...)
    try:
        return torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=True)
    except Exception:
        # Fallback для старых версий
        if device.type == "cuda":
            return torch.cuda.amp.autocast(enabled=True)
        return nullcontext()


def make_grad_scaler(device: torch.device):
    """
    Возвращает GradScaler, если он нужен.

    GradScaler имеет смысл в основном на CUDA.
    На CPU/MPS чаще всего его не используют.
    """
    if not (bool(USE_AMP) and device.type == "cuda"):
        return None
    try:
        # Новый API
        return torch.amp.GradScaler("cuda", enabled=True)
    except Exception:
        # Старый API
        return torch.cuda.amp.GradScaler(enabled=True)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    """
    Оценка модели на val/test:
    - loss
    - top-1 accuracy
    - top-3 accuracy
    """
    model.eval()
    total = 0
    correct1 = 0
    correct3 = 0
    loss_sum = 0.0
    ce = nn.CrossEntropyLoss()

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = ce(logits, y)

        total += int(y.numel())
        loss_sum += float(loss.item()) * int(y.numel())

        # top-1
        pred1 = logits.argmax(dim=1)
        correct1 += int((pred1 == y).sum().item())

        # top-3
        top3 = torch.topk(logits, k=min(3, logits.shape[1]), dim=1).indices
        correct3 += int((top3 == y[:, None]).any(dim=1).sum().item())

    if total <= 0:
        return {"loss": 0.0, "acc1": 0.0, "acc3": 0.0, "n": 0}

    return {
        "loss": loss_sum / total,
        "acc1": correct1 / total,
        "acc3": correct3 / total,
        "n": total,
    }


def _pick_next_run_dir(runs_root: str, prefix: str = "run_") -> str:
    """
    Выбираем следующую папку вида run_<n> внутри runs_root.

    Пример:
    - если есть run_1, run_2, run_7 -> вернём run_8
    - если нет ни одной -> вернём run_1
    """
    os.makedirs(runs_root, exist_ok=True)
    best = 0
    for name in os.listdir(runs_root):
        if not name.startswith(prefix):
            continue
        tail = name[len(prefix) :]
        if not tail.isdigit():
            continue
        n = int(tail)
        if n > best:
            best = n
    return os.path.join(runs_root, f"{prefix}{best + 1}")


def main():
    """
    - читает ImageFolder датасет (train/val/test)
    - обучает MobileNetV3-small
    - сохраняет best.pt (по val acc1) и last.pt
    """
    _set_seed(int(SEED))
    device = torch.device(str(DEVICE))

    # Папка вывода: по умолчанию каждый запуск — новая run_<n>, чтобы не затирать прошлые результаты.
    if RUN_NAME and str(RUN_NAME).strip():
        out_dir = os.path.join(str(RUNS_ROOT), str(RUN_NAME).strip())
    else:
        out_dir = _pick_next_run_dir(str(RUNS_ROOT))

    # Конфиг датасета (важен image_size и Normalize в dataset.py).
    cfg = DatasetConfig(data_dir=str(DATA_DIR), image_size=int(IMAGE_SIZE))
    # labels.txt теперь используется как источник истины для порядка классов (см. dataset.py).
    labels = read_labels_txt(cfg.data_dir)

    # Загружаем датасеты.
    train_ds = load_split_dataset(cfg, "train")
    val_ds = load_split_dataset(cfg, "val") if os.path.isdir(os.path.join(cfg.data_dir, "val")) else None
    test_ds = load_split_dataset(cfg, "test") if os.path.isdir(os.path.join(cfg.data_dir, "test")) else None

    # Количество классов берём из датасета (он уже построен с class_to_idx из labels.txt).
    num_classes = len(train_ds.classes)
    if labels and len(labels) != num_classes:
        # В норме это не должно происходить, потому что load_split_dataset использует labels.txt.
        print("[WARN] labels.txt size != dataset classes size:", len(labels), num_classes)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(BATCH_SIZE),
        shuffle=True,
        num_workers=int(NUM_WORKERS),
        # pin_memory имеет смысл в основном для CUDA. На MPS будет warning, поэтому выключаем.
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=int(BATCH_SIZE),
            shuffle=False,
            num_workers=int(NUM_WORKERS),
            pin_memory=(device.type == "cuda"),
        )
        if val_ds is not None
        else None
    )
    test_loader = (
        DataLoader(
            test_ds,
            batch_size=int(BATCH_SIZE),
            shuffle=False,
            num_workers=int(NUM_WORKERS),
            pin_memory=(device.type == "cuda"),
        )
        if test_ds is not None
        else None
    )

    model = create_model(num_classes, pretrained=bool(PRETRAINED)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(LEARNING_RATE), weight_decay=1e-2)
    ce = nn.CrossEntropyLoss()
    scaler = make_grad_scaler(device)
    autocast_ctx = make_autocast(device)

    # Пишем конфиг запуска (параметры + список классов), чтобы потом было легко повторить.
    os.makedirs(str(out_dir), exist_ok=True)
    with open(os.path.join(str(out_dir), "config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "params": {
                    "DATA_DIR": DATA_DIR,
                    "RUNS_ROOT": RUNS_ROOT,
                    "RUN_NAME": RUN_NAME,
                    "OUT_DIR": out_dir,
                    "IMAGE_SIZE": IMAGE_SIZE,
                    "EPOCHS": EPOCHS,
                    "BATCH_SIZE": BATCH_SIZE,
                    "LEARNING_RATE": LEARNING_RATE,
                    "NUM_WORKERS": NUM_WORKERS,
                    "SEED": SEED,
                    "DEVICE": DEVICE,
                    "PRETRAINED": PRETRAINED,
                },
                "dataset": asdict(cfg),
                "classes": train_ds.classes,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("[run]", os.path.abspath(str(out_dir)))

    best_acc1 = -1.0
    best_path = os.path.join(str(out_dir), "best.pt")
    last_path = os.path.join(str(out_dir), "last.pt")

    for epoch in range(1, int(EPOCHS) + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        seen = 0
        train_correct1 = 0
        train_correct3 = 0

        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{int(EPOCHS)}", leave=False)
        for x, y in pbar:
            # non_blocking имеет смысл в основном для CUDA
            x = x.to(device, non_blocking=(device.type == "cuda"))
            y = y.to(device, non_blocking=(device.type == "cuda"))

            opt.zero_grad(set_to_none=True)
            with autocast_ctx:
                logits = model(x)
                loss = ce(logits, y)

            # Метрики на train (просто чтобы понимать, что модель реально учится)
            with torch.no_grad():
                pred1 = logits.argmax(dim=1)
                train_correct1 += int((pred1 == y).sum().item())
                top3 = torch.topk(logits, k=min(3, logits.shape[1]), dim=1).indices
                train_correct3 += int((top3 == y[:, None]).any(dim=1).sum().item())

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()

            bs = int(y.numel())
            seen += bs
            running += float(loss.item()) * bs
            pbar.set_postfix(loss=(running / max(1, seen)))

        train_loss = running / max(1, seen)
        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc1": (train_correct1 / max(1, seen)),
            "train_acc3": (train_correct3 / max(1, seen)),
            "seconds": time.time() - t0,
        }

        if val_loader is not None:
            val_m = evaluate(model, val_loader, device)
            metrics.update({f"val_{k}": v for k, v in val_m.items()})

            acc1 = float(val_m["acc1"])
            if acc1 > best_acc1:
                best_acc1 = acc1
                torch.save(
                    {"model": model.state_dict(), "classes": train_ds.classes, "image_size": cfg.image_size},
                    best_path,
                )

        torch.save(
            {"model": model.state_dict(), "classes": train_ds.classes, "image_size": cfg.image_size},
            last_path,
        )

        # Пишем метрики в jsonl (по строке на эпоху).
        with open(os.path.join(str(out_dir), "metrics.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics, ensure_ascii=False) + "\n")

        print("[epoch]", json.dumps(metrics, ensure_ascii=False))

    # Final eval on test
    if test_loader is not None:
        # Финальная оценка на test (если он есть).
        m = evaluate(model, test_loader, device)
        with open(os.path.join(str(out_dir), "test_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(m, f, ensure_ascii=False, indent=2)
        print("[test]", m)


if __name__ == "__main__":
    main()
