from __future__ import annotations

"""
Оценка (eval) классификатора скорости.

Что делает скрипт:
- загружает чекпоинт PyTorch (best.pt/last.pt)
- прогоняет val или test
- сохраняет confusion matrix (png)
- сохраняет подробный CSV "файл -> предсказание":
    class, image_path, predicted_class, conf

Запуск:
  python training/speed_classifier/eval.py
"""

import csv
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm

# Импорты сделаны с fallback, чтобы работало и так:
# - python training/speed_classifier/train.py
# - python -m training.speed_classifier.train
try:
    from training.speed_classifier.dataset import DatasetConfig, load_split_dataset
    from training.speed_classifier.model import create_model
    from training.speed_classifier.utils import pick_device
except Exception:  # pragma: no cover
    from dataset import DatasetConfig, load_split_dataset
    from model import create_model
    from utils import pick_device


# ============================================================
# ПАРАМЕТРЫ (ПРАВЬТЕ ТОЛЬКО ЭТОТ БЛОК)
# ============================================================

# Где лежит датасет (папки train/val/test внутри)
DATA_DIR = "datasets/speed_cls_v1"

# Какой сплит оценивать: 'val' или 'test'
SPLIT = "test"

# Какой чекпоинт оценивать
CKPT_PATH = "training/speed_classifier/runs/run1/best.pt"

# Размер входа (должен совпадать с train.py)
IMAGE_SIZE = 128

# Даталоадер
BATCH_SIZE = 64
NUM_WORKERS = 0  # 0 = надёжно и кроссплатформенно (без shared memory проблем)

# Куда сохранить артефакты eval (confusion matrix, txt, csv).
# Если оставить пустым "", то OUT_DIR будет равен папке чекпоинта (dirname(CKPT_PATH)).
OUT_DIR = ""

# Сколько самых частых ошибок вывести/сохранить
TOP_K_CONFUSIONS = 30

# Файл со всеми предсказаниями (по каждому изображению): class, image_path, predicted_class, conf
PREDICTIONS_CSV_NAME = "predictions_test.csv"  # можно поменять на predictions_val.csv

# Устройство
DEVICE = pick_device()


def parse_speed_label(label: str):
    """
    Пытаемся распарсить "<base>_<speed>", например "3.24_60".
    Возвращаем (base, speed_int) или (label, None), если не получилось.
    """
    s = str(label).strip()
    if "_" not in s:
        return s, None
    base, speed = s.rsplit("_", 1)
    try:
        return base, int(speed)
    except Exception:
        return s, None


def write_top_confusions_txt(out_path: str, cm: np.ndarray, class_names: list[str], top_k: int):
    """
    Сохраняем в txt самые частые ошибки (off-diagonal).
    """
    n = int(cm.shape[0])
    pairs = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            c = int(cm[i, j])
            if c <= 0:
                continue
            true_name = class_names[i] if i < len(class_names) else str(i)
            pred_name = class_names[j] if j < len(class_names) else str(j)
            tb, ts = parse_speed_label(true_name)
            pb, ps = parse_speed_label(pred_name)
            same_base = tb == pb and ts is not None and ps is not None
            delta = (ps - ts) if (ts is not None and ps is not None) else None
            pairs.append((c, same_base, true_name, pred_name, delta))

    pairs.sort(key=lambda x: (-x[0], x[2], x[3]))
    top = pairs[: max(0, int(top_k))]

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Top confusions (count, true -> pred)\n")
        f.write("\n")
        for (c, same_base, t, p, delta) in top:
            if delta is None:
                extra = ""
            else:
                extra = f"  (same_base, Δspeed={delta})" if same_base else f"  (Δspeed={delta})"
            f.write(f"{c}\t{t}\t->\t{p}{extra}\n")


class ImageFolderWithPaths(Dataset):
    """
    Обёртка над ImageFolder, чтобы вместе с (x, y) вернуть ещё и путь до файла.
    Так мы можем сохранить полный отчёт "какой файл -> какой предикт".
    """

    def __init__(self, base_ds):
        self.base = base_ds

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int):
        x, y = self.base[idx]
        path = self.base.samples[idx][0] if hasattr(self.base, "samples") else ""
        return x, y, path


def write_predictions_csv(out_path: str, rows: list[tuple[str, str, str, float]]):
    """
    Пишем CSV с колонками:
      class, image_path, predicted_class, conf
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "image_path", "predicted_class", "conf"])
        for (true_cls, image_path, pred_cls, conf) in rows:
            w.writerow([true_cls, image_path, pred_cls, f"{float(conf):.6f}"])


def confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Простейшая confusion matrix без sklearn.
    cm[i, j] = сколько раз истинный класс i был предсказан как j
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for yt, yp in zip(y_true.tolist(), y_pred.tolist()):
        if 0 <= int(yt) < num_classes and 0 <= int(yp) < num_classes:
            cm[int(yt), int(yp)] += 1
    return cm


@torch.no_grad()
def predict_all(model: torch.nn.Module, loader: DataLoader, device: torch.device):
    """
    Прогоняем весь датасет и собираем:
    - y: истинные классы
    - p: предсказанные классы (argmax)
    - rows: полный отчёт по изображениям (class, image_path, predicted_class, conf)
    """
    model.eval()
    ys = []
    ps = []

    # Имена классов (берём из ImageFolder)
    base_ds = getattr(loader.dataset, "base", loader.dataset)
    class_names = getattr(base_ds, "classes", None)
    if not isinstance(class_names, list):
        class_names = []

    rows: list[tuple[str, str, str, float]] = []

    for x, y, paths in tqdm(loader, desc="predict", leave=False):
        x = x.to(device, non_blocking=(device.type == "cuda"))
        logits = model(x)
        pred = logits.argmax(dim=1)
        prob = torch.softmax(logits, dim=1)
        conf = prob[torch.arange(pred.shape[0]), pred]

        pred_np = pred.cpu().numpy()
        ys.append(y.numpy())
        ps.append(pred_np)

        # Построчно сохраняем "что по конкретному файлу"
        for i in range(int(pred.shape[0])):
            yi = int(y[i].item()) if hasattr(y[i], "item") else int(y[i])
            pi = int(pred_np[i])
            true_cls = class_names[yi] if 0 <= yi < len(class_names) else str(yi)
            pred_cls = class_names[pi] if 0 <= pi < len(class_names) else str(pi)
            path_i = str(paths[i]) if isinstance(paths, (list, tuple)) else str(paths)
            rows.append((true_cls, path_i, pred_cls, float(conf[i].item())))

    return np.concatenate(ys, axis=0), np.concatenate(ps, axis=0), rows


def main():
    device = torch.device(str(DEVICE))
    split = str(SPLIT).strip().lower()
    if split not in ("val", "test"):
        raise ValueError("SPLIT должен быть 'val' или 'test'.")

    # Если OUT_DIR не задан — складываем артефакты рядом с чекпоинтом.
    # Так удобнее: одна папка run_<n> содержит и чекпоинты, и результаты eval.
    out_dir = str(OUT_DIR).strip()
    if not out_dir:
        out_dir = os.path.dirname(os.path.abspath(str(CKPT_PATH)))

    cfg = DatasetConfig(data_dir=str(DATA_DIR), image_size=int(IMAGE_SIZE))
    base_ds = load_split_dataset(cfg, split)
    ds = ImageFolderWithPaths(base_ds)
    loader = DataLoader(
        ds,
        batch_size=int(BATCH_SIZE),
        shuffle=False,
        num_workers=int(NUM_WORKERS),
        pin_memory=(device.type == "cuda"),
    )

    model = create_model(len(base_ds.classes), pretrained=False).to(device)

    ckpt = torch.load(str(CKPT_PATH), map_location="cpu")
    model.load_state_dict(ckpt["model"])

    # Проверка на рассинхрон классов:
    # Если labels.txt поменяли после обучения или датасет собран иначе — индексы будут неправильные.
    ckpt_classes = ckpt.get("classes")
    if isinstance(ckpt_classes, list):
        ckpt_classes = [str(x) for x in ckpt_classes]
        if ckpt_classes != list(base_ds.classes):
            raise RuntimeError(
                "Рассинхрон классов между чекпоинтом и датасетом.\n"
                f"ckpt_classes_len={len(ckpt_classes)} dataset_classes_len={len(base_ds.classes)}\n"
                "Это означает, что labels.txt / порядок папок отличается от того, с чем обучали модель.\n"
                "Решение: использовать тот же labels.txt, что был при обучении, или переобучить модель."
            )

    y, p, rows = predict_all(model, loader, device)
    cm = confusion_matrix_np(y, p, num_classes=len(base_ds.classes))

    # Matplotlib иногда пытается писать кеш в домашнюю директорию.
    # В некоторых окружениях (sandbox/CI) это запрещено, поэтому задаём кеш внутри проекта.
    mpl_cache = os.path.join(os.path.dirname(__file__), ".mplconfig")
    os.environ.setdefault("MPLCONFIGDIR", mpl_cache)

    import matplotlib.pyplot as plt  # noqa: E402

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(f"Confusion matrix ({split})")
    ax.set_xlabel("pred")
    ax.set_ylabel("true")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Если классов слишком много, подписи станут нечитаемыми — в этом случае убираем тики.
    n = len(base_ds.classes)
    if n <= 40:
        ax.set_xticks(list(range(n)))
        ax.set_yticks(list(range(n)))
        ax.set_xticklabels(base_ds.classes, rotation=90, fontsize=7)
        ax.set_yticklabels(base_ds.classes, fontsize=7)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()

    os.makedirs(str(out_dir), exist_ok=True)
    out_path = os.path.join(str(out_dir), f"confusion_{split}.png")
    fig.savefig(out_path, dpi=160)
    print("[OK] wrote", out_path)

    # Пишем самые частые ошибки в txt (удобнее, чем смотреть на картинку)
    txt_path = os.path.join(str(out_dir), f"confusions_{split}.txt")
    write_top_confusions_txt(txt_path, cm, base_ds.classes, top_k=int(TOP_K_CONFUSIONS))
    print("[OK] wrote", txt_path)

    # Пишем полный CSV-отчёт: по каждому изображению что предсказали и с какой уверенностью
    csv_name = str(PREDICTIONS_CSV_NAME).strip() or f"predictions_{split}.csv"
    csv_path = os.path.join(str(out_dir), csv_name)
    write_predictions_csv(csv_path, rows)
    print("[OK] wrote", csv_path)


if __name__ == "__main__":
    main()

