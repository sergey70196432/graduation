"""
Обучение (дообучение) детектора дорожных знаков на базе Ultralytics YOLOv8n.

Запуск (из корня репозитория):
  python training/signs_detection/train.py

Что делает скрипт:
- запускает обучение предобученной модели `yolov8n.pt` (COCO) на датасете YOLO;
- сохраняет артефакты Ultralytics в `models/<experiment>/...`;
- дополнительно копирует лучший чекпоинт в `models/signs_detector_best.pt`;
- пишет краткую сводку в `models/signs_detector_info.txt`.

Скрипт intentionally простой и "линейный": без CLI, параметры — вверху файла.
"""

from __future__ import annotations

import csv
import os
import platform
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import torch
    from ultralytics import YOLO
except ModuleNotFoundError as e:
    print(
        "[ОШИБКА] Не найдена зависимость: "
        f"{e}.\n"
        "Установите зависимости командой:\n"
        "  python -m pip install -r training/signs_detection/requirements.txt\n",
        file=sys.stderr,
    )
    raise SystemExit(1)

REPO_ROOT = Path(__file__).resolve().parents[2]
os.chdir(REPO_ROOT)
# Важно: чтобы работали импорты вида `from training...` при запуске:
#   python training/signs_detection/train.py
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ============================================================
# ПАРАМЕТРЫ (ПРАВЬТЕ ТОЛЬКО ЭТОТ БЛОК)
# ============================================================

# Путь к датасету Ultralytics (dataset.yaml).
# Обычно генерируется скриптом make_dataset/generate_synth_yolo_dataset.py
DATASET_YAML = Path("datasets/dataset_9/dataset.yaml")

# Предобученные веса (COCO). Скачиваются Ultralytics автоматически при первом запуске.
MODEL_PRETRAINED = "yolov8n.pt"

# Параметры обучения (подбирайте под ваш датасет/железо):
EPOCHS = 50
BATCH = 16
IMGSZ = 640
OPTIMIZER = "AdamW"
LR0 = 0.001
SEED = 1337

# Аугментации (Ultralytics):
AUGMENT = True
COS_LR = True
MIXUP = 0.2
DEGREES = 5.0
SCALE = 0.5

# Куда сохранять артефакты Ultralytics:
PROJECT_DIR = Path("models/signs_detection")
EXPERIMENT_NAME = "experiment"

# Доп. стабильные артефакты (чтобы всегда было понятно, где "лучшая модель"):
STABLE_BEST_PATH = PROJECT_DIR / "signs_detector_best.pt"
MODEL_INFO_PATH = PROJECT_DIR / "signs_detector_info.txt"

# Устройство (cuda / mps / cpu). По умолчанию выбираем лучшее доступное.
# Импорт с fallback, чтобы работало и так:
# - python training/signs_detection/train.py
# - python -m training.signs_detection.train
try:
    from training.utils import pick_device
except Exception:  # pragma: no cover
    from utils import pick_device  # type: ignore

DEVICE = pick_device()


def _die(message: str, exit_code: int = 1) -> None:
    print(f"[ОШИБКА] {message}", file=sys.stderr)
    raise SystemExit(exit_code)


def _ensure_dir(path: Path) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        _die(f"Не удалось создать папку `{path}`: {e}")


def _safe_get(d: Any, key: str) -> Any:
    try:
        return d.get(key)
    except Exception:
        return None


def try_extract_metrics_from_train_result(train_result: Any) -> dict[str, Any]:
    metrics: dict[str, Any] = {}

    if hasattr(train_result, "results_dict"):
        rd = getattr(train_result, "results_dict", None)
        if isinstance(rd, dict):
            metrics.update(rd)

    if isinstance(train_result, dict):
        metrics.update(train_result)

    return metrics


def try_extract_map50_from_results_csv(run_dir: Path) -> float | None:
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return None

    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except OSError:
        return None

    if not rows:
        return None

    last = rows[-1]
    v = last.get("metrics/mAP50(B)")
    if v is None:
        return None

    try:
        return float(v)
    except ValueError:
        return None


def get_ultralytics_save_dir(train_result: Any, fallback_run_dir: Path) -> Path:
    if hasattr(train_result, "save_dir"):
        sd = getattr(train_result, "save_dir")
        if sd:
            try:
                p = Path(sd)
                if p.exists():
                    return p
            except Exception:
                pass
    return fallback_run_dir


def find_best_checkpoint(save_dir: Path) -> Path | None:
    direct = save_dir / "weights" / "best.pt"
    if direct.exists():
        return direct

    candidates: list[Path] = []
    if save_dir.exists():
        candidates.extend(save_dir.rglob("weights/best.pt"))

    if not candidates:
        return None

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _unique_dir(path: Path) -> Path:
    """
    Возвращает уникальный путь директории, добавляя суффикс _1, _2, ...
    если директория уже существует.
    """
    if not path.exists():
        return path
    for i in range(1, 10_000):
        p2 = path.with_name(f"{path.name}_{i}")
        if not p2.exists():
            return p2
    raise RuntimeError(f"Не удалось подобрать уникальный путь для: {path}")


def relocate_ultralytics_run_dir(save_dir: Path, runs_root: Path) -> Path:
    """
    Ultralytics обычно создаёт `.../<task>/<name>` (например `detect/...`).
    Мы хотим плоскую структуру: `models/signs_detection/<name>/...`.
    """
    desired = runs_root / save_dir.name
    if desired.resolve() == save_dir.resolve():
        return save_dir

    _ensure_dir(runs_root)
    desired = _unique_dir(desired)

    try:
        shutil.move(str(save_dir), str(desired))
    except Exception as e:
        _die(f"Не удалось перенести папку эксперимента `{save_dir}` -> `{desired}`: {e}")

    # Пытаемся подчистить пустые task/project директории
    try:
        p = save_dir.parent
        while p != runs_root and p.exists():
            if any(p.iterdir()):
                break
            p.rmdir()
            p = p.parent
    except Exception:
        pass

    return desired


def main() -> None:
    print("[INFO] === YOLOv8: обучение детектора дорожных знаков ===")
    print(f"[INFO] Repo root: {REPO_ROOT}")

    if not DATASET_YAML.exists():
        _die(
            f"Не найден файл `{DATASET_YAML}`.\n"
            "Сначала сгенерируйте датасет:\n"
            "  python make_dataset/generate_synth_yolo_dataset.py\n"
            "или исправьте DATASET_YAML вверху training/signs_detection/train.py."
        )

    _ensure_dir(PROJECT_DIR)

    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] Dataset: {DATASET_YAML}")
    print(f"[INFO] Pretrained: {MODEL_PRETRAINED}")

    model = YOLO(MODEL_PRETRAINED)

    # Гарантируем, что Ultralytics будет писать артефакты в `models/`, а не в `runs/`.
    try:
        from ultralytics import settings  # type: ignore

        if hasattr(settings, "update"):
            settings.update({"runs_dir": str(PROJECT_DIR)})
    except Exception:
        pass

    print("[INFO] Старт обучения...")
    train_result = model.train(
        data=str(DATASET_YAML),
        epochs=EPOCHS,
        batch=BATCH,
        imgsz=IMGSZ,
        optimizer=OPTIMIZER,
        lr0=LR0,
        seed=SEED,
        augment=AUGMENT,
        # В Ultralytics save_dir собирается примерно как:
        #   runs_dir / task / project / name
        # При runs_dir=PROJECT_DIR используем нейтральный project,
        # а затем "расплющиваем" save_dir до `models/signs_detection/<name>`.
        project=".",
        name=EXPERIMENT_NAME,
        save=True,
        plots=True,
        cos_lr=COS_LR,
        mixup=MIXUP,
        degrees=DEGREES,
        scale=SCALE,
        device=DEVICE,
    )
    print("[INFO] Обучение завершено.")

    fallback_run_dir = PROJECT_DIR / EXPERIMENT_NAME
    save_dir = get_ultralytics_save_dir(train_result, fallback_run_dir=fallback_run_dir)
    print(f"[INFO] Папка эксперимента (save_dir): {save_dir}")
    save_dir = relocate_ultralytics_run_dir(save_dir, runs_root=PROJECT_DIR)
    print(f"[INFO] Папка эксперимента (relocated): {save_dir}")

    best_path = find_best_checkpoint(save_dir)
    if best_path is None or not best_path.exists():
        _die(
            "Не найден файл лучшей модели `best.pt`.\n"
            "Проверьте, куда Ultralytics сохранил эксперимент (save_dir) и содержимое папки weights/."
        )

    try:
        STABLE_BEST_PATH.write_bytes(best_path.read_bytes())
    except OSError as e:
        _die(f"Не удалось скопировать `{best_path}` -> `{STABLE_BEST_PATH}`: {e}")

    print(f"[INFO] Лучшая модель сохранена: {STABLE_BEST_PATH}")

    metrics = try_extract_metrics_from_train_result(train_result)
    map50 = (
        _safe_get(metrics, "metrics/mAP50(B)")
        or _safe_get(metrics, "metrics/mAP50")
        or _safe_get(metrics, "map50")
    )

    map50_float: float | None = None
    if map50 is not None:
        try:
            map50_float = float(map50)
        except (TypeError, ValueError):
            map50_float = None

    if map50_float is None:
        map50_float = try_extract_map50_from_results_csv(save_dir)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    info_lines = [
        "YOLOv8 training summary (road signs detector)",
        f"datetime: {now}",
        f"dataset_yaml: {DATASET_YAML.as_posix()}",
        f"pretrained: {MODEL_PRETRAINED}",
        f"epochs: {EPOCHS}",
        f"batch: {BATCH}",
        f"imgsz: {IMGSZ}",
        f"optimizer: {OPTIMIZER}",
        f"lr0: {LR0}",
        f"seed: {SEED}",
        f"augment: {AUGMENT}",
        f"project_dir: {PROJECT_DIR.as_posix()}",
        f"experiment_name: {EXPERIMENT_NAME}",
        f"device: {DEVICE}",
        f"save_dir: {save_dir.as_posix()}",
        f"best_checkpoint: {best_path.as_posix()}",
        f"stable_best: {STABLE_BEST_PATH.as_posix()}",
        f"python: {sys.version.split()[0]}",
        f"platform: {platform.platform()}",
        f"torch: {torch.__version__}",
    ]

    if map50_float is not None:
        info_lines.append(f"metrics/mAP50(B): {map50_float:.6f}")
    else:
        info_lines.append("metrics/mAP50(B): (не удалось извлечь автоматически)")

    try:
        import ultralytics  # type: ignore

        info_lines.append(f"ultralytics: {getattr(ultralytics, '__version__', '(unknown)')}")
    except Exception:
        pass

    try:
        MODEL_INFO_PATH.write_text("\n".join(info_lines) + "\n", encoding="utf-8")
    except OSError as e:
        _die(f"Не удалось записать `{MODEL_INFO_PATH}`: {e}")

    print(f"[INFO] Информация о модели сохранена: {MODEL_INFO_PATH}")
    print(f"[INFO] Артефакты Ultralytics лежат в: {save_dir}")


if __name__ == "__main__":
    main()
