"""
Что делает:
1) Берёт чекпоинт PyTorch (best.pt или last.pt)
2) Экспортирует модель в ONNX (NCHW: [1, 3, H, W])
3) Конвертирует ONNX -> TensorFlow SavedModel + TFLite (через onnx2tf)
4) Кладёт итоговый .tflite и labels.txt в OUT_DIR

Запуск:
  python training/speed_classifier/export_to_tflite.py

Зависимости для конвертации (ставятся отдельно):
  pip install -r training/speed_classifier/requirements_convert.txt

Важно:
- classes/labels берём из чекпоинта (ckpt["classes"]), это порядок классов в ImageFolder.
- В мобильном приложении labels.txt должен совпадать 1:1 с этим порядком,
  иначе индексы будут указывать не на те классы.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

import torch

# Импорты с fallback, чтобы работало и так:
# - python training/speed_classifier/export_to_tflite.py
# - python -m training.speed_classifier.export_to_tflite (если training станет пакетом)
try:
    from training.speed_classifier.model import create_model
    from training.speed_classifier.dataset import read_labels_txt
except Exception:  # pragma: no cover
    from model import create_model
    from dataset import read_labels_txt


# ============================================================
# ПАРАМЕТРЫ (ПРАВЬТЕ ТОЛЬКО ЭТОТ БЛОК)
# ============================================================

# Откуда брать чекпоинт (обычно best.pt из папки run_<n>)
CKPT_PATH = "training/speed_classifier/runs/run1/best.pt"

# Папка датасета, где лежит labels.txt (источник истины порядка классов).
# Если оставить пустым "", проверка рассинхрона будет пропущена.
DATA_DIR = "datasets/speed_cls_v1"

# Папка, куда сложить артефакты экспорта.
# Если оставить пустым "", то будет "<папка_чекпоинта>/export".
OUT_DIR = ""

# Размер входа модели (тот же, что был при обучении)
IMAGE_SIZE = 128

# ONNX opset
# На новых версиях PyTorch/onnxscript экспортёр часто работает в opset>=18.
# Но onnx2tf зачастую стабильнее на opset 17 для "классических" CNN.
ONNX_OPSET = 17

# Экспорт ONNX через новый exporter (dynamo=True) иногда даёт граф,
# который onnx2tf хуже переваривает. Для совместимости используем legacy exporter.
ONNX_DYNAMO = False

# Квантование TFLite:
# - 'fp16'        : обычно лучший компромисс размер/скорость
# - 'fp32'        : без оптимизаций
# - 'int8_dynamic': динамическое int8 (без representative dataset)
TFLITE_QUANT = "fp16"  # 'fp16' | 'fp32' | 'int8_dynamic'


# ============================================================
# ВНУТРЕННЯЯ ЛОГИКА
# ============================================================


def ensure_dir(p: str):
    """
    Создаём папку, если её нет.
    """
    os.makedirs(p, exist_ok=True)


def export_onnx(ckpt_path: str, out_onnx: str, image_size: int, opset: int) -> list[str]:
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    classes = ckpt.get("classes")
    if not isinstance(classes, list) or len(classes) < 2:
        raise RuntimeError("В чекпоинте нет списка classes. Ожидается ключ 'classes'.")

    num_classes = len(classes)
    model = create_model(num_classes, pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Экспортируем в "классическом" формате NCHW.
    # onnx2tf лучше всего работает именно с NCHW-ONNX и сам расставляет нужные transpose под TF/TFLite (NHWC).
    dummy = torch.zeros((1, 3, int(image_size), int(image_size)), dtype=torch.float32)

    ensure_dir(os.path.dirname(os.path.abspath(out_onnx)))

    torch.onnx.export(
        model,
        dummy,
        str(out_onnx),
        opset_version=int(opset),
        input_names=["input"],
        output_names=["logits"],
        dynamo=bool(ONNX_DYNAMO),
        # Для совместимости с тулзами, которые плохо работают с graph-level оптимизациями экспорта.
        optimize=False,
        # Важно для onnx2tf: лучше один .onnx без external data,
        # иначе часть тулов плохо подхватывает initializers.
        external_data=False,
        dynamic_axes=None,
    )

    return [str(x) for x in classes]


def run_onnx2tf(onnx_path: str, saved_model_dir: str):
    ensure_dir(saved_model_dir)

    # ВАЖНО: используем тот же Python, которым запущен этот скрипт (venv),
    # иначе можно случайно подхватить глобальный onnx2tf (pyenv shim) с другими зависимостями.
    cmd = [sys.executable, "-m", "onnx2tf", "-i", onnx_path, "-o", saved_model_dir]

    # onnx2tf иногда вызывает внешние бинарники (onnxsim и т.п.).
    # Убедимся, что PATH содержит bin текущего интерпретатора (venv).
    env = os.environ.copy()
    py_bin = os.path.dirname(os.path.abspath(sys.executable))
    env["PATH"] = py_bin + os.pathsep + env.get("PATH", "")

    print("[run]", " ".join(cmd))
    res = subprocess.run(cmd, check=False, env=env)
    if res.returncode != 0:
        raise RuntimeError(f"onnx2tf упал с кодом {res.returncode}")


def convert_saved_model_to_tflite(saved_model_dir: str, out_tflite: str, quant: str):
    # Heavy import — делаем в момент конвертации
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

    q = str(quant).strip().lower()
    if q == "fp16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif q == "int8_dynamic":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif q == "fp32":
        pass
    else:
        raise ValueError(f"Неизвестный TFLITE_QUANT: {quant}")

    tflite_model = converter.convert()
    ensure_dir(os.path.dirname(os.path.abspath(out_tflite)))
    with open(out_tflite, "wb") as f:
        f.write(tflite_model)


def pick_onnx2tf_tflite_path(saved_model_dir: str, quant: str) -> str:
    """
    onnx2tf обычно сам генерирует TFLite рядом с SavedModel:
      - model_float32.tflite
      - model_float16.tflite

    Берём нужный файл и копируем в OUT_DIR как model_<quant>.tflite.
    """
    q = str(quant).strip().lower()
    if q == "fp16":
        cand = os.path.join(saved_model_dir, "model_float16.tflite")
    elif q == "fp32":
        cand = os.path.join(saved_model_dir, "model_float32.tflite")
    else:
        # int8_dynamic пока оставляем через TF converter (если понадобится)
        cand = ""
    return cand


def main():
    ckpt_path = os.path.abspath(CKPT_PATH)
    # Если OUT_DIR не задан — экспортируем в "<папка_чекпоинта>/export".
    out_dir = str(OUT_DIR).strip()
    if not out_dir:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(ckpt_path)), "export")
    out_dir = os.path.abspath(out_dir)
    ensure_dir(out_dir)

    onnx_path = os.path.join(out_dir, "model.onnx")
    saved_model_dir = os.path.join(out_dir, "saved_model")
    tflite_path = os.path.join(out_dir, f"model_{TFLITE_QUANT}.tflite")
    labels_path = os.path.join(out_dir, "labels.txt")

    print("[1/4] Export ONNX:", onnx_path)
    classes = export_onnx(ckpt_path, onnx_path, image_size=int(IMAGE_SIZE), opset=int(ONNX_OPSET))

    # Проверка на рассинхрон:
    # DATA_DIR/labels.txt считаем источником истины порядка классов.
    # Если чекпоинт обучали с другим порядком — экспортировать нельзя, иначе в приложении
    # индексы будут указывать на неправильные строки.
    data_dir = str(DATA_DIR).strip()
    if data_dir:
        expected = read_labels_txt(os.path.abspath(data_dir))
        if expected and list(expected) != list(classes):
            raise RuntimeError(
                "Рассинхрон классов между DATA_DIR/labels.txt и ckpt['classes'].\n"
                f"DATA_DIR={os.path.abspath(data_dir)}\n"
                f"CKPT_PATH={ckpt_path}\n"
                f"labels_len={len(expected)} ckpt_classes_len={len(classes)}\n"
                "Решение: экспортировать модель только с тем labels.txt, с которым она обучалась, "
                "или переобучить модель на актуальном датасете."
            )

    with open(labels_path, "w", encoding="utf-8") as f:
        for c in classes:
            f.write(c + "\n")
    print("[OK] labels:", labels_path)

    print("[2/4] ONNX -> SavedModel (onnx2tf):", saved_model_dir)
    run_onnx2tf(onnx_path, saved_model_dir)

    # onnx2tf уже мог сгенерировать .tflite; это самый надёжный путь (без проблем с signatures).
    onnx2tf_tfl = pick_onnx2tf_tflite_path(saved_model_dir, quant=TFLITE_QUANT)
    if onnx2tf_tfl and os.path.isfile(onnx2tf_tfl):
        print("[3/4] Copy onnx2tf TFLite:", onnx2tf_tfl, "->", tflite_path)
        ensure_dir(os.path.dirname(os.path.abspath(tflite_path)))
        shutil.copyfile(onnx2tf_tfl, tflite_path)
    else:
        print("[3/4] SavedModel -> TFLite (TF converter):", tflite_path)
        convert_saved_model_to_tflite(saved_model_dir, tflite_path, quant=TFLITE_QUANT)

    print("[4/4] Done")
    print("[OK] tflite:", tflite_path)
    print("[OK] labels:", labels_path)


if __name__ == "__main__":
    main()

