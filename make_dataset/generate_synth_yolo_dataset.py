import os
import re
import io
import csv
import math
import random
import shutil
import subprocess
import tempfile
from glob import glob

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance

# ============================================================
# Скрипт генерации синтетического датасета дорожных знаков (YOLO)
# ============================================================
# Требования:
# - одинаковое число изображений для всех классов
# - для классов с вариациями: каждая вариация минимум 2 раза
# - TARGET_COUNT = max(MIN_IMAGES_PER_CLASS, 2 * max(V)) по всем вариативным классам
# - выход: dataset_<N>/images, dataset_<N>/labels, classes.txt, annotations.csv, dataset.yaml, train.txt, val.txt
#
# Важно про SVG:
# - лучше поставить cairosvg: pip install cairosvg
# - если cairosvg нет, попробуем rsvg-convert или inkscape (если они установлены в системе)


# ===================== Параметры (правьте здесь) =====================
BACKGROUNDS_DIR = "make_dataset/backgrounds"
# Кадры с видеорегистратора (лучшие фоны для домена). Можно указать папку с jpg/png.
# Файлы ищутся РЕКУРСИВНО. Если папка не задана или пустая — используем BACKGROUNDS_DIR.
DASHCAM_FRAMES_DIR = "make_dataset/dashcam_frames"
SIGNS_IMAGES_DIR = "shared/signs/images"  # папка с базовыми изображениями знаков (если у класса нет splits-вариаций)
SPLITS_DIR = "shared/signs/splits"        # папка с подпапками вариаций: splits/<name>/*.(png/jpg/svg)
CSV_PATH = "shared/signs/signs.csv"       # CSV с колонками filename,class_id,class_name

OUTPUT_BASE = "datasets/dataset"  # папки будут dataset_1, dataset_2, ...
MIN_IMAGES_PER_CLASS = 20         # минимум на класс
VAL_RATIO = 0.2                   # доля val на класс (для train.txt/val.txt)
WEATHER_PROB = 0.7                # вероятность погодного эффекта на кадре (дождь/снег/туман)
RANDOM_SEED = 1337                # фиксируем seed для воспроизводимости

# Если список пустой — генерируем все классы из CSV.
# Если указать, например [0, 4, 10], то будут использоваться только эти class_id.
SELECT_CLASS_IDS = [67,68]  # пример: [0, 4, 10]

# Диапазоны аугментаций
# Для видеорегистратора знаки чаще маленькие, поэтому диапазон обычно ниже.
SCALE_RANGE = (0.05, 0.20)        # ширина знака как доля ширины фона
SCALE_BIAS_POWER = 2.2            # >1 => чаще маленькие (u**power)
ROLL_ANGLE_RANGE = (-30.0, 30.0)  # поворот в плоскости
PERSPECTIVE_STRENGTH = 0.10       # перспектива (наклон)
SHIFT_FRACTION = 0.08             # небольшой сдвиг после преобразований

# Размещение
RIGHT_HALF_PROB = 0.7              # вероятность, что знак будет в правой половине кадра (x_center > 0.5)
MIN_VISIBLE_AREA_FRACTION = 0.60   # минимум видимой площади знака (0..1), иначе пробуем другую позицию
MAX_PLACEMENT_TRIES = 10           # число попыток разместить знак на фоне, прежде чем сдаться

# Негативные примеры (кадры без знаков). Для YOLO это важно, чтобы не было ложных срабатываний.
NEGATIVE_RATIO = 0.1  # доля негативных изображений (без знаков) относительно числа сгенерированных кадров (примерно)

# Несколько знаков на одном изображении (как на реальных кадрах видеорегистратора)
MULTI_OBJECT_ENABLED = True
EXTRA_OBJECTS_RANGE = (0, 3)       # сколько ДОПОЛНИТЕЛЬНЫХ знаков добавлять к основному (0..5)
EXTRA_SAME_CLASS_PROB = 0.15       # шанс, что доп. знак будет того же класса, что и основной
MAX_IOU_BETWEEN_SIGNS = 0.25       # ограничение пересечений bbox между знаками
MAX_EXTRA_TRIES = 25               # попыток размещения одного доп. знака

# Эффекты "камеры" видеорегистратора (умеренно)
CAMERA_EFFECTS_PROB = 0.90           # вероятность применить набор "камерных" эффектов к кадру
JPEG_QUALITY_RANGE = (35, 95)        # качество JPEG при пережатии кадра (меньше => сильнее артефакты)
NOISE_STD_RANGE = (0.0, 8.0)         # sigma шума (в единицах яркости 0..255)
GAUSS_BLUR_PROB = 0.25               # вероятность лёгкого Gaussian blur (как defocus/стабилизация)
MOTION_BLUR_PROB = 0.25              # вероятность motion blur (смаз от движения)
VIGNETTE_PROB = 0.20                 # вероятность виньетки (потемнение по краям)
COLOR_JITTER_PROB = 0.35             # вероятность сдвигов яркости/контраста/насыщенности (автоэкспозиция/ББ)

# Небольшая тень и "смягчение" краёв знака (чтобы не было идеально вырезанного контура)
SHADOW_PROB = 0.70                      # вероятность добавить тень от знака на фон
SHADOW_STRENGTH_RANGE = (0.20, 0.55)  # насколько затемнять фон
SHADOW_BLUR_RANGE = (3, 11)           # нечётное число
SHADOW_OFFSET_X_RANGE = (-6, 6)         # сдвиг тени по X (пиксели, в ROI знака)
SHADOW_OFFSET_Y_RANGE = (2, 10)         # сдвиг тени по Y (пиксели, в ROI знака)
EDGE_BLUR_PROB = 0.30                   # вероятность дополнительно размыть RGB знака (альфу не трогаем)
EDGE_BLUR_K_RANGE = (1, 3)              # диапазон sigma для Gaussian blur знака (в пикселях)

# Ухудшение качества знака в зависимости от размера (для маленьких знаков сильнее)
SIGN_DEGRADE_ENABLED = True
SIGN_DEGRADE_PROB = 0.95               # вероятность применить деградацию качества к каждому знаку
# "Размер знака" — ширина bbox по альфе (в пикселях). Всё ниже MIN будет как "очень мелко",
# всё выше MAX — как "крупно".
SIGN_DEGRADE_MIN_PX = 35
SIGN_DEGRADE_MAX_PX = 160
# JPEG качество для ROI знака (мелкий знак => ниже качество)
SIGN_JPEG_QUALITY_SMALL = (12, 40)
SIGN_JPEG_QUALITY_LARGE = (65, 95)
# Потеря деталей: downscale->upscale (мелкий => сильнее)
SIGN_DOWNSCALE_SMALL = (0.28, 0.55)
SIGN_DOWNSCALE_LARGE = (0.85, 1.00)
# Размытие: мелкий => сильнее
SIGN_BLUR_SIGMA_SMALL = (0.6, 1.8)
SIGN_BLUR_SIGMA_LARGE = (0.0, 0.6)
# Шум: мелкий => сильнее
SIGN_NOISE_STD_SMALL = (1.0, 10.0)
SIGN_NOISE_STD_LARGE = (0.0, 3.0)

# Цветокоррекция по яркости фона под знаком
BRIGHTNESS_DARK_THRESH = 90.0
BRIGHTNESS_BRIGHT_THRESH = 190.0
SIGN_DARKEN_FACTOR = 0.85
SIGN_DESAT_FACTOR = 0.80
SIGN_BRIGHTEN_FACTOR = 1.10
SIGN_CONTRAST_FACTOR = 1.10

# Для bbox
ALPHA_THRESHOLD = 8

# Чтобы не зациклиться (если много пропусков SVG/плохих размещений)
MAX_ATTEMPTS_MULT = 12  # попыток на класс = TARGET_COUNT * MAX_ATTEMPTS_MULT

TEMPLATE_EXTS = (".png", ".jpg", ".jpeg", ".svg")
BACKGROUND_EXTS = (".png", ".jpg", ".jpeg")


try:
    import cairosvg  # type: ignore

    HAS_CAIROSVG = True
except Exception:
    HAS_CAIROSVG = False


def info(msg):
    print("[INFO]", msg)


def warn(msg):
    print("[WARN]", msg)


def base_name_no_ext(s):
    """Берём имя без расширения (если вдруг в CSV оно указано с расширением)."""
    s = os.path.basename(str(s)).strip()
    return os.path.splitext(s)[0]


def list_files_by_ext(folder, exts):
    """Список файлов в папке по наборам расширений (НЕ рекурсивно)."""
    if not os.path.isdir(folder):
        return []
    out = []
    for ext in exts:
        out += glob(os.path.join(folder, "*" + ext))
        out += glob(os.path.join(folder, "*" + ext.upper()))
    out = sorted(set([p for p in out if os.path.isfile(p)]))
    return out


def list_files_by_ext_recursive(folder, exts):
    """Список файлов в папке по расширениям (рекурсивно)."""
    if not folder or not os.path.isdir(folder):
        return []
    out = []
    for ext in exts:
        out += glob(os.path.join(folder, "**", "*" + ext), recursive=True)
        out += glob(os.path.join(folder, "**", "*" + ext.upper()), recursive=True)
    out = sorted(set([p for p in out if os.path.isfile(p)]))
    return out


def split_dir_candidates(name):
    """
    Основное правило: splits/<name>/.
    В репозитории часто встречается name вида '3.24_template' при папке splits/3.24,
    поэтому делаем простой fallback на удаление суффиксов.
    """
    cands = [name]
    for suf in ("_template", "_temp"):
        if name.endswith(suf):
            cands.append(name[: -len(suf)])
        if suf in name:
            cands.append(name.split(suf, 1)[0])
    # уберём повторы, сохраним порядок
    res = []
    seen = set()
    for c in cands:
        c = c.strip()
        if not c or c in seen:
            continue
        seen.add(c)
        res.append(c)
    return res


def find_template_in_images(images_dir, name):
    """Если нет вариаций, ищем файл, который начинается с name и имеет поддерживаемое расширение."""
    if not os.path.isdir(images_dir):
        return None
    for ext in TEMPLATE_EXTS:
        cands = sorted(glob(os.path.join(images_dir, f"{name}*{ext}")))
        cands += sorted(glob(os.path.join(images_dir, f"{name}*{ext.upper()}")))
        if cands:
            return cands[0]
    return None


def load_classes():
    """
    Читаем CSV и для каждого класса определяем список шаблонов:
    - если есть splits/<name>/ с файлами -> класс с вариациями
    - иначе ищем один файл в images/
    """
    df = pd.read_csv(CSV_PATH)
    for col in ("filename", "class_id", "class_name"):
        if col not in df.columns:
            raise ValueError(f"В CSV нет колонки '{col}': {CSV_PATH}")

    classes = []
    for _, row in df.iterrows():
        name = base_name_no_ext(row["filename"])
        try:
            class_id = int(row["class_id"])
        except Exception:
            warn(f"Плохой class_id: {row}")
            continue
        class_name = str(row["class_name"])

        # 1) пробуем найти вариации
        templates = []
        for cand in split_dir_candidates(name):
            d = os.path.join(SPLITS_DIR, cand)
            t = list_files_by_ext(d, TEMPLATE_EXTS)
            if t:
                templates = sorted(t, key=lambda p: os.path.basename(p).lower())
                break

        if templates:
            classes.append(
                {
                    "name": name,
                    "class_id": class_id,
                    "class_name": class_name,
                    "templates": templates,
                    "has_variations": True,
                }
            )
            continue

        # 2) иначе один шаблон в images/
        single = find_template_in_images(SIGNS_IMAGES_DIR, name)
        if not single:
            warn(f"Не найден шаблон для '{name}' (id={class_id}) в {SIGNS_IMAGES_DIR}. Пропускаю класс.")
            continue

        classes.append(
            {
                "name": name,
                "class_id": class_id,
                "class_name": class_name,
                "templates": [single],
                "has_variations": False,
            }
        )

    if not classes:
        raise RuntimeError("Не удалось загрузить ни одного класса (все пропущены). Проверьте пути.")

    classes = sorted(classes, key=lambda c: c["class_id"])

    # Фильтр по class_id (если задан)
    if SELECT_CLASS_IDS:
        want = set(int(x) for x in SELECT_CLASS_IDS)
        classes = [c for c in classes if c["class_id"] in want]
        if not classes:
            raise RuntimeError("После фильтра SELECT_CLASS_IDS не осталось ни одного класса.")
    return classes


def compute_target_count(classes):
    """TARGET_COUNT = max(MIN_IMAGES_PER_CLASS, max(2*V) среди вариативных классов)."""
    max_variation_target = 0
    for c in classes:
        if c["has_variations"]:
            v = len(c["templates"])
            max_variation_target = max(max_variation_target, v * 2)
    return int(max(MIN_IMAGES_PER_CLASS, max_variation_target))


def build_template_sequence(cls, target_count):
    """
    Для вариативного класса:
    - каждый шаблон 2 раза (по сортировке)
    - затем добиваем циклически до TARGET_COUNT
    Для не вариативного:
    - один шаблон TARGET_COUNT раз
    """
    if not cls["has_variations"]:
        return [cls["templates"][0]] * target_count

    t = sorted(cls["templates"], key=lambda p: os.path.basename(p).lower())
    out = []
    for p in t:
        out += [p, p]
    i = 0
    while len(out) < target_count:
        out.append(t[i % len(t)])
        i += 1
    return out[:target_count]


def ensure_output_dir(base):
    """Создаём dataset_<N>/images и dataset_<N>/labels."""
    n = 1
    while True:
        out_dir = f"{base}_{n}"
        if not os.path.exists(out_dir):
            os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
            os.makedirs(os.path.join(out_dir, "labels"), exist_ok=True)
            return out_dir
        n += 1


def yaml_quote(s):
    """Всегда пишем имена в одинарных кавычках, экранируя одиночную кавычку."""
    s = str(s)
    return "'" + s.replace("'", "''") + "'"


def write_dataset_yaml_and_splits(out_dir, classes, train_list, val_list):
    """
    Пишем:
    - dataset.yaml (как в примере)
    - train.txt / val.txt (списки путей к изображениям, относительно out_dir)
    """
    # train/val txt
    train_txt = os.path.join(out_dir, "train.txt")
    val_txt = os.path.join(out_dir, "val.txt")
    with open(train_txt, "w", encoding="utf-8") as f:
        for p in train_list:
            f.write(p + "\n")
    with open(val_txt, "w", encoding="utf-8") as f:
        for p in val_list:
            f.write(p + "\n")

    # names должны индексироваться по class_id, поэтому делаем список длиной max_id+1
    max_id = max(c["class_id"] for c in classes)
    names = [""] * (max_id + 1)
    for c in classes:
        names[c["class_id"]] = c["class_name"]
    for i in range(len(names)):
        if not names[i]:
            names[i] = f"class_{i}"

    yaml_path = os.path.join(out_dir, "dataset.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("path: .\n")
        f.write("train: train.txt\n")
        f.write("val: val.txt\n")
        f.write(f"nc: {len(names)}\n")
        f.write("names:\n")
        for n in names:
            f.write(f"- {yaml_quote(n)}\n")


def write_classes_txt(out_dir, classes):
    """classes.txt: строка = class_id, значение = class_name."""
    max_id = max(c["class_id"] for c in classes)
    lines = [""] * (max_id + 1)
    for c in classes:
        lines[c["class_id"]] = c["class_name"]
    for i in range(len(lines)):
        if not lines[i]:
            lines[i] = f"class_{i}"
    with open(os.path.join(out_dir, "classes.txt"), "w", encoding="utf-8") as f:
        for s in lines:
            f.write(s + "\n")


def load_background_paths():
    dashcam = list_files_by_ext_recursive(DASHCAM_FRAMES_DIR, BACKGROUND_EXTS)
    generic = list_files_by_ext(BACKGROUNDS_DIR, BACKGROUND_EXTS)

    paths = []
    if dashcam:
        paths += dashcam
    if generic:
        paths += generic

    if len(paths) < 1:
        raise FileNotFoundError(
            f"Не найдено фонов. Проверьте BACKGROUNDS_DIR='{BACKGROUNDS_DIR}' и DASHCAM_FRAMES_DIR='{DASHCAM_FRAMES_DIR}'."
        )

    if dashcam:
        info(f"Фоны: {len(dashcam)} кадров видеорегистратора + {len(generic)} дополнительных фонов.")
    else:
        warn("DASHCAM_FRAMES_DIR не задан или пустой — используем только BACKGROUNDS_DIR.")
        info(f"Фоны: {len(generic)} изображений.")

    return paths


def load_bg_bgr(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Не удалось прочитать фон: {path}")
    return img


def pil_enhance_rgba(sign_rgba, brightness=1.0, contrast=1.0, color=1.0):
    """Простая коррекция яркости/контраста/насыщенности через PIL."""
    pil = Image.fromarray(sign_rgba, mode="RGBA")
    r, g, b, a = pil.split()
    rgb = Image.merge("RGB", (r, g, b))
    if abs(brightness - 1.0) > 1e-3:
        rgb = ImageEnhance.Brightness(rgb).enhance(brightness)
    if abs(contrast - 1.0) > 1e-3:
        rgb = ImageEnhance.Contrast(rgb).enhance(contrast)
    if abs(color - 1.0) > 1e-3:
        rgb = ImageEnhance.Color(rgb).enhance(color)
    out = Image.merge("RGBA", (*rgb.split(), a))
    return np.array(out, dtype=np.uint8)


def apply_jpeg_compression(img_bgr, quality):
    """Имитируем пережатие видеорегистратора (JPEG)."""
    quality = int(np.clip(quality, 5, 100))
    ok, enc = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return img_bgr
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return dec if dec is not None else img_bgr


def add_gaussian_noise(img_bgr, std):
    """Добавляем гауссов шум."""
    if std <= 0:
        return img_bgr
    noise = np.random.normal(0.0, std, img_bgr.shape).astype(np.float32)
    out = img_bgr.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def clamp(x, a, b):
    return max(a, min(b, x))


def lerp(a, b, t):
    return a + (b - a) * t


def sample_interp_range(r_small, r_large, t):
    """
    Интерполируем диапазоны (low, high) между "мелко" и "крупно",
    потом выбираем случайное значение внутри интерполированного диапазона.
    t=0 => мелкий знак, t=1 => крупный знак.
    """
    lo = lerp(float(r_small[0]), float(r_large[0]), float(t))
    hi = lerp(float(r_small[1]), float(r_large[1]), float(t))
    if hi < lo:
        lo, hi = hi, lo
    return random.uniform(lo, hi)


def degrade_sign_by_size(sign_rgba, sign_bbox_w_px):
    """
    Ухудшаем качество знака (только в области непрозрачных пикселей), причём
    чем меньше знак по размеру — тем сильнее деградация.
    """
    if not SIGN_DEGRADE_ENABLED:
        return sign_rgba
    if random.random() > SIGN_DEGRADE_PROB:
        return sign_rgba

    alpha = sign_rgba[:, :, 3]
    bb = alpha_bbox(alpha, thr=ALPHA_THRESHOLD)
    if bb is None:
        return sign_rgba
    x1, y1, x2, y2 = bb
    w = int(x2 - x1 + 1)
    h = int(y2 - y1 + 1)
    if w < 2 or h < 2:
        return sign_rgba

    # Нормализуем размер: t=0 (мелко) .. t=1 (крупно)
    t = 0.0
    if SIGN_DEGRADE_MAX_PX > SIGN_DEGRADE_MIN_PX:
        t = (float(sign_bbox_w_px) - float(SIGN_DEGRADE_MIN_PX)) / float(SIGN_DEGRADE_MAX_PX - SIGN_DEGRADE_MIN_PX)
    t = clamp(t, 0.0, 1.0)

    roi = sign_rgba[y1 : y2 + 1, x1 : x2 + 1].copy()
    roi_rgb = roi[:, :, :3]
    roi_a = roi[:, :, 3:4]

    # Работаем в BGR для OpenCV
    roi_bgr = roi_rgb[:, :, ::-1].copy()

    # 1) Downscale -> Upscale (потеря деталей)
    s = sample_interp_range(SIGN_DOWNSCALE_SMALL, SIGN_DOWNSCALE_LARGE, t)
    s = clamp(s, 0.15, 1.0)
    if s < 0.999:
        nw = max(2, int(w * s))
        nh = max(2, int(h * s))
        tmp = cv2.resize(roi_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        roi_bgr = cv2.resize(tmp, (w, h), interpolation=cv2.INTER_LINEAR)

    # 2) Blur (мелкий знак сильнее)
    sigma = sample_interp_range(SIGN_BLUR_SIGMA_SMALL, SIGN_BLUR_SIGMA_LARGE, t)
    if sigma > 0.05:
        roi_bgr = cv2.GaussianBlur(roi_bgr, (0, 0), sigmaX=float(sigma))

    # 3) Noise (мелкий знак сильнее)
    std = sample_interp_range(SIGN_NOISE_STD_SMALL, SIGN_NOISE_STD_LARGE, t)
    roi_bgr = add_gaussian_noise(roi_bgr, std)

    # 4) JPEG артефакты (локально)
    q = int(round(sample_interp_range(SIGN_JPEG_QUALITY_SMALL, SIGN_JPEG_QUALITY_LARGE, t)))
    roi_bgr = apply_jpeg_compression(roi_bgr, q)

    # Возвращаем обратно в RGBA, альфу не трогаем
    out = sign_rgba.copy()
    out_roi_rgb = roi_bgr[:, :, ::-1]
    out[y1 : y2 + 1, x1 : x2 + 1, :3] = out_roi_rgb
    out[y1 : y2 + 1, x1 : x2 + 1, 3:4] = roi_a
    return out


def apply_motion_blur(img_bgr, k):
    """Простая смаз по движению (линейное ядро)."""
    k = int(k)
    if k < 3:
        return img_bgr
    if k % 2 == 0:
        k += 1
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0
    kernel /= kernel.sum()
    angle = random.uniform(-25.0, 25.0)
    M = cv2.getRotationMatrix2D((k / 2, k / 2), angle, 1.0)
    kernel = cv2.warpAffine(kernel, M, (k, k))
    s = kernel.sum()
    if s > 0:
        kernel /= s
    return cv2.filter2D(img_bgr, -1, kernel)


def apply_vignette(img_bgr, strength=0.35):
    """Лёгкая виньетка."""
    h, w = img_bgr.shape[:2]
    y = np.linspace(-1, 1, h).reshape(-1, 1)
    x = np.linspace(-1, 1, w).reshape(1, -1)
    r2 = x * x + y * y
    mask = 1.0 - strength * np.clip(r2, 0, 1)
    mask = np.clip(mask, 0.2, 1.0).astype(np.float32)
    out = img_bgr.astype(np.float32)
    out *= mask[:, :, None]
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_color_jitter(img_bgr):
    """Небольшие сдвиги яркости/контраста/насыщенности (как автоэкспозиция/ББ)."""
    out = img_bgr.astype(np.float32)
    # контраст
    c = random.uniform(0.90, 1.15)
    out = (out - 127.5) * c + 127.5
    # яркость
    b = random.uniform(-12.0, 12.0)
    out = out + b
    out = np.clip(out, 0, 255).astype(np.uint8)

    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= random.uniform(0.85, 1.20)  # saturation
    hsv[:, :, 2] *= random.uniform(0.90, 1.10)  # value
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def apply_camera_effects(img_bgr):
    """Компактный набор эффектов, похожих на видеорегистратор."""
    if random.random() > CAMERA_EFFECTS_PROB:
        return img_bgr

    out = img_bgr

    # иногда "ресайз туда-сюда" (лёгкая потеря деталей)
    if random.random() < 0.30:
        h, w = out.shape[:2]
        s = random.uniform(0.70, 0.95)
        nw, nh = max(2, int(w * s)), max(2, int(h * s))
        tmp = cv2.resize(out, (nw, nh), interpolation=cv2.INTER_AREA)
        out = cv2.resize(tmp, (w, h), interpolation=cv2.INTER_LINEAR)

    if random.random() < COLOR_JITTER_PROB:
        out = apply_color_jitter(out)

    if random.random() < GAUSS_BLUR_PROB:
        sigma = random.uniform(0.6, 1.6)
        out = cv2.GaussianBlur(out, (0, 0), sigmaX=sigma)

    if random.random() < MOTION_BLUR_PROB:
        k = random.choice([5, 7, 9, 11])
        out = apply_motion_blur(out, k)

    std = random.uniform(NOISE_STD_RANGE[0], NOISE_STD_RANGE[1])
    out = add_gaussian_noise(out, std)

    if random.random() < VIGNETTE_PROB:
        out = apply_vignette(out, strength=random.uniform(0.20, 0.45))

    # JPEG в конце (как финальный кодек)
    q = random.randint(JPEG_QUALITY_RANGE[0], JPEG_QUALITY_RANGE[1])
    out = apply_jpeg_compression(out, q)
    return out


def render_svg_to_rgba(svg_path, render_px=1024):
    """
    SVG -> RGBA.
    1) cairosvg (если есть)
    2) rsvg-convert (если установлен)
    3) inkscape (если установлен)
    """
    if HAS_CAIROSVG:
        with open(svg_path, "rb") as f:
            svg_bytes = f.read()
        png_bytes = cairosvg.svg2png(bytestring=svg_bytes, output_width=render_px, output_height=render_px)
        pil = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
        return np.array(pil, dtype=np.uint8)

    rsvg = shutil.which("rsvg-convert")
    if rsvg:
        proc = subprocess.run(
            [rsvg, "-w", str(render_px), "-h", str(render_px), svg_path],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if proc.returncode == 0 and proc.stdout:
            pil = Image.open(io.BytesIO(proc.stdout)).convert("RGBA")
            return np.array(pil, dtype=np.uint8)
        warn("rsvg-convert не смог отрендерить SVG: " + os.path.basename(svg_path))
        return None

    inkscape = shutil.which("inkscape")
    if inkscape:
        with tempfile.TemporaryDirectory() as td:
            out_png = os.path.join(td, "out.png")
            proc = subprocess.run(
                [
                    inkscape,
                    svg_path,
                    "--export-type=png",
                    f"--export-filename={out_png}",
                    f"--export-width={render_px}",
                    f"--export-height={render_px}",
                ],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if proc.returncode == 0 and os.path.exists(out_png):
                pil = Image.open(out_png).convert("RGBA")
                return np.array(pil, dtype=np.uint8)
        warn("Inkscape не смог отрендерить SVG: " + os.path.basename(svg_path))
        return None

    return None


def load_template_rgba(path, render_px=1024):
    """Грузим PNG/JPG как RGBA или рендерим SVG."""
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".svg":
            rgba = render_svg_to_rgba(path, render_px=render_px)
            if rgba is None:
                warn(
                    f"SVG '{path}', но нет cairosvg и не найден/не сработал rsvg-convert/inkscape. "
                    f"Поставьте: pip install cairosvg (или brew install librsvg)."
                )
            return rgba
        pil = Image.open(path).convert("RGBA")
        return np.array(pil, dtype=np.uint8)
    except Exception as e:
        warn(f"Не удалось загрузить шаблон {path}: {e}")
        return None


def resize_sign(sign_rgba, bg_w):
    """Масштабируем знак так, чтобы его ширина была 5..25% ширины фона."""
    h, w = sign_rgba.shape[:2]
    if w <= 0 or h <= 0:
        return sign_rgba
    # Смещаем распределение в сторону маленьких знаков (актуально для видеорегистратора)
    u = random.random() ** float(SCALE_BIAS_POWER)
    rel = float(SCALE_RANGE[0]) + (float(SCALE_RANGE[1]) - float(SCALE_RANGE[0])) * u
    target_w = int(bg_w * rel)
    target_w = max(8, target_w)
    scale = target_w / float(w)
    target_h = max(8, int(h * scale))
    return cv2.resize(sign_rgba, (target_w, target_h), interpolation=cv2.INTER_AREA)


def augment_affine(sign_rgba):
    """
    Поворот + перспектива + небольшой сдвиг.
    Делаем подложку (padding), чтобы при повороте не обрезать знак.
    """
    h, w = sign_rgba.shape[:2]
    pad = int(max(h, w) * 0.30) + 2
    ch, cw = h + 2 * pad, w + 2 * pad
    canvas = np.zeros((ch, cw, 4), dtype=np.uint8)
    canvas[pad : pad + h, pad : pad + w] = sign_rgba

    # roll
    angle = random.uniform(ROLL_ANGLE_RANGE[0], ROLL_ANGLE_RANGE[1])
    center = (cw / 2.0, ch / 2.0)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        canvas,
        M,
        (cw, ch),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    # perspective
    dx = PERSPECTIVE_STRENGTH * cw
    dy = PERSPECTIVE_STRENGTH * ch
    src = np.float32([[0, 0], [cw - 1, 0], [cw - 1, ch - 1], [0, ch - 1]])
    dst = src + np.float32([[random.uniform(-dx, dx), random.uniform(-dy, dy)] for _ in range(4)])
    P = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(
        rotated,
        P,
        (cw, ch),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    # shift
    tx = random.uniform(-SHIFT_FRACTION, SHIFT_FRACTION) * cw
    ty = random.uniform(-SHIFT_FRACTION, SHIFT_FRACTION) * ch
    T = np.float32([[1, 0, tx], [0, 1, ty]])
    shifted = cv2.warpAffine(
        warped,
        T,
        (cw, ch),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    return shifted


def alpha_bbox(alpha, thr=ALPHA_THRESHOLD):
    """bbox по альфа-каналу (в координатах изображения)."""
    ys, xs = np.where(alpha > thr)
    if xs.size == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def bbox_iou(b1, b2):
    """IoU для bbox (x1,y1,x2,y2), координаты в пикселях."""
    x11, y11, x12, y12 = b1
    x21, y21, x22, y22 = b2
    ix1 = max(x11, x21)
    iy1 = max(y11, y21)
    ix2 = min(x12, x22)
    iy2 = min(y12, y22)
    iw = max(0, ix2 - ix1 + 1)
    ih = max(0, iy2 - iy1 + 1)
    inter = iw * ih
    a1 = max(0, x12 - x11 + 1) * max(0, y12 - y11 + 1)
    a2 = max(0, x22 - x21 + 1) * max(0, y22 - y21 + 1)
    denom = a1 + a2 - inter
    if denom <= 0:
        return 0.0
    return inter / float(denom)


def bbox_from_sign_alpha_at_pos(sign_alpha, top, left, bg_h, bg_w):
    """Быстрая оценка bbox на фоне по альфе знака и позиции (без композитинга)."""
    bb = alpha_bbox(sign_alpha, thr=ALPHA_THRESHOLD)
    if bb is None:
        return None
    x1, y1, x2, y2 = bb
    x1 += left
    x2 += left
    y1 += top
    y2 += top
    x1 = max(0, min(bg_w - 1, x1))
    x2 = max(0, min(bg_w - 1, x2))
    y1 = max(0, min(bg_h - 1, y1))
    y2 = max(0, min(bg_h - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def visible_area_fraction(alpha, top, left, bg_h, bg_w, thr=ALPHA_THRESHOLD):
    """
    Доля видимой части знака после размещения на фоне.
    Считаем по альфе: сколько "непрозрачных" пикселей попадает в границы фона.
    """
    total = int(np.count_nonzero(alpha > thr))
    if total <= 0:
        return 0.0

    a_h, a_w = alpha.shape[:2]
    y1 = max(0, top)
    x1 = max(0, left)
    y2 = min(bg_h, top + a_h)
    x2 = min(bg_w, left + a_w)
    if y2 <= y1 or x2 <= x1:
        return 0.0

    ay1, ax1 = y1 - top, x1 - left
    ay2, ax2 = ay1 + (y2 - y1), ax1 + (x2 - x1)
    vis = int(np.count_nonzero(alpha[ay1:ay2, ax1:ax2] > thr))
    return vis / float(total)


def sample_position(bg_h, bg_w, sign_h, sign_w):
    """С вероятностью 70% центр знака в правой половине."""
    if random.random() < RIGHT_HALF_PROB:
        x_center = random.uniform(0.52, 0.95)
    else:
        x_center = random.uniform(0.05, 0.48)
    # На видеорегистраторе знаки чаще в верхней части кадра (не на капоте/дороге).
    y_center = 0.10 + (0.85 - 0.10) * (random.random() ** 1.35)
    left = int(x_center * bg_w - sign_w / 2.0)
    top = int(y_center * bg_h - sign_h / 2.0)
    return top, left


def place_one_object(bg, state, template_cache, existing_bboxes, max_tries=MAX_PLACEMENT_TRIES):
    """
    Размещаем один знак на фоне.
    state = {"cls": cls_dict, "seq": [...], "ptr": int, "count": int}
    existing_bboxes: список bbox уже размещённых знаков, чтобы контролировать пересечения.
    Возвращаем (bg2, label, bbox_px) или None.
    label = (class_id, x_center, y_center, w, h) в нормализованных координатах YOLO.
    """
    bg_h, bg_w = bg.shape[:2]

    cls = state["cls"]
    # Текущий шаблон для этого класса (важно для баланса вариаций)
    if state["ptr"] >= len(state["seq"]):
        tpl = state["seq"][-1]
    else:
        tpl = state["seq"][state["ptr"]]
    if tpl not in template_cache:
        rgba = load_template_rgba(tpl)
        if rgba is None:
            return None
        template_cache[tpl] = rgba
    else:
        rgba = template_cache[tpl]

    sign = resize_sign(rgba, bg_w)
    sign = augment_affine(sign)

    # Деградация качества знака в зависимости от его размера (после аугментаций)
    bb0 = alpha_bbox(sign[:, :, 3], thr=ALPHA_THRESHOLD)
    if bb0 is not None:
        w0 = int(bb0[2] - bb0[0] + 1)
        sign = degrade_sign_by_size(sign, w0)

    # Иногда слегка размываем знак, чтобы края не были "идеально вырезанными"
    if random.random() < EDGE_BLUR_PROB:
        sigma = random.uniform(EDGE_BLUR_K_RANGE[0], EDGE_BLUR_K_RANGE[1])
        rgb = sign[:, :, :3]
        a = sign[:, :, 3:4]
        rgb2 = cv2.GaussianBlur(rgb, (0, 0), sigmaX=sigma)
        sign = np.concatenate([rgb2, a], axis=2)

    alpha = sign[:, :, 3]
    if np.count_nonzero(alpha > ALPHA_THRESHOLD) == 0:
        return None

    s_h, s_w = sign.shape[:2]
    for _ in range(max_tries):
        top, left = sample_position(bg_h, bg_w, s_h, s_w)
        frac = visible_area_fraction(alpha, top, left, bg_h, bg_w)
        if frac < MIN_VISIBLE_AREA_FRACTION:
            continue

        bb_fast = bbox_from_sign_alpha_at_pos(alpha, top, left, bg_h, bg_w)
        if bb_fast is None:
            continue

        # ограничим пересечение с уже размещёнными
        ok = True
        for bb2 in existing_bboxes:
            if bbox_iou(bb_fast, bb2) > MAX_IOU_BETWEEN_SIGNS:
                ok = False
                break
        if not ok:
            continue

        y1 = max(0, top)
        x1 = max(0, left)
        y2 = min(bg_h, top + s_h)
        x2 = min(bg_w, left + s_w)
        if y2 <= y1 or x2 <= x1:
            continue

        patch = bg[y1:y2, x1:x2]
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        mean_brightness = float(np.mean(gray)) if gray.size else 127.0

        if mean_brightness < BRIGHTNESS_DARK_THRESH:
            sign2 = pil_enhance_rgba(sign, brightness=SIGN_DARKEN_FACTOR, contrast=1.0, color=SIGN_DESAT_FACTOR)
        elif mean_brightness > BRIGHTNESS_BRIGHT_THRESH:
            sign2 = pil_enhance_rgba(sign, brightness=SIGN_BRIGHTEN_FACTOR, contrast=SIGN_CONTRAST_FACTOR, color=1.0)
        else:
            sign2 = sign

        composed, alpha_placed = composite_rgba_over_bgr(bg, sign2, top, left)
        yolo_box = yolo_box_from_alpha(alpha_placed, img_w=bg_w, img_h=bg_h)
        bb_final = alpha_bbox(alpha_placed, thr=ALPHA_THRESHOLD)
        if yolo_box is None or bb_final is None:
            continue

        existing_bboxes.append(bb_final)
        label = (int(cls["class_id"]), *yolo_box)

        # Успешно поставили объект => двигаем счётчики экземпляров
        state["ptr"] += 1
        state["count"] += 1
        return composed, label, bb_final

    return None


def generate_multi(bg_paths, main_state, states, template_cache):
    """
    Генерируем изображение с несколькими объектами.
    main_state: состояние основного класса (обязательный объект).
    states: список состояний всех классов (для выбора доп. объектов).
    Возвращаем (img_bgr, labels) или None.
    """
    bg = load_bg_bgr(random.choice(bg_paths))
    labels = []
    bboxes = []

    # 1) основной объект обязателен
    res = place_one_object(bg, main_state, template_cache, bboxes, max_tries=MAX_PLACEMENT_TRIES)
    if res is None:
        return None
    bg, label, _bb = res
    labels.append(label)

    # 2) дополнительные объекты
    if MULTI_OBJECT_ENABLED:
        extra_n = random.randint(EXTRA_OBJECTS_RANGE[0], EXTRA_OBJECTS_RANGE[1])
    else:
        extra_n = 0

    for _ in range(extra_n):
        # выбираем, какой класс добавить
        if random.random() < EXTRA_SAME_CLASS_PROB:
            st = main_state
        else:
            # лучше выбирать из тех, у кого ещё не достигнут target (баланс по экземплярам)
            candidates = [s for s in states if s["count"] < s["target"]]
            if not candidates:
                candidates = states
            # вес = сколько ещё осталось добрать
            weights = [max(1, s.get("target", 0) - s.get("count", 0)) for s in candidates]
            st = random.choices(candidates, weights=weights, k=1)[0]

        res2 = place_one_object(bg, st, template_cache, bboxes, max_tries=MAX_EXTRA_TRIES)
        if res2 is None:
            continue
        bg, label2, _bb2 = res2
        labels.append(label2)

    if not labels:
        return None
    return bg, labels


def composite_rgba_over_bgr(bg_bgr, sign_rgba, top, left):
    """
    Вставляем RGBA знак в BGR фон по альфе.
    Возвращаем:
    - новое изображение BGR
    - alpha_placed (альфа знака в координатах фона) для bbox
    """
    bg_h, bg_w = bg_bgr.shape[:2]
    s_h, s_w = sign_rgba.shape[:2]

    y1 = max(0, top)
    x1 = max(0, left)
    y2 = min(bg_h, top + s_h)
    x2 = min(bg_w, left + s_w)
    if y2 <= y1 or x2 <= x1:
        return bg_bgr, np.zeros((bg_h, bg_w), dtype=np.uint8)

    sy1, sx1 = y1 - top, x1 - left
    sy2, sx2 = sy1 + (y2 - y1), sx1 + (x2 - x1)

    roi = bg_bgr[y1:y2, x1:x2].astype(np.float32)
    sroi = sign_rgba[sy1:sy2, sx1:sx2].astype(np.float32)

    # Тень: затемняем фон под знаком (небольшой сдвиг + размытие)
    if random.random() < SHADOW_PROB:
        strength = random.uniform(SHADOW_STRENGTH_RANGE[0], SHADOW_STRENGTH_RANGE[1])
        k = random.randint(SHADOW_BLUR_RANGE[0], SHADOW_BLUR_RANGE[1])
        if k % 2 == 0:
            k += 1
        dx = random.randint(SHADOW_OFFSET_X_RANGE[0], SHADOW_OFFSET_X_RANGE[1])
        dy = random.randint(SHADOW_OFFSET_Y_RANGE[0], SHADOW_OFFSET_Y_RANGE[1])

        a = (sroi[:, :, 3] / 255.0).astype(np.float32)
        a = cv2.GaussianBlur(a, (k, k), 0)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        a = cv2.warpAffine(a, M, (a.shape[1], a.shape[0]), flags=cv2.INTER_LINEAR, borderValue=0.0)
        a = np.clip(a * strength, 0.0, 0.9)
        roi *= (1.0 - a[:, :, None])

    alpha = sroi[:, :, 3:4] / 255.0
    sign_bgr = sroi[:, :, :3][:, :, ::-1]  # RGBA->BGR

    out = roi * (1.0 - alpha) + sign_bgr * alpha
    out_img = bg_bgr.copy()
    out_img[y1:y2, x1:x2] = np.clip(out, 0, 255).astype(np.uint8)

    alpha_placed = np.zeros((bg_h, bg_w), dtype=np.uint8)
    alpha_placed[y1:y2, x1:x2] = sroi[:, :, 3].astype(np.uint8)
    return out_img, alpha_placed


def apply_weather(img_bgr):
    """Умеренные эффекты: дождь/снег/туман."""
    if random.random() > WEATHER_PROB:
        return img_bgr

    h, w = img_bgr.shape[:2]
    effect = random.choice(["rain", "snow", "fog"])

    if effect == "rain":
        overlay = img_bgr.copy()
        n = random.randint(120, 260)
        for _ in range(n):
            x = random.randint(0, w - 1)
            y = random.randint(0, h - 1)
            length = random.randint(int(0.02 * h), int(0.08 * h))
            angle = random.uniform(-math.pi / 5, math.pi / 5)
            dx = int(math.cos(angle) * length)
            dy = int(math.sin(angle) * length + length)
            x2 = int(np.clip(x + dx, 0, w - 1))
            y2 = int(np.clip(y + dy, 0, h - 1))
            color = (random.randint(160, 220),) * 3
            thickness = random.randint(1, 2)
            cv2.line(overlay, (x, y), (x2, y2), color, thickness, lineType=cv2.LINE_AA)
        overlay = cv2.GaussianBlur(overlay, (3, 3), 0)
        a = random.uniform(0.18, 0.30)
        return cv2.addWeighted(img_bgr, 1.0 - a, overlay, a, 0.0)

    if effect == "snow":
        overlay = img_bgr.copy()
        n = random.randint(400, 1200)
        for _ in range(n):
            x = random.randint(0, w - 1)
            y = random.randint(0, h - 1)
            r = random.randint(1, 2)
            cv2.circle(overlay, (x, y), r, (255, 255, 255), -1, lineType=cv2.LINE_AA)
        overlay = cv2.GaussianBlur(overlay, (5, 5), 0)
        a = random.uniform(0.10, 0.22)
        return cv2.addWeighted(img_bgr, 1.0 - a, overlay, a, 0.0)

    # fog
    blur = cv2.GaussianBlur(img_bgr, (0, 0), sigmaX=random.uniform(1.0, 2.6))
    fog = np.full_like(img_bgr, random.randint(200, 235))
    a1 = random.uniform(0.10, 0.18)
    a2 = random.uniform(0.08, 0.16)
    out = cv2.addWeighted(img_bgr, 1.0 - a1, blur, a1, 0.0)
    out = cv2.addWeighted(out, 1.0 - a2, fog, a2, 0.0)
    return out


def yolo_box_from_alpha(alpha_placed, img_w, img_h):
    """YOLO bbox в нормализованных координатах (xc, yc, w, h)."""
    bb = alpha_bbox(alpha_placed, thr=ALPHA_THRESHOLD)
    if bb is None:
        return None
    x1, y1, x2, y2 = bb
    bw = (x2 - x1 + 1)
    bh = (y2 - y1 + 1)
    xc = x1 + bw / 2.0
    yc = y1 + bh / 2.0
    return (xc / img_w, yc / img_h, bw / img_w, bh / img_h)


def sanitize_for_filename(name, max_len=120):
    """Делаем безопасное имя файла из названия класса."""
    name = str(name).strip()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^\w\-.]+", "_", name, flags=re.UNICODE)
    name = name.strip("._-")
    if not name:
        name = "class"
    return name[:max_len]


def build_image_stem(labels, id_to_code):
    """
    Имя изображения из использованных классов.
    Примеры:
    - 1.1_1.4
    - 1.1__2
    - 1.1__2_2.3
    """
    cnt = {}
    for (cid, _xc, _yc, _w, _h) in labels:
        cid = int(cid)
        cnt[cid] = cnt.get(cid, 0) + 1
    parts = []
    for cid in sorted(cnt.keys()):
        code = id_to_code.get(cid, f"class_{cid}")
        if cnt[cid] == 1:
            parts.append(code)
        else:
            parts.append(f"{code}__{cnt[cid]}")
    stem = "_".join(parts) if parts else "image"
    return sanitize_for_filename(stem, max_len=180)


def save_sample(out_dir, image_bgr, unique_idx, labels, id_to_code):
    """
    Сохраняем:
    - PNG в images/
    - TXT в labels/ (YOLO), несколько строк если объектов несколько
    Возвращаем относительный путь до картинки (для train/val txt).
    """
    stem = build_image_stem(labels, id_to_code)
    img_file = f"{stem}__{unique_idx:07d}.png"
    img_path = os.path.join(out_dir, "images", img_file)
    lbl_path = os.path.join(out_dir, "labels", os.path.splitext(img_file)[0] + ".txt")

    ok = cv2.imwrite(img_path, image_bgr)
    if not ok:
        raise RuntimeError(f"Не удалось записать изображение: {img_path}")

    with open(lbl_path, "w", encoding="utf-8") as f:
        for (cid, xc, yc, w, h) in labels:
            f.write(f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

    # Для train.txt/val.txt: путь относительно out_dir, как в примере
    return os.path.join("images", img_file)


def save_negative_sample(out_dir, image_bgr, unique_idx):
    """Сохраняем негативный кадр: картинка + пустой label-файл."""
    img_file = f"negative_{unique_idx:07d}.png"
    img_path = os.path.join(out_dir, "images", img_file)
    lbl_path = os.path.join(out_dir, "labels", os.path.splitext(img_file)[0] + ".txt")

    ok = cv2.imwrite(img_path, image_bgr)
    if not ok:
        raise RuntimeError(f"Не удалось записать изображение: {img_path}")

    # пустая разметка (нет объектов)
    with open(lbl_path, "w", encoding="utf-8") as f:
        f.write("")

    return os.path.join("images", img_file)


def generate_dataset():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    bg_paths = load_background_paths()
    classes = load_classes()

    # Балансируем по ЧИСЛУ ЭКЗЕМПЛЯРОВ (bbox) на класс
    target_instances = compute_target_count(classes)
    info(f"Классов загружено: {len(classes)}. TARGET (экземпляров на класс) = {target_instances}.")

    if not HAS_CAIROSVG:
        # Предупреждение не фатальное, т.к. есть fallback на rsvg/inkscape
        warn("cairosvg не установлен. Будем пробовать rsvg-convert/inkscape для SVG.")

    out_dir = ensure_output_dir(OUTPUT_BASE)
    info(f"Выходная папка: {out_dir}")

    write_classes_txt(out_dir, classes)

    # Итоговые списки для Ultralytics (как в вашем примере)
    train_list = []
    val_list = []

    # Сводный CSV
    ann_csv_path = os.path.join(out_dir, "annotations.csv")
    ann_f = open(ann_csv_path, "w", encoding="utf-8", newline="")
    ann_w = csv.writer(ann_f)
    ann_w.writerow(["image_file", "class_id", "x_center", "y_center", "width", "height"])

    template_cache = {}
    unique_idx = 0

    # Для имён файлов: class_id -> "код" (base filename без расширения, например "1.1")
    id_to_code = {int(c["class_id"]): str(c["name"]) for c in classes}

    # Состояние для каждого класса: прогресс по экземплярам и последовательность шаблонов
    states = []
    for c in classes:
        seq = build_template_sequence(c, target_instances)
        states.append({"cls": c, "seq": seq, "ptr": 0, "count": 0, "target": target_instances})

    try:
        # Генерация, пока каждый класс не наберёт target_instances объектов
        total_target = len(states) * target_instances
        total_made = 0
        attempts = 0
        max_attempts = total_target * MAX_ATTEMPTS_MULT

        while total_made < total_target and attempts < max_attempts:
            attempts += 1

            need = [s for s in states if s["count"] < s["target"]]
            if not need:
                break

            weights = [max(1, s["target"] - s["count"]) for s in need]
            main_state = random.choices(need, weights=weights, k=1)[0]

            res = generate_multi(bg_paths, main_state, states, template_cache)
            if res is None:
                continue

            img_bgr, labels = res
            img_bgr = apply_weather(img_bgr)
            img_bgr = apply_camera_effects(img_bgr)

            unique_idx += 1
            rel_img_path = save_sample(out_dir, img_bgr, unique_idx, labels, id_to_code)

            # train/val — просто рандомно, чтобы не было утечки по фонам
            if random.random() < VAL_RATIO:
                val_list.append(rel_img_path)
            else:
                train_list.append(rel_img_path)

            for (cid, xc, yc, ww, hh) in labels:
                ann_w.writerow([rel_img_path, cid, f"{xc:.6f}", f"{yc:.6f}", f"{ww:.6f}", f"{hh:.6f}"])

            total_made = sum(s["count"] for s in states)

        # Итоги по классам
        for s in states:
            c = s["cls"]
            if s["count"] < s["target"]:
                warn(f"Класс id={c['class_id']} набрал {s['count']}/{s['target']} экземпляров.")
            else:
                info(f"Класс id={c['class_id']}: {s['count']}/{s['target']} экземпляров.")
    finally:
        ann_f.close()

    # Негативные изображения (без знаков)
    # Это помогает модели не "видеть" знаки везде на реальных кадрах.
    # Негативы считаем от ожидаемого числа изображений (примерно).
    if MULTI_OBJECT_ENABLED:
        exp_objs = 1.0 + (EXTRA_OBJECTS_RANGE[0] + EXTRA_OBJECTS_RANGE[1]) / 2.0
    else:
        exp_objs = 1.0
    exp_images = (len(classes) * target_instances) / max(1.0, exp_objs)
    neg_count = int(round(exp_images * float(NEGATIVE_RATIO)))
    if neg_count > 0:
        info(f"Генерируем негативные изображения: {neg_count}")
        for i in range(neg_count):
            bg = load_bg_bgr(random.choice(bg_paths))
            bg = apply_weather(bg)
            bg = apply_camera_effects(bg)
            unique_idx += 1
            rel_img_path = save_negative_sample(out_dir, bg, unique_idx)
            if random.random() < VAL_RATIO:
                val_list.append(rel_img_path)
            else:
                train_list.append(rel_img_path)

    # YAML + train/val списки
    write_dataset_yaml_and_splits(out_dir, classes, train_list, val_list)
    info("Готово.")
    return out_dir


if __name__ == "__main__":
    generate_dataset()
