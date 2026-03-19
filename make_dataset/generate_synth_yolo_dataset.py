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
SIGNS_IMAGES_DIR = "make_dataset/signs/images"
SPLITS_DIR = "make_dataset/signs/splits"
CSV_PATH = "make_dataset/signs/signs.csv"

OUTPUT_BASE = "datasets/dataset"  # папки будут dataset_1, dataset_2, ...
MIN_IMAGES_PER_CLASS = 20         # минимум на класс
VAL_RATIO = 0.2                   # доля val на класс (для train.txt/val.txt)
WEATHER_PROB = 0.7                # вероятность погодного эффекта
RANDOM_SEED = 1337

# Диапазоны аугментаций
SCALE_RANGE = (0.05, 0.25)        # ширина знака как доля ширины фона
ROLL_ANGLE_RANGE = (-30.0, 30.0)  # поворот в плоскости
PERSPECTIVE_STRENGTH = 0.10       # перспектива (наклон)
SHIFT_FRACTION = 0.08             # небольшой сдвиг после преобразований

# Размещение
RIGHT_HALF_PROB = 0.7
MIN_VISIBLE_AREA_FRACTION = 0.60
MAX_PLACEMENT_TRIES = 10

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
    """Список файлов в папке по наборам расширений."""
    if not os.path.isdir(folder):
        return []
    out = []
    for ext in exts:
        out += glob(os.path.join(folder, "*" + ext))
        out += glob(os.path.join(folder, "*" + ext.upper()))
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
    paths = list_files_by_ext(BACKGROUNDS_DIR, BACKGROUND_EXTS)
    if len(paths) < 1:
        raise FileNotFoundError(f"Не найдено фонов в {BACKGROUNDS_DIR}")
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
    target_w = int(bg_w * random.uniform(SCALE_RANGE[0], SCALE_RANGE[1]))
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
    y_center = random.uniform(0.10, 0.90)
    left = int(x_center * bg_w - sign_w / 2.0)
    top = int(y_center * bg_h - sign_h / 2.0)
    return top, left


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


def generate_one(bg_paths, template_path, template_cache):
    """
    Генерируем 1 изображение:
    - выбираем фон
    - грузим/кешируем шаблон
    - масштаб + аффинные преобразования
    - выбираем позицию с ограничением по видимой площади
    - коррекция знака по яркости области фона
    - вставка по альфе
    """
    bg_path = random.choice(bg_paths)
    bg = load_bg_bgr(bg_path)
    bg_h, bg_w = bg.shape[:2]

    if template_path not in template_cache:
        rgba = load_template_rgba(template_path)
        if rgba is None:
            return None
        template_cache[template_path] = rgba
    else:
        rgba = template_cache[template_path]

    sign = resize_sign(rgba, bg_w)
    sign = augment_affine(sign)

    alpha = sign[:, :, 3]
    if np.count_nonzero(alpha > ALPHA_THRESHOLD) == 0:
        return None

    s_h, s_w = sign.shape[:2]
    for _ in range(MAX_PLACEMENT_TRIES):
        top, left = sample_position(bg_h, bg_w, s_h, s_w)
        frac = visible_area_fraction(alpha, top, left, bg_h, bg_w)
        if frac < MIN_VISIBLE_AREA_FRACTION:
            continue

        # яркость области фона (по прямоугольнику размещения)
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
        return composed, alpha_placed

    warn(f"Не удалось нормально разместить знак (шаблон {os.path.basename(template_path)}).")
    return None


def save_sample(out_dir, image_bgr, class_id, class_name, unique_idx, yolo_box):
    """
    Сохраняем:
    - PNG в images/
    - TXT в labels/ (YOLO)
    Возвращаем относительный путь до картинки (для train/val txt).
    """
    safe = sanitize_for_filename(class_name)
    img_file = f"{safe}_{unique_idx:07d}.png"
    img_path = os.path.join(out_dir, "images", img_file)
    lbl_path = os.path.join(out_dir, "labels", os.path.splitext(img_file)[0] + ".txt")

    ok = cv2.imwrite(img_path, image_bgr)
    if not ok:
        raise RuntimeError(f"Не удалось записать изображение: {img_path}")

    xc, yc, w, h = yolo_box
    with open(lbl_path, "w", encoding="utf-8") as f:
        f.write(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

    # Для train.txt/val.txt: путь относительно out_dir, как в примере
    return os.path.join("images", img_file)


def generate_dataset():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    bg_paths = load_background_paths()
    classes = load_classes()
    target = compute_target_count(classes)
    info(f"Классов загружено: {len(classes)}. TARGET_COUNT на класс = {target}.")

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

    # Сколько примеров на val для каждого класса (одинаково для всех)
    val_count = int(target * VAL_RATIO)
    if target >= 2:
        val_count = max(1, val_count)
    val_count = min(val_count, target)
    train_count = target - val_count

    try:
        for cls in classes:
            seq = build_template_sequence(cls, target)
            made = 0
            attempts = 0
            max_attempts = target * MAX_ATTEMPTS_MULT

            while made < target and attempts < max_attempts:
                attempts += 1
                template_path = seq[made]
                res = generate_one(bg_paths, template_path, template_cache)
                if res is None:
                    continue

                img_bgr, alpha_placed = res
                img_bgr = apply_weather(img_bgr)

                h, w = img_bgr.shape[:2]
                yolo_box = yolo_box_from_alpha(alpha_placed, img_w=w, img_h=h)
                if yolo_box is None:
                    continue

                unique_idx += 1
                rel_img_path = save_sample(out_dir, img_bgr, cls["class_id"], cls["class_name"], unique_idx, yolo_box)

                # split train/val по индексу внутри класса
                if made < train_count:
                    train_list.append(rel_img_path)
                else:
                    val_list.append(rel_img_path)

                ann_w.writerow([rel_img_path, cls["class_id"], *[f"{v:.6f}" for v in yolo_box]])
                made += 1

            if made < target:
                warn(
                    f"Класс id={cls['class_id']} сгенерировал {made}/{target} (попыток {attempts}). "
                    f"Если много пропусков, поставьте cairosvg или rsvg-convert."
                )
            else:
                info(f"Класс id={cls['class_id']}: {made}/{target}")
    finally:
        ann_f.close()

    # YAML + train/val списки
    write_dataset_yaml_and_splits(out_dir, classes, train_list, val_list)
    info("Готово.")
    return out_dir


if __name__ == "__main__":
    generate_dataset()
