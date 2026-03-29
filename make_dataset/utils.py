import os
import re
import io
import csv
import math
import random
import shutil
import subprocess
import tempfile
import ast
import threading
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from collections import OrderedDict
from glob import glob

import cv2
import numpy as np
from PIL import Image, ImageEnhance

from make_dataset import config as cfg
from make_dataset import effects


try:
    import cairosvg  # type: ignore

    HAS_CAIROSVG = True
except Exception:
    HAS_CAIROSVG = False


def info(*msg, end=None):
    print("[INFO]", *msg, end=end)


def warn(msg):
    print("[WARN]", msg)


def base_name_no_ext(s):
    """Берём имя без расширения."""
    s = os.path.basename(str(s)).strip()
    return os.path.splitext(s)[0]


def sanitize_for_filename(name, max_len=120):
    """Делаем безопасное имя файла."""
    name = str(name).strip()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^\w\-.]+", "_", name, flags=re.UNICODE)
    name = name.strip("._-")
    if not name:
        name = "class"
    return name[:max_len]


def normalize_class_code(s):
    """
    Нормализация "кода" класса для сопоставления внешнего датасета с нашим.
    Примеры:
    - '1.11.1.' -> '1.11.1'
    - '3.24._10' -> '3.24_10'
    - '6.2._80' -> '6.2_80'
    """
    s = str(s).strip()
    s = s.strip("'\"")
    while s.endswith("."):
        s = s[:-1].rstrip()
    s = s.replace("._", "_")
    s = re.sub(r"\s+", "", s)
    return s


def list_files_by_ext(folder, exts):
    """Список файлов в папке по расширениям (НЕ рекурсивно)."""
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
    поэтому делаем fallback на удаление суффиксов.
    """
    cands = [name]
    for suf in ("_template", "_temp"):
        if name.endswith(suf):
            cands.append(name[: -len(suf)])
        if suf in name:
            cands.append(name.split(suf, 1)[0])
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
    """Если нет вариаций, ищем файл, который начинается с name."""
    if not os.path.isdir(images_dir):
        return None
    for ext in cfg.TEMPLATE_EXTS:
        cands = sorted(glob(os.path.join(images_dir, f"{name}*{ext}")))
        cands += sorted(glob(os.path.join(images_dir, f"{name}*{ext.upper()}")))
        if cands:
            return cands[0]
    return None


def load_classes(filter_ids=None):
    """
    Читаем CSV и для каждого класса определяем список шаблонов:
    - если есть splits/<name>/ с файлами -> класс с вариациями
    - иначе ищем один файл в images/
    """
    classes = []
    with open(cfg.CSV_PATH, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for col in ("filename", "class_id", "class_name"):
            if col not in (reader.fieldnames or []):
                raise ValueError(f"В CSV нет колонки '{col}': {cfg.CSV_PATH}")

        for row in reader:
            name = base_name_no_ext(row["filename"])
            try:
                class_id = int(row["class_id"])
            except Exception:
                warn(f"Плохой class_id: {row}")
                continue
            class_name = str(row["class_name"])

            templates = []
            for cand in split_dir_candidates(name):
                d = os.path.join(cfg.SPLITS_DIR, cand)
                t = list_files_by_ext(d, cfg.TEMPLATE_EXTS)
                if t:
                    templates = sorted(t, key=lambda p: os.path.basename(p).lower())
                    break

            if templates:
                classes.append(
                    {"name": name, "class_id": class_id, "class_name": class_name, "templates": templates, "has_variations": True}
                )
                continue

            single = find_template_in_images(cfg.SIGNS_IMAGES_DIR, name)
            if not single:
                warn(f"Не найден шаблон для '{name}' (id={class_id}) в {cfg.SIGNS_IMAGES_DIR}. Пропускаю класс.")
                continue

            classes.append(
                {"name": name, "class_id": class_id, "class_name": class_name, "templates": [single], "has_variations": False}
            )

    if not classes:
        raise RuntimeError("Не удалось загрузить ни одного класса (все пропущены). Проверьте пути.")

    classes = sorted(classes, key=lambda c: c["class_id"])

    if filter_ids:
        want = set(int(x) for x in filter_ids)
        classes = [c for c in classes if int(c["class_id"]) in want]
        if not classes:
            raise RuntimeError("После фильтра классов не осталось ни одного класса.")

    return classes


def compute_target_count(selected_classes, external_instances_per_class=None):
    """
    max(MIN_IMAGES_PER_CLASS, max(2*V) среди вариативных классов, max_instances_external).
    """
    max_variation_target = 0
    for c in selected_classes:
        if c["has_variations"]:
            v = len(c["templates"])
            max_variation_target = max(max_variation_target, v * 2)

    max_external = 0
    if isinstance(external_instances_per_class, dict) and external_instances_per_class:
        for c in selected_classes:
            cid = int(c["class_id"])
            max_external = max(max_external, int(external_instances_per_class.get(cid, 0)))

    return int(max(cfg.MIN_IMAGES_PER_CLASS, max_variation_target, max_external))


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
    train_txt = os.path.join(out_dir, "train.txt")
    val_txt = os.path.join(out_dir, "val.txt")

    # Для больших датасетов списки путей могут занимать много памяти.
    # Если train_list/val_list = None, считаем что файлы train.txt/val.txt уже записаны потоково.
    if train_list is not None:
        with open(train_txt, "w", encoding="utf-8") as f:
            for p in train_list:
                f.write(p + "\n")
    if val_list is not None:
        with open(val_txt, "w", encoding="utf-8") as f:
            for p in val_list:
                f.write(p + "\n")

    max_id = max(int(c["class_id"]) for c in classes)
    names = [""] * (max_id + 1)
    for c in classes:
        names[int(c["class_id"])] = c["class_name"]
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
    max_id = max(int(c["class_id"]) for c in classes)
    lines = [""] * (max_id + 1)
    for c in classes:
        lines[int(c["class_id"])] = c["class_name"]
    for i in range(len(lines)):
        if not lines[i]:
            lines[i] = f"class_{i}"
    with open(os.path.join(out_dir, "classes.txt"), "w", encoding="utf-8") as f:
        for s in lines:
            f.write(s + "\n")


def load_background_paths():
    dashcam = list_files_by_ext_recursive(cfg.DASHCAM_FRAMES_DIR, cfg.BACKGROUND_EXTS)
    generic = list_files_by_ext(cfg.BACKGROUNDS_DIR, cfg.BACKGROUND_EXTS)

    paths = []
    if dashcam:
        paths += dashcam
    if generic:
        paths += generic

    if len(paths) < 1:
        raise FileNotFoundError(
            f"Не найдено фонов. Проверьте BACKGROUNDS_DIR='{cfg.BACKGROUNDS_DIR}' и DASHCAM_FRAMES_DIR='{cfg.DASHCAM_FRAMES_DIR}'."
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
    """Коррекция яркости/контраста/насыщенности через PIL."""
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


def render_svg_to_rgba(svg_path, render_px=None):
    """
    SVG -> RGBA.
    1) cairosvg (если есть)
    2) rsvg-convert (если установлен)
    3) inkscape (если установлен)
    """
    if render_px is None:
        render_px = int(getattr(cfg, "SVG_RENDER_PX", 512))

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
        return None

    return None


def load_template_rgba(path, render_px=None):
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".svg":
            rgba = render_svg_to_rgba(path, render_px=render_px)
            return rgba
        pil = Image.open(path).convert("RGBA")
        return np.array(pil, dtype=np.uint8)
    except Exception as e:
        warn(f"Не удалось загрузить шаблон {path}: {e}")
        return None


def resize_sign(sign_rgba, bg_w):
    """Масштабируем знак."""
    h, w = sign_rgba.shape[:2]
    if w <= 0 or h <= 0:
        return sign_rgba
    u = random.random() ** float(cfg.SCALE_BIAS_POWER)
    rel = float(cfg.SCALE_RANGE[0]) + (float(cfg.SCALE_RANGE[1]) - float(cfg.SCALE_RANGE[0])) * u
    target_w = int(bg_w * rel)
    target_w = max(8, target_w)
    scale = target_w / float(w)
    target_h = max(8, int(h * scale))
    return cv2.resize(sign_rgba, (target_w, target_h), interpolation=cv2.INTER_AREA)


def augment_affine(sign_rgba):
    """Поворот + перспектива + небольшой сдвиг."""
    h, w = sign_rgba.shape[:2]
    pad = int(max(h, w) * 0.30) + 2
    ch, cw = h + 2 * pad, w + 2 * pad
    canvas = np.zeros((ch, cw, 4), dtype=np.uint8)
    canvas[pad : pad + h, pad : pad + w] = sign_rgba

    angle = random.uniform(cfg.ROLL_ANGLE_RANGE[0], cfg.ROLL_ANGLE_RANGE[1])
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

    dx = cfg.PERSPECTIVE_STRENGTH * cw
    dy = cfg.PERSPECTIVE_STRENGTH * ch
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

    tx = random.uniform(-cfg.SHIFT_FRACTION, cfg.SHIFT_FRACTION) * cw
    ty = random.uniform(-cfg.SHIFT_FRACTION, cfg.SHIFT_FRACTION) * ch
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


def visible_area_fraction(alpha, top, left, bg_h, bg_w, thr=cfg.ALPHA_THRESHOLD):
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
    if random.random() < cfg.RIGHT_HALF_PROB:
        x_center = random.uniform(0.52, 0.95)
    else:
        x_center = random.uniform(0.05, 0.48)
    y_center = 0.10 + (0.85 - 0.10) * (random.random() ** 1.35)
    left = int(x_center * bg_w - sign_w / 2.0)
    top = int(y_center * bg_h - sign_h / 2.0)
    return top, left


def composite_rgba_over_bgr(bg_bgr, sign_rgba, top, left):
    """Вставляем RGBA знак в BGR фон по альфе."""
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

    # тень
    if random.random() < cfg.SHADOW_PROB:
        strength = random.uniform(cfg.SHADOW_STRENGTH_RANGE[0], cfg.SHADOW_STRENGTH_RANGE[1])
        k = random.randint(cfg.SHADOW_BLUR_RANGE[0], cfg.SHADOW_BLUR_RANGE[1])
        if k % 2 == 0:
            k += 1
        dx = random.randint(cfg.SHADOW_OFFSET_X_RANGE[0], cfg.SHADOW_OFFSET_X_RANGE[1])
        dy = random.randint(cfg.SHADOW_OFFSET_Y_RANGE[0], cfg.SHADOW_OFFSET_Y_RANGE[1])

        a = (sroi[:, :, 3] / 255.0).astype(np.float32)
        a = cv2.GaussianBlur(a, (k, k), 0)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        a = cv2.warpAffine(a, M, (a.shape[1], a.shape[0]), flags=cv2.INTER_LINEAR, borderValue=0.0)
        a = np.clip(a * strength, 0.0, 0.9)
        roi *= (1.0 - a[:, :, None])

    alpha = sroi[:, :, 3:4] / 255.0
    sign_bgr = sroi[:, :, :3][:, :, ::-1]
    out = roi * (1.0 - alpha) + sign_bgr * alpha
    out_img = bg_bgr.copy()
    out_img[y1:y2, x1:x2] = np.clip(out, 0, 255).astype(np.uint8)

    alpha_placed = np.zeros((bg_h, bg_w), dtype=np.uint8)
    alpha_placed[y1:y2, x1:x2] = sroi[:, :, 3].astype(np.uint8)
    return out_img, alpha_placed


def yolo_box_from_alpha(alpha_placed, img_w, img_h):
    bb = effects.alpha_bbox(alpha_placed, thr=cfg.ALPHA_THRESHOLD)
    if bb is None:
        return None
    x1, y1, x2, y2 = bb
    bw = (x2 - x1 + 1)
    bh = (y2 - y1 + 1)
    xc = x1 + bw / 2.0
    yc = y1 + bh / 2.0
    return (xc / img_w, yc / img_h, bw / img_w, bh / img_h)


def bbox_iou(b1, b2):
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


def bbox_intersection_area(b1, b2):
    """Площадь пересечения bbox (x1,y1,x2,y2) в пикселях (с +1 как в bbox_iou)."""
    x11, y11, x12, y12 = b1
    x21, y21, x22, y22 = b2
    ix1 = max(x11, x21)
    iy1 = max(y11, y21)
    ix2 = min(x12, x22)
    iy2 = min(y12, y22)
    iw = max(0, ix2 - ix1 + 1)
    ih = max(0, iy2 - iy1 + 1)
    return iw * ih


def bbox_from_sign_alpha_at_pos(sign_alpha, top, left, bg_h, bg_w):
    bb = effects.alpha_bbox(sign_alpha, thr=cfg.ALPHA_THRESHOLD)
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


def place_one_object(bg, state, template_cache, existing_bboxes, max_tries=cfg.MAX_PLACEMENT_TRIES):
    """
    state = {"cls": cls_dict, "seq": [...], "ptr": int, "count": int}
    Возвращаем (bg2, label, bbox_px) или None.
    """
    bg_h, bg_w = bg.shape[:2]
    cls = state["cls"]

    seq = state.get("seq") or []
    ptr = int(state.get("ptr", 0))
    tpl = seq[-1] if (seq and ptr >= len(seq)) else (seq[ptr] if seq else None)
    if tpl is None:
        return None

    rgba = template_cache.get(tpl)
    if rgba is None:
        rgba = load_template_rgba(tpl)
        if rgba is None:
            return None
        template_cache[tpl] = rgba

    # LRU-очистка кэша шаблонов (важно для RAM)
    if hasattr(template_cache, "move_to_end"):
        try:
            template_cache.move_to_end(tpl)  # type: ignore[attr-defined]
        except Exception:
            pass
        max_items = int(getattr(cfg, "TEMPLATE_CACHE_MAX_ITEMS_PER_THREAD", 0) or 0)
        if max_items > 0:
            try:
                while len(template_cache) > max_items:
                    template_cache.popitem(last=False)  # type: ignore[arg-type]
            except Exception:
                pass

    sign = resize_sign(rgba, bg_w)
    sign = augment_affine(sign)

    bb0 = effects.alpha_bbox(sign[:, :, 3], thr=cfg.ALPHA_THRESHOLD)
    if bb0 is not None:
        w0 = int(bb0[2] - bb0[0] + 1)
        sign = effects.degrade_sign_by_size(sign, w0)

    if random.random() < cfg.EDGE_BLUR_PROB:
        sigma = random.uniform(cfg.EDGE_BLUR_K_RANGE[0], cfg.EDGE_BLUR_K_RANGE[1])
        rgb = sign[:, :, :3]
        a = sign[:, :, 3:4]
        rgb2 = cv2.GaussianBlur(rgb, (0, 0), sigmaX=sigma)
        sign = np.concatenate([rgb2, a], axis=2)

    alpha = sign[:, :, 3]
    if np.count_nonzero(alpha > cfg.ALPHA_THRESHOLD) == 0:
        return None

    s_h, s_w = sign.shape[:2]
    for _ in range(max_tries):
        top, left = sample_position(bg_h, bg_w, s_h, s_w)
        frac = visible_area_fraction(alpha, top, left, bg_h, bg_w)
        if frac < cfg.MIN_VISIBLE_AREA_FRACTION:
            continue

        bb_fast = bbox_from_sign_alpha_at_pos(alpha, top, left, bg_h, bg_w)
        if bb_fast is None:
            continue

        ok = True
        for bb2 in existing_bboxes:
            if bbox_iou(bb_fast, bb2) > cfg.MAX_IOU_BETWEEN_SIGNS:
                ok = False
                break

            # Запрещаем сильное перекрытие меньшего объекта (боремся с “большой накрыл маленький”)
            inter = bbox_intersection_area(bb_fast, bb2)
            if inter > 0:
                a1 = max(0, bb_fast[2] - bb_fast[0] + 1) * max(0, bb_fast[3] - bb_fast[1] + 1)
                a2 = max(0, bb2[2] - bb2[0] + 1) * max(0, bb2[3] - bb2[1] + 1)
                denom = float(max(1, min(a1, a2)))
                io_min = float(inter) / denom
                if io_min > float(getattr(cfg, "MAX_IO_MINAREA_BETWEEN_SIGNS", 0.25)):
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

        if mean_brightness < cfg.BRIGHTNESS_DARK_THRESH:
            sign2 = pil_enhance_rgba(sign, brightness=cfg.SIGN_DARKEN_FACTOR, contrast=1.0, color=cfg.SIGN_DESAT_FACTOR)
        elif mean_brightness > cfg.BRIGHTNESS_BRIGHT_THRESH:
            sign2 = pil_enhance_rgba(
                sign, brightness=cfg.SIGN_BRIGHTEN_FACTOR, contrast=cfg.SIGN_CONTRAST_FACTOR, color=1.0
            )
        else:
            sign2 = sign

        composed, alpha_placed = composite_rgba_over_bgr(bg, sign2, top, left)
        yolo_box = yolo_box_from_alpha(alpha_placed, img_w=bg_w, img_h=bg_h)
        bb_final = effects.alpha_bbox(alpha_placed, thr=cfg.ALPHA_THRESHOLD)
        if yolo_box is None or bb_final is None:
            continue

        existing_bboxes.append(bb_final)
        label = (int(cls["class_id"]), *yolo_box)
        state["ptr"] = ptr + 1
        state["count"] = int(state.get("count", 0)) + 1
        return composed, label, bb_final

    return None


def generate_multi(bg_paths, main_state, states, extra_pool_states, template_cache):
    """Генерируем изображение с несколькими объектами."""
    bg = load_bg_bgr(random.choice(bg_paths))
    labels = []
    bboxes = []

    res = place_one_object(bg, main_state, template_cache, bboxes, max_tries=cfg.MAX_PLACEMENT_TRIES)
    if res is None:
        return None
    bg, label, _bb = res
    labels.append(label)

    extra_n = random.randint(cfg.EXTRA_OBJECTS_RANGE[0], cfg.EXTRA_OBJECTS_RANGE[1]) if cfg.MULTI_OBJECT_ENABLED else 0
    for _ in range(extra_n):
        if random.random() < cfg.EXTRA_SAME_CLASS_PROB:
            st = main_state
        else:
            candidates = [s for s in extra_pool_states if s.get("count", 0) < s.get("target", 0)]
            if not candidates:
                candidates = list(extra_pool_states)
            weights = [max(1, int(s.get("target", 0)) - int(s.get("count", 0))) for s in candidates]
            st = random.choices(candidates, weights=weights, k=1)[0]

        res2 = place_one_object(bg, st, template_cache, bboxes, max_tries=cfg.MAX_EXTRA_TRIES)
        if res2 is None:
            continue
        bg, label2, _bb2 = res2
        labels.append(label2)

    if not labels:
        return None
    return bg, labels


def build_image_stem(labels, id_to_code):
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
    stem = build_image_stem(labels, id_to_code)
    img_file = f"{stem}__{unique_idx:07d}.png"
    img_path = os.path.join(out_dir, "images", img_file)
    lbl_path = os.path.join(out_dir, "labels", os.path.splitext(img_file)[0] + ".txt")

    ok = cv2.imwrite(
        img_path,
        image_bgr,
        [int(cv2.IMWRITE_PNG_COMPRESSION), int(getattr(cfg, "PNG_COMPRESSION", 3))],
    )
    if not ok:
        raise RuntimeError(f"Не удалось записать изображение: {img_path}")

    with open(lbl_path, "w", encoding="utf-8") as f:
        for (cid, xc, yc, w, h) in labels:
            f.write(f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

    return os.path.join("images", img_file)


def save_negative_sample(out_dir, image_bgr, unique_idx):
    img_file = f"negative_{unique_idx:07d}.png"
    img_path = os.path.join(out_dir, "images", img_file)
    lbl_path = os.path.join(out_dir, "labels", os.path.splitext(img_file)[0] + ".txt")

    ok = cv2.imwrite(
        img_path,
        image_bgr,
        [int(cv2.IMWRITE_PNG_COMPRESSION), int(getattr(cfg, "PNG_COMPRESSION", 3))],
    )
    if not ok:
        raise RuntimeError(f"Не удалось записать изображение: {img_path}")
    with open(lbl_path, "w", encoding="utf-8") as f:
        f.write("")
    return os.path.join("images", img_file)


_thread_local = threading.local()


def _get_thread_template_cache():
    if not hasattr(_thread_local, "template_cache"):
        _thread_local.template_cache = OrderedDict()
    return _thread_local.template_cache


def reserve_object_specs(main_state, extra_pool_states):
    """
    Резервируем шаблоны по seq/ptr у классов в главном потоке.
    Возвращаем [{"cid": int, "tpl": str}, ...]
    """
    specs = []

    def reserve_one(st):
        seq = st.get("seq") or []
        ptr = int(st.get("ptr", 0))
        if not seq:
            return None
        tpl = seq[-1] if ptr >= len(seq) else seq[ptr]
        st["ptr"] = ptr + 1
        return tpl

    tpl0 = reserve_one(main_state)
    if tpl0 is None:
        return None
    specs.append({"cid": int(main_state["cls"]["class_id"]), "tpl": tpl0})

    extra_n = random.randint(cfg.EXTRA_OBJECTS_RANGE[0], cfg.EXTRA_OBJECTS_RANGE[1]) if cfg.MULTI_OBJECT_ENABLED else 0
    for _ in range(extra_n):
        if random.random() < cfg.EXTRA_SAME_CLASS_PROB:
            st = main_state
        else:
            candidates = [s for s in extra_pool_states if s.get("count", 0) < s.get("target", 0)]
            if not candidates:
                candidates = list(extra_pool_states)
            weights = [max(1, int(s.get("target", 0)) - int(s.get("count", 0))) for s in candidates]
            st = random.choices(candidates, weights=weights, k=1)[0]

        tpl = reserve_one(st)
        if tpl is None:
            continue
        specs.append({"cid": int(st["cls"]["class_id"]), "tpl": tpl})

    return specs


def worker_generate_and_save(out_dir, bg_paths, specs, unique_idx, id_to_code):
    """Тяжёлая часть: генерация + эффекты + сохранение (в worker-потоке)."""
    if not specs:
        return None

    template_cache = _get_thread_template_cache()
    bg = load_bg_bgr(random.choice(bg_paths))

    labels = []
    bboxes = []
    for i, sp in enumerate(specs):
        cid = int(sp["cid"])
        tpl = sp["tpl"]
        tries = cfg.MAX_PLACEMENT_TRIES if i == 0 else cfg.MAX_EXTRA_TRIES
        st = {"cls": {"class_id": cid}, "seq": [tpl], "ptr": 0, "count": 0}
        res = place_one_object(bg, st, template_cache, bboxes, max_tries=tries)
        if res is None:
            if i == 0:
                return None
            continue
        bg, label, _bb = res
        labels.append(label)

    if not labels:
        return None

    bg = effects.apply_weather(bg)
    bg = effects.apply_camera_effects(bg)
    rel_img_path = save_sample(out_dir, bg, unique_idx, labels, id_to_code)
    return rel_img_path, labels


def read_external_data_yaml(yaml_path):
    """Читаем data.yaml внешнего датасета (без PyYAML)."""
    data = {"train": None, "val": None, "names": None, "path": None}
    if not os.path.exists(yaml_path):
        return data

    with open(yaml_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("path:"):
            data["path"] = line.split(":", 1)[1].strip()
        elif line.startswith("train:"):
            data["train"] = line.split(":", 1)[1].strip()
        elif line.startswith("val:"):
            data["val"] = line.split(":", 1)[1].strip()
        elif line.startswith("names:"):
            rhs = line.split(":", 1)[1].strip()
            if rhs.startswith("[") and rhs.endswith("]"):
                try:
                    data["names"] = ast.literal_eval(rhs)
                except Exception:
                    data["names"] = None
            else:
                data["names"] = []

    if isinstance(data.get("names"), list) and data["names"] == []:
        collect = False
        for line in lines:
            if line.strip().startswith("names:"):
                collect = True
                continue
            if collect:
                s = line.strip()
                if not s.startswith("-"):
                    continue
                item = s[1:].strip().strip("'\"")
                data["names"].append(item)

    return data


def external_label_to_image_path(label_path):
    base = os.path.splitext(os.path.basename(label_path))[0]
    img_dir = label_path.replace(os.sep + "labels" + os.sep, os.sep + "images" + os.sep)
    img_dir = os.path.dirname(img_dir)
    for ext in cfg.EXTERNAL_IMAGE_EXTS:
        p = os.path.join(img_dir, base + ext)
        if os.path.exists(p):
            return p
        p2 = os.path.join(img_dir, base + ext.upper())
        if os.path.exists(p2):
            return p2
    return None


def load_external_index(external_dir, our_code_to_id):
    index = {"images": {}, "images_by_class": {}, "instances_per_class": {}, "enabled": False}
    yaml_path = os.path.join(external_dir, "data.yaml")
    meta = read_external_data_yaml(yaml_path)
    names = meta.get("names")
    if not isinstance(names, list) or not names:
        warn(f"Не удалось прочитать names из внешнего data.yaml: {yaml_path}")
        return index

    label_files = glob(os.path.join(external_dir, "**", "labels", "*.txt"), recursive=True)
    label_files = sorted(set(label_files))
    if not label_files:
        warn(f"Во внешнем датасете не найдено label-файлов: {external_dir}")
        return index

    unknown_names = set()
    ambiguous_names = set()
    our_id_set = set(int(v) for v in our_code_to_id.values())

    def map_external_code_to_our_id(code):
        if code in our_code_to_id:
            return int(our_code_to_id[code])

        def _split_suffix_num(s: str):
            """
            Разделяем строку на (prefix, number) по последнему числовому суффиксу.
            Примеры:
            - '1.20.3' -> ('1.20.', 3)
            - '2.3.7'  -> ('2.3.', 7)
            Возвращаем None, если суффикс не распарсился.
            """
            m = re.match(r"^(.*?)(\d+)$", str(s))
            if not m:
                return None
            return m.group(1), int(m.group(2))

        # 1.5) диапазоны вида '1.20.1-1.20.3' (внешний может содержать '1.20.2')
        range_hits = []
        for k, cid in our_code_to_id.items():
            kk = str(k)
            if "-" not in kk:
                continue
            parts = kk.split("-")
            if len(parts) != 2:
                continue
            a, b = parts[0].strip(), parts[1].strip()
            sa = _split_suffix_num(a)
            sb = _split_suffix_num(b)
            sc = _split_suffix_num(code)
            if not sa or not sb or not sc:
                continue
            pa, na = sa
            pb, nb = sb
            pc, nc = sc
            if pa != pb or pa != pc:
                continue
            lo = min(na, nb)
            hi = max(na, nb)
            if lo <= nc <= hi:
                range_hits.append(int(cid))
        range_hits = sorted(set(range_hits))
        if len(range_hits) == 1:
            return range_hits[0]
        if len(range_hits) > 1:
            return None

        token_hits = []
        for k, cid in our_code_to_id.items():
            tokens = re.split(r"[-,; ]+", str(k))
            if code in tokens:
                token_hits.append(int(cid))
        token_hits = sorted(set(token_hits))
        if len(token_hits) == 1:
            return token_hits[0]
        if len(token_hits) > 1:
            return None

        sub_hits = []
        for k, cid in our_code_to_id.items():
            if code and code in str(k):
                sub_hits.append(int(cid))
        sub_hits = sorted(set(sub_hits))
        if len(sub_hits) == 1:
            return sub_hits[0]
        return None

    for lp in label_files:
        img_path = external_label_to_image_path(lp)
        if not img_path:
            continue

        labels = []
        with open(lp, "r", encoding="utf-8") as f:
            for line in f.read().splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                try:
                    ext_idx = int(float(parts[0]))
                    xc = float(parts[1])
                    yc = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])
                except Exception:
                    continue

                if ext_idx < 0 or ext_idx >= len(names):
                    continue

                ext_name = names[ext_idx]
                code = normalize_class_code(ext_name)
                cid = map_external_code_to_our_id(code)
                if cid is None:
                    has_any = any(code and (code in str(k) or code in re.split(r"[-,; ]+", str(k))) for k in our_code_to_id.keys())
                    if has_any:
                        ambiguous_names.add(code)
                    else:
                        unknown_names.add(code)
                    continue

                # защита: берём только те class_id, которые реально есть в shared/signs/signs.csv
                if int(cid) not in our_id_set:
                    unknown_names.add(code)
                    continue

                labels.append((cid, xc, yc, w, h))

        if not labels:
            continue

        index["images"][img_path] = labels
        for (cid, _xc, _yc, _w, _h) in labels:
            index["images_by_class"].setdefault(int(cid), set()).add(img_path)
            index["instances_per_class"][int(cid)] = index["instances_per_class"].get(int(cid), 0) + 1

    if unknown_names:
        warn(f"Внешние классы не сопоставились с нашими (первые 10): {sorted(list(unknown_names))[:10]}")
    if ambiguous_names:
        warn(f"Внешние классы сопоставились неоднозначно (первые 10): {sorted(list(ambiguous_names))[:10]}")

    index["enabled"] = True
    return index


def import_external_images_for_selected(
    out_dir,
    external_index,
    selected_ids,
    id_to_code,
    train_list,
    val_list,
    ann_w,
    unique_idx_ref,
    train_f=None,
    val_f=None,
):
    imported_counts = {}
    imported_class_ids = set()
    imported_images = 0

    images_to_import = set()
    for cid in selected_ids:
        imgs = external_index["images_by_class"].get(int(cid))
        if imgs:
            images_to_import.update(imgs)

    if not images_to_import:
        return imported_counts, imported_class_ids, imported_images

    info(f"Подмешиваем внешние изображения: {len(images_to_import)} шт.")

    for img_path in sorted(images_to_import):
        labels = external_index["images"].get(img_path, [])
        if not labels:
            continue

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            continue

        unique_idx_ref[0] += 1
        unique_idx = unique_idx_ref[0]

        stem = build_image_stem(labels, id_to_code)
        img_file = f"{stem}__ext__{unique_idx:07d}.png"
        out_img_path = os.path.join(out_dir, "images", img_file)
        out_lbl_path = os.path.join(out_dir, "labels", os.path.splitext(img_file)[0] + ".txt")

        ok = cv2.imwrite(
            out_img_path,
            img,
            [int(cv2.IMWRITE_PNG_COMPRESSION), int(getattr(cfg, "PNG_COMPRESSION", 3))],
        )
        if not ok:
            continue

        with open(out_lbl_path, "w", encoding="utf-8") as f:
            for (cid, xc, yc, w, h) in labels:
                f.write(f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

        rel_img_path = os.path.join("images", img_file)
        if random.random() < cfg.VAL_RATIO:
            if val_list is not None:
                val_list.append(rel_img_path)
            elif val_f is not None:
                val_f.write(rel_img_path + "\n")
        else:
            if train_list is not None:
                train_list.append(rel_img_path)
            elif train_f is not None:
                train_f.write(rel_img_path + "\n")

        for (cid, xc, yc, w, h) in labels:
            imported_class_ids.add(int(cid))
            imported_counts[int(cid)] = imported_counts.get(int(cid), 0) + 1
            ann_w.writerow([rel_img_path, cid, f"{xc:.6f}", f"{yc:.6f}", f"{w:.6f}", f"{h:.6f}"])

        imported_images += 1

    return imported_counts, imported_class_ids, imported_images


def load_negative_image_paths():
    """Список путей к негативным изображениям (можно рекурсивно)."""
    return list_files_by_ext_recursive(cfg.NEGATIVE_IMAGES_DIR, cfg.NEGATIVE_IMAGE_EXTS)


def import_negative_images(
    out_dir,
    negative_paths,
    unique_idx_ref,
    train_list=None,
    val_list=None,
    train_f=None,
    val_f=None,
):
    """
    Копируем/конвертируем негативные изображения в датасет:
    - пишем PNG в images/
    - создаём пустой label-файл
    - добавляем в train/val
    Возвращаем количество реально добавленных негативов.
    """
    added = 0
    for src in negative_paths:
        img = cv2.imread(src, cv2.IMREAD_COLOR)
        if img is None:
            continue

        unique_idx_ref[0] += 1
        unique_idx = unique_idx_ref[0]

        img_file = f"negative__{unique_idx:07d}.png"
        out_img_path = os.path.join(out_dir, "images", img_file)
        out_lbl_path = os.path.join(out_dir, "labels", os.path.splitext(img_file)[0] + ".txt")

        ok = cv2.imwrite(
            out_img_path,
            img,
            [int(cv2.IMWRITE_PNG_COMPRESSION), int(getattr(cfg, "PNG_COMPRESSION", 3))],
        )
        if not ok:
            continue

        with open(out_lbl_path, "w", encoding="utf-8") as f:
            f.write("")

        rel_img_path = os.path.join("images", img_file)
        if random.random() < cfg.VAL_RATIO:
            if val_list is not None:
                val_list.append(rel_img_path)
            elif val_f is not None:
                val_f.write(rel_img_path + "\n")
        else:
            if train_list is not None:
                train_list.append(rel_img_path)
            elif train_f is not None:
                train_f.write(rel_img_path + "\n")

        added += 1

    return added


def generate_dataset():
    """
    Оставляем публичную функцию `generate_dataset()` здесь для обратной совместимости,
    но основная оркестрация теперь находится в `make_dataset/generate_synth_yolo_dataset.py`.
    """
    from make_dataset.generate_synth_yolo_dataset import generate_dataset as _generate_dataset

    return _generate_dataset()


def fmt_progress(made: int, target: int) -> str:
    if target <= 0:
        return f"{made}/{target}"
    pct = 100.0 * float(made) / float(target)
    return f"{made}/{target} ({pct:.1f}%)"


def count_dir_files(dir_path: str, exts_lower: tuple) -> int:
    try:
        return sum(1 for n in os.listdir(dir_path) if os.path.splitext(n)[1].lower() in exts_lower)
    except Exception:
        return 0


def write_data_stats(
    out_dir: str,
    all_classes: list,
    id_to_code: dict,
    imported_counts: dict,
    generated_counts: dict,
    images_count: int,
    label_files_count: int,
):
    imported_total = int(sum(int(v) for v in (imported_counts or {}).values()))
    generated_total = int(sum(int(v) for v in (generated_counts or {}).values()))
    total_objects = imported_total + generated_total

    lines = []
    lines.append("Детальная статистика датасета")
    lines.append("")
    lines.append("Сводка:")
    lines.append(f"- Сколько объектов сгенерировано: {generated_total}")
    lines.append(f"- Сколько объектов взято из внешнего датасета: {imported_total}")
    lines.append(f"- Сколько всего объектов: {total_objects}")
    lines.append(f"- Сколько изображений в датасете: {images_count}")
    lines.append(f"- Сколько лейблов (label-файлов): {label_files_count}")
    lines.append("")
    lines.append("Таблица по классам (только классы с total > 0):")
    lines.append("class_id\tcode\tclass_name\texternal\tgenerated\ttotal")

    rows = []
    for c in sorted(all_classes, key=lambda x: int(x["class_id"])):
        cid = int(c["class_id"])
        ext_n = int((imported_counts or {}).get(cid, 0))
        gen_n = int((generated_counts or {}).get(cid, 0))
        tot = ext_n + gen_n
        if tot <= 0:
            continue
        code = str(id_to_code.get(cid, c.get("name", f"class_{cid}")))
        rows.append((cid, code, ext_n, gen_n, tot))

    for (cid, code, ext_n, gen_n, tot) in rows:
        lines.append(f"{cid}\t{code}\t{ext_n}\t{gen_n}\t{tot}")

    lines.append("")
    lines.append(f"Всего классов с total > 0: {len(rows)}")

    stats_path = os.path.join(out_dir, "data_stats.txt")
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")

