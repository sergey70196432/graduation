"""
Идея:
- PNG-кропы знаков в папке `shared/signs/speed_png/pngs`.
- Внутри лежат подпапки с именем класса, например `3.24_60` (base_speed).
- Скрипт собирает структуру, удобную для PyTorch ImageFolder:
  datasets/speed_cls_v1/train/<class>/*.png
  datasets/speed_cls_v1/val/<class>/*.png
  datasets/speed_cls_v1/test/<class>/*.png
- Пишет labels.txt (список классов) и stats.txt (статистика).

Важно про `labels.txt`:
- Это "источник истины" для порядка классов (см. training/speed_classifier/dataset.py).
- Если порядок классов меняется, индексы на выходе модели начинают соответствовать другим строкам,
  и в приложении/экспорте появляются "не те" классы.

Про версии датасета:
- По умолчанию каждый запуск создаёт НОВУЮ папку `datasets/speed_cls_v<n>` (v1, v2, v3, ...),
  чтобы не затирать предыдущие версии и иметь воспроизводимость.
"""

import json
import os
import random
import re
import shutil
import sys

import cv2
import numpy as np

# Позволяет запускать так:
#   python make_dataset/generate_speed_classifier_dataset.py
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ============================================================
# ПАРАМЕТРЫ
# ============================================================

# Откуда брать исходные PNG-кропы (подпапки = классы, например 3.24_60)
SRC_ROOT = "shared/signs/speed_png/pngs"

# Куда сохранить итоговый датасет.
# Мы хотим, чтобы каждый запуск генерировал НОВУЮ версию (speed_cls_v1, speed_cls_v2, ...),
# чтобы не затирать предыдущие датасеты и не ловить рассинхроны.
#
# - Если AUTO_VERSION=True, то OUT_ROOT задаётся как префикс без номера, например:
#     OUT_ROOT_PREFIX="datasets/speed_cls_v"
#   и скрипт сам выберет следующую свободную папку speed_cls_v<n>.
# - Если AUTO_VERSION=False, то используем OUT_ROOT_PREFIX как полный путь (как раньше).
AUTO_VERSION = True
OUT_ROOT_PREFIX = "datasets/speed_cls_v"

# Размер входа классификатора (квадрат)
IMAGE_SIZE = 128

# Сохранять ли выходные картинки в фиксированном размере IMAGE_SIZE.
# - False: сохраняем в "родном" размере (разные разрешения реально используются)
# - True : все выходные картинки будут IMAGE_SIZE x IMAGE_SIZE
RESIZE_OUTPUT_TO_IMAGE_SIZE = False

# Если RESIZE_OUTPUT_TO_IMAGE_SIZE=False, то всё равно ограничим максимальный размер,
# чтобы файлы не были слишком большими.
MAX_OUTPUT_SIDE = 512

# Сплиты
VAL_RATIO = 0.20
TEST_RATIO = 0.10

# Балансировка классов.
# Проблема: в SRC_ROOT у разных классов разное число исходных PNG, из-за этого train получается несбалансированным.
# Решение: генерировать одинаковое число примеров на класс.
#
# Как именно балансируем:
# - downsample: берём "как у самого маленького класса" (датасет меньше, но без перекоса)
# - upsample  : берём "как у самого большого класса" (датасет больше, но используем всё)
BALANCE_CLASSES = True
BALANCE_STRATEGY = "upsample"  # 'downsample' | 'upsample'

# Можно задать явные количества (0 = авто).
# TRAIN_BASE_PER_CLASS — сколько РАЗНЫХ исходных картинок (из SRC_ROOT) взять в train на класс
# (потом для каждой делается TRAIN_COPIES_PER_IMAGE и bad-crops).
TRAIN_BASE_PER_CLASS = 0

# Сколько файлов выбрать в val/test на класс (0 = авто из VAL_RATIO/TEST_RATIO и train-base target).
VAL_PER_CLASS = 0
TEST_PER_CLASS = 0

# Минимум картинок на класс в val/test.
# Если исходных картинок мало, скрипт будет дублировать (это нормально для простого пайплайна).
MIN_VAL_PER_CLASS = 10
MIN_TEST_PER_CLASS = 10

# Сид для воспроизводимости
SEED = 1337

# Если True — перед запуском удаляем папку вывода целиком (чтобы не смешивать старые/новые файлы).
# Когда AUTO_VERSION=True, папка вывода всегда новая, поэтому удаление обычно не нужно.
CLEAN_OUTPUT_DIR = True

# Ограничение исходных картинок на класс (0 = без лимита)
MAX_PER_CLASS = 0

# Сколько "вариантов" сделать из одной картинки в train.
# Вариант = тот же знак, но с другим фоном (и размером/паддингом как есть).
TRAIN_COPIES_PER_IMAGE = 24

# Максимум разных исходных картинок на класс, которые пойдут в train (0 = все).
# Удобно, если у вас очень много исходников, и датасет получается слишком большой.
MAX_TRAIN_BASE_PER_CLASS = 0

# Небольшая деградация качества прямо в генераторе.
# Держим её умеренной, но уже ближе к реальным видео-ROI.
GEN_BLUR_PROB = 0.28
GEN_BLUR_SIGMA_MIN = 0.2
GEN_BLUR_SIGMA_MAX = 1.6
GEN_NOISE_PROB = 0.32
GEN_NOISE_STD_MIN = 0.0
GEN_NOISE_STD_MAX = 9.0  # в uint8

# Генерация "плохих кропов" (только для train).
# Имитируем ситуации, когда YOLO дал неточный ROI:
# - знак занимает меньшую часть кадра (много фона),
# - знак смещён (не по центру),
# - края чуть обрезаны,
# - качество похоже на видео (JPEG/смаз).
ENABLE_BAD_CROPS = True
# Доля train-копий, которые превращаем в "bad crop"
BAD_CROP_PROB = 0.55
# Дополнительно "плохих" вариантов на одну исходную картинку (помимо TRAIN_COPIES_PER_IMAGE)
BAD_CROP_EXTRA_PER_IMAGE = 4

# "Zoom-out" (знак меньше в кадре): масштаб уменьшенной вставки относительно исходного размера
BAD_CROP_MIN_SCALE = 0.40
BAD_CROP_MAX_SCALE = 0.92

# Подрезание краёв (имитация промаха bbox)
BAD_CROP_EDGE_CUT_PROB = 0.38
BAD_CROP_MAX_CUT_FRAC = 0.18

# JPEG-артефакты
BAD_CROP_JPEG_PROB = 0.35
BAD_CROP_JPEG_Q_MIN = 18
BAD_CROP_JPEG_Q_MAX = 82

# Motion blur
BAD_CROP_MOTION_BLUR_PROB = 0.28
BAD_CROP_MOTION_BLUR_K_MIN = 5
BAD_CROP_MOTION_BLUR_K_MAX = 15

# Реальные фоны (важно для качества на реальных ROI).
# Если True — будем класть знак не на серый фон, а на случайный кроп из реальных фотографий/кадров.
# Это сильно снижает доменный сдвиг между синтетикой и реальными ROI.
USE_REAL_BACKGROUNDS = True
BACKGROUND_DIRS = [
    "make_dataset/dashcam_frames",
    "make_dataset/backgrounds",
]
# Сколько фонов держать в RAM (простая LRU-кешировка чтения).
BG_CACHE_MAX_ITEMS = 64
# Если фон очень большой — уменьшаем для скорости (0 = не ограничивать).
BG_MAX_SIDE_FOR_CACHE = 1280

# Добавление реальных ROI в train поверх синтетики.
# Это позволяет уменьшить доменный сдвиг, сохранив ту же 105-классовую постановку.
INCLUDE_REAL_ROI_IN_TRAIN = True
REAL_ROI_INDEX_JSONL = "speed_test_roi/roi_index.jsonl"
# 1 = один раз добавить каждый ROI, 40 = агрессивный overweight реальных примеров.
REAL_ROI_REPEAT = 40


# ============================================================
# РЕАЛИЗАЦИЯ
# ============================================================

_CLASS_RE = re.compile(r"^(?P<base>[\d.]+)_(?P<speed>\d+)$")


def pick_next_version_dir(prefix_path: str) -> str:
    """
    Выбираем следующую папку вида "<prefix><n>".

    Пример:
      prefix_path="datasets/speed_cls_v"
      если существуют speed_cls_v1, speed_cls_v2, speed_cls_v7 -> вернём speed_cls_v8
      если не существует ничего -> вернём speed_cls_v1
    """
    prefix_path = os.path.abspath(str(prefix_path))
    parent = os.path.dirname(prefix_path)
    base_prefix = os.path.basename(prefix_path)
    os.makedirs(parent, exist_ok=True)

    best = 0
    for name in os.listdir(parent):
        if not name.startswith(base_prefix):
            continue
        tail = name[len(base_prefix) :]
        if not tail.isdigit():
            continue
        n = int(tail)
        if n > best:
            best = n
    return os.path.join(parent, f"{base_prefix}{best + 1}")


def is_image_file(name: str) -> bool:
    """
    Быстрая проверка по расширению, что файл похож на картинку.

    Используется при рекурсивном обходе папок классов, чтобы собрать список исходных изображений.
    """
    n = name.lower()
    return n.endswith(".png") or n.endswith(".jpg") or n.endswith(".jpeg") or n.endswith(".webp")


def iter_images_recursive(root: str):
    """
    Рекурсивно обходим папку и возвращаем пути до всех изображений.

    Почему рекурсивно:
    - в исходных данных иногда бывает вложенная структура (например разные размеры в подпапках),
      и мы хотим собрать всё.
    """
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for fn in sorted(filenames):
            if is_image_file(fn):
                yield os.path.join(dirpath, fn)


def parse_class_name(folder_name: str):
    """
    Ожидаем имя класса вида "<base>_<speed>", например "3.24_60".
    Возвращаем (base, speed) или None, если формат не подошёл.
    """
    m = _CLASS_RE.match(str(folder_name).strip())
    if not m:
        return None
    return m.group("base"), int(m.group("speed"))


def class_sort_key(class_name: str):
    """
    Ключ сортировки классов для labels.txt и для прохода по классам.

    Мы хотим "человеческий" порядок:
    - сначала по base (например 3.24, 3.25, 4.6, ...)
    - потом по speed как числу (10, 20, 30, ..., 100, 110, ...)

    Это важно, потому что лексикографическая сортировка строк даёт порядок 10, 100, 110, ..., 20, ...
    """
    parsed = parse_class_name(class_name)
    if not parsed:
        return (class_name, 10**9)
    base, speed = parsed
    return (base, speed)


def ensure_dir(p: str):
    """
    Создаём директорию (и родителей), если её нет.
    """
    os.makedirs(p, exist_ok=True)


def read_rgba(path: str) -> np.ndarray:
    """
    Читаем картинку как RGBA uint8.
    Если альфы нет, делаем альфу 255.
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Не удалось прочитать изображение: {path}")

    if img.ndim != 3:
        raise RuntimeError(f"Неожиданная форма изображения: {img.shape} ({path})")

    if img.shape[2] == 4:
        # BGRA -> RGBA
        b, g, r, a = cv2.split(img)
        return cv2.merge([r, g, b, a])

    if img.shape[2] == 3:
        # BGR -> RGBA (opaque)
        b, g, r = cv2.split(img)
        a = np.full_like(b, 255)
        return cv2.merge([r, g, b, a])

    raise RuntimeError(f"Неожиданное число каналов: {img.shape[2]} ({path})")


def rgba_to_bgr_over_solid(rgba: np.ndarray, bgr_color: tuple[int, int, int]) -> np.ndarray:
    """
    Накладываем RGBA на сплошной фон (BGR).
    Выход: BGR uint8.

    Важно:
    - вход rgba в формате RGBA (как читаем через read_rgba)
    - OpenCV по умолчанию работает в BGR, поэтому фон задаём как BGR,
      а rgb из rgba разворачиваем в bgr при смешивании.
    """
    h, w = rgba.shape[:2]
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    bg[:, :] = np.array(bgr_color, dtype=np.uint8)[None, None, :]

    rgb = rgba[:, :, :3].astype(np.float32)
    a = (rgba[:, :, 3:4].astype(np.float32)) / 255.0

    # rgb -> bgr
    out = bg.astype(np.float32) * (1.0 - a) + rgb[:, :, ::-1] * a
    return np.clip(out, 0, 255).astype(np.uint8)


def resize_to_square(bgr: np.ndarray, out_size: int) -> np.ndarray:
    """
    Делаем квадрат: если картинка прямоугольная — дополняем краями,
    затем ресайзим до out_size x out_size.

    "Дополняем краями" (BORDER_REPLICATE) вместо чёрных полей:
    - чтобы не добавлять в датасет неестественные рамки
    - чтобы край знака выглядел ближе к реальному кропу
    """
    h, w = bgr.shape[:2]
    if h == w:
        return cv2.resize(bgr, (out_size, out_size), interpolation=cv2.INTER_AREA)

    m = max(h, w)
    pad_y = (m - h) // 2
    pad_x = (m - w) // 2
    padded = cv2.copyMakeBorder(
        bgr,
        top=pad_y,
        bottom=m - h - pad_y,
        left=pad_x,
        right=m - w - pad_x,
        borderType=cv2.BORDER_REPLICATE,
    )
    return cv2.resize(padded, (out_size, out_size), interpolation=cv2.INTER_AREA)


def pad_to_square(bgr: np.ndarray) -> np.ndarray:
    """
    Делаем квадрат без ресайза: дополняем края (border replicate).

    Зачем:
    - если RESIZE_OUTPUT_TO_IMAGE_SIZE=False, мы хотим сохранить исходное (разное) разрешение,
      но при этом приводим к квадрату, чтобы дальше проще было резать ROI и учить модель.
    """
    h, w = bgr.shape[:2]
    if h == w:
        return bgr
    m = max(h, w)
    pad_y = (m - h) // 2
    pad_x = (m - w) // 2
    padded = cv2.copyMakeBorder(
        bgr,
        top=pad_y,
        bottom=m - h - pad_y,
        left=pad_x,
        right=m - w - pad_x,
        borderType=cv2.BORDER_REPLICATE,
    )
    return padded


def limit_max_side(bgr: np.ndarray, max_side: int) -> np.ndarray:
    """
    Если картинка слишком большая — уменьшаем так, чтобы max(h,w) == max_side.

    Это нужно, когда RESIZE_OUTPUT_TO_IMAGE_SIZE=False:
    - иначе можно получить очень большие файлы (медленно читать и много места на диске).
    """
    h, w = bgr.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return bgr
    scale = float(max_side) / float(m)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

def random_bg_color(rng: random.Random) -> tuple[int, int, int]:
    """
    Простой серый фон (BGR) с небольшим шумом.

    Мы специально используем "не яркий" фон:
    - на реальных кадрах фон не бывает идеально белым,
    - и нам важно, чтобы модель не переобучилась на один конкретный цвет.
    """
    v = int(rng.uniform(25, 230))
    jitter = int(rng.uniform(-18, 18))
    v2 = max(0, min(255, v + jitter))
    v3 = max(0, min(255, v - jitter))
    return (v, v2, v3)


# Пул фоновых картинок и кеш чтения.
_BG_PATHS: list[str] = []
_BG_CACHE: dict[str, np.ndarray] = {}
_BG_CACHE_KEYS: list[str] = []
_NP_RNG: np.random.Generator | None = None


def _get_np_rng() -> np.random.Generator:
    if _NP_RNG is None:
        raise RuntimeError("NumPy RNG is not initialized. Call main() first.")
    return _NP_RNG


def _read_bgr_cached(path: str) -> np.ndarray | None:
    """
    Читаем фон (BGR) с простым кешем, чтобы не читать один и тот же файл много раз.
    """
    p = str(path)
    if p in _BG_CACHE:
        return _BG_CACHE[p]
    img = cv2.imread(p, cv2.IMREAD_COLOR)
    if img is None:
        return None
    if int(BG_MAX_SIDE_FOR_CACHE) > 0:
        img = limit_max_side(img, int(BG_MAX_SIDE_FOR_CACHE))
    # LRU eviction
    _BG_CACHE[p] = img
    _BG_CACHE_KEYS.append(p)
    if len(_BG_CACHE_KEYS) > int(BG_CACHE_MAX_ITEMS):
        k = _BG_CACHE_KEYS.pop(0)
        _BG_CACHE.pop(k, None)
    return img


def _pick_background_patch(size: int, rng: random.Random) -> np.ndarray:
    """
    Возвращаем квадратный BGR-патч размера size x size.
    Если фоновых картинок нет — используем серый фон.
    """
    s = int(max(8, size))
    if not _BG_PATHS:
        bg = np.zeros((s, s, 3), dtype=np.uint8)
        bg[:, :] = np.array(random_bg_color(rng), dtype=np.uint8)[None, None, :]
        return bg

    # Несколько попыток найти читаемый фон.
    for _ in range(6):
        p = rng.choice(_BG_PATHS)
        img = _read_bgr_cached(p)
        if img is None:
            continue
        h, w = img.shape[:2]
        if h < 8 or w < 8:
            continue
        # Если фон меньше нужного — апскейлим (лучше так, чем падать).
        if h < s or w < s:
            img2 = cv2.resize(img, (max(s, w), max(s, h)), interpolation=cv2.INTER_LINEAR)
            img = img2
            h, w = img.shape[:2]
        x0 = int(rng.randint(0, max(0, w - s)))
        y0 = int(rng.randint(0, max(0, h - s)))
        patch = img[y0 : y0 + s, x0 : x0 + s].copy()
        return patch

    bg = np.zeros((s, s, 3), dtype=np.uint8)
    bg[:, :] = np.array(random_bg_color(rng), dtype=np.uint8)[None, None, :]
    return bg


def rgba_over_bgr_patch(rgba: np.ndarray, bgr_patch: np.ndarray) -> np.ndarray:
    """
    Накладываем RGBA-объект на готовый BGR-патч того же размера.
    """
    h, w = rgba.shape[:2]
    if bgr_patch.shape[0] != h or bgr_patch.shape[1] != w:
        bgr_patch = cv2.resize(bgr_patch, (w, h), interpolation=cv2.INTER_LINEAR)
    rgb = rgba[:, :, :3].astype(np.float32)
    a = (rgba[:, :, 3:4].astype(np.float32)) / 255.0
    out = bgr_patch.astype(np.float32) * (1.0 - a) + rgb[:, :, ::-1] * a
    return np.clip(out, 0, 255).astype(np.uint8)


def _jpeg_compress_bgr(bgr: np.ndarray, quality: int) -> np.ndarray:
    """
    JPEG encode/decode через OpenCV (BGR).

    Имитируем артефакты видео/пересжатия:
    - кодируем в JPEG (с quality)
    - декодируем обратно в BGR
    """
    q = int(max(1, min(100, quality)))
    ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(q)])
    if not ok:
        return bgr
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return bgr if dec is None else dec


def _motion_blur_bgr(bgr: np.ndarray, rng: random.Random, k_min: int, k_max: int) -> np.ndarray:
    """
    Простой motion blur: линейное ядро, повернутое на случайный угол.

    Это имитация смаза:
    - движение камеры
    - движение объекта
    - сильное размытие при низком FPS/освещении
    """
    h, w = bgr.shape[:2]
    if h < 3 or w < 3:
        return bgr

    k = int(rng.randint(int(k_min), int(k_max)))
    if k % 2 == 0:
        k += 1
    k = max(3, min(31, k))
    angle = float(rng.uniform(0.0, 180.0))

    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((k / 2.0 - 0.5, k / 2.0 - 0.5), angle, 1.0)
    kernel = cv2.warpAffine(kernel, M, (k, k))
    s = float(kernel.sum())
    if s > 1e-6:
        kernel /= s
    out = cv2.filter2D(bgr, -1, kernel)
    return out


def _zoom_out_to_background(bgr: np.ndarray, rng: random.Random, min_scale: float, max_scale: float) -> np.ndarray:
    """
    Уменьшаем картинку и вставляем в случайную позицию на фоне того же размера.

    Это имитация "плохого ROI":
    - знак занимает меньшую часть кропа
    - знак смещён (bbox промахнулся)
    - вокруг появляется фон
    """
    h, w = bgr.shape[:2]
    if h < 8 or w < 8:
        return bgr

    s = float(rng.uniform(float(min_scale), float(max_scale)))
    s = max(0.10, min(1.0, s))
    nh = max(2, int(round(h * s)))
    nw = max(2, int(round(w * s)))
    if nh >= h and nw >= w:
        return bgr

    small = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)

    bg = np.zeros((h, w, 3), dtype=np.uint8)
    bg[:, :] = np.array(random_bg_color(rng), dtype=np.uint8)[None, None, :]
    # лёгкий шум на фон (через numpy, он уже seeded выше)
    n = _get_np_rng().normal(0.0, 6.0, size=bg.shape).astype(np.float32)
    bg = np.clip(bg.astype(np.float32) + n, 0, 255).astype(np.uint8)

    x0 = int(rng.randint(0, max(0, w - nw)))
    y0 = int(rng.randint(0, max(0, h - nh)))
    bg[y0 : y0 + nh, x0 : x0 + nw] = small
    return bg


def _edge_cut_and_resize_back(bgr: np.ndarray, rng: random.Random, max_cut_frac: float) -> np.ndarray:
    """
    Случайно обрезаем края и ресайзим назад (имитация промаха bbox).

    Идея:
    - bbox иногда "съедает" часть знака
    - после обрезания мы ресайзим обратно, чтобы сохранить размер входа модели
    """
    h, w = bgr.shape[:2]
    if h < 16 or w < 16:
        return bgr

    mcf = float(max(0.0, min(0.45, max_cut_frac)))
    max_l = int(round(w * mcf))
    max_r = int(round(w * mcf))
    max_t = int(round(h * mcf))
    max_b = int(round(h * mcf))

    l = int(rng.randint(0, max(0, max_l)))
    r = int(rng.randint(0, max(0, max_r)))
    t = int(rng.randint(0, max(0, max_t)))
    b = int(rng.randint(0, max(0, max_b)))

    new_w = w - l - r
    new_h = h - t - b
    if new_w < max(8, int(round(w * 0.65))) or new_h < max(8, int(round(h * 0.65))):
        return bgr

    cropped = bgr[t : t + new_h, l : l + new_w]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_AREA)


def make_bad_crop_example(bgr: np.ndarray, rng: random.Random) -> np.ndarray:
    """
    Превращаем "нормальный" пример в "плохой кроп".

    Порядок шагов важен:
    1) сначала делаем "больше фона" (zoom-out), потому что это меняет композицию
    2) потом иногда подрезаем края (edge cut)
    3) и в конце добавляем "качество видео" (смаз + JPEG)
    """
    out = bgr

    # 1) знак становится меньше в кадре + смещение
    out = _zoom_out_to_background(out, rng, float(BAD_CROP_MIN_SCALE), float(BAD_CROP_MAX_SCALE))

    # 2) иногда подрезаем края
    if rng.random() < float(BAD_CROP_EDGE_CUT_PROB):
        out = _edge_cut_and_resize_back(out, rng, float(BAD_CROP_MAX_CUT_FRAC))

    # 3) немного "видео" деградации
    if rng.random() < float(BAD_CROP_MOTION_BLUR_PROB):
        out = _motion_blur_bgr(out, rng, int(BAD_CROP_MOTION_BLUR_K_MIN), int(BAD_CROP_MOTION_BLUR_K_MAX))
    if rng.random() < float(BAD_CROP_JPEG_PROB):
        q = int(rng.randint(int(BAD_CROP_JPEG_Q_MIN), int(BAD_CROP_JPEG_Q_MAX)))
        out = _jpeg_compress_bgr(out, q)

    return out


def make_plain_example(rgba: np.ndarray, out_size: int, rng: random.Random) -> np.ndarray:
    """
    Самый простой вариант: RGBA -> BGR на случайном сером фоне + квадратный resize.

    Это "базовый" генератор примера без сложной синтетики:
    - берём прозрачный PNG (кроп знака)
    - кладём на простой фон (чтобы убрать альфу и получить обычную RGB-картинку)
    - приводим к квадрату
    - опционально ресайзим до фиксированного размера
    """
    # 1) Делаем квадратный RGBA (важно: альфа должна соответствовать новой геометрии).
    # Примечание: pad_to_square выше работает для BGR, поэтому для RGBA делаем вручную через BORDER_REPLICATE.
    h, w = rgba.shape[:2]
    m = max(h, w)
    pad_y = (m - h) // 2
    pad_x = (m - w) // 2
    rgba_sq = cv2.copyMakeBorder(
        rgba,
        top=pad_y,
        bottom=m - h - pad_y,
        left=pad_x,
        right=m - w - pad_x,
        borderType=cv2.BORDER_REPLICATE,
    )

    # 2) Выбираем фон (реальный или серый) и накладываем RGBA.
    if bool(USE_REAL_BACKGROUNDS):
        bg = _pick_background_patch(int(rgba_sq.shape[0]), rng)
        bgr = rgba_over_bgr_patch(rgba_sq, bg)
    else:
        bgr = rgba_to_bgr_over_solid(rgba_sq, random_bg_color(rng))

    # 3) Опциональный фиксированный resize до out_size.
    if bool(RESIZE_OUTPUT_TO_IMAGE_SIZE):
        return cv2.resize(bgr, (int(out_size), int(out_size)), interpolation=cv2.INTER_AREA)

    # 4) Ограничим максимальный размер, чтобы не раздувать датасет.
    bgr = limit_max_side(bgr, int(MAX_OUTPUT_SIDE))

    # Лёгкая деградация качества (очень простая)
    if rng.random() < float(GEN_BLUR_PROB):
        sigma = rng.uniform(float(GEN_BLUR_SIGMA_MIN), float(GEN_BLUR_SIGMA_MAX))
        bgr = cv2.GaussianBlur(bgr, (0, 0), sigmaX=float(sigma))
    if rng.random() < float(GEN_NOISE_PROB):
        std = rng.uniform(float(GEN_NOISE_STD_MIN), float(GEN_NOISE_STD_MAX))
        if std > 0:
            n = _get_np_rng().normal(0.0, float(std), size=bgr.shape).astype(np.float32)
            bgr = np.clip(bgr.astype(np.float32) + n, 0, 255).astype(np.uint8)

    return bgr


def write_labels_txt(out_root: str, class_names_sorted: list[str]):
    """
    Пишем labels.txt — список классов, 1 строка = 1 класс.

    Важно:
    - порядок строк = порядок индексов классов (class_to_idx)
    - этот порядок должен совпадать между:
      - датасетом
      - чекпоинтом (ckpt["classes"])
      - export_to_tflite.py
      - приложением (labels.txt рядом с .tflite)
    """
    with open(os.path.join(out_root, "labels.txt"), "w", encoding="utf-8") as f:
        for c in class_names_sorted:
            f.write(c + "\n")


def make_splits(imgs: list[str], rng: random.Random) -> dict[str, list[str]]:
    """
    Делаем сплиты "независимо", чтобы train не терял разнообразие:
    - train: берём (почти) все исходные картинки (или ограничиваем MAX_TRAIN_BASE_PER_CLASS)
    - val/test: семплим нужное число картинок С ПОВТОРАМИ из исходного пула

    Почему так:
    - если у класса мало исходных картинок, и мы делим 80/10/10,
      то в train может остаться 1 картинка => модель не видит разных разрешений.
    - независимые сплиты позволяют train видеть максимум вариантов,
      а val/test дают стабильную метрику.
    """
    n = len(imgs)
    if n <= 0:
        return {"train": [], "val": [], "test": []}

    raise RuntimeError("make_splits() больше не используется напрямую. Используйте make_splits_balanced().")


def _sample_exact(imgs: list[str], rng: random.Random, k: int, *, replace: bool) -> list[str]:
    """
    Выбираем ровно k элементов из imgs:
    - replace=False: без повторов (если imgs >= k)
    - replace=True : с повторами (если imgs < k или если так явно попросили)
    """
    k = int(max(0, k))
    if k <= 0:
        return []
    if not imgs:
        return []
    if (not replace) and len(imgs) >= k:
        tmp = list(imgs)
        rng.shuffle(tmp)
        return tmp[:k]
    return [rng.choice(imgs) for _ in range(k)]


def make_splits_balanced(
    imgs: list[str],
    rng: random.Random,
    *,
    train_base_k: int,
    val_k: int,
    test_k: int,
    downsample: bool,
) -> dict[str, list[str]]:
    """
    Балансированные сплиты: одинаковое число элементов на каждый класс.

    Важно:
    - train/val/test выбираются независимо
    - train_base_k — это число исходных картинок (до TRAIN_COPIES_PER_IMAGE и bad-crops)
    - val/test семплим с повторами (это нормально для маленьких классов)
    """
    n = len(imgs)
    if n <= 0:
        return {"train": [], "val": [], "test": []}

    train = _sample_exact(imgs, rng, int(train_base_k), replace=(not downsample))
    if not train:
        train = [rng.choice(imgs)]

    val = _sample_exact(imgs, rng, int(val_k), replace=True) if int(val_k) > 0 else []
    test = _sample_exact(imgs, rng, int(test_k), replace=True) if int(test_k) > 0 else []

    return {"train": train, "val": val, "test": test}


def main():
    # ============================================================
    # 0) Инициализация (сид, пути, базовые проверки параметров)
    # ============================================================
    rng = random.Random(int(SEED))
    # Чтобы результат был воспроизводимым, держим отдельный numpy RNG.
    global _NP_RNG
    _NP_RNG = np.random.default_rng(int(SEED))
    src_root = os.path.abspath(SRC_ROOT)
    # Выбор папки вывода.
    # По умолчанию каждый запуск — новая версия speed_cls_v<n>.
    if bool(AUTO_VERSION):
        out_root = pick_next_version_dir(str(OUT_ROOT_PREFIX))
    else:
        out_root = os.path.abspath(str(OUT_ROOT_PREFIX))

    if not os.path.isdir(src_root):
        raise FileNotFoundError(f"Не найдена папка с PNG-кропами: {src_root}")

    # ============================================================
    # 0.5) Собираем список фоновых картинок (если включено)
    # ============================================================
    global _BG_PATHS
    _BG_PATHS = []
    if bool(USE_REAL_BACKGROUNDS):
        for d in BACKGROUND_DIRS:
            p = os.path.abspath(str(d))
            if not os.path.isdir(p):
                continue
            _BG_PATHS.extend(list(iter_images_recursive(p)))
        # Чтобы генерация была воспроизводимее, фиксируем порядок, а выбор делаем через rng.
        _BG_PATHS = sorted(list({str(x) for x in _BG_PATHS}))
        if not _BG_PATHS:
            print("[WARN] USE_REAL_BACKGROUNDS=True, но фоновые картинки не найдены. Будет серый фон.")

    if VAL_RATIO < 0 or TEST_RATIO < 0 or (VAL_RATIO + TEST_RATIO) >= 1.0:
        raise ValueError("Плохие значения VAL_RATIO / TEST_RATIO (нужно VAL+TEST < 1).")

    # ============================================================
    # 1) Сканируем классы в SRC_ROOT (подпапки = классы)
    # ============================================================
    class_dirs = []
    for name in sorted(os.listdir(src_root)):
        p = os.path.join(src_root, name)
        if os.path.isdir(p):
            class_dirs.append((name, p))
    if not class_dirs:
        raise RuntimeError(f"В {src_root} не найдено подпапок-классов.")

    # ============================================================
    # 2) Фильтруем классы по "разрешённым" скоростям (если вдруг в SRC_ROOT есть лишнее)
    # ============================================================
    want_speeds = set(range(10, 151, 10))
    filtered = []
    for cname, p in class_dirs:
        parsed = parse_class_name(cname)
        if not parsed:
            continue
        _base, speed = parsed
        if speed in want_speeds:
            filtered.append((cname, p))
    if filtered:
        class_dirs = filtered

    # ============================================================
    # 3) Собираем список файлов по каждому классу.
    # В labels.txt попадут только классы, у которых есть реальные изображения.
    # ============================================================
    class_to_imgs: dict[str, list[str]] = {}
    for class_name, class_dir in class_dirs:
        imgs = list(iter_images_recursive(class_dir))
        if not imgs:
            print("[WARN] В классе нет изображений, пропускаю:", class_name)
            continue
        rng.shuffle(imgs)
        if MAX_PER_CLASS and int(MAX_PER_CLASS) > 0:
            imgs = imgs[: int(MAX_PER_CLASS)]
        class_to_imgs[class_name] = imgs

    # Порядок классов фиксируем через class_sort_key (см. комментарий внутри).
    class_names_sorted = sorted(list(class_to_imgs.keys()), key=class_sort_key)
    if not class_names_sorted:
        raise RuntimeError("Не найдено ни одного класса с изображениями (проверь SRC_ROOT).")

    # ============================================================
    # 3.5) Балансировка: выбираем одинаковые размеры сплитов для всех классов
    # ============================================================
    lens = [len(class_to_imgs[c]) for c in class_names_sorted]
    min_n = min(lens) if lens else 0
    max_n = max(lens) if lens else 0

    # Учитываем ограничение MAX_TRAIN_BASE_PER_CLASS (оно применяется к train-пулу).
    cap = int(MAX_TRAIN_BASE_PER_CLASS) if int(MAX_TRAIN_BASE_PER_CLASS) > 0 else None
    if cap is not None:
        min_n = min(min_n, cap)
        max_n = min(max_n, cap)

    strategy = str(BALANCE_STRATEGY).strip().lower()
    if strategy not in ("downsample", "upsample"):
        raise ValueError("BALANCE_STRATEGY должен быть 'downsample' или 'upsample'.")
    downsample = strategy == "downsample"

    if not bool(BALANCE_CLASSES):
        # Старое поведение (небалансированное): размеры зависят от размера класса.
        # Оставляем как опцию, но по умолчанию лучше балансировать.
        train_base_k = None
        val_k = None
        test_k = None
    else:
        # Сколько исходных картинок (до копий/аугментаций) взять в train на класс.
        if int(TRAIN_BASE_PER_CLASS) > 0:
            train_base_k = int(TRAIN_BASE_PER_CLASS)
        else:
            train_base_k = int(min_n if downsample else max_n)
        train_base_k = max(1, int(train_base_k))

        # Val/test: либо явно, либо авто от train_base_k и ratio.
        if float(VAL_RATIO) <= 0:
            val_k = 0
        elif int(VAL_PER_CLASS) > 0:
            val_k = int(VAL_PER_CLASS)
        else:
            val_k = max(int(MIN_VAL_PER_CLASS), int(round(float(train_base_k) * float(VAL_RATIO))))

        if float(TEST_RATIO) <= 0:
            test_k = 0
        elif int(TEST_PER_CLASS) > 0:
            test_k = int(TEST_PER_CLASS)
        else:
            test_k = max(int(MIN_TEST_PER_CLASS), int(round(float(train_base_k) * float(TEST_RATIO))))

    # ============================================================
    # 4) Подготовка выходной папки и базовых файлов (labels.txt)
    # ============================================================
    if bool(CLEAN_OUTPUT_DIR) and os.path.isdir(out_root):
        shutil.rmtree(out_root, ignore_errors=True)

    # Создаём структуру split-папок заранее, чтобы результат был очевидным.
    for split in ("train", "val", "test"):
        ensure_dir(os.path.join(out_root, split))
    write_labels_txt(out_root, class_names_sorted)

    # ============================================================
    # 5) Генерация итоговых изображений (train/val/test)
    # ============================================================
    # Счётчики, чтобы в конце записать статистику.
    counts = {"train": 0, "val": 0, "test": 0}

    for class_name in class_names_sorted:
        imgs = class_to_imgs[class_name]
        # Выбираем какие исходные файлы пойдут в train/val/test.
        # Важно: при BALANCE_CLASSES=True делаем одинаковое число на каждый класс.
        if bool(BALANCE_CLASSES):
            split_items = make_splits_balanced(
                imgs,
                rng,
                train_base_k=int(train_base_k),
                val_k=int(val_k),
                test_k=int(test_k),
                downsample=bool(downsample),
            )
        else:
            split_items = make_splits(imgs, rng)

        for split, paths in split_items.items():
            out_dir = os.path.join(out_root, split, class_name)
            ensure_dir(out_dir)

            for i, img_path in enumerate(paths):
                # 1) Читаем исходный PNG (обычно RGBA с альфой)
                rgba = read_rgba(img_path)
                # 2) В train делаем несколько "копий" одного исходника (разные фоны/шумы),
                #    чтобы увеличить разнообразие данных.
                base_copies = int(TRAIN_COPIES_PER_IMAGE) if split == "train" else 1
                extra_bad = int(BAD_CROP_EXTRA_PER_IMAGE) if (split == "train" and bool(ENABLE_BAD_CROPS)) else 0

                total = int(base_copies) + int(extra_bad)
                for k in range(total):
                    # Базовый пример: знак на фоне + квадрат (и опциональный resize/лимит размера).
                    bgr_out = make_plain_example(rgba, IMAGE_SIZE, rng)

                    is_bad = False
                    if split == "train" and bool(ENABLE_BAD_CROPS):
                        # В первых base_copies делаем "bad" по вероятности.
                        # В оставшихся extra_bad — всегда делаем "bad".
                        if k >= int(base_copies) or (rng.random() < float(BAD_CROP_PROB)):
                            bgr_out = make_bad_crop_example(bgr_out, rng)
                            is_bad = True

                    # Имена делаем информативными, чтобы при необходимости можно было глазами
                    # понять: это ok-версия или bad-версия, какой split, какой исходник.
                    if total > 1:
                        tag = "bad" if is_bad else "ok"
                        out_name = f"{class_name}_{split}_{i:05d}_{k:02d}_{tag}.png"
                    else:
                        out_name = f"{class_name}_{split}_{i:05d}.png"

                    out_path = os.path.join(out_dir, out_name)
                    ok = cv2.imwrite(out_path, bgr_out)
                    if not ok:
                        raise RuntimeError(f"Не удалось записать файл: {out_path}")
                    counts[split] += 1

    # ============================================================
    # 5.5) Добавляем реальные ROI в train (если включено)
    # ============================================================
    real_roi_added_unique = 0
    real_roi_added_total = 0
    if bool(INCLUDE_REAL_ROI_IN_TRAIN):
        roi_index = os.path.abspath(str(REAL_ROI_INDEX_JSONL))
        if not os.path.isfile(roi_index):
            print("[WARN] INCLUDE_REAL_ROI_IN_TRAIN=True, но roi_index.jsonl не найден:", roi_index)
        else:
            allowed_classes = set(class_names_sorted)
            roi_items: list[tuple[str, str]] = []
            with open(roi_index, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s:
                        continue
                    obj = json.loads(s)
                    roi_rel = str(obj.get("roi_path") or "").strip()
                    true_cls = str(obj.get("true_class_name") or "").strip()
                    if not roi_rel or not true_cls:
                        continue
                    if true_cls not in allowed_classes:
                        continue
                    src = os.path.abspath(os.path.join(os.path.dirname(roi_index), roi_rel))
                    if not os.path.isfile(src):
                        continue
                    roi_items.append((true_cls, src))

            roi_items = sorted(roi_items, key=lambda x: (x[0], os.path.basename(x[1]), x[1]))
            for idx, (true_cls, src) in enumerate(roi_items):
                out_dir = os.path.join(out_root, "train", true_cls)
                ensure_dir(out_dir)
                src_base = os.path.basename(src)
                src_stem, src_ext = os.path.splitext(src_base)
                src_ext = src_ext.lower() if src_ext else ".jpg"
                for rep in range(int(max(1, REAL_ROI_REPEAT))):
                    out_name = f"{true_cls}_realroi_{idx:05d}_{rep:03d}_{src_stem}{src_ext}"
                    out_path = os.path.join(out_dir, out_name)
                    shutil.copy2(src, out_path)
                    counts["train"] += 1
                    real_roi_added_total += 1
                real_roi_added_unique += 1

    # ============================================================
    # 6) Пишем статистику (чтобы понимать, что получилось)
    # ============================================================
    stats_path = os.path.join(out_root, "stats.txt")
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write("speed classifier dataset stats\n")
        f.write(f"src={src_root}\n")
        f.write(f"out={out_root}\n")
        f.write("mode=plain\n")
        f.write(f"image_size={IMAGE_SIZE}\n")
        f.write(f"resize_output_to_image_size={bool(RESIZE_OUTPUT_TO_IMAGE_SIZE)}\n")
        f.write(f"max_output_side={int(MAX_OUTPUT_SIDE)}\n")
        f.write(f"val_ratio={VAL_RATIO}\n")
        f.write(f"test_ratio={TEST_RATIO}\n")
        f.write(f"balance_classes={bool(BALANCE_CLASSES)}\n")
        f.write(f"balance_strategy={str(BALANCE_STRATEGY).strip().lower()}\n")
        if bool(BALANCE_CLASSES):
            f.write(f"train_base_per_class={int(train_base_k)}\n")
            f.write(f"val_per_class={int(val_k)}\n")
            f.write(f"test_per_class={int(test_k)}\n")
        f.write(f"min_val_per_class={int(MIN_VAL_PER_CLASS)}\n")
        f.write(f"min_test_per_class={int(MIN_TEST_PER_CLASS)}\n")
        f.write(f"max_per_class={MAX_PER_CLASS}\n")
        f.write("extra_train_aug=0\n")
        f.write(f"train_copies_per_image={int(TRAIN_COPIES_PER_IMAGE)}\n")
        f.write(f"max_train_base_per_class={int(MAX_TRAIN_BASE_PER_CLASS)}\n")
        f.write(f"enable_bad_crops={bool(ENABLE_BAD_CROPS)}\n")
        f.write(f"bad_crop_prob={float(BAD_CROP_PROB)}\n")
        f.write(f"bad_crop_extra_per_image={int(BAD_CROP_EXTRA_PER_IMAGE)}\n")
        f.write(f"bad_crop_min_scale={float(BAD_CROP_MIN_SCALE)}\n")
        f.write(f"bad_crop_max_scale={float(BAD_CROP_MAX_SCALE)}\n")
        f.write(f"bad_crop_edge_cut_prob={float(BAD_CROP_EDGE_CUT_PROB)}\n")
        f.write(f"bad_crop_max_cut_frac={float(BAD_CROP_MAX_CUT_FRAC)}\n")
        f.write(f"bad_crop_jpeg_prob={float(BAD_CROP_JPEG_PROB)}\n")
        f.write(f"bad_crop_motion_blur_prob={float(BAD_CROP_MOTION_BLUR_PROB)}\n")
        f.write(f"use_real_backgrounds={bool(USE_REAL_BACKGROUNDS)}\n")
        f.write(f"background_dirs={BACKGROUND_DIRS}\n")
        f.write(f"num_background_images={len(_BG_PATHS)}\n")
        f.write(f"seed={int(SEED)}\n")
        f.write(f"include_real_roi_in_train={bool(INCLUDE_REAL_ROI_IN_TRAIN)}\n")
        f.write(f"real_roi_index_jsonl={os.path.abspath(str(REAL_ROI_INDEX_JSONL))}\n")
        f.write(f"real_roi_repeat={int(REAL_ROI_REPEAT)}\n")
        f.write(f"real_roi_added_unique={int(real_roi_added_unique)}\n")
        f.write(f"real_roi_added_total={int(real_roi_added_total)}\n")
        f.write(f"num_classes={len(class_names_sorted)}\n")
        f.write(f"train={counts['train']}\n")
        f.write(f"val={counts['val']}\n")
        f.write(f"test={counts['test']}\n")

    print("[OK] Датасет собран:", out_root)
    print("[OK] Train/Val/Test:", counts)
    print("[OK] Labels:", os.path.join(out_root, "labels.txt"))
    if bool(INCLUDE_REAL_ROI_IN_TRAIN):
        print("[OK] Real ROI added:", real_roi_added_total, "(unique:", real_roi_added_unique, ")")


if __name__ == "__main__":
    main()
