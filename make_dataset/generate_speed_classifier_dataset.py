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
TRAIN_COPIES_PER_IMAGE = 20

# Максимум разных исходных картинок на класс, которые пойдут в train (0 = все).
# Удобно, если у вас очень много исходников, и датасет получается слишком большой.
MAX_TRAIN_BASE_PER_CLASS = 0

# Небольшая деградация качества прямо в генераторе (помогает, когда train-aug выключены/слабые)
GEN_BLUR_PROB = 0.20
GEN_BLUR_SIGMA_MIN = 0.2
GEN_BLUR_SIGMA_MAX = 1.2
GEN_NOISE_PROB = 0.25
GEN_NOISE_STD_MIN = 0.0
GEN_NOISE_STD_MAX = 7.0  # в uint8

# Генерация "плохих кропов" (только для train).
# Имитируем ситуации, когда YOLO дал неточный ROI:
# - знак занимает меньшую часть кадра (много фона),
# - знак смещён (не по центру),
# - края чуть обрезаны,
# - качество похоже на видео (JPEG/смаз).
ENABLE_BAD_CROPS = True
# Доля train-копий, которые превращаем в "bad crop"
BAD_CROP_PROB = 0.35
# Дополнительно "плохих" вариантов на одну исходную картинку (помимо TRAIN_COPIES_PER_IMAGE)
BAD_CROP_EXTRA_PER_IMAGE = 0

# "Zoom-out" (знак меньше в кадре): масштаб уменьшенной вставки относительно исходного размера
BAD_CROP_MIN_SCALE = 0.55
BAD_CROP_MAX_SCALE = 0.95

# Подрезание краёв (имитация промаха bbox)
BAD_CROP_EDGE_CUT_PROB = 0.25
BAD_CROP_MAX_CUT_FRAC = 0.12

# JPEG-артефакты
BAD_CROP_JPEG_PROB = 0.22
BAD_CROP_JPEG_Q_MIN = 30
BAD_CROP_JPEG_Q_MAX = 90

# Motion blur
BAD_CROP_MOTION_BLUR_PROB = 0.18
BAD_CROP_MOTION_BLUR_K_MIN = 5
BAD_CROP_MOTION_BLUR_K_MAX = 13


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
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
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
    n = np.random.normal(0.0, 6.0, size=bg.shape).astype(np.float32)
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
    bgr = rgba_to_bgr_over_solid(rgba, random_bg_color(rng))
    # Сначала делаем квадрат (без ресайза), чтобы сохранить "родное" разрешение,
    # а затем уже по настройке либо ресайзим до IMAGE_SIZE, либо оставляем как есть.
    bgr = pad_to_square(bgr)
    if bool(RESIZE_OUTPUT_TO_IMAGE_SIZE):
        return resize_to_square(bgr, out_size)
    # Ограничим максимальный размер, чтобы не раздувать датасет.
    bgr = limit_max_side(bgr, int(MAX_OUTPUT_SIDE))

    # Лёгкая деградация качества (очень простая)
    if rng.random() < float(GEN_BLUR_PROB):
        sigma = rng.uniform(float(GEN_BLUR_SIGMA_MIN), float(GEN_BLUR_SIGMA_MAX))
        bgr = cv2.GaussianBlur(bgr, (0, 0), sigmaX=float(sigma))
    if rng.random() < float(GEN_NOISE_PROB):
        std = rng.uniform(float(GEN_NOISE_STD_MIN), float(GEN_NOISE_STD_MAX))
        if std > 0:
            n = np.random.normal(0.0, float(std), size=bgr.shape).astype(np.float32)
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

    want_val = 0 if float(VAL_RATIO) <= 0 else max(int(MIN_VAL_PER_CLASS), int(round(n * float(VAL_RATIO))))
    want_test = 0 if float(TEST_RATIO) <= 0 else max(int(MIN_TEST_PER_CLASS), int(round(n * float(TEST_RATIO))))

    # Train base: все, либо ограничение
    train = list(imgs)
    if int(MAX_TRAIN_BASE_PER_CLASS) > 0:
        train = train[: int(MAX_TRAIN_BASE_PER_CLASS)]
    if not train:
        train = [rng.choice(imgs)]

    val = [rng.choice(imgs) for _ in range(int(want_val))] if want_val > 0 else []
    test = [rng.choice(imgs) for _ in range(int(want_test))] if want_test > 0 else []

    return {"train": train, "val": val, "test": test}


def main():
    # ============================================================
    # 0) Инициализация (сид, пути, базовые проверки параметров)
    # ============================================================
    rng = random.Random(int(SEED))
    # Чтобы результат был воспроизводимым (шум генерим через numpy)
    np.random.seed(int(SEED))
    src_root = os.path.abspath(SRC_ROOT)
    # Выбор папки вывода.
    # По умолчанию каждый запуск — новая версия speed_cls_v<n>.
    if bool(AUTO_VERSION):
        out_root = pick_next_version_dir(str(OUT_ROOT_PREFIX))
    else:
        out_root = os.path.abspath(str(OUT_ROOT_PREFIX))

    if not os.path.isdir(src_root):
        raise FileNotFoundError(f"Не найдена папка с PNG-кропами: {src_root}")

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
        # Логика сплитов описана внутри make_splits().
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
        f.write(f"num_classes={len(class_names_sorted)}\n")
        f.write(f"train={counts['train']}\n")
        f.write(f"val={counts['val']}\n")
        f.write(f"test={counts['test']}\n")

    print("[OK] Датасет собран:", out_root)
    print("[OK] Train/Val/Test:", counts)
    print("[OK] Labels:", os.path.join(out_root, "labels.txt"))


if __name__ == "__main__":
    main()
