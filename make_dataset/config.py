import os

# ============================================================
# Конфиг генератора синтетического датасета (YOLO)
# ============================================================
# Правьте параметры тут.

# ===================== Количество объектов для отчета прогресса =====================
PROGRESS_EVERY_N_OBJECTS = 25

# ===================== Пути =====================
BACKGROUNDS_DIR = "make_dataset/backgrounds"

# Кадры с видеорегистратора (лучшие фоны для домена). Можно указать папку с jpg/png.
# Файлы ищутся РЕКУРСИВНО. Если папка не задана или пустая — используем BACKGROUNDS_DIR.
DASHCAM_FRAMES_DIR = "make_dataset/dashcam_frames"

# Папки со знаками/CSV
SIGNS_IMAGES_DIR = "shared/signs/images"  # базовые изображения знаков (если у класса нет splits-вариаций)
SPLITS_DIR = "shared/signs/splits"        # вариации: splits/<name>/*.(png/jpg/svg)
CSV_PATH = "shared/signs/signs.csv"       # CSV: filename,class_id,class_name

# ===================== Выход =====================
OUTPUT_BASE = "datasets/dataset"  # папки будут dataset_1, dataset_2, ...
MIN_IMAGES_PER_CLASS = 200         # минимум экземпляров (bbox) на класс
VAL_RATIO = 0.2                   # доля val (для train.txt/val.txt)
WEATHER_PROB = 0.7                # вероятность погодного эффекта на кадре
RANDOM_SEED = 1337                # фиксируем seed для воспроизводимости

# Если список пустой — генерируем все классы из CSV.
# Если указать, например [0, 4, 10], то будут использоваться только эти class_id.
SELECT_CLASS_IDS = []  # пример: [0, 4, 10]

# ===================== Внешний датасет (YOLO) =====================
EXTERNAL_MIX_ENABLED = True
EXTERNAL_DATASET_DIR = "make_dataset/external_dataset/Road Sign-3"
EXTERNAL_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# ===================== Геометрия / аугментации =====================
# Для видеорегистратора знаки чаще маленькие, поэтому диапазон обычно ниже.
SCALE_RANGE = (0.05, 0.20)        # ширина знака как доля ширины фона
SCALE_BIAS_POWER = 2.2            # >1 => чаще маленькие (u**power)
ROLL_ANGLE_RANGE = (-30.0, 30.0)  # поворот в плоскости
PERSPECTIVE_STRENGTH = 0.10       # перспектива (наклон)
SHIFT_FRACTION = 0.08             # небольшой сдвиг после преобразований

# Размещение
RIGHT_HALF_PROB = 0.7              # шанс, что знак окажется в правой половине кадра
MIN_VISIBLE_AREA_FRACTION = 0.60   # минимум видимой площади знака (0..1)
MAX_PLACEMENT_TRIES = 10           # попыток размещения знака

# Негативные примеры (кадры без знаков)
NEGATIVE_RATIO = 0.1

# Несколько знаков на одном изображении
MULTI_OBJECT_ENABLED = True
EXTRA_OBJECTS_RANGE = (0, 3)       # сколько ДОП. знаков добавлять к основному
EXTRA_SAME_CLASS_PROB = 0.15       # шанс, что доп. знак будет того же класса
MAX_IOU_BETWEEN_SIGNS = 0.25       # ограничение пересечений bbox между знаками
MAX_EXTRA_TRIES = 25               # попыток разместить доп. знак

# Если выбран конкретный список классов (SELECT_CLASS_IDS), то по умолчанию дополнительные знаки
# будут выбираться ТОЛЬКО из этих классов (иначе можно случайно "зацепить" любой class_id).
ALLOW_EXTRA_NON_SELECTED_CLASSES = False

# ===================== Многопоточность =====================
USE_MULTITHREADING = True
NUM_WORKERS = max(1, (os.cpu_count() or 4) - 1)
MAX_INFLIGHT_TASKS = None  # None => NUM_WORKERS

# ===================== Эффекты камеры (видеорегистратор) =====================
CAMERA_EFFECTS_PROB = 0.90
JPEG_QUALITY_RANGE = (35, 95)
NOISE_STD_RANGE = (0.0, 8.0)
GAUSS_BLUR_PROB = 0.25
MOTION_BLUR_PROB = 0.25
VIGNETTE_PROB = 0.20
COLOR_JITTER_PROB = 0.35

# Тень / смягчение
SHADOW_PROB = 0.70
SHADOW_STRENGTH_RANGE = (0.20, 0.55)
SHADOW_BLUR_RANGE = (3, 11)
SHADOW_OFFSET_X_RANGE = (-6, 6)
SHADOW_OFFSET_Y_RANGE = (2, 10)
EDGE_BLUR_PROB = 0.30
EDGE_BLUR_K_RANGE = (1, 3)

# ===================== Деградация качества знака =====================
SIGN_DEGRADE_ENABLED = True
SIGN_DEGRADE_PROB = 0.95
SIGN_BASE_QUALITY = 0.90
SIGN_DEGRADE_MIN_PX = 35
SIGN_DEGRADE_MAX_PX = 160
SIGN_JPEG_QUALITY_SMALL = (12, 40)
SIGN_JPEG_QUALITY_LARGE = (65, 95)
SIGN_DOWNSCALE_SMALL = (0.28, 0.55)
SIGN_DOWNSCALE_LARGE = (0.85, 1.00)
SIGN_BLUR_SIGMA_SMALL = (0.6, 1.8)
SIGN_BLUR_SIGMA_LARGE = (0.0, 0.6)
SIGN_NOISE_STD_SMALL = (1.0, 10.0)
SIGN_NOISE_STD_LARGE = (0.0, 3.0)

SIGN_PIXELATE_ENABLED = True
SIGN_PIXELATE_PROB = 0.90
SIGN_PIXELATE_SCALE_SMALL = (0.18, 0.45)
SIGN_PIXELATE_SCALE_LARGE = (0.75, 0.95)

# ===================== Цветокоррекция знака по яркости фона =====================
BRIGHTNESS_DARK_THRESH = 90.0
BRIGHTNESS_BRIGHT_THRESH = 190.0
SIGN_DARKEN_FACTOR = 0.85
SIGN_DESAT_FACTOR = 0.80
SIGN_BRIGHTEN_FACTOR = 1.10
SIGN_CONTRAST_FACTOR = 1.10

# ===================== Форматы / bbox =====================
ALPHA_THRESHOLD = 8
MAX_ATTEMPTS_MULT = 12  # попыток = total_target * MAX_ATTEMPTS_MULT

TEMPLATE_EXTS = (".png", ".jpg", ".jpeg", ".svg")
BACKGROUND_EXTS = (".png", ".jpg", ".jpeg")
