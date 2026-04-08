from __future__ import annotations

"""
Датасет и аугментации для классификатора скорости.

Важно понимать, что здесь происходят 3 большие вещи:
1) Загрузка данных через ImageFolder (класс = имя папки)
2) Аугментации (только для split == "train")
3) Препроцессинг, который ДОЛЖЕН совпадать на обучении и на inference:
   - ToTensor() переводит uint8 RGB в float 0..1
   - Normalize(mean,std) (ImageNet) приводит распределение к тому, что ждёт MobileNet

Если в приложении не сделать такой же Normalize — качество резко падает.
"""

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
from torchvision import datasets, transforms
from torchvision.transforms import functional as F


@dataclass(frozen=True)
class DatasetConfig:
    data_dir: str
    image_size: int = 128
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)


class ImageFolderFixedClasses(datasets.ImageFolder):
    """
    ImageFolder, который берёт порядок классов из заранее заданного списка.

    Зачем это нужно:
    - В torchvision ImageFolder порядок классов = сортировка имён папок (лексикографически).
      Для классов вида "3.24_10", "3.24_100" это даёт порядок 10,100,110,...,20,...
    - У нас есть labels.txt, который задаёт "истинный" порядок классов (обычно человеческий:
      10,20,30,...,100,110,...). Если модель обучить с другим порядком, то на inference
      индексы будут маппиться в неправильные строки.

    Правило:
    - labels.txt считается источником истины для class_to_idx.
    - Если какой-то класс из labels.txt отсутствует в split-папке — падаем с понятной ошибкой.
    - Любые "лишние" подпапки, которых нет в labels.txt, будут проигнорированы.
    """

    def __init__(self, root: str, *, classes: list[str], transform=None):
        self._fixed_classes = [str(c).strip() for c in classes if str(c).strip()]
        super().__init__(root=root, transform=transform)

    def find_classes(self, directory: str):  # type: ignore[override]
        classes = list(self._fixed_classes)
        class_to_idx = {c: i for i, c in enumerate(classes)}

        missing = []
        for c in classes:
            p = os.path.join(directory, c)
            if not os.path.isdir(p):
                missing.append(c)
        if missing:
            raise FileNotFoundError(
                "В split не найдены папки для классов из labels.txt.\n"
                f"split_dir={os.path.abspath(directory)}\n"
                f"missing_first_10={missing[:10]}\n"
                "Проверь, что датасет сгенерирован полностью и labels.txt соответствует папкам."
            )

        return classes, class_to_idx


class RandomDownscaleUpscale:
    """
    Простая аугментация "понизить разрешение -> повысить обратно".
    Помогает при мелких/размытых знаках и имитирует сильное сжатие/дальность.
    """

    def __init__(
        self,
        min_scale: float = 0.35,
        max_scale: float = 0.95,
        min_side_for_strong: int = 160,
        min_scale_if_small: float = 0.80,
    ):
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        self.min_side_for_strong = int(min_side_for_strong)
        self.min_scale_if_small = float(min_scale_if_small)

    def __call__(self, img):
        # img — PIL.Image
        w, h = img.size
        if w <= 2 or h <= 2:
            return img

        import random

        # Если изображение уже маленькое, сильный downscale почти гарантированно "убьёт" цифры.
        # Поэтому для маленьких картинок ограничиваем минимальный scale.
        min_side = min(int(w), int(h))
        min_s = float(self.min_scale)
        if min_side < int(self.min_side_for_strong):
            min_s = max(min_s, float(self.min_scale_if_small))

        s = random.uniform(min_s, float(self.max_scale))
        nw = max(2, int(round(w * s)))
        nh = max(2, int(round(h * s)))

        # downscale -> upscale
        small = img.resize((nw, nh))
        out = small.resize((w, h))
        return out


class SmartPreResizeSquare:
    """
    "Умный" pre-resize до квадрата перед геометрическими аугментациями.

    Идея:
    - Мы хотим чуть больший размер (pre_size), чтобы перспективы/аффинные искажения были стабильнее.
    - Но если исходник очень маленький, нет смысла сильно апскейлить (появится мыло),
      и последующий downscale будет только вредить.
    """

    def __init__(self, pre_size: int, small_threshold: int, small_pre_size: int):
        self.pre_size = int(pre_size)
        self.small_threshold = int(small_threshold)
        self.small_pre_size = int(small_pre_size)

    def __call__(self, img):
        # img — PIL.Image
        w, h = img.size
        if w <= 2 or h <= 2:
            return img
        target = int(self.pre_size)
        if min(int(w), int(h)) < int(self.small_threshold):
            target = int(self.small_pre_size)
        target = max(8, target)
        if int(w) == target and int(h) == target:
            return img
        return img.resize((target, target))


class RandomDirtAndStains:
    """
    Простая "грязь": случайные пятна/точки поверх изображения.
    Делается в PIL-режиме через numpy, чтобы было максимально просто.
    """

    def __init__(
        self,
        p: float = 0.35,
        max_stains: int = 6,
        center_radius_frac: float = 0.28,
        max_center_stain_frac: float = 0.06,
        snow_p: float = 0.25,
        max_snow: int = 45,
    ):
        self.p = float(p)
        self.max_stains = int(max_stains)
        # Центральная зона (где обычно цифры). В ней запрещаем большие "комки".
        self.center_radius_frac = float(center_radius_frac)
        self.max_center_stain_frac = float(max_center_stain_frac)
        # Мелкие "снежинки/крошки" (частичное перекрытие цифр, но не полностью)
        self.snow_p = float(snow_p)
        self.max_snow = int(max_snow)

    def __call__(self, img):
        import random

        if random.random() > self.p:
            return img

        arr = np.array(img)
        if arr.ndim != 3 or arr.shape[2] < 3:
            return img

        h, w = arr.shape[:2]
        cx0 = float(w - 1) * 0.5
        cy0 = float(h - 1) * 0.5
        cr = float(min(h, w)) * float(self.center_radius_frac)
        cr2 = cr * cr

        n = random.randint(1, max(1, self.max_stains))

        for _ in range(n):
            # Случайное пятно (круг/эллипс)
            cx = random.randint(0, w - 1)
            cy = random.randint(0, h - 1)

            # В центре — только мелкие пятна, чтобы не "убивать" цифры.
            dx = float(cx) - cx0
            dy = float(cy) - cy0
            in_center = (dx * dx + dy * dy) <= cr2

            rx_min = max(1, w // 40)
            ry_min = max(1, h // 40)
            rx_max = max(2, w // 10)
            ry_max = max(2, h // 10)
            if in_center:
                rx_max = max(rx_min, int(round(float(min(h, w)) * float(self.max_center_stain_frac))))
                ry_max = max(ry_min, int(round(float(min(h, w)) * float(self.max_center_stain_frac))))

            rx = random.randint(int(rx_min), int(rx_max))
            ry = random.randint(int(ry_min), int(ry_max))

            # Цвет пятна: серо-коричневый + прозрачность
            base = random.randint(30, 220)
            col = np.array(
                [base, max(0, base - random.randint(0, 25)), max(0, base - random.randint(0, 45))],
                dtype=np.uint8,
            )
            # В центре делаем пятна более прозрачными (ещё один "предохранитель")
            if in_center:
                alpha = random.uniform(0.06, 0.22)
            else:
                alpha = random.uniform(0.10, 0.45)

            y0 = max(0, cy - ry)
            y1 = min(h, cy + ry + 1)
            x0 = max(0, cx - rx)
            x1 = min(w, cx + rx + 1)

            yy, xx = np.ogrid[y0:y1, x0:x1]
            mask = ((xx - cx) / float(max(1, rx))) ** 2 + ((yy - cy) / float(max(1, ry))) ** 2 <= 1.0

            roi = arr[y0:y1, x0:x1, :3].astype(np.float32)
            roi[mask] = roi[mask] * (1.0 - alpha) + col.astype(np.float32) * alpha
            arr[y0:y1, x0:x1, :3] = np.clip(roi, 0, 255).astype(np.uint8)

        # Мелкие "снежинки/крошки": маленькие светлые точки, которые могут частично перекрывать цифры.
        if random.random() < float(self.snow_p):
            n2 = random.randint(8, max(8, int(self.max_snow)))
            # чуть чаще в центре (но очень мелкие)
            for _ in range(n2):
                if random.random() < 0.55:
                    # центр-биас
                    sx = int(np.clip(np.random.normal(cx0, cr * 0.45), 0, w - 1))
                    sy = int(np.clip(np.random.normal(cy0, cr * 0.45), 0, h - 1))
                else:
                    sx = random.randint(0, w - 1)
                    sy = random.randint(0, h - 1)

                r = random.randint(1, 2)  # маленькие точки
                alpha = random.uniform(0.10, 0.28)
                white = np.array([245, 245, 245], dtype=np.float32)

                y0 = max(0, sy - r)
                y1 = min(h, sy + r + 1)
                x0 = max(0, sx - r)
                x1 = min(w, sx + r + 1)
                yy, xx = np.ogrid[y0:y1, x0:x1]
                mask = (xx - sx) ** 2 + (yy - sy) ** 2 <= (r * r)
                roi = arr[y0:y1, x0:x1, :3].astype(np.float32)
                roi[mask] = roi[mask] * (1.0 - alpha) + white * alpha
                arr[y0:y1, x0:x1, :3] = np.clip(roi, 0, 255).astype(np.uint8)

        return F.to_pil_image(arr)


class RandomGlareLines:
    """
    Простые "блики/полосы": несколько ярких полупрозрачных линий.
    """

    def __init__(self, p: float = 0.25, max_lines: int = 3):
        self.p = float(p)
        self.max_lines = int(max_lines)

    def __call__(self, img):
        import random

        if random.random() > self.p:
            return img

        arr = np.array(img)
        if arr.ndim != 3 or arr.shape[2] < 3:
            return img

        h, w = arr.shape[:2]
        n = random.randint(1, max(1, self.max_lines))

        for _ in range(n):
            # Линия задана двумя точками
            x0 = random.randint(0, w - 1)
            y0 = random.randint(0, h - 1)
            x1 = random.randint(0, w - 1)
            y1 = random.randint(0, h - 1)
            thickness = random.randint(1, max(1, min(w, h) // 40))
            alpha = random.uniform(0.10, 0.35)

            # Рисуем линию через OpenCV на маске
            import cv2

            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.line(mask, (x0, y0), (x1, y1), color=255, thickness=thickness)

            base = arr[:, :, :3].astype(np.float32)
            # Альфа-маска (H,W,1) -> корректное смешивание без булевой индексации
            a = ((mask > 0).astype(np.float32) * float(alpha))[:, :, None]
            white = np.array([255.0, 255.0, 255.0], dtype=np.float32)[None, None, :]
            mixed = base * (1.0 - a) + white * a
            arr[:, :, :3] = np.clip(mixed, 0, 255).astype(np.uint8)

        return F.to_pil_image(arr)


class RandomGaussianNoise:
    """
    Гауссов шум уже после ToTensor (на float tensor).
    """

    def __init__(self, p: float = 0.30, std_min: float = 0.0, std_max: float = 0.05):
        self.p = float(p)
        self.std_min = float(std_min)
        self.std_max = float(std_max)

    def __call__(self, x):
        import random
        import torch

        if random.random() > self.p:
            return x
        std = random.uniform(self.std_min, self.std_max)
        if std <= 0:
            return x
        noise = torch.randn_like(x) * float(std)
        y = x + noise
        return torch.clamp(y, 0.0, 1.0)


class RandomMotionBlur:
    """
    Motion blur (похоже на смаз от движения/видео).
    Делается через OpenCV свёртку линейным ядром.
    """

    def __init__(self, p: float = 0.18, k_min: int = 5, k_max: int = 13):
        self.p = float(p)
        self.k_min = int(k_min)
        self.k_max = int(k_max)

    def __call__(self, img):
        import random

        if random.random() > self.p:
            return img

        arr = np.array(img)
        if arr.ndim != 3 or arr.shape[2] < 3:
            return img

        import cv2

        k = random.randint(self.k_min, self.k_max)
        if k % 2 == 0:
            k += 1
        k = max(3, min(31, k))

        angle = random.uniform(0.0, 180.0)
        # базовое горизонтальное ядро
        kernel = np.zeros((k, k), dtype=np.float32)
        kernel[k // 2, :] = 1.0

        # вращаем ядро
        M = cv2.getRotationMatrix2D((k / 2.0 - 0.5, k / 2.0 - 0.5), angle, 1.0)
        kernel = cv2.warpAffine(kernel, M, (k, k))
        s = float(kernel.sum())
        if s > 1e-6:
            kernel /= s

        out = cv2.filter2D(arr[:, :, :3], -1, kernel)
        arr[:, :, :3] = np.clip(out, 0, 255).astype(np.uint8)
        return F.to_pil_image(arr)


class RandomJpegCompression:
    """
    JPEG-артефакты (типичная потеря качества на видео/скриншотах).
    """

    def __init__(self, p: float = 0.22, q_min: int = 25, q_max: int = 85):
        self.p = float(p)
        self.q_min = int(q_min)
        self.q_max = int(q_max)

    def __call__(self, img):
        import random

        if random.random() > self.p:
            return img

        arr = np.array(img)
        if arr.ndim != 3 or arr.shape[2] < 3:
            return img

        import cv2

        q = random.randint(self.q_min, self.q_max)
        # PIL -> numpy обычно RGB, OpenCV ожидает BGR
        bgr = arr[:, :, :3][:, :, ::-1]
        ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(q)])
        if not ok:
            return img
        dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        if dec is None:
            return img
        rgb = dec[:, :, ::-1]
        arr[:, :, :3] = np.clip(rgb, 0, 255).astype(np.uint8)
        return F.to_pil_image(arr)


class RandomZoomOutToBackground:
    """
    "Плохой кроп": знак занимает меньшую часть кадра + случайное смещение.
    Мы уменьшаем изображение и вставляем его на фон того же размера.
    Это помогает, когда ROI включает много окружения, а цифры получаются мелкими.
    """

    def __init__(self, p: float = 0.25, min_scale: float = 0.55, max_scale: float = 1.0, noise: int = 12):
        self.p = float(p)
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        self.noise = int(noise)

    def __call__(self, img):
        import random

        if random.random() > self.p:
            return img

        w, h = img.size
        if w < 8 or h < 8:
            return img

        s = random.uniform(self.min_scale, self.max_scale)
        nw = max(2, int(round(w * s)))
        nh = max(2, int(round(h * s)))
        if nw >= w and nh >= h:
            return img

        small = img.resize((nw, nh))
        small_arr = np.array(small)
        if small_arr.ndim != 3 or small_arr.shape[2] < 3:
            return img

        # Фон: почти однотонный + лёгкий шум, чтобы не было "идеального" бэкграунда
        base = random.randint(10, 245)
        bg = np.full((h, w, 3), base, dtype=np.uint8)
        if self.noise > 0:
            n = np.random.randint(-self.noise, self.noise + 1, size=(h, w, 3), dtype=np.int16)
            bg = np.clip(bg.astype(np.int16) + n, 0, 255).astype(np.uint8)

        # Вставляем уменьшенный знак в случайную позицию
        x0 = random.randint(0, max(0, w - nw))
        y0 = random.randint(0, max(0, h - nh))
        bg[y0 : y0 + nh, x0 : x0 + nw, :] = small_arr[:, :, :3]

        return F.to_pil_image(bg)


class RandomEdgeCutAndResizeBack:
    """
    "Плохой кроп": bbox чуть промахнулся и обрезал края знака.
    Случайно отрезаем края и приводим назад к исходному размеру.
    """

    def __init__(self, p: float = 0.18, max_cut_frac: float = 0.12):
        self.p = float(p)
        self.max_cut_frac = float(max_cut_frac)

    def __call__(self, img):
        import random

        if random.random() > self.p:
            return img

        w, h = img.size
        if w < 16 or h < 16:
            return img

        max_l = int(round(w * self.max_cut_frac))
        max_r = int(round(w * self.max_cut_frac))
        max_t = int(round(h * self.max_cut_frac))
        max_b = int(round(h * self.max_cut_frac))

        l = random.randint(0, max(0, max_l))
        r = random.randint(0, max(0, max_r))
        t = random.randint(0, max(0, max_t))
        b = random.randint(0, max(0, max_b))

        # не допускаем слишком маленького остатка
        if (w - l - r) < max(8, int(round(w * 0.65))) or (h - t - b) < max(8, int(round(h * 0.65))):
            return img

        cropped = img.crop((l, t, w - r, h - b))
        return cropped.resize((w, h))


def build_transforms(cfg: DatasetConfig, split: str):
    if split not in ("train", "val", "test"):
        raise ValueError(f"Unknown split: {split}")

    if split == "train":
        # Делаем небольшой "оверсэмплинг" по размеру, чтобы геометрия работала на чуть более
        # детальном изображении, а потом уже приводим к финальному 128x128.
        pre_size = int(round(float(cfg.image_size) * 1.25))
        # Если исходник совсем маленький, не апскейлим слишком сильно.
        small_threshold = int(round(float(cfg.image_size) * 0.75))  # например, <96 при IMAGE_SIZE=128
        small_pre_size = int(round(float(cfg.image_size) * 1.05))   # например, ~134 при IMAGE_SIZE=128

        aug = [
            transforms.RandomApply(
                [RandomDownscaleUpscale(0.55, 0.95, min_side_for_strong=pre_size, min_scale_if_small=0.85)],
                p=0.28,
            ),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.20, hue=0.02)],
                p=0.35,
            ),
            # Перспективные искажения (умеренно, чтобы не "убивать" читаемость цифр)
            transforms.RandomPerspective(distortion_scale=0.4, p=0.28),
            # Наклоны/сдвиги/масштаб/повороты (в одном месте)
            transforms.RandomAffine(
                degrees=10,
                translate=(0.06, 0.06),
                scale=(0.90, 1.10),
                shear=(-6, 6),
            ),
            # Имитация плохого ROI: много фона/смещение/обрезанные края
            RandomZoomOutToBackground(p=0.25, min_scale=0.55, max_scale=1.0, noise=12),
            RandomEdgeCutAndResizeBack(p=0.18, max_cut_frac=0.12),
            RandomDirtAndStains(p=0.18, max_stains=5),
            RandomGlareLines(p=0.10, max_lines=2),
        ]
    else:
        pre_size = int(cfg.image_size)
        small_threshold = int(cfg.image_size)
        small_pre_size = int(cfg.image_size)
        aug = []

    # ВАЖНО:
    # - На val/test трансформы должны быть детерминированными, иначе метрики и отчёты будут "прыгать".
    # - Поэтому noise/erasing/blur/jpeg/motionblur применяем только на train.
    if split == "train":
        tail = [
            # Артефакты качества лучше применять уже на финальном размере (как "камера/видео")
            RandomJpegCompression(p=0.12, q_min=35, q_max=95),
            RandomMotionBlur(p=0.10, k_min=5, k_max=11),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.4))], p=0.14),
            transforms.ToTensor(),  # uint8 HWC -> float CHW in 0..1
            RandomGaussianNoise(p=0.25, std_min=0.0, std_max=0.03),
            # RandomErasing работает по tensor, поэтому ставим после ToTensor
            transforms.RandomErasing(p=0.08, scale=(0.02, 0.08), ratio=(0.3, 3.3), value="random"),
            transforms.Normalize(cfg.mean, cfg.std),
        ]
    else:
        tail = [
            transforms.ToTensor(),
            transforms.Normalize(cfg.mean, cfg.std),
        ]

    return transforms.Compose(
        [
            # Сначала приводим к чуть большему размеру (train), чтобы геометрия была стабильнее.
            SmartPreResizeSquare(pre_size=pre_size, small_threshold=small_threshold, small_pre_size=small_pre_size),
            *aug,
            # Финальный resize к размеру модели
            transforms.Resize((cfg.image_size, cfg.image_size), antialias=True),
            *tail,
        ]
    )


def load_split_dataset(cfg: DatasetConfig, split: str):
    root = os.path.join(cfg.data_dir, split)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Split folder not found: {root}")

    tfm = build_transforms(cfg, split)
    # Источник истины для порядка классов — labels.txt в корне DATA_DIR.
    # Это защищает от рассинхрона "индекс -> label" между обучением и inference.
    labels = read_labels_txt(cfg.data_dir)
    if labels:
        ds = ImageFolderFixedClasses(root=root, classes=labels, transform=tfm)
    else:
        ds = datasets.ImageFolder(root=root, transform=tfm)
    return ds


def read_labels_txt(data_dir: str) -> Optional[list[str]]:
    p = os.path.join(data_dir, "labels.txt")
    if not os.path.isfile(p):
        return None
    with open(p, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.read().splitlines()]
    return [x for x in lines if x]

