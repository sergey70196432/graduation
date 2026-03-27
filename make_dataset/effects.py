import math
import random

import cv2
import numpy as np

from make_dataset import config as cfg


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


def alpha_bbox(alpha, thr=cfg.ALPHA_THRESHOLD):
    """bbox по альфа-каналу (x1,y1,x2,y2) в координатах alpha."""
    ys, xs = np.where(alpha > thr)
    if xs.size == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def apply_jpeg_compression(img_bgr, quality):
    """JPEG артефакты."""
    quality = int(np.clip(quality, 5, 100))
    ok, enc = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return img_bgr
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return dec if dec is not None else img_bgr


def add_gaussian_noise(img_bgr, std):
    """Гауссов шум."""
    if std <= 0:
        return img_bgr
    noise = np.random.normal(0.0, std, img_bgr.shape).astype(np.float32)
    out = img_bgr.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_motion_blur(img_bgr, k):
    """Простая смаз по движению."""
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
    """Сдвиги яркости/контраста/насыщенности."""
    out = img_bgr.astype(np.float32)
    c = random.uniform(0.90, 1.15)
    out = (out - 127.5) * c + 127.5
    b = random.uniform(-12.0, 12.0)
    out = out + b
    out = np.clip(out, 0, 255).astype(np.uint8)

    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= random.uniform(0.85, 1.20)
    hsv[:, :, 2] *= random.uniform(0.90, 1.10)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def apply_camera_effects(img_bgr):
    """Набор эффектов, похожих на видеорегистратор."""
    if random.random() > cfg.CAMERA_EFFECTS_PROB:
        return img_bgr

    out = img_bgr
    if random.random() < 0.30:
        h, w = out.shape[:2]
        s = random.uniform(0.70, 0.95)
        nw, nh = max(2, int(w * s)), max(2, int(h * s))
        tmp = cv2.resize(out, (nw, nh), interpolation=cv2.INTER_AREA)
        out = cv2.resize(tmp, (w, h), interpolation=cv2.INTER_LINEAR)

    if random.random() < cfg.COLOR_JITTER_PROB:
        out = apply_color_jitter(out)

    if random.random() < cfg.GAUSS_BLUR_PROB:
        sigma = random.uniform(0.6, 1.6)
        out = cv2.GaussianBlur(out, (0, 0), sigmaX=sigma)

    if random.random() < cfg.MOTION_BLUR_PROB:
        k = random.choice([5, 7, 9, 11])
        out = apply_motion_blur(out, k)

    std = random.uniform(cfg.NOISE_STD_RANGE[0], cfg.NOISE_STD_RANGE[1])
    out = add_gaussian_noise(out, std)

    if random.random() < cfg.VIGNETTE_PROB:
        out = apply_vignette(out, strength=random.uniform(0.20, 0.45))

    q = random.randint(cfg.JPEG_QUALITY_RANGE[0], cfg.JPEG_QUALITY_RANGE[1])
    out = apply_jpeg_compression(out, q)
    return out


def apply_weather(img_bgr):
    """Умеренные эффекты: дождь/снег/туман."""
    if random.random() > cfg.WEATHER_PROB:
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

    blur = cv2.GaussianBlur(img_bgr, (0, 0), sigmaX=random.uniform(1.0, 2.6))
    fog = np.full_like(img_bgr, random.randint(200, 235))
    a1 = random.uniform(0.10, 0.18)
    a2 = random.uniform(0.08, 0.16)
    out = cv2.addWeighted(img_bgr, 1.0 - a1, blur, a1, 0.0)
    out = cv2.addWeighted(out, 1.0 - a2, fog, a2, 0.0)
    return out


def pixelate_bgr(img_bgr, scale):
    """Пикселизация: downscale (area) -> upscale (nearest)."""
    h, w = img_bgr.shape[:2]
    scale = float(scale)
    scale = clamp(scale, 0.05, 1.0)
    if scale >= 0.999 or w < 2 or h < 2:
        return img_bgr
    nw = max(2, int(w * scale))
    nh = max(2, int(h * scale))
    small = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    pix = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    return pix


def degrade_sign_by_size(sign_rgba, sign_bbox_w_px):
    """
    Ухудшаем качество знака (только в области непрозрачных пикселей), причём
    чем меньше знак по размеру — тем сильнее деградация.
    """
    if not cfg.SIGN_DEGRADE_ENABLED:
        return sign_rgba
    if random.random() > cfg.SIGN_DEGRADE_PROB:
        return sign_rgba

    alpha = sign_rgba[:, :, 3]
    bb = alpha_bbox(alpha, thr=cfg.ALPHA_THRESHOLD)
    if bb is None:
        return sign_rgba
    x1, y1, x2, y2 = bb
    w = int(x2 - x1 + 1)
    h = int(y2 - y1 + 1)
    if w < 2 or h < 2:
        return sign_rgba

    t = 0.0
    if cfg.SIGN_DEGRADE_MAX_PX > cfg.SIGN_DEGRADE_MIN_PX:
        t = (float(sign_bbox_w_px) - float(cfg.SIGN_DEGRADE_MIN_PX)) / float(cfg.SIGN_DEGRADE_MAX_PX - cfg.SIGN_DEGRADE_MIN_PX)
    t = clamp(t, 0.0, 1.0)
    t = clamp(t * float(cfg.SIGN_BASE_QUALITY), 0.0, 1.0)

    roi = sign_rgba[y1 : y2 + 1, x1 : x2 + 1].copy()
    roi_a = roi[:, :, 3:4]
    roi_bgr = roi[:, :, :3][:, :, ::-1].copy()

    if cfg.SIGN_PIXELATE_ENABLED and random.random() < cfg.SIGN_PIXELATE_PROB:
        s_pix = sample_interp_range(cfg.SIGN_PIXELATE_SCALE_SMALL, cfg.SIGN_PIXELATE_SCALE_LARGE, t)
        roi_bgr = pixelate_bgr(roi_bgr, s_pix)

    s = sample_interp_range(cfg.SIGN_DOWNSCALE_SMALL, cfg.SIGN_DOWNSCALE_LARGE, t)
    s = clamp(s, 0.15, 1.0)
    if s < 0.999:
        nw = max(2, int(w * s))
        nh = max(2, int(h * s))
        tmp = cv2.resize(roi_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        roi_bgr = cv2.resize(tmp, (w, h), interpolation=cv2.INTER_LINEAR)

    sigma = sample_interp_range(cfg.SIGN_BLUR_SIGMA_SMALL, cfg.SIGN_BLUR_SIGMA_LARGE, t)
    if sigma > 0.05:
        roi_bgr = cv2.GaussianBlur(roi_bgr, (0, 0), sigmaX=float(sigma))

    std = sample_interp_range(cfg.SIGN_NOISE_STD_SMALL, cfg.SIGN_NOISE_STD_LARGE, t)
    roi_bgr = add_gaussian_noise(roi_bgr, std)

    q = int(round(sample_interp_range(cfg.SIGN_JPEG_QUALITY_SMALL, cfg.SIGN_JPEG_QUALITY_LARGE, t)))
    roi_bgr = apply_jpeg_compression(roi_bgr, q)

    out = sign_rgba.copy()
    out[y1 : y2 + 1, x1 : x2 + 1, :3] = roi_bgr[:, :, ::-1]
    out[y1 : y2 + 1, x1 : x2 + 1, 3:4] = roi_a
    return out

