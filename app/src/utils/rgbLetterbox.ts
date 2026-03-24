import type { FrameSize } from '../types/detection';
import type { LetterboxMeta } from './yoloPostprocess';
import type { RgbaImage } from './jpegDecode';

export function buildLetterboxedFloatInput(
  img: RgbaImage,
  inputSize: number
): { input: Float32Array; letterbox: LetterboxMeta; srcFrameSize: FrameSize } {
  const srcW = img.width;
  const srcH = img.height;

  const scale = Math.min(inputSize / srcW, inputSize / srcH);
  const resizedWidth = Math.max(1, Math.round(srcW * scale));
  const resizedHeight = Math.max(1, Math.round(srcH * scale));
  const padX = Math.floor((inputSize - resizedWidth) / 2);
  const padY = Math.floor((inputSize - resizedHeight) / 2);

  const input = new Float32Array(inputSize * inputSize * 3);

  // Nearest-neighbor resize в Letterbox. Для тестового режима (видео) этого достаточно.
  for (let y = 0; y < resizedHeight; y++) {
    const srcY = Math.min(srcH - 1, Math.max(0, Math.round(y / scale)));
    for (let x = 0; x < resizedWidth; x++) {
      const srcX = Math.min(srcW - 1, Math.max(0, Math.round(x / scale)));

      const si = (srcY * srcW + srcX) * 4;
      const r = img.data[si] ?? 0;
      const g = img.data[si + 1] ?? 0;
      const b = img.data[si + 2] ?? 0;

      const dx = x + padX;
      const dy = y + padY;
      const di = (dy * inputSize + dx) * 3;

      input[di] = r / 255;
      input[di + 1] = g / 255;
      input[di + 2] = b / 255;
    }
  }

  const letterbox: LetterboxMeta = {
    inputSize,
    scale,
    padX,
    padY,
    resizedWidth,
    resizedHeight,
  };

  return { input, letterbox, srcFrameSize: { width: srcW, height: srcH } };
}

