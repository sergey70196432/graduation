import type { Detection, FrameSize } from '../types/detection';

export type LetterboxMeta = {
  inputSize: number;
  scale: number;
  padX: number;
  padY: number;
  resizedWidth: number;
  resizedHeight: number;
};

function clip(v: number, min: number, max: number): number {
  'worklet';
  return Math.max(min, Math.min(max, v));
}

/**
 * Декодер для модели со "вшитым" NMS.
 *
 * Ожидаемый формат выходного тензора: [1, N, 6] или [N, 6]
 * где каждая строка: [x1, y1, x2, y2, score, classId]
 *
 * Координаты обычно в пикселях относительно letterbox inputSize, но иногда бывают 0..1.
 * Мы делаем простую проверку и при необходимости домножаем на inputSize.
 */
export function decodeYoloV8DetectionsEmbeddedNms(
  output0: Float32Array,
  labels: readonly string[],
  frameSize: FrameSize,
  letterbox: LetterboxMeta,
  confidenceThreshold: number
): Detection[] {
  'worklet';
  const STRIDE = 6;
  const n = Math.floor(output0.length / STRIDE);
  if (n <= 0) return [];

  // Эвристика: нормализованы ли координаты (0..1).
  // Сканируем немного первых боксов.
  let maxCoord = 0;
  const scan = Math.min(n, 20);
  for (let i = 0; i < scan; i++) {
    const base = i * STRIDE;
    const x2 = output0[base + 2] ?? 0;
    const y2 = output0[base + 3] ?? 0;
    if (x2 > maxCoord) maxCoord = x2;
    if (y2 > maxCoord) maxCoord = y2;
  }
  const coordsLookNormalized = maxCoord <= 1.5;

  const out: Detection[] = [];
  for (let i = 0; i < n; i++) {
    const base = i * STRIDE;
    let x1 = output0[base + 0] ?? 0;
    let y1 = output0[base + 1] ?? 0;
    let x2 = output0[base + 2] ?? 0;
    let y2 = output0[base + 3] ?? 0;
    const score = output0[base + 4] ?? 0;
    const clsRaw = output0[base + 5] ?? -1;

    if (!(score > confidenceThreshold)) continue;

    const classId = Math.round(clsRaw);
    if (classId < 0) continue;

    if (coordsLookNormalized) {
      x1 *= letterbox.inputSize;
      y1 *= letterbox.inputSize;
      x2 *= letterbox.inputSize;
      y2 *= letterbox.inputSize;
    }

    // Иногда экспорт отдаёт x2<x1 или y2<y1 (редко, но бывает).
    if (x2 < x1) {
      const t = x1;
      x1 = x2;
      x2 = t;
    }
    if (y2 < y1) {
      const t = y1;
      y1 = y2;
      y2 = t;
    }

    const fx1 = (x1 - letterbox.padX) / letterbox.scale;
    const fy1 = (y1 - letterbox.padY) / letterbox.scale;
    const fx2 = (x2 - letterbox.padX) / letterbox.scale;
    const fy2 = (y2 - letterbox.padY) / letterbox.scale;

    const clippedX1 = clip(fx1, 0, frameSize.width);
    const clippedY1 = clip(fy1, 0, frameSize.height);
    const clippedX2 = clip(fx2, 0, frameSize.width);
    const clippedY2 = clip(fy2, 0, frameSize.height);

    const bw = clippedX2 - clippedX1;
    const bh = clippedY2 - clippedY1;
    if (!(bw > 1) || !(bh > 1)) continue;

    const label0 = labels[classId];
    const label = label0 === undefined ? String(classId) : label0;
    out.push({
      bbox: { x: clippedX1, y: clippedY1, width: bw, height: bh },
      classId,
      label,
      confidence: score,
    });
  }

  return out;
}

