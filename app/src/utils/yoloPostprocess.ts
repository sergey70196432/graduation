import type { Detection, FrameSize } from '../types/detection';
import { nonMaxSuppression } from './nms';

export type YoloOutputLayout = 'CN' | 'NC';

export type YoloOutputInfo = {
  // Кол-во боксов (например 8400)
  numBoxes: number;
  // Кол-во каналов на бокс (4 + numClasses)
  channels: number;
  // Как лежит массив: [channels, numBoxes] или [numBoxes, channels]
  layout: YoloOutputLayout;
};

export type LetterboxMeta = {
  inputSize: number;
  scale: number;
  padX: number;
  padY: number;
  resizedWidth: number;
  resizedHeight: number;
};

function inferYoloOutputInfo(
  shape: readonly number[] | undefined,
  numClasses: number
): YoloOutputInfo | null {
  'worklet';
  const channels = 4 + numClasses;
  if (!shape || shape.length === 0) return null;

  // Убираем ведущие "1", они нам не важны.
  const dims: number[] = [];
  for (let i = 0; i < shape.length; i++) {
    const d = shape[i];
    if (d === undefined) continue;
    if (d !== 1) dims.push(d);
  }

  if (dims.length === 2) {
    const a0 = dims[0];
    const b0 = dims[1];
    const a = a0 === undefined ? 0 : a0;
    const b = b0 === undefined ? 0 : b0;
    if (a === channels) return { numBoxes: b, channels, layout: 'CN' };
    if (b === channels) return { numBoxes: a, channels, layout: 'NC' };
  }

  if (dims.length === 3) {
    // Типичный случай: [C, N, ?] не встречается, но на всякий случай.
    // Чаще всего бывает [1, C, N] или [1, N, C], после фильтрации 1 останется [C, N] или [N, C]
    const a0 = dims[0];
    const b0 = dims[1];
    const c0 = dims[2];
    const a = a0 === undefined ? 0 : a0;
    const b = b0 === undefined ? 0 : b0;
    const c = c0 === undefined ? 0 : c0;
    if (a === channels) return { numBoxes: b * c, channels, layout: 'CN' };
    if (c === channels) return { numBoxes: a * b, channels, layout: 'NC' };
  }

  // TODO(рус): если твой экспорт даёт другую форму, посмотри логи shape и поправь это место.
  return null;
}

function getValue(
  data: Float32Array,
  info: YoloOutputInfo,
  boxIndex: number,
  channelIndex: number
): number {
  'worklet';
  if (info.layout === 'NC') {
    const idx = boxIndex * info.channels + channelIndex;
    const v = data[idx];
    return v === undefined ? 0 : v;
  }
  // 'CN'
  const idx = channelIndex * info.numBoxes + boxIndex;
  const v = data[idx];
  return v === undefined ? 0 : v;
}

function clip(v: number, min: number, max: number): number {
  'worklet';
  return Math.max(min, Math.min(max, v));
}

function sigmoid(x: number): number {
  'worklet';
  // Защита от overflow на очень больших значениях.
  if (x >= 20) return 1;
  if (x <= -20) return 0;
  return 1 / (1 + Math.exp(-x));
}

export function decodeYoloV8Detections(
  output0: Float32Array,
  output0Shape: readonly number[] | undefined,
  labels: readonly string[],
  frameSize: FrameSize,
  letterbox: LetterboxMeta,
  confidenceThreshold: number,
  iouThreshold: number,
  preNmsTopK: number,
  postNmsTopK: number
): Detection[] {
  'worklet';
  const numClasses = labels.length;
  const info = inferYoloOutputInfo(output0Shape, numClasses);
  if (!info) return [];

  // Эвристики под разные экспорты:
  // - иногда class scores — это logits (не 0..1), тогда нужно sigmoid
  // - иногда bbox координаты нормализованные (0..1), тогда нужно умножить на inputSize
  const scanBoxes = Math.min(info.numBoxes, 40);
  const scanClasses = Math.min(numClasses, 20);

  let minScore = 1e9;
  let maxScore = -1e9;
  let maxWH = 0;

  for (let i = 0; i < scanBoxes; i++) {
    const w = getValue(output0, info, i, 2);
    const h = getValue(output0, info, i, 3);
    if (w > maxWH) maxWH = w;
    if (h > maxWH) maxWH = h;

    for (let c = 0; c < scanClasses; c++) {
      const s = getValue(output0, info, i, 4 + c);
      if (s < minScore) minScore = s;
      if (s > maxScore) maxScore = s;
    }
  }

  const scoresLookLikeLogits = minScore < -0.01 || maxScore > 1.01;
  const coordsLookNormalized = maxWH <= 2.0;

  const candidates: Detection[] = [];

  for (let i = 0; i < info.numBoxes; i++) {
    let cx = getValue(output0, info, i, 0);
    let cy = getValue(output0, info, i, 1);
    let w = getValue(output0, info, i, 2);
    let h = getValue(output0, info, i, 3);

    if (coordsLookNormalized) {
      cx *= letterbox.inputSize;
      cy *= letterbox.inputSize;
      w *= letterbox.inputSize;
      h *= letterbox.inputSize;
    }

    let bestClass = -1;
    let bestScore = 0;
    for (let c = 0; c < numClasses; c++) {
      const raw = getValue(output0, info, i, 4 + c);
      const s = scoresLookLikeLogits ? sigmoid(raw) : raw;
      if (s > bestScore) {
        bestScore = s;
        bestClass = c;
      }
    }

    if (bestClass < 0 || bestScore < confidenceThreshold) continue;

    // xywh -> x1y1x2y2 на входе модели (letterbox inputSize x inputSize)
    const x1 = cx - w / 2;
    const y1 = cy - h / 2;
    const x2 = cx + w / 2;
    const y2 = cy + h / 2;

    // Маппинг обратно из letterbox в координаты исходного кадра камеры.
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
    if (bw <= 1 || bh <= 1) continue;

    const label0 = labels[bestClass];
    const label = label0 === undefined ? String(bestClass) : label0;
    candidates.push({
      bbox: { x: clippedX1, y: clippedY1, width: bw, height: bh },
      classId: bestClass,
      label,
      confidence: bestScore,
    });
  }

  // Обрезаем до topK до NMS по confidence.
  // Сортируем по confidence по убыванию (простая сортировка, без колбэков).
  for (let i = 0; i < candidates.length; i++) {
    let bestI = i;
    let bestS = candidates[i] ? candidates[i]!.confidence : -1;
    for (let j = i + 1; j < candidates.length; j++) {
      const s = candidates[j] ? candidates[j]!.confidence : -1;
      if (s > bestS) {
        bestS = s;
        bestI = j;
      }
    }
    if (bestI !== i) {
      const tmp = candidates[i]!;
      candidates[i] = candidates[bestI]!;
      candidates[bestI] = tmp;
    }
  }

  const top: Detection[] = [];
  const k = Math.min(preNmsTopK, candidates.length);
  for (let i = 0; i < k; i++) {
    top.push(candidates[i]!);
  }

  const nms = nonMaxSuppression(top, iouThreshold, postNmsTopK);
  return nms;
}

