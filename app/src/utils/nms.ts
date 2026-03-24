import type { Detection } from '../types/detection';

function iou(a: Detection['bbox'], b: Detection['bbox']): number {
  'worklet';
  const ax1 = a.x;
  const ay1 = a.y;
  const ax2 = a.x + a.width;
  const ay2 = a.y + a.height;

  const bx1 = b.x;
  const by1 = b.y;
  const bx2 = b.x + b.width;
  const by2 = b.y + b.height;

  const interX1 = Math.max(ax1, bx1);
  const interY1 = Math.max(ay1, by1);
  const interX2 = Math.min(ax2, bx2);
  const interY2 = Math.min(ay2, by2);

  const interW = Math.max(0, interX2 - interX1);
  const interH = Math.max(0, interY2 - interY1);
  const interArea = interW * interH;

  const areaA = Math.max(0, ax2 - ax1) * Math.max(0, ay2 - ay1);
  const areaB = Math.max(0, bx2 - bx1) * Math.max(0, by2 - by1);
  const union = areaA + areaB - interArea;
  if (union <= 0) return 0;
  return interArea / union;
}

export function nonMaxSuppression(
  detections: Detection[],
  iouThreshold: number,
  limit: number
): Detection[] {
  'worklet';
  if (detections.length === 0) return [];

  // Сортируем по confidence по убыванию (без колбэков, чтобы было worklet-safe).
  const sorted: Detection[] = [];
  for (let i = 0; i < detections.length; i++) sorted.push(detections[i]!);
  for (let i = 0; i < sorted.length; i++) {
    let bestI = i;
    let bestS = sorted[i] ? sorted[i]!.confidence : -1;
    for (let j = i + 1; j < sorted.length; j++) {
      const s = sorted[j] ? sorted[j]!.confidence : -1;
      if (s > bestS) {
        bestS = s;
        bestI = j;
      }
    }
    if (bestI !== i) {
      const tmp = sorted[i]!;
      sorted[i] = sorted[bestI]!;
      sorted[bestI] = tmp;
    }
  }

  const selected: Detection[] = [];
  const used = new Array(sorted.length).fill(false) as boolean[];

  for (let i = 0; i < sorted.length; i++) {
    if (used[i]) continue;
    const cur = sorted[i]!;
    selected.push(cur);
    if (selected.length >= limit) break;

    for (let j = i + 1; j < sorted.length; j++) {
      if (used[j]) continue;
      const other = sorted[j]!;
      if (cur.classId !== other.classId) continue; // NMS отдельно по классам
      if (iou(cur.bbox, other.bbox) >= iouThreshold) used[j] = true;
    }
  }

  return selected;
}

