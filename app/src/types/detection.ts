export type BoundingBox = {
  // Координаты в пикселях исходного кадра камеры (frame).
  // (0,0) — левый верхний угол кадра.
  x: number;
  y: number;
  width: number;
  height: number;
};

export type Detection = {
  bbox: BoundingBox;
  classId: number;
  label: string;
  confidence: number; // 0..1
  /**
   * Optional refined classification (e.g. speed value).
   * When present, UI can prefer this label for display/grouping.
   */
  refinedLabel?: string;
  refinedConfidence?: number; // 0..1
};

export type FrameSize = {
  width: number;
  height: number;
};

export type DetectorStats = {
  fps: number;
  lastInferenceMs: number;
  lastTotalMs: number;
  resizeMs: number;
  letterboxMs: number;
  decodeMs: number;
  lastSpeedClsMs: number;
  lastSpeedClsRan: boolean;
  lastNumDetections: number;
  lastUpdatedAtMs: number;
  inputSize: number;
};

export type SpeedModelState = 'missing' | 'loading' | 'loaded' | 'error';

