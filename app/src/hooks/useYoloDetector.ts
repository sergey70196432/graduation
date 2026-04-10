import { useEffect, useRef, useState } from 'react';
import { Alert, Platform, Share } from 'react-native';
import {
  useFrameProcessor,
  type Frame,
} from 'react-native-vision-camera';
import { useResizePlugin } from 'vision-camera-resize-plugin';
import { useRunOnJS, useSharedValue } from 'react-native-worklets-core';
import * as RNFS from 'react-native-fs';
import type { Detection, DetectorStats, FrameSize } from '../types/detection';
import {
  decodeYoloV8DetectionsEmbeddedNms,
  type LetterboxMeta,
} from '../utils/yoloPostprocess';
import { useYoloModel } from './useYoloModel';
import { useSpeedClassifierModel } from './useSpeedClassifierModel';
import Config from 'react-native-config';

const SPEED_CLS_INPUT_SIZE = 128;
const SPEED_CLS_MIN_ROI_PX = 10;
const SPEED_CLS_MARGIN = 0.05;
const SPEED_CLS_MIN_CONF = 0.01;

// ВАЖНО: на обучении используется torchvision Normalize(ImageNet):
// x = (x/255 - mean) / std
const SPEED_CLS_MEAN_R = 0.485;
const SPEED_CLS_MEAN_G = 0.456;
const SPEED_CLS_MEAN_B = 0.406;
const SPEED_CLS_STD_R = 0.229;
const SPEED_CLS_STD_G = 0.224;
const SPEED_CLS_STD_B = 0.225;

function clamp(v: number, lo: number, hi: number): number {
  'worklet';
  return Math.max(lo, Math.min(hi, v));
}

function isBaseInList(base: string, bases: readonly string[]): boolean {
  'worklet';
  for (let i = 0; i < bases.length; i++) {
    if (bases[i] === base) return true;
  }
  return false;
}

function argmax(a: Float32Array): number {
  'worklet';
  let bestI = 0;
  let bestV = a[0] ?? -1e9;
  for (let i = 1; i < a.length; i++) {
    const v = a[i] ?? -1e9;
    if (v > bestV) {
      bestV = v;
      bestI = i;
    }
  }
  return bestI;
}

function softmaxProbAt(a: Float32Array, idx: number): number {
  'worklet';
  // stable softmax probability for one index
  let m = -1e9;
  for (let i = 0; i < a.length; i++) {
    const v = a[i] ?? -1e9;
    if (v > m) m = v;
  }
  let sum = 0;
  let num = 0;
  for (let i = 0; i < a.length; i++) {
    const v = a[i] ?? -1e9;
    const e = Math.exp(v - m);
    sum += e;
    if (i === idx) num = e;
  }
  if (sum <= 0) return 0;
  return num / sum;
}

// NOTE: Раньше ROI для speed-cls резали из YOLO input tensor через bilinear.
// Сейчас ROI берём из оригинального кадра через resize-plugin, поэтому эта функция не нужна.

function readBuildFlag(name: string): boolean {
  const env = Config;
  const raw =
    env?.[name] ??
    env?.[`EXPO_PUBLIC_${name}`] ??
    env?.[`REACT_NATIVE_${name}`];

  if (raw == null) return false;
  const s = String(raw).trim().toLowerCase();
  return s === '1' || s === 'true' || s === 'yes' || s === 'on';
}

const WRITE_LOGS = readBuildFlag('WRITE_LOGS');

type FramePerf = {
  frameW: number;
  frameH: number;
  resizedW: number;
  resizedH: number;
  padX: number;
  padY: number;
  scale: number;
  resizeMs: number;
  letterboxMs: number;
  inferenceMs: number;
  decodeMs: number;
  speedClsMs: number;
  speedClsRan: boolean;
  totalMs: number;
  droppedFramesSinceLastReport: number;
  numDetections: number;
};

type UiPayload = {
  detections: Detection[];
  inferenceMs: number;
  frameSize: FrameSize;
  updatedAtMs: number;
  perf: FramePerf;
};

export function useYoloDetector() {
  const [isDetecting, setIsDetecting] = useState(false);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [lastFrameSize, setLastFrameSize] = useState<FrameSize | null>(null);

  const {
    model,
    labels,
    isModelLoaded,
    modelState,
    modelErrorMessage,
    activeYoloParams,
    activeModelItem,
    models,
    catalogState,
    catalogErrorMessage,
    refreshManifest,
    setActiveModel,
    ensureDownloaded,
    deleteDownloaded,
    isClearingStorage,
    clearStorage,
    downloadProgressById,
    manifestGeneratedAt,
    manifestUrl,
    storageRoot,
  } = useYoloModel();

  const {
    model: speedModel,
    labels: speedLabels,
    speedBases,
    isLoaded: isSpeedModelLoaded,
    state: speedModelState,
    errorMessage: speedModelErrorMessage,
  } = useSpeedClassifierModel();

  const [stats, setStats] = useState<DetectorStats>({
    fps: 0,
    lastInferenceMs: 0,
    lastTotalMs: 0,
    resizeMs: 0,
    letterboxMs: 0,
    decodeMs: 0,
    lastSpeedClsMs: 0,
    lastSpeedClsRan: false,
    lastNumDetections: 0,
    lastUpdatedAtMs: 0,
    inputSize: activeYoloParams.inputSize,
  });

  useEffect(() => {
    setStats(s => ({ ...s, inputSize: activeYoloParams.inputSize }));
  }, [activeYoloParams.inputSize]);

  const { resize } = useResizePlugin();

  const isLoggingRef = useRef(false);
  const logLinesRef = useRef<string[]>([]);
  const wasDetectingRef = useRef(false);
  const [lastLogFilePath, setLastLogFilePath] = useState<string | null>(null);

  // Считаем FPS на стороне JS (по тому, как часто приходят обновления).
  const fpsRef = useRef({
    windowStartMs: 0,
    frames: 0,
    fps: 0,
  });

  useEffect(() => {
    if (!WRITE_LOGS) {
      isLoggingRef.current = false;
      logLinesRef.current = [];
      if (isDetecting) setLastLogFilePath(null);
      return;
    }

    const was = wasDetectingRef.current;
    wasDetectingRef.current = isDetecting;

    if (isDetecting && !was) {
      isLoggingRef.current = true;
      setLastLogFilePath(null);

      const startedAtIso = new Date().toISOString();
      const header = [
        `# YOLO camera perf log`,
        `# startedAt=${startedAtIso}`,
        `# inputSize=${activeYoloParams.inputSize}`,
        `# confidenceThreshold=${activeYoloParams.confidenceThreshold}`,
        `# preNmsTopK=${activeYoloParams.preNmsTopK}`,
        `# postNmsTopK=${activeYoloParams.postNmsTopK}`,
        `# platform=${Platform.OS}`,
        `#`,
        `t_ms\tframe_w\tframe_h\tresized_w\tresized_h\tpad_x\tpad_y\tscale\tresize_ms\tletterbox_ms\tinference_ms\tdecode_ms\tspeed_cls_ms\tspeed_cls_ran\ttotal_ms\tdropped\tobjects`,
      ];
      logLinesRef.current = header;
      return;
    }

    if (!isDetecting && was) {
      isLoggingRef.current = false;
      const lines = logLinesRef.current;
      logLinesRef.current = [];
      if (lines.length <= 1) return;

      (async () => {
        try {
          const stoppedAtIso = new Date().toISOString();
          const content = `${lines.join('\n')}\n# stoppedAt=${stoppedAtIso}\n`;
          const fileName = `yolo-camera-perf-${stoppedAtIso.replace(/[:.]/g, '-')}.txt`;
          const filePath = `${RNFS.DocumentDirectoryPath}/${fileName}`;
          await RNFS.writeFile(filePath, content, 'utf8');

          const fileUrl = `file://${filePath}`;
          setLastLogFilePath(fileUrl);

          // На iOS это позволяет “Сохранить в Файлы”. На Android зависит от shell/провайдера,
          // но хотя бы не теряем файл в sandbox.
          try {
            await Share.share({
              title: 'YOLO camera perf log',
              url: fileUrl,
              message: 'Лог производительности детектора с камеры (txt).',
            });
          } catch (e) {
            Alert.alert('Логи сохранены', filePath, [
              { text: 'OK' },
            ]);
            console.warn('[logs] Share failed:', e);
          }
        } catch (e) {
          console.warn('[logs] export failed:', e);
          Alert.alert('Не удалось сохранить логи', String(e));
        }
      })().catch(() => {
        // ignore
      });
    }
  }, [activeYoloParams, isDetecting]);

  const onWorkletResult = useRunOnJS((payload: UiPayload) => {
    const now = payload.updatedAtMs;
    if (fpsRef.current.windowStartMs === 0) {
      fpsRef.current.windowStartMs = now;
      fpsRef.current.frames = 0;
      fpsRef.current.fps = 0;
    }

    fpsRef.current.frames += 1;
    const dt = now - fpsRef.current.windowStartMs;
    if (dt >= 1000) {
      fpsRef.current.fps = (fpsRef.current.frames * 1000) / dt;
      fpsRef.current.frames = 0;
      fpsRef.current.windowStartMs = now;
    }

    if (isLoggingRef.current) {
      const p = payload.perf;
      // Ограничиваем, чтобы случайно не сожрать память при долгом прогоне.
      if (logLinesRef.current.length < 20000) {
        logLinesRef.current.push(
          [
            now,
            p.frameW,
            p.frameH,
            p.resizedW,
            p.resizedH,
            p.padX,
            p.padY,
            p.scale.toFixed(6),
            p.resizeMs.toFixed(2),
            p.letterboxMs.toFixed(2),
            p.inferenceMs.toFixed(2),
            p.decodeMs.toFixed(2),
            p.speedClsMs.toFixed(2),
            p.speedClsRan ? 1 : 0,
            p.totalMs.toFixed(2),
            p.droppedFramesSinceLastReport,
            p.numDetections,
          ].join('\t')
        );
      }
    }

    setDetections(payload.detections);
    setLastFrameSize(payload.frameSize);
    setStats({
      fps: fpsRef.current.fps,
      lastInferenceMs: payload.inferenceMs,
      lastTotalMs: payload.perf.totalMs,
      resizeMs: payload.perf.resizeMs,
      letterboxMs: payload.perf.letterboxMs,
      decodeMs: payload.perf.decodeMs,
      lastSpeedClsMs: payload.perf.speedClsMs,
      lastSpeedClsRan: payload.perf.speedClsRan,
      lastNumDetections: payload.detections.length,
      lastUpdatedAtMs: payload.updatedAtMs,
      inputSize: activeYoloParams.inputSize,
    });
  }, [activeYoloParams.inputSize]);

  const onWorkletError = useRunOnJS((message: string) => {
    console.warn(message);
  }, []);

  const isProcessing = useSharedValue(false);
  const lastReportAtMs = useSharedValue(0);
  const droppedFrames = useSharedValue(0);
  const lastErrorAtMs = useSharedValue(0);

  const workletMinIntervalMs = Platform.OS === 'android' ? 800 : 300;

  const frameProcessor = useFrameProcessor(
    (frame: Frame) => {
      'worklet';
      if (!isDetecting) return;
      if (model == null) return;
      if (labels.length === 0) return;

      // Важно: не грузим video-thread постоянной инференс-работой.
      // Иначе на Android возможен abort из-за "SuspendAll timeout" (VisionCamera.video).
      const now0 = Date.now();
      if (now0 - lastReportAtMs.value < workletMinIntervalMs) {
        droppedFrames.value = droppedFrames.value + 1;
        return;
      }

      // Запускаем синхронно прямо в frameProcessor (без runAsync),
      // чтобы избежать "cannot be shared" при передаче closure в async runtime.
      let stage: string = 'start';
      try {
        // Не обрабатываем несколько кадров параллельно.
        if (isProcessing.value) {
          droppedFrames.value = droppedFrames.value + 1;
          return;
        }
        isProcessing.value = true;

        const tStart = now0;
        const frameW = frame.width;
        const frameH = frame.height;

        // ROI debug previews were removed.

        const inputSize = activeYoloParams.inputSize;
        const scale = Math.min(inputSize / frameW, inputSize / frameH);
        const resizedWidth = Math.max(1, Math.round(frameW * scale));
        const resizedHeight = Math.max(1, Math.round(frameH * scale));
        const padX = Math.floor((inputSize - resizedWidth) / 2);
        const padY = Math.floor((inputSize - resizedHeight) / 2);

        // 1) Resize в RGB uint8
        stage = 'resize';
        const tResize0 = Date.now();
        const resized = resize(frame, {
          scale: { width: resizedWidth, height: resizedHeight },
          pixelFormat: 'rgb',
          dataType: 'uint8',
        }) as Uint8Array;
        const resizeMs = Date.now() - tResize0;

        // 2) Letterbox + нормализация 0..1
        stage = 'letterbox';
        const tLetter0 = Date.now();
        const len = inputSize * inputSize * 3;
        const gg = globalThis as unknown as {
          __yoloInputFloat?: Float32Array;
        };
        let inputFloat = gg.__yoloInputFloat;
        if (!(inputFloat instanceof Float32Array) || inputFloat.length !== len) {
          inputFloat = new Float32Array(len);
          gg.__yoloInputFloat = inputFloat;
        } else {
          inputFloat.fill(0);
        }
        const srcRowStride = resizedWidth * 3;
        const dstRowStride = inputSize * 3;
        const inv255 = 1 / 255;

        for (let y = 0; y < resizedHeight; y++) {
          const srcRow = y * srcRowStride;
          const dstRow = (y + padY) * dstRowStride + padX * 3;
          for (let x = 0; x < resizedWidth; x++) {
            const si = srcRow + x * 3;
            const di = dstRow + x * 3;
            const r = resized[si] ?? 0;
            const g = resized[si + 1] ?? 0;
            const b = resized[si + 2] ?? 0;
            inputFloat[di] = r * inv255;
            inputFloat[di + 1] = g * inv255;
            inputFloat[di + 2] = b * inv255;
          }
        }
        const letterboxMs = Date.now() - tLetter0;

        // 3) Inference
        stage = 'inference';
        const g = globalThis as unknown as {
          performance?: { now?: () => number };
        };
        const perfObj = g.performance;
        const nowFn = perfObj ? perfObj.now : undefined;
        const t0 = typeof nowFn === 'function' ? nowFn() : 0;
        const outputs = model.runSync([inputFloat]);
        const t1 = typeof nowFn === 'function' ? nowFn() : 0;
        const inferenceMs = t1 > 0 && t0 > 0 ? t1 - t0 : 0;

        stage = 'outputs';
        const out0 = outputs[0];
        if (!(out0 instanceof Float32Array)) {
          return;
        }

        const letterbox: LetterboxMeta = {
          inputSize,
          scale,
          padX,
          padY,
          resizedWidth,
          resizedHeight,
        };

        stage = 'decode';
        const tDecode0 = Date.now();
        const decoded = decodeYoloV8DetectionsEmbeddedNms(
          out0,
          labels,
          { width: frameW, height: frameH },
          letterbox,
          activeYoloParams.confidenceThreshold
        );
        const decodeMs = Date.now() - tDecode0;

        let speedClsMs = 0;
        let speedClsRan = false;

        // 4) Optional refinement: speed value classifier
        // Требования:
        // 1) прогоняем для всех детекций "скоростных" баз
        // 2) если уверенность < 50% => удаляем детекцию целиком
        // 3) ROI берём из оригинального кадра (frame), а не из YOLO input tensor
        stage = 'speed_cls';
        const tSpeed0 = Date.now();
        let decodedOut = decoded;
        if (
          speedModel != null &&
          speedLabels.length > 0 &&
          speedBases.length > 0 &&
          decoded.length > 0
        ) {
          const filtered: Detection[] = [];
          const gg2 = globalThis as unknown as {
            __speedClsInputFloat?: Float32Array;
          };
          const dstLen = SPEED_CLS_INPUT_SIZE * SPEED_CLS_INPUT_SIZE * 3;
          let speedInput = gg2.__speedClsInputFloat;
          if (!(speedInput instanceof Float32Array) || speedInput.length !== dstLen) {
            speedInput = new Float32Array(dstLen);
            gg2.__speedClsInputFloat = speedInput;
          }

          const inv255Speed = 1 / 255;
          for (let i = 0; i < decoded.length; i++) {
            const d = decoded[i];
            if (!d) continue;

            const raw = String(d.label ?? '');
            const base = raw.split('_')[0] ?? '';
            const isSpeedBase = !!base && isBaseInList(base, speedBases);
            if (!isSpeedBase) {
              filtered.push(d);
              continue;
            }

            // Если знак слишком маленький — не удаляем детекцию (важно для дебага).
            if (d.bbox.width < SPEED_CLS_MIN_ROI_PX || d.bbox.height < SPEED_CLS_MIN_ROI_PX) {
              filtered.push(d);
              continue;
            }

            // ROI в координатах ОРИГИНАЛЬНОГО кадра (frame), с небольшим margin.
            let x1 = d.bbox.x;
            let y1 = d.bbox.y;
            let x2 = d.bbox.x + d.bbox.width;
            let y2 = d.bbox.y + d.bbox.height;

            const w0 = Math.max(1, x2 - x1);
            const h0 = Math.max(1, y2 - y1);
            x1 = x1 - w0 * SPEED_CLS_MARGIN;
            y1 = y1 - h0 * SPEED_CLS_MARGIN;
            x2 = x2 + w0 * SPEED_CLS_MARGIN;
            y2 = y2 + h0 * SPEED_CLS_MARGIN;

            x1 = clamp(x1, 0, frameW - 1);
            y1 = clamp(y1, 0, frameH - 1);
            x2 = clamp(x2, 0, frameW - 1);
            y2 = clamp(y2, 0, frameH - 1);

            const roiW = Math.max(1, Math.round(x2 - x1));
            const roiH = Math.max(1, Math.round(y2 - y1));
            if (roiW < 2 || roiH < 2) {
              filtered.push(d);
              continue;
            }

            try {
              // Берём кроп из оригинального кадра и сразу ресайзим в вход классификатора.
              const roiU8 = resize(frame, {
                crop: {
                  x: Math.round(x1),
                  y: Math.round(y1),
                  width: roiW,
                  height: roiH,
                },
                scale: { width: SPEED_CLS_INPUT_SIZE, height: SPEED_CLS_INPUT_SIZE },
                pixelFormat: 'rgb',
                dataType: 'uint8',
              }) as Uint8Array;

              const invStdR = 1 / SPEED_CLS_STD_R;
              const invStdG = 1 / SPEED_CLS_STD_G;
              const invStdB = 1 / SPEED_CLS_STD_B;
              // roiU8 — HWC, RGB uint8
              for (let j = 0; j < dstLen; j += 3) {
                const r = (roiU8[j] ?? 0) * inv255Speed;
                const g1 = (roiU8[j + 1] ?? 0) * inv255Speed;
                const b = (roiU8[j + 2] ?? 0) * inv255Speed;
                speedInput[j] = (r - SPEED_CLS_MEAN_R) * invStdR;
                speedInput[j + 1] = (g1 - SPEED_CLS_MEAN_G) * invStdG;
                speedInput[j + 2] = (b - SPEED_CLS_MEAN_B) * invStdB;
              }

              const out2 = speedModel.runSync([speedInput]);
              const logits = out2[0];
              if (logits instanceof Float32Array) {
                speedClsRan = true;
                // ВАЖНО: YOLO уже хорошо различает "базу" знака (например 3.24 vs 3.25).
                // Поэтому ограничиваем выбор класса только этой базой, а speed-cls уточняет именно значение.
                // Это резко снижает путаницу между базами при похожих цифрах (50, 60, ...).
                let bestIdx = -1;
                let bestV = -1e9;
                for (let k = 0; k < logits.length && k < speedLabels.length; k++) {
                  const lbl = speedLabels[k];
                  if (typeof lbl !== 'string') continue;
                  const b = lbl.split('_')[0] ?? '';
                  if (b !== base) continue;
                  const v = logits[k] ?? -1e9;
                  if (v > bestV) {
                    bestV = v;
                    bestIdx = k;
                  }
                }

                let idx = bestIdx;
                if (idx < 0) {
                  // fallback: если по какой-то причине labels не содержат эту базу
                  idx = argmax(logits);
                }

                // Probability считаем по softmax. Если нашли базу — считаем softmax только по классам этой базы.
                let prob = 0;
                if (bestIdx >= 0) {
                  let m = -1e9;
                  for (let k = 0; k < logits.length && k < speedLabels.length; k++) {
                    const lbl = speedLabels[k];
                    if (typeof lbl !== 'string') continue;
                    const b = lbl.split('_')[0] ?? '';
                    if (b !== base) continue;
                    const v = logits[k] ?? -1e9;
                    if (v > m) m = v;
                  }
                  let sum = 0;
                  let num = 0;
                  for (let k = 0; k < logits.length && k < speedLabels.length; k++) {
                    const lbl = speedLabels[k];
                    if (typeof lbl !== 'string') continue;
                    const b = lbl.split('_')[0] ?? '';
                    if (b !== base) continue;
                    const v = logits[k] ?? -1e9;
                    const e = Math.exp(v - m);
                    sum += e;
                    if (k === bestIdx) num = e;
                  }
                  prob = sum > 0 ? num / sum : 0;
                } else {
                  prob = softmaxProbAt(logits, idx);
                }

                if (prob >= SPEED_CLS_MIN_CONF) {
                  const refined = speedLabels[idx];
                  if (typeof refined === 'string' && refined.length > 0) {
                    d.refinedLabel = refined;
                    d.refinedConfidence = prob;
                  }
                }
              }
            } catch {
              // Если ROI-кроп/ресайз на устройстве не поддержан/упал — не ломаем весь пайплайн.
              // Оставляем исходную детекцию без refined-результата.
              filtered.push(d);
              continue;
            }

            // Если дошли сюда — либо успешно уточнили, либо logits не Float32Array.
            // В обоих случаях оставляем детекцию (если logits невалидны — без refined).
            filtered.push(d);
          }
          decodedOut = filtered;
        }

        speedClsMs = Date.now() - tSpeed0;
        const totalMs = Date.now() - tStart;

        stage = 'report';
        const now = Date.now();
        if (now - lastReportAtMs.value >= workletMinIntervalMs) {
          lastReportAtMs.value = now0;
          const dropped = droppedFrames.value;
          droppedFrames.value = 0;
          onWorkletResult({
            detections: decodedOut,
            inferenceMs,
            frameSize: { width: frameW, height: frameH },
            updatedAtMs: now,
            perf: {
              frameW,
              frameH,
              resizedW: resizedWidth,
              resizedH: resizedHeight,
              padX,
              padY,
              scale,
              resizeMs,
              letterboxMs,
              inferenceMs,
              decodeMs,
              speedClsMs,
              speedClsRan,
              totalMs,
              droppedFramesSinceLastReport: dropped,
              numDetections: decodedOut.length,
            },
          });
        }
      } catch (e) {
        let msg = 'unknown';
        const anyE = e as unknown as { message?: unknown };
        if (typeof anyE === 'string') msg = anyE;
        else if (anyE && typeof anyE === 'object' && 'message' in anyE) {
          msg = String((anyE as { message?: unknown }).message);
        }
        const now = Date.now();
        if (now - lastErrorAtMs.value > 500) {
          lastErrorAtMs.value = now;
          onWorkletError(`[yolo] Ошибка в frame processor (stage=${stage}): ${msg}`);
        }
      } finally {
        isProcessing.value = false;
      }
    },
    [
      isDetecting,
      model,
      labels,
      resize,
      onWorkletResult,
      isProcessing,
      lastReportAtMs,
      droppedFrames,
      lastErrorAtMs,
      onWorkletError,
      activeYoloParams,
      speedModel,
      speedLabels,
      speedBases,
    ]
  );

  return {
    frameProcessor,
    detections,
    stats,
    isDetecting,
    setIsDetecting,
    isModelLoaded,
    modelState,
    modelErrorMessage,
    labels,
    lastFrameSize,
    lastLogFilePath,

    // models ui/actions
    hasSelectedModel: activeModelItem != null,
    activeModelItem,
    activeYoloParams,
    models,
    catalogState,
    catalogErrorMessage,
    refreshManifest,
    setActiveModel,
    ensureDownloaded,
    deleteDownloaded,
    isClearingStorage,
    clearStorage,
    downloadProgressById,
    manifestGeneratedAt,
    manifestUrl,
    storageRoot,

    // speed classifier (optional)
    isSpeedModelLoaded,
    speedModelState,
    speedModelErrorMessage,
    speedLabels,
    speedBases,
  };
}

