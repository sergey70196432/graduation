import { useEffect, useMemo, useRef, useState } from 'react';
import { Alert, Platform, Share } from 'react-native';
import {
  useFrameProcessor,
  type Frame,
} from 'react-native-vision-camera';
import { useResizePlugin } from 'vision-camera-resize-plugin';
import { useRunOnJS, useSharedValue } from 'react-native-worklets-core';
import * as RNFS from 'react-native-fs';
import type { Detection, DetectorStats, FrameSize } from '../types/detection';
import { decodeYoloV8Detections, type LetterboxMeta } from '../utils/yoloPostprocess';
import { useYoloModel } from './useYoloModel';

function readBuildFlag(name: string): boolean {
  // RN/Expo: env может быть “вшит” на этапе сборки, либо отсутствовать в рантайме.
  const g = globalThis as unknown as {
    process?: { env?: Record<string, unknown> };
  };
  const env = g.process?.env;
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
    downloadProgressById,
    manifestGeneratedAt,
    manifestUrl,
    storageRoot,
  } = useYoloModel();

  const [stats, setStats] = useState<DetectorStats>({
    fps: 0,
    lastInferenceMs: 0,
    lastTotalMs: 0,
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
        `# iouThreshold=${activeYoloParams.iouThreshold}`,
        `# preNmsTopK=${activeYoloParams.preNmsTopK}`,
        `# postNmsTopK=${activeYoloParams.postNmsTopK}`,
        `# platform=${Platform.OS}`,
        `#`,
        `t_ms\tframe_w\tframe_h\tresized_w\tresized_h\tpad_x\tpad_y\tscale\tresize_ms\tletterbox_ms\tinference_ms\tdecode_ms\ttotal_ms\tdropped\tobjects`,
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

  const output0Shape = useMemo(() => {
    const s = model?.outputs?.[0]?.shape;
    return Array.isArray(s) ? s : undefined;
  }, [model]);

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
        const decoded = decodeYoloV8Detections(
          out0,
          output0Shape,
          labels,
          { width: frameW, height: frameH },
          letterbox,
          activeYoloParams.confidenceThreshold,
          activeYoloParams.iouThreshold,
          activeYoloParams.preNmsTopK,
          activeYoloParams.postNmsTopK
        );
        const decodeMs = Date.now() - tDecode0;
        const totalMs = Date.now() - tStart;

        stage = 'report';
        const now = Date.now();
        if (now - lastReportAtMs.value >= workletMinIntervalMs) {
          lastReportAtMs.value = now0;
          const dropped = droppedFrames.value;
          droppedFrames.value = 0;
          onWorkletResult({
            detections: decoded,
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
              totalMs,
              droppedFramesSinceLastReport: dropped,
              numDetections: decoded.length,
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
      output0Shape,
      droppedFrames,
      lastErrorAtMs,
      onWorkletError,
      activeYoloParams,
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
    activeModelItem,
    activeYoloParams,
    models,
    catalogState,
    catalogErrorMessage,
    refreshManifest,
    setActiveModel,
    ensureDownloaded,
    deleteDownloaded,
    downloadProgressById,
    manifestGeneratedAt,
    manifestUrl,
    storageRoot,
  };
}

