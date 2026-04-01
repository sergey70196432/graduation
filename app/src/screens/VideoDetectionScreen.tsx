import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  ActivityIndicator,
  Pressable,
  StyleSheet,
  Text,
  useWindowDimensions,
  View,
} from 'react-native';
import DocumentPicker from 'react-native-document-picker';
import { createThumbnail } from 'react-native-create-thumbnail';
import type { TensorflowModel } from 'react-native-fast-tflite';
import { SafeAreaView, useSafeAreaInsets } from 'react-native-safe-area-context';
import * as RNFS from 'react-native-fs';
import { launchImageLibrary } from 'react-native-image-picker';
import Video from 'react-native-video';
import { DetectionOverlay } from '../components/DetectionOverlay';
import type { Detection, FrameSize } from '../types/detection';
import { decodeJpegToRgba } from '../utils/jpegDecode.ts';
import { buildLetterboxedFloatInput } from '../utils/rgbLetterbox.ts';
import { decodeYoloV8DetectionsEmbeddedNms } from '../utils/yoloPostprocess';
import { useYoloModel } from '../hooks/useYoloModel';

type Thumb = {
  path: string;
  width: number;
  height: number;
};

function toFileUriMaybe(path: string) {
  if (path.startsWith('file://')) return path;
  return `file://${path}`;
}

function stripFileScheme(p: string) {
  return p.startsWith('file://') ? p.slice('file://'.length) : p;
}

async function ensureVideoIsLocalFileUri(picked: {
  uri?: string;
  fileCopyUri?: string | null;
  name?: string | null;
}): Promise<string> {
  // 1) Лучший кейс: DocumentPicker сам скопировал файл в cachesDirectory.
  const copyUri = picked.fileCopyUri ?? undefined;
  if (copyUri) return toFileUriMaybe(copyUri);

  // 2) Иначе пробуем скопировать сами (пока у нас ещё есть доступ к security-scoped URL).
  const srcUri = picked.uri;
  if (!srcUri) throw new Error('Не удалось получить uri для видео.');
  if (!srcUri.startsWith('file://')) {
    throw new Error(
      `Ожидался file:// uri. Получено: ${srcUri}. Попробуй выбрать видео из "Фото" или сохранить файл локально.`
    );
  }

  const name = picked.name?.trim() || `video-${Date.now()}.mp4`;
  const dstPath = `${RNFS.CachesDirectoryPath}/yolo-video-test-${Date.now()}-${name}`;
  await RNFS.copyFile(stripFileScheme(srcUri), dstPath);
  return toFileUriMaybe(dstPath);
}

function humanizeVideoOpenError(err: unknown): string {
  const s = String(err);
  // Частая ошибка AVFoundation: iOS не смог открыть файл (кодек/формат/файл не локальный).
  if (s.includes('AVFoundationErrorDomain') && s.includes('Code=-11829')) {
    return (
      'iOS не смог открыть это видео (AVFoundation -11829 "Cannot Open").\n\n' +
      'Чаще всего причины такие:\n' +
      '- файл в iCloud/Files ещё не скачан локально\n' +
      '- формат/кодек не поддерживается iOS (например .mkv / hevc в некоторых кейсах)\n\n' +
      'Попробуй видео в MP4 (H.264) или сначала сохрани файл локально на устройство.'
    );
  }
  if (s.includes('NSCocoaErrorDomain') && s.includes('Code=257')) {
    return (
      'iOS не дал доступ к файлу (NSCocoaErrorDomain 257).\n\n' +
      'Так бывает, когда видео выбрано из Files/iCloud и доступ "security-scoped".\n\n' +
      'Что попробовать:\n' +
      '- выбери видео из "Фото" (Camera Roll)\n' +
      '- или сначала "Сохранить видео" локально на устройство\n' +
      '- или другой файл (MP4 H.264)\n\n' +
      'Мы уже пытаемся копировать видео в cache приложения, но iOS иногда не даёт доступ к источнику.'
    );
  }
  return s;
}

export function VideoDetectionScreen(props: { onBack?: () => void }) {
  const insets = useSafeAreaInsets();
  const { width: windowW, height: windowH } = useWindowDimensions();
  const isLandscape = windowW > windowH;
  const sidePanelWidth = useMemo(() => {
    if (!isLandscape) return undefined;
    const w = Math.round(windowW * 0.34);
    return Math.max(280, Math.min(380, w));
  }, [isLandscape, windowW]);

  const {
    model,
    labels,
    modelState,
    modelErrorMessage,
    activeYoloParams,
  } = useYoloModel();
  const isLoadingModel = modelState === 'loading';
  const modelError = modelState === 'error' ? modelErrorMessage : null;

  const [videoUri, setVideoUri] = useState<string | null>(null);
  const [timestampMs, setTimestampMs] = useState<number>(0);

  const [thumb, setThumb] = useState<Thumb | null>(null);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [frameSize, setFrameSize] = useState<FrameSize | null>(null);
  const [viewSize, setViewSize] = useState<{ width: number; height: number }>({
    width: 0,
    height: 0,
  });
  const [lastInferenceMs, setLastInferenceMs] = useState<number>(0);
  const [_isProcessing, setIsProcessing] = useState(false);
  const [processingError, setProcessingError] = useState<string | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [videoPositionMs, setVideoPositionMs] = useState(0);

  const processingRef = useRef(false);
  const loopTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const lastInferredTsMsRef = useRef<number>(-1);

  const onPickVideoFromFiles = useCallback(async () => {
    try {
      setProcessingError(null);
      // Важно для iOS: файл из Files/iCloud может быть "security-scoped" и не читаться нативными API.
      // Поэтому просим DocumentPicker СКОПИРОВАТЬ файл в sandbox приложения (cache).
      const picked = await DocumentPicker.pickSingle({
        type: [DocumentPicker.types.video],
        // На iOS 'import' чаще всего даёт доступ + копирование в sandbox.
        mode: 'import',
        copyTo: 'cachesDirectory',
      });

      const uri = await ensureVideoIsLocalFileUri({
        uri: picked.uri,
        fileCopyUri: picked.fileCopyUri,
        name: picked.name,
      });

      setVideoUri(uri);
      setTimestampMs(0);
      setThumb(null);
      setDetections([]);
      setFrameSize(null);
      setVideoPositionMs(0);
      setIsPlaying(false);
    } catch (e) {
      if (DocumentPicker.isCancel(e)) return;
      setProcessingError(humanizeVideoOpenError(e));
    }
  }, []);

  const onPickVideoFromPhotos = useCallback(async () => {
    try {
      setProcessingError(null);
      const res = await launchImageLibrary({
        mediaType: 'video',
        selectionLimit: 1,
      });
      if (res.didCancel) return;
      if (res.errorCode) {
        throw new Error(`${res.errorCode}: ${res.errorMessage ?? 'no message'}`);
      }
      const asset = res.assets?.[0];
      const uri = asset?.uri;
      if (!uri) throw new Error('Не удалось получить uri выбранного видео.');

      // Обычно image-picker отдаёт локальный file:// путь, который нормально читается нативкой.
      setVideoUri(toFileUriMaybe(uri));
      setTimestampMs(0);
      setThumb(null);
      setDetections([]);
      setFrameSize(null);
      setVideoPositionMs(0);
      setIsPlaying(false);
    } catch (e) {
      setProcessingError(humanizeVideoOpenError(e));
    }
  }, []);

  const inferFrame = useCallback(async (params: {
    labels: string[];
    model: TensorflowModel | undefined;
    timestampMs: number;
    videoUri: string | null;
  }) => {
    const labs = params.labels;
    const m = params.model;
    const tsMs = params.timestampMs;
    const vUri = params.videoUri;

    if (!vUri) return;
    if (!m) return;
    if (labs.length === 0) return;

    try {
      // Явные логи, чтобы было видно, что инференс реально идёт.
      console.log(`[video] infer frame start @${tsMs}ms`);
      const startInferFrame = Date.now();
      const startThumbnail = Date.now();
      const t = await createThumbnail({
        url: vUri,
        timeStamp: tsMs,
        format: 'jpeg',
        maxWidth: activeYoloParams.inputSize,
        maxHeight: activeYoloParams.inputSize,
        timeToleranceMs: 500,
      });
      setThumb({ path: t.path, width: t.width, height: t.height });
      console.log('thumbnail time', Date.now() - startThumbnail + 'ms');

      const startDecodeJpeg = Date.now();
      const decoded = await decodeJpegToRgba(t.path);
      console.log('decode time', Date.now() - startDecodeJpeg + 'ms');

      const startBuildLetterboxedFloatInput = Date.now();
      const { input, letterbox, srcFrameSize } = buildLetterboxedFloatInput(
        decoded,
        activeYoloParams.inputSize
      );
      console.log('build letterboxed float input time', Date.now() - startBuildLetterboxedFloatInput + 'ms');

      const t0 = Date.now();
      const startInference = Date.now();
      const outputs = m.runSync([input]);
      const inferenceMs = Date.now() - t0;
      setLastInferenceMs(inferenceMs);
      console.log('inference time', Date.now() - startInference + 'ms');

      const out0 = outputs[0];
      if (!(out0 instanceof Float32Array)) {
        throw new Error('Output[0] is not Float32Array. Для MVP ожидается float32 output.');
      }

      const startDecode = Date.now();
      const decodedDetections = decodeYoloV8DetectionsEmbeddedNms(
        out0,
        labs,
        srcFrameSize,
        letterbox,
        activeYoloParams.confidenceThreshold
      );

      console.log('decode time', Date.now() - startDecode + 'ms');

      setDetections(decodedDetections);
      setFrameSize(srcFrameSize);

      console.log(
        `[video] infer done @${tsMs}ms: ${decodedDetections.length} objects, ${inferenceMs}ms`
      );
      console.log('infer frame time', Date.now() - startInferFrame + 'ms');
    } catch (e) {
      setProcessingError(humanizeVideoOpenError(e));
    } finally {
      setIsProcessing(false);
      processingRef.current = false;
    }
  }, [activeYoloParams]);

  const onTogglePlay = useCallback(() => {
    setIsPlaying(v => !v);
  }, []);

  // Непрерывный авто-инференс:
  // - инференс всегда включён и не отключается
  // - как только обработка закончилась, берём текущий кадр (по таймкоду) и запускаем следующий прогон
  // - защита: если таймкод не меняется (например, видео на паузе) — не крутимся в tight-loop
  useEffect(() => {
    // стартуем только когда всё готово
    if (!videoUri) return;
    if (!model) return;
    if (labels.length === 0) return;

    let cancelled = false;

    const schedule = (ms: number) => {
      if (loopTimeoutRef.current) clearTimeout(loopTimeoutRef.current);
      loopTimeoutRef.current = setTimeout(tick, ms);
    };

    const tick = async () => {
      if (cancelled) return;
      if (processingRef.current) {
        schedule(30);
        return;
      }

      const ts = Math.max(0, Math.round(isPlaying ? videoPositionMs : timestampMs));
      const lastTs = lastInferredTsMsRef.current;

      // Если кадр "не сдвинулся", не делаем бесконечный прогон одного и того же.
      if (ts === lastTs) {
        schedule(isPlaying ? 30 : 200);
        return;
      }

      lastInferredTsMsRef.current = ts;
      processingRef.current = true;
      setIsProcessing(true);
      setProcessingError(null);
      try {
        await inferFrame({
          labels,
          model,
          timestampMs: ts,
          videoUri,
        });
      } catch (e) {
        setProcessingError(humanizeVideoOpenError(e));
      } finally {
        processingRef.current = false;
        setIsProcessing(false);
        schedule(0);
      }
    };

    schedule(0);
    return () => {
      cancelled = true;
      if (loopTimeoutRef.current) clearTimeout(loopTimeoutRef.current);
      loopTimeoutRef.current = null;
    };
  }, [
    inferFrame,
    isPlaying,
    labels,
    model,
    timestampMs,
    videoPositionMs,
    videoUri,
  ]);

  const onPrev = useCallback(() => {
    setTimestampMs(t => Math.max(0, t - 500));
  }, []);
  const onNext = useCallback(() => {
    setTimestampMs(t => t + 500);
  }, []);

  console.log('Render');
  console.log('isPlaying', isPlaying);
  
  return (
    <View style={styles.root}>
      <SafeAreaView edges={['top']} style={styles.topBarSafeArea}>
        <View style={styles.topBar}>
          <Pressable
            style={({ pressed }) => [
              styles.smallButton,
              pressed && styles.buttonPressed,
            ]}
            onPress={props.onBack}
          >
            <Text style={styles.smallButtonText}>Back</Text>
          </Pressable>

          <Pressable
            style={({ pressed }) => [
              styles.smallButton,
              pressed && styles.buttonPressed,
            ]}
            onPress={onPickVideoFromPhotos}
          >
            <Text style={styles.smallButtonText}>Pick (Photos)</Text>
          </Pressable>
        </View>
      </SafeAreaView>

      <View style={[styles.content, isLandscape && styles.contentLandscape]}>
        <View
          style={styles.preview}
          onLayout={e => {
            const { width, height } = e.nativeEvent.layout;
            setViewSize({ width, height });
          }}
        >
          {videoUri ? (
            <>
              <Video
                source={{ uri: videoUri }}
                style={StyleSheet.absoluteFill}
                resizeMode="contain"
                paused={!isPlaying}
                controls={true}
                onProgress={p => {
                  // react-native-video отдаёт секунды
                  setVideoPositionMs(p.currentTime * 1000);
                }}
                onError={e => {
                  setProcessingError(`Video error: ${JSON.stringify(e)}`);
                }}
              />
              <DetectionOverlay
                detections={detections}
                frameSize={frameSize}
                viewSize={viewSize}
              />
            </>
          ) : (
            <View style={styles.center}>
              <Text style={styles.title}>Тест по видео</Text>
              <Text style={styles.text}>
                Выбери видео с телефона, затем нажми "Infer frame".
              </Text>
            </View>
          )}
        </View>

        <View
          style={[
            styles.panel,
            isLandscape ? styles.panelLandscape : styles.panelPortrait,
            isLandscape ? { width: sidePanelWidth } : null,
            {
              paddingBottom: Math.max(insets.bottom, 12),
              paddingRight: Math.max(insets.right, 12),
            },
          ]}
        >
          {isLoadingModel && (
            <View style={styles.loadingInline}>
              <ActivityIndicator color="#fff" size="small" />
              <Text style={styles.loadingInlineText}>Загружаю модель…</Text>
            </View>
          )}

          {modelError && (
            <View style={styles.errorBox}>
              <Text style={styles.errorTitle}>Модель не загрузилась</Text>
              <Text style={styles.errorText}>{modelError}</Text>
            </View>
          )}

          <View style={styles.metrics}>
            <Metric label="Video" value={videoUri ? 'selected' : '—'} />
            <Metric label="t, ms" value={String(Math.round(videoPositionMs))} />
            <Metric label="Infer, ms" value={lastInferenceMs.toFixed(1)} />
            <Metric label="Objects" value={String(detections.length)} />
          </View>

          <View style={[styles.row, styles.rowMt10]}>
            <Pressable
              style={({ pressed }) => [styles.button, pressed && styles.buttonPressed]}
              onPress={onTogglePlay}
              disabled={!videoUri}
            >
              <Text style={styles.buttonText}>{isPlaying ? 'Pause' : 'Play'}</Text>
            </Pressable>
          </View>

          <View style={[styles.row, styles.rowMt10]}>
            <Pressable
              style={({ pressed }) => [styles.button, pressed && styles.buttonPressed]}
              onPress={onPickVideoFromFiles}
            >
              <Text style={styles.buttonText}>Pick (Files)</Text>
            </Pressable>
          </View>


          <View style={[styles.row, styles.rowMt10]}>
            <Pressable
              style={({ pressed }) => [styles.button, pressed && styles.buttonPressed]}
              onPress={onPrev}
              disabled={!videoUri}
            >
              <Text style={styles.buttonText}>-0.5s</Text>
            </Pressable>
            <Pressable
              style={({ pressed }) => [styles.button, pressed && styles.buttonPressed]}
              onPress={onNext}
              disabled={!videoUri}
            >
              <Text style={styles.buttonText}>+0.5s</Text>
            </Pressable>
          </View>

          {processingError && (
            <View style={styles.errorBox}>
              <Text style={styles.errorTitle}>Ошибка</Text>
              <Text style={styles.errorText}>{processingError}</Text>
            </View>
          )}

          {thumb && (
            <Text style={styles.hint}>
              Thumb: {thumb.width}x{thumb.height}
            </Text>
          )}
        </View>
      </View>
    </View>
  );
}

function Metric(props: { label: string; value: string }) {
  return (
    <View style={styles.metric}>
      <Text style={styles.metricLabel}>{props.label}</Text>
      <Text style={styles.metricValue}>{props.value}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: '#000' },
  topBarSafeArea: {
    backgroundColor: 'rgba(15, 15, 18, 0.95)',
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: 'rgba(255,255,255,0.15)',
  },
  topBar: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingHorizontal: 12,
    paddingVertical: 10,
  },
  smallButton: {
    paddingVertical: 8,
    paddingHorizontal: 10,
    borderRadius: 10,
    backgroundColor: 'rgba(255,255,255,0.08)',
  },
  smallButtonText: { color: '#fff', fontSize: 13, fontWeight: '700' },
  content: {
    flex: 1,
    flexDirection: 'column',
  },
  contentLandscape: {
    flexDirection: 'row',
  },
  preview: { flex: 1, backgroundColor: '#000' },
  panel: {
    backgroundColor: 'rgba(15, 15, 18, 0.95)',
    paddingHorizontal: 12,
    paddingTop: 10,
  },
  panelPortrait: {
    borderTopWidth: StyleSheet.hairlineWidth,
    borderTopColor: 'rgba(255,255,255,0.15)',
  },
  panelLandscape: {
    borderLeftWidth: StyleSheet.hairlineWidth,
    borderLeftColor: 'rgba(255,255,255,0.15)',
  },
  row: {
    flexDirection: 'row',
    gap: 10,
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  rowMt10: {
    marginTop: 10,
  },
  button: {
    flex: 1,
    borderRadius: 10,
    paddingVertical: 12,
    paddingHorizontal: 10,
    backgroundColor: '#2b2b33',
    alignItems: 'center',
  },
  buttonPressed: { opacity: 0.85 },
  buttonDisabled: { opacity: 0.5 },
  buttonText: { color: '#fff', fontSize: 14, fontWeight: '600' },
  metrics: {
    marginTop: 10,
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
  },
  metric: {
    minWidth: 120,
    paddingVertical: 8,
    paddingHorizontal: 10,
    borderRadius: 10,
    backgroundColor: 'rgba(255,255,255,0.06)',
  },
  metricLabel: { color: 'rgba(255,255,255,0.65)', fontSize: 12 },
  metricValue: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '700',
    marginTop: 2,
  },
  center: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 20,
  },
  title: { color: '#fff', fontSize: 20, fontWeight: '800', marginBottom: 8 },
  text: {
    color: 'rgba(255,255,255,0.75)',
    fontSize: 14,
    textAlign: 'center',
    marginBottom: 14,
  },
  loadingInline: {
    marginTop: 10,
    paddingVertical: 10,
    paddingHorizontal: 12,
    borderRadius: 12,
    backgroundColor: 'rgba(255,255,255,0.06)',
    flexDirection: 'row',
    gap: 10,
    alignItems: 'center',
  },
  loadingInlineText: { color: '#fff', fontSize: 13, fontWeight: '600' },
  errorBox: {
    marginTop: 10,
    paddingVertical: 10,
    paddingHorizontal: 12,
    borderRadius: 12,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: 'rgba(255,255,255,0.15)',
    backgroundColor: 'rgba(220, 38, 38, 0.12)',
  },
  errorTitle: { color: '#fff', fontSize: 14, fontWeight: '800', marginBottom: 6 },
  errorText: { color: 'rgba(255,255,255,0.8)', fontSize: 12, lineHeight: 16 },
  hint: { marginTop: 8, color: 'rgba(255,255,255,0.5)', fontSize: 12 },
});

