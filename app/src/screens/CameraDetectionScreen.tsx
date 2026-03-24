import React, { useCallback, useMemo, useState } from 'react';
import {
  ActivityIndicator,
  LayoutChangeEvent,
  Pressable,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import {
  Camera,
  useCameraDevice,
  useCameraPermission,
} from 'react-native-vision-camera';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { DetectionOverlay } from '../components/DetectionOverlay';
import { useYoloDetector } from '../hooks/useYoloDetector';
import type { FrameSize } from '../types/detection';

export function CameraDetectionScreen(props: { onOpenVideoTest?: () => void }) {
  const insets = useSafeAreaInsets();
  const { hasPermission, requestPermission } = useCameraPermission();

  const [cameraPosition, setCameraPosition] = useState<'back' | 'front'>(
    'back'
  );
  const device = useCameraDevice(cameraPosition);

  const {
    frameProcessor,
    detections,
    stats,
    isDetecting,
    setIsDetecting,
    isModelLoaded,
    modelState,
    modelErrorMessage,
    lastFrameSize,
    lastLogFilePath,
  } = useYoloDetector();

  const [previewSize, setPreviewSize] = useState<{ width: number; height: number }>({
    width: 0,
    height: 0,
  });

  const onPreviewLayout = useCallback((e: LayoutChangeEvent) => {
    const { width, height } = e.nativeEvent.layout;
    setPreviewSize({ width, height });
  }, []);

  const canShowCamera = hasPermission && device != null;

  const frameSize: FrameSize | null = useMemo(() => {
    if (!lastFrameSize) return null;
    return lastFrameSize;
  }, [lastFrameSize]);

  const onToggleDetection = useCallback(() => {
    setIsDetecting(v => !v);
  }, [setIsDetecting]);

  const onSwitchCamera = useCallback(() => {
    setCameraPosition(p => (p === 'back' ? 'front' : 'back'));
  }, []);

  const onOpenVideoTest = useCallback(() => {
    props.onOpenVideoTest?.();
  }, [props]);

  const permissionUi = (
    <View style={styles.center}>
      <Text style={styles.title}>Нужна камера</Text>
      <Text style={styles.text}>
        Разреши доступ к камере, чтобы запустить детекцию.
      </Text>
      <Pressable
        style={({ pressed }) => [styles.button, pressed && styles.buttonPressed]}
        onPress={requestPermission}
      >
        <Text style={styles.buttonText}>Разрешить</Text>
      </Pressable>
    </View>
  );

  const noDeviceUi = (
    <View style={styles.center}>
      <Text style={styles.title}>Камера недоступна</Text>
      <Text style={styles.text}>
        Похоже, камера не найдена. На iOS Simulator камеры обычно нет, поэтому
        детекция с камеры там не заработает.
        {'\n\n'}
        Запусти приложение на реальном iPhone.
      </Text>
      <Pressable
        style={({ pressed }) => [styles.button, pressed && styles.buttonPressed]}
        onPress={onSwitchCamera}
      >
        <Text style={styles.buttonText}>Попробовать другую камеру</Text>
      </Pressable>
    </View>
  );

  return (
    <View style={styles.root}>
      <View style={styles.preview} onLayout={onPreviewLayout}>
        {canShowCamera ? (
          <>
            <Camera
              style={StyleSheet.absoluteFill}
              device={device}
              isActive={true}
              frameProcessor={frameProcessor}
              resizeMode="contain"
            />

            <DetectionOverlay
              detections={detections}
              frameSize={frameSize}
              viewSize={previewSize}
            />
          </>
        ) : !hasPermission ? (
          permissionUi
        ) : (
          noDeviceUi
        )}
      </View>

      <View style={[styles.panel, { paddingBottom: Math.max(insets.bottom, 12) }]}>
        <View style={styles.row}>
          <Pressable
            style={({ pressed }) => [
              styles.button,
              isDetecting ? styles.buttonStop : styles.buttonStart,
              pressed && styles.buttonPressed,
              (!isModelLoaded || !canShowCamera) && styles.buttonDisabled,
            ]}
            onPress={onToggleDetection}
            disabled={!isModelLoaded || !canShowCamera}
          >
            <Text style={styles.buttonText}>
              {isDetecting ? 'Stop detection' : 'Start detection'}
            </Text>
          </Pressable>

          <Pressable
            style={({ pressed }) => [styles.button, pressed && styles.buttonPressed]}
            onPress={onSwitchCamera}
            disabled={!canShowCamera}
          >
            <Text style={styles.buttonText}>Switch camera</Text>
          </Pressable>
        </View>

              <View style={[styles.row, styles.rowMt10]}>
                <Pressable
                  style={({ pressed }) => [styles.button, pressed && styles.buttonPressed]}
                  onPress={onOpenVideoTest}
                >
                  <Text style={styles.buttonText}>Load video (test)</Text>
                </Pressable>
              </View>

        {modelState === 'loading' && (
          <View style={styles.loadingInline}>
            <ActivityIndicator color="#fff" size="small" />
            <Text style={styles.loadingInlineText}>Загружаю модель…</Text>
          </View>
        )}

        <View style={styles.metrics}>
          <Metric label="FPS" value={stats.fps.toFixed(1)} />
          <Metric label="Inference, ms" value={stats.lastInferenceMs.toFixed(1)} />
          <Metric label="Objects" value={String(stats.lastNumDetections)} />
          <Metric label="Input" value={`${stats.inputSize}x${stats.inputSize}`} />
        </View>

        {modelState === 'error' && (
          <View style={styles.errorBox}>
            <Text style={styles.errorTitle}>Модель не загрузилась</Text>
            <Text style={styles.errorText}>
              Проверь, что файл модели реально лежит в `app/assets/models/` и не пустой.
              {'\n\n'}
              {modelErrorMessage ?? 'Нет текста ошибки.'}
            </Text>
          </View>
        )}

        {lastLogFilePath && (
          <View style={styles.hintBox}>
            <Text style={styles.hintTitle}>Логи сохранены</Text>
            <Text style={styles.hintText} numberOfLines={2}>
              {lastLogFilePath}
            </Text>
          </View>
        )}
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
  root: {
    flex: 1,
    backgroundColor: '#000',
  },
  preview: {
    flex: 1,
    backgroundColor: '#000',
  },
  panel: {
    backgroundColor: 'rgba(15, 15, 18, 0.95)',
    paddingHorizontal: 12,
    paddingTop: 10,
    borderTopWidth: StyleSheet.hairlineWidth,
    borderTopColor: 'rgba(255,255,255,0.15)',
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
  buttonPressed: {
    opacity: 0.85,
  },
  buttonStart: {
    backgroundColor: '#2563eb',
  },
  buttonStop: {
    backgroundColor: '#dc2626',
  },
  buttonDisabled: {
    opacity: 0.5,
  },
  buttonText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '600',
  },
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
  metricLabel: {
    color: 'rgba(255,255,255,0.65)',
    fontSize: 12,
  },
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
  title: {
    color: '#fff',
    fontSize: 20,
    fontWeight: '800',
    marginBottom: 8,
  },
  text: {
    color: 'rgba(255,255,255,0.75)',
    fontSize: 14,
    textAlign: 'center',
    marginBottom: 14,
  },
  loading: {
    // (не используется) раньше лоадер был поверх камеры, но Dynamic Island мог его закрывать
    // оставляем стиль пустым, чтобы не ломать историю правок
  },
  loadingText: {
    // (не используется)
    color: '#fff',
    fontSize: 13,
    fontWeight: '600',
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
  loadingInlineText: {
    color: '#fff',
    fontSize: 13,
    fontWeight: '600',
  },
  errorBox: {
    marginTop: 10,
    paddingVertical: 10,
    paddingHorizontal: 12,
    borderRadius: 12,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: 'rgba(255,255,255,0.15)',
    backgroundColor: 'rgba(220, 38, 38, 0.12)',
  },
  errorTitle: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '800',
    marginBottom: 6,
  },
  errorText: {
    color: 'rgba(255,255,255,0.8)',
    fontSize: 12,
    lineHeight: 16,
  },
  hintBox: {
    marginTop: 10,
    paddingVertical: 10,
    paddingHorizontal: 12,
    borderRadius: 12,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: 'rgba(255,255,255,0.15)',
    backgroundColor: 'rgba(34, 197, 94, 0.12)',
  },
  hintTitle: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '800',
    marginBottom: 6,
  },
  hintText: {
    color: 'rgba(255,255,255,0.8)',
    fontSize: 12,
    lineHeight: 16,
  },
});

