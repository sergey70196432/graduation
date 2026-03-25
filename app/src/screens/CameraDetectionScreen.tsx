import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  ActivityIndicator,
  LayoutChangeEvent,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  useWindowDimensions,
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
import type { Detection } from '../types/detection';
import type { FrameSize } from '../types/detection';
import Svg, { Circle, Path, Rect as SvgRect, Text as SvgText } from 'react-native-svg';

type CameraExtraProps = {
  /**
   * On Android, using TextureView avoids SurfaceView overlay issues where
   * the camera can appear above other React Native views.
   */
  androidPreviewViewType?: 'surface-view' | 'texture-view';
};

const CameraView = Camera as unknown as React.ComponentType<
  React.ComponentProps<typeof Camera> & CameraExtraProps
>;

export function CameraDetectionScreen(props: { onOpenVideoTest?: () => void }) {
  const insets = useSafeAreaInsets();
  const { width: windowW, height: windowH } = useWindowDimensions();
  const isLandscape = windowW > windowH;
  const debugPanelWidth = useMemo(() => {
    if (!isLandscape) return undefined;
    const w = Math.round(windowW * 0.34);
    return Math.max(300, Math.min(420, w));
  }, [isLandscape, windowW]);
  const { hasPermission, requestPermission } = useCameraPermission();

  const [cameraPosition, setCameraPosition] = useState<'back' | 'front'>(
    'back'
  );
  const device = useCameraDevice(cameraPosition);

  const [isDebug, setIsDebug] = useState(false);

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

  const frameProcessorFps = useMemo(() => {
    // На Android (особенно на эмуляторе) тяжёлый frameProcessor легко приводит к ANR.
    // Ограничиваем частоту вызова worklet на уровне VisionCamera.
    if (Platform.OS === 'android') return isDebug ? 2 : 1;
    return isDebug ? 10 : 6;
  }, [isDebug]);

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

  // В обычном режиме (не debug) детекция включена всегда, чтобы колонка "знаки" жила своей жизнью.
  useEffect(() => {
    if (isDebug) return;
    if (!hasPermission) return;
    if (!device) return;
    if (!isModelLoaded) return;
    if (!isDetecting) setIsDetecting(true);
  }, [device, hasPermission, isDebug, isDetecting, isModelLoaded, setIsDetecting]);

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

  const signs = useMemo(() => {
    // Группируем детекции по label, берём max confidence + количество.
    const m = new Map<
      string,
      { label: string; classId: number; bestConfidence: number; count: number }
    >();
    for (let i = 0; i < detections.length; i++) {
      const d = detections[i] as Detection | undefined;
      if (!d) continue;
      const key = d.label;
      const prev = m.get(key);
      if (!prev) {
        m.set(key, {
          label: d.label,
          classId: d.classId,
          bestConfidence: d.confidence,
          count: 1,
        });
      } else {
        prev.count += 1;
        if (d.confidence > prev.bestConfidence) prev.bestConfidence = d.confidence;
      }
    }
    const arr = Array.from(m.values());
    arr.sort((a, b) => b.bestConfidence - a.bestConfidence);
    return arr;
  }, [detections]);

  return (
    <View style={styles.root}>
      <View style={styles.mainRow}>
        <View
          style={[
            styles.signsColumn,
            { paddingLeft: Math.max(insets.left, 12) },
          ]}
        >
          <View style={styles.signsHeader}>
            <Text style={styles.signsTitle}>Знаки</Text>
            <Pressable
              style={({ pressed }) => [
                styles.smallButton,
                pressed && styles.buttonPressed,
                isDebug && styles.smallButtonActive,
              ]}
              onPress={() => setIsDebug(v => !v)}
            >
              <Text style={styles.smallButtonText}>Debug</Text>
            </Pressable>
          </View>

          <ScrollView
            style={styles.signsList}
            contentContainerStyle={{ paddingBottom: Math.max(insets.bottom, 12) }}
          >
            {signs.length === 0 ? (
              <View style={styles.signEmpty}>
                <Text style={styles.signEmptyText}>
                  Пока ничего не найдено.
                </Text>
                {!hasPermission && (
                  <Text style={[styles.signEmptyText, styles.signEmptyTextMt]}>
                    Нужен доступ к камере.
                  </Text>
                )}
              </View>
            ) : (
              signs.slice(0, 30).map(s => (
                <View key={s.label} style={styles.signTile}>
                  <SignSvg label={s.label} confidence={s.bestConfidence} />
                  <View style={styles.signMeta}>
                    <Text style={styles.signLabel} numberOfLines={2}>
                      {s.label}
                    </Text>
                    <Text style={styles.signSub}>
                      {(s.bestConfidence * 100).toFixed(0)}% · x{s.count}
                    </Text>
                  </View>
                </View>
              ))
            )}
          </ScrollView>
        </View>

        <View style={styles.cameraArea}>
          <View style={styles.preview} onLayout={onPreviewLayout}>
            {canShowCamera ? (
              <>
                <CameraView
                  style={StyleSheet.absoluteFill}
                  device={device}
                  isActive={true}
                  frameProcessor={isDetecting ? frameProcessor : undefined}
                  frameProcessorFps={isDetecting ? frameProcessorFps : undefined}
                  androidPreviewViewType={Platform.OS === 'android' ? 'texture-view' : undefined}
                  resizeMode="contain"
                />

                {isDebug && (
                  <DetectionOverlay
                    detections={detections}
                    frameSize={frameSize}
                    viewSize={previewSize}
                  />
                )}
              </>
            ) : !hasPermission ? (
              permissionUi
            ) : (
              noDeviceUi
            )}
          </View>

          {isDebug && (
            <View
              style={[
                styles.debugPanel,
                isLandscape ? { width: debugPanelWidth } : null,
                {
                  paddingRight: Math.max(insets.right, 12),
                },
              ]}
            >
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
                  style={({ pressed }) => [
                    styles.button,
                    pressed && styles.buttonPressed,
                  ]}
                  onPress={onSwitchCamera}
                  disabled={!canShowCamera}
                >
                  <Text style={styles.buttonText}>Switch camera</Text>
                </Pressable>
              </View>

              <View style={[styles.row, styles.rowMt10]}>
                <Pressable
                  style={({ pressed }) => [
                    styles.button,
                    pressed && styles.buttonPressed,
                  ]}
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
                <Metric
                  label="Total, ms"
                  value={stats.lastTotalMs.toFixed(1)}
                />
                <Metric
                  label="Inference, ms"
                  value={stats.lastInferenceMs.toFixed(1)}
                />
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

function SignSvg(props: { label: string; confidence: number }) {
  const conf = Math.max(0, Math.min(1, props.confidence));
  const pct = Math.round(conf * 100);
  const short =
    props.label.length > 10 ? props.label.slice(0, 10).trim() + '…' : props.label;

  return (
    <Svg width={54} height={54} viewBox="0 0 54 54">
      <SvgRect
        x={2}
        y={2}
        width={50}
        height={50}
        rx={14}
        ry={14}
        fill="rgba(255,255,255,0.06)"
        stroke="rgba(255,255,255,0.18)"
        strokeWidth={1}
      />

      {/* Простой пиктограммный "знак" */}
      <Circle cx={27} cy={23} r={11} fill="rgba(0, 255, 170, 0.10)" />
      <Path
        d="M27 13.5c5.25 0 9.5 4.25 9.5 9.5s-4.25 9.5-9.5 9.5-9.5-4.25-9.5-9.5 4.25-9.5 9.5-9.5Z"
        stroke="rgba(0, 255, 170, 0.95)"
        strokeWidth={2}
        fill="transparent"
      />

      <SvgText
        x={27}
        y={42}
        fill="rgba(255,255,255,0.92)"
        fontSize="10"
        fontWeight="700"
        textAnchor="middle"
      >
        {pct}%
      </SvgText>

      <SvgText
        x={27}
        y={29}
        fill="rgba(0, 255, 170, 0.95)"
        fontSize="9"
        fontWeight="800"
        textAnchor="middle"
      >
        {short}
      </SvgText>
    </Svg>
  );
}

const styles = StyleSheet.create({
  root: {
    flex: 1,
    backgroundColor: '#000',
  },
  mainRow: {
    flex: 1,
    flexDirection: 'row',
  },
  signsColumn: {
    width: 240,
    backgroundColor: 'rgba(15, 15, 18, 0.95)',
    borderRightWidth: StyleSheet.hairlineWidth,
    borderRightColor: 'rgba(255,255,255,0.15)',
    paddingTop: 10,
    paddingRight: 12,
  },
  signsHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingBottom: 10,
  },
  signsTitle: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '900',
  },
  signsList: {
    flex: 1,
  },
  signEmpty: {
    paddingVertical: 12,
    paddingHorizontal: 10,
    borderRadius: 12,
    backgroundColor: 'rgba(255,255,255,0.06)',
  },
  signEmptyText: {
    color: 'rgba(255,255,255,0.75)',
    fontSize: 12,
    lineHeight: 16,
  },
  signEmptyTextMt: {
    marginTop: 6,
  },
  signTile: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    paddingVertical: 10,
    paddingHorizontal: 10,
    borderRadius: 12,
    backgroundColor: 'rgba(255,255,255,0.04)',
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: 'rgba(255,255,255,0.12)',
    marginBottom: 10,
  },
  signMeta: {
    flex: 1,
    minWidth: 0,
  },
  signLabel: {
    color: '#fff',
    fontSize: 13,
    fontWeight: '800',
  },
  signSub: {
    marginTop: 4,
    color: 'rgba(255,255,255,0.65)',
    fontSize: 12,
    fontWeight: '700',
  },
  cameraArea: {
    flex: 1,
    backgroundColor: '#000',
  },
  preview: {
    flex: 1,
    backgroundColor: '#000',
  },
  debugPanel: {
    position: 'absolute',
    top: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(15, 15, 18, 0.92)',
    paddingHorizontal: 12,
    paddingTop: 10,
    paddingBottom: 12,
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
  smallButton: {
    paddingVertical: 8,
    paddingHorizontal: 10,
    borderRadius: 10,
    backgroundColor: 'rgba(255,255,255,0.08)',
  },
  smallButtonActive: {
    backgroundColor: 'rgba(37, 99, 235, 0.35)',
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: 'rgba(37, 99, 235, 0.6)',
  },
  smallButtonText: { color: '#fff', fontSize: 13, fontWeight: '800' },
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

