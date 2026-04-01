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
import { signByLabel } from '../signs/signRegistry';
import DebugMetrics from '../widgets/DebugMetrics';

type StickySign = {
  label: string;
  classId: number;
  bestConfidence: number;
  speedConfidence: number | undefined;
  count: number;
  lastSeenMs: number;
};

type CameraExtraProps = {
  /**
   * On Android, using TextureView avoids SurfaceView overlay issues where
   * the camera can appear above other React Native views.
   */
  androidPreviewViewType?: 'surface-view' | 'texture-view';

  /**
   * Some VisionCamera versions expose this prop, but typings can lag behind.
   * We keep it here to be able to throttle frameProcessor calls.
   */
  frameProcessorFps?: number;
};

const CameraView = Camera as unknown as React.ComponentType<
  React.ComponentProps<typeof Camera> & CameraExtraProps
>;

export function CameraDetectionScreen(props: { onOpenVideoTest?: () => void }) {
  const insets = useSafeAreaInsets();
  const { width: windowW, height: windowH } = useWindowDimensions();
  const isLandscape = windowW > windowH;
  const isDevBuild = __DEV__ === true;
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
  const [isModelPickerOpen, setIsModelPickerOpen] = useState(false);

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
    isSpeedModelLoaded,
    speedModelState,

    activeModelItem,
    models,
    catalogState,
    catalogErrorMessage,
    refreshManifest,
    setActiveModel,
    ensureDownloaded,
    deleteDownloaded,
    downloadProgressById,
    isClearingStorage,
    clearStorage,
    manifestGeneratedAt,
  } = useYoloDetector();

  const hasSelectedModel = activeModelItem != null;

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
    return isDebug ? 6 : 3;
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

  const modelPickerSafePadding = useMemo(() => {
    // В landscape "ноуч" (Dynamic Island) уезжает влево/вправо, поэтому учитываем safe-area по всем сторонам.
    return {
      paddingTop: Math.max(insets.top, 16),
      paddingBottom: Math.max(insets.bottom, 16),
      paddingLeft: Math.max(insets.left, 16),
      paddingRight: Math.max(insets.right, 16),
    };
  }, [insets.bottom, insets.left, insets.right, insets.top]);

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

  const SIGN_DEBOUNCE_MS = 3000;
  const [stickySigns, setStickySigns] = useState<StickySign[]>([]);

  // Обновляем "последний раз увидели" по новым detections.
  useEffect(() => {
    const now = Date.now();

    // Группируем текущие детекции по label.
    const frame = new Map<
      string,
      { label: string; classId: number; bestConfidence: number; count: number; speedConfidence: number | undefined }
    >();
    for (let i = 0; i < detections.length; i++) {
      const d = detections[i] as Detection | undefined;
      if (!d) continue;
      const label = d.refinedLabel ?? d.label;
      const prev = frame.get(label);
      if (!prev) {
        frame.set(label, {
          label,
          classId: d.classId,
          bestConfidence: d.confidence,
          count: 1,
          speedConfidence: d.refinedConfidence,
        });
      } else {
        prev.count += 1;
        if (d.confidence > prev.bestConfidence) prev.bestConfidence = d.confidence;
        if (d.refinedConfidence) prev.speedConfidence = d.refinedConfidence;
      }
    }

    setStickySigns(prev => {
      const byLabel = new Map<string, StickySign>();
      for (const s of prev) byLabel.set(s.label, s);

      for (const v of frame.values()) {
        const existing = byLabel.get(v.label);
        if (existing) {
          existing.lastSeenMs = now;
          existing.bestConfidence = Math.max(existing.bestConfidence, v.bestConfidence);
          existing.speedConfidence = v.speedConfidence;
          existing.count = v.count;
          existing.classId = v.classId;
        } else {
          byLabel.set(v.label, {
            label: v.label,
            classId: v.classId,
            bestConfidence: v.bestConfidence,
            speedConfidence: v.speedConfidence,
            count: v.count,
            lastSeenMs: now,
          });
        }
      }

      const out = Array.from(byLabel.values()).filter(
        s => now - s.lastSeenMs <= SIGN_DEBOUNCE_MS
      );
      out.sort((a, b) => b.lastSeenMs - a.lastSeenMs);
      return out;
    });
  }, [detections]);

  // Таймер нужен, чтобы знак исчез ровно через 3с даже если новых детекций нет.
  useEffect(() => {
    if (stickySigns.length === 0) return;
    const id = setInterval(() => {
      const now = Date.now();
      setStickySigns(prev => {
        const out = prev.filter(s => now - s.lastSeenMs <= SIGN_DEBOUNCE_MS);
        if (out.length === prev.length) return prev;
        return out;
      });
    }, 200);
    return () => clearInterval(id);
  }, [stickySigns.length]);

  const signs = stickySigns;

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

            <View style={styles.signsHeaderButtonsRow}>
              <View style={styles.modelButtonWrap}>
                <Pressable
                  style={({ pressed }) => [
                    styles.smallButton,
                    pressed && styles.buttonPressed,
                  ]}
                  onPress={() => setIsModelPickerOpen(true)}
                >
                  <Text style={styles.smallButtonText}>Модель</Text>
                </Pressable>
                <Text style={styles.modelButtonSubText} numberOfLines={2}>
                  {activeModelItem?.title ?? '—'}
                </Text>
              </View>

              {isDevBuild && (
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
              )}
            </View>
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
                      {(s.bestConfidence * 100).toFixed(0)}% · x{s.count};
                      {(s.speedConfidence ? s.speedConfidence * 100 : 0).toFixed(0)}% · x{s.count}
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
                  {...(isDetecting ? { frameProcessor, frameProcessorFps } : {})}
                  {...(Platform.OS === 'android'
                    ? ({ androidPreviewViewType: 'texture-view' } as const)
                    : {})}
                  resizeMode="contain"
                />

                {!hasSelectedModel && (
                  <View pointerEvents="none" style={styles.noModelOverlay}>
                    <View style={styles.noModelCard}>
                      <Text style={styles.noModelTitle}>Нет выбранной модели</Text>
                      <Text style={styles.noModelText}>
                        Открой “Модель”, обнови список и выбери модель для детекции.
                      </Text>
                    </View>
                  </View>
                )}

                {(isDevBuild || isDebug) && (
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

              <DebugMetrics
                stats={stats}
                isSpeedModelLoaded={isSpeedModelLoaded}
                speedModelState={speedModelState}
              />

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

      {isModelPickerOpen && (
        <View style={[styles.modalRoot, modelPickerSafePadding]} pointerEvents="box-none">
          <Pressable
            style={styles.modalBackdrop}
            onPress={() => {
              if (isClearingStorage) return;
              setIsModelPickerOpen(false);
            }}
          />

          <View style={styles.modalCard}>
            <ScrollView
              style={styles.modalScroll}
              contentContainerStyle={styles.modalScrollContent}
              showsVerticalScrollIndicator={true}
            >
              <View style={styles.modalHeader}>
                <Text style={styles.modalTitle}>Выбор модели</Text>
                <Pressable
                  style={({ pressed }) => [
                    styles.smallButton,
                    pressed && styles.buttonPressed,
                  ]}
                  onPress={() => refreshManifest()}
                  disabled={isClearingStorage}
                >
                  <Text style={styles.smallButtonText}>Обновить</Text>
                </Pressable>
              </View>

              <View style={styles.modalTopActions}>
                <Pressable
                  style={({ pressed }) => [
                    styles.modelActionButton,
                    styles.modelActionButtonDanger,
                    pressed && styles.buttonPressed,
                    isClearingStorage && styles.buttonDisabled,
                  ]}
                  onPress={async () => {
                    await clearStorage();
                  }}
                  disabled={isClearingStorage}
                >
                  <View style={styles.clearRow}>
                    {isClearingStorage && (
                      <ActivityIndicator color="#fff" size="small" />
                    )}
                    <Text style={styles.modelActionText}>
                      {isClearingStorage ? 'Очищаю…' : 'Очистить хранилище'}
                    </Text>
                  </View>
                </Pressable>
              </View>

              <Text style={styles.modalSubText} numberOfLines={2}>
                {manifestGeneratedAt
                  ? `manifest generatedAt: ${manifestGeneratedAt}`
                  : 'manifest: (ещё не загружен)'}
              </Text>
              {(catalogState === 'error' || catalogErrorMessage) && (
                <Text style={[styles.modalSubText, styles.modalErrorText]}>
                  {catalogErrorMessage ?? 'Ошибка загрузки манифеста'}
                </Text>
              )}

              <View style={styles.modalList}>
                {models.map(m => {
                  const isDownloading = downloadProgressById[m.id] != null;
                  const progress = downloadProgressById[m.id] ?? 0;
                  const canDelete = m.isDownloaded && !m.isActive;
                  const disabledAll = isClearingStorage;

                  return (
                    <View key={m.id} style={styles.modelRow}>
                      <View style={styles.modelRowMeta}>
                        <Text style={styles.modelRowTitle} numberOfLines={1}>
                          {m.title}
                        </Text>
                        <Text style={styles.modelRowSub} numberOfLines={2}>
                        {m.isDownloaded
                          ? `скачана · imgsz=${m.inputSize}`
                          : `не скачана · imgsz=${m.inputSize}`}
                          {m.isActive ? ' · active' : ''}
                        </Text>
                        {isDownloading && (
                          <Text style={styles.modelRowSub}>
                            download: {(progress * 100).toFixed(0)}%
                          </Text>
                        )}
                      </View>

                      <View style={styles.modelRowActions}>
                        {!m.isDownloaded ? (
                          <Pressable
                            style={({ pressed }) => [
                              styles.modelActionButton,
                              pressed && styles.buttonPressed,
                            ]}
                            onPress={async () => {
                              try {
                                await ensureDownloaded(m.id);
                              } catch {
                                // Ошибка уже будет показана через Alert внутри хука/или здесь.
                              }
                            }}
                            disabled={isDownloading || disabledAll}
                          >
                            <Text style={styles.modelActionText}>
                              {isDownloading ? '...' : 'Скачать'}
                            </Text>
                          </Pressable>
                        ) : (
                          <Pressable
                            style={({ pressed }) => [
                              styles.modelActionButton,
                              pressed && styles.buttonPressed,
                              m.isActive && styles.modelActionButtonActive,
                            ]}
                            onPress={async () => {
                              if (!m.isDownloaded) return;
                              await setActiveModel(m.id);
                            }}
                            disabled={
                              disabledAll ||
                              m.isActive ||
                              !m.isDownloaded
                            }
                          >
                            <Text style={styles.modelActionText}>
                              {m.isActive ? 'Выбрана' : 'Выбрать'}
                            </Text>
                          </Pressable>
                        )}

                        {canDelete && (
                          <Pressable
                            style={({ pressed }) => [
                              styles.modelActionButton,
                              styles.modelActionButtonDanger,
                              pressed && styles.buttonPressed,
                            ]}
                            onPress={async () => {
                              await deleteDownloaded(m.id);
                            }}
                            disabled={disabledAll}
                          >
                            <Text style={styles.modelActionText}>Удалить</Text>
                          </Pressable>
                        )}
                      </View>
                    </View>
                  );
                })}
              </View>

              <Pressable
                style={({ pressed }) => [
                  styles.button,
                  pressed && styles.buttonPressed,
                  isClearingStorage && styles.buttonDisabled,
                ]}
                onPress={() => {
                  if (isClearingStorage) return;
                  setIsModelPickerOpen(false);
                }}
                disabled={isClearingStorage}
              >
                <Text style={styles.buttonText}>Закрыть</Text>
              </Pressable>
            </ScrollView>
          </View>
        </View>
      )}
    </View>
  );
}



function SignSvg(props: { label: string; confidence: number }) {
  const conf = Math.max(0, Math.min(1, props.confidence));
  const pct = Math.round(conf * 100);
  const short =
    props.label.length > 10 ? props.label.slice(0, 10).trim() + '…' : props.label;

  const base = props.label.split('_')[0] ?? props.label;
  const Sign = signByLabel[props.label] ?? signByLabel[base];
  if (Sign) {
    // Большинство наших SVG из датасета идут без viewBox (только width/height=100x86).
    // Если просто поставить width/height=54, картинка будет клипаться/выходить за границы.
    // Поэтому задаём viewBox и вписываем в квадрат 54x54 с обрезкой по контейнеру.
    return (
      <View style={styles.signIconBox}>
        <Sign
          width="100%"
          height="100%"
          viewBox="0 0 100 86"
          preserveAspectRatio="xMidYMid meet"
        />
      </View>
    );
  }

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
    flexDirection: 'column',
    paddingBottom: 10,
  },
  signsHeaderButtonsRow: {
    marginTop: 8,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 10,
  },
  signsTitle: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '900',
  },
  modelButtonWrap: { flex: 1, minWidth: 0 },
  modelButtonSubText: {
    marginTop: 4,
    color: 'rgba(255,255,255,0.65)',
    fontSize: 11,
    fontWeight: '800',
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
  signIconBox: {
    width: 54,
    height: 54,
    overflow: 'hidden',
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
  noModelOverlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(0,0,0,0.55)',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 16,
  },
  noModelCard: {
    maxWidth: 420,
    paddingVertical: 14,
    paddingHorizontal: 14,
    borderRadius: 14,
    backgroundColor: 'rgba(15, 15, 18, 0.92)',
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: 'rgba(255,255,255,0.18)',
  },
  noModelTitle: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '900',
  },
  noModelText: {
    marginTop: 6,
    color: 'rgba(255,255,255,0.72)',
    fontSize: 13,
    fontWeight: '700',
    lineHeight: 18,
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

  modalBackdrop: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(0,0,0,0.55)',
  },
  modalRoot: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    justifyContent: 'center',
    alignItems: 'center',
  },
  modalCard: {
    backgroundColor: 'rgba(15, 15, 18, 0.98)',
    borderRadius: 14,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: 'rgba(255,255,255,0.16)',
    padding: 12,
    width: '100%',
    maxWidth: 520,
    maxHeight: '85%',
  },
  modalScroll: {
    flexGrow: 0,
  },
  modalScrollContent: {
    paddingBottom: 10,
  },
  modalHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 8,
    gap: 10,
  },
  modalTitle: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '900',
  },
  modalSubText: {
    color: 'rgba(255,255,255,0.70)',
    fontSize: 12,
    marginBottom: 6,
  },
  modalErrorText: {
    color: 'rgba(255, 120, 120, 0.95)',
  },
  modalTopActions: {
    marginTop: 4,
    marginBottom: 6,
  },
  clearRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  modalList: {
    marginTop: 6,
    marginBottom: 10,
  },
  modelRow: {
    paddingVertical: 10,
    paddingHorizontal: 10,
    borderRadius: 12,
    backgroundColor: 'rgba(255,255,255,0.05)',
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: 'rgba(255,255,255,0.12)',
    marginBottom: 10,
    flexDirection: 'row',
    gap: 10,
  },
  modelRowMeta: {
    flex: 1,
    minWidth: 0,
  },
  modelRowTitle: {
    color: '#fff',
    fontSize: 13,
    fontWeight: '900',
  },
  modelRowSub: {
    marginTop: 4,
    color: 'rgba(255,255,255,0.70)',
    fontSize: 12,
    fontWeight: '700',
  },
  modelRowActions: {
    alignItems: 'flex-end',
    gap: 8,
  },
  modelActionButton: {
    paddingVertical: 8,
    paddingHorizontal: 10,
    borderRadius: 10,
    backgroundColor: 'rgba(255,255,255,0.10)',
  },
  modelActionButtonActive: {
    backgroundColor: 'rgba(37, 99, 235, 0.35)',
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: 'rgba(37, 99, 235, 0.6)',
  },
  modelActionButtonDanger: {
    backgroundColor: 'rgba(220, 38, 38, 0.22)',
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: 'rgba(220, 38, 38, 0.45)',
  },
  modelActionText: {
    color: '#fff',
    fontSize: 12,
    fontWeight: '900',
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

