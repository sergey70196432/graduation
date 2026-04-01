import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Alert, Platform } from 'react-native';
import { loadTensorflowModel, type TensorflowModel } from 'react-native-fast-tflite';
import { loadLabelsFromAsset } from '../utils/labels';
import { SpeedModelState } from '../types/detection';

const BUNDLED_SPEED_MODEL_ASSET = require('../../assets/models/speed_classifier/model_float32.tflite');
const BUNDLED_SPEED_LABELS_ASSET = require('../../assets/models/speed_classifier/labels.txt');

export function useSpeedClassifierModel() {
  const [model, setModel] = useState<TensorflowModel | undefined>(undefined);
  const [labels, setLabels] = useState<string[]>([]);
  const [state, setState] = useState<SpeedModelState>('loading');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const isLoaded = state === 'loaded' && model != null && labels.length > 0;

  const loadTokenRef = useRef(0);

  const load = useCallback(async () => {
    const token = ++loadTokenRef.current;
    setState('loading');
    setErrorMessage(null);

    try {
      const [m, labs] = await Promise.all([
        loadTensorflowModel(BUNDLED_SPEED_MODEL_ASSET),
        loadLabelsFromAsset(BUNDLED_SPEED_LABELS_ASSET),
      ]);
      if (loadTokenRef.current !== token) return;

      setModel(m);
      setLabels(labs);
      setState('loaded');
    } catch (e) {
      const msg = String(e);
      console.warn('[speed-cls] load failed', e);
      if (loadTokenRef.current !== token) return;
      setModel(undefined);
      setLabels([]);
      setState('error');
      setErrorMessage(msg);
    }
  }, []);

  useEffect(() => {
    load().catch(() => {
      // handled
    });
  }, [load]);

  const speedBases = useMemo(() => {
    const set = new Set<string>();
    for (const s of labels) {
      const base = String(s).split('_')[0]?.trim();
      if (base) set.add(base);
    }
    return Array.from(set.values()).sort((a, b) => a.localeCompare(b));
  }, [labels]);

  const missingHint = useMemo(() => {
    return `Классификатор скорости не найден (bundled).\n\nПроверь, что файлы есть в проекте:\n- app/assets/models/speed_classifier/model.tflite\n- app/assets/models/speed_classifier/labels.txt\n\nПосле изменения ассетов нужно пересобрать приложение.`;
  }, []);

  const showMissingHint = useCallback(() => {
    Alert.alert('Speed classifier', missingHint);
  }, [missingHint]);

  return {
    // runtime
    model,
    labels,
    speedBases,
    isLoaded,
    state,
    errorMessage,

    // actions
    reload: load,
    showMissingHint,

    // diag
    platform: Platform.OS,
    source: 'bundled' as const,
  };
}

