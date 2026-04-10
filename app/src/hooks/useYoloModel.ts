import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Alert, Platform } from 'react-native';
import * as RNFS from 'react-native-fs';
import { loadTensorflowModel, type TensorflowModel } from 'react-native-fast-tflite';
import { YOLO } from '../constants/yolo';
import { MODELS } from '../constants/models';
import { loadLabelsFromFile } from '../utils/labels';
import type {
  ModelCatalogState,
  ModelListItem,
  RemoteModelManifestV1,
  RemoteYoloModel,
} from '../types/models';

type PersistedState = {
  activeModelId: string | null;
  updatedAtIso?: string;
};

function toFileUri(p: string) {
  return p.startsWith('file://') ? p : `file://${p}`;
}

function modelsRootDir() {
  return `${RNFS.DocumentDirectoryPath}/${MODELS.deviceDirName}`;
}

function manifestCachePath() {
  return `${modelsRootDir()}/manifest-cache.json`;
}

function statePath() {
  return `${modelsRootDir()}/state.json`;
}

function modelDir(id: string) {
  return `${modelsRootDir()}/${id}`;
}

function localModelPath(id: string) {
  return `${modelDir(id)}/model.tflite`;
}

function localLabelsPath(id: string) {
  return `${modelDir(id)}/labels.txt`;
}

async function readJsonFile<T>(path: string): Promise<T | null> {
  try {
    const exists = await RNFS.exists(path);
    if (!exists) return null;
    const raw = await RNFS.readFile(path, 'utf8');
    return JSON.parse(raw) as T;
  } catch (e) {
    console.warn('[models] readJsonFile failed', path, e);
    return null;
  }
}

async function writeJsonFile(path: string, value: unknown) {
  await RNFS.mkdir(modelsRootDir());
  await RNFS.writeFile(path, JSON.stringify(value, null, 2), 'utf8');
}

function normalizeRemoteModel(m: RemoteYoloModel): ModelListItem {
  return {
    id: m.id,
    title: m.title,
    ...(typeof m.description === 'string' ? { description: m.description } : {}),
    inputSize: m.inputSize ?? YOLO.inputSize,
    confidenceThreshold: m.confidenceThreshold ?? YOLO.confidenceThreshold,
    preNmsTopK: m.preNmsTopK ?? YOLO.preNmsTopK,
    postNmsTopK: m.postNmsTopK ?? YOLO.postNmsTopK,
    isActive: false,
    isDownloaded: false,
    localModelPath: localModelPath(m.id),
    localLabelsPath: localLabelsPath(m.id),
    ...(typeof m.model.bytes === 'number' ? { remoteModelBytes: m.model.bytes } : {}),
    ...(typeof m.labels.bytes === 'number' ? { remoteLabelsBytes: m.labels.bytes } : {}),
  };
}

async function computeDownloadedFlag(item: ModelListItem): Promise<boolean> {
  const mp = item.localModelPath;
  const lp = item.localLabelsPath;
  if (!mp || !lp) return false;
  const [mExists, lExists] = await Promise.all([RNFS.exists(mp), RNFS.exists(lp)]);
  if (!mExists || !lExists) return false;

  // Минимальная проверка "докачалось ли" по bytes (если есть в манифесте).
  try {
    if (typeof item.remoteModelBytes === 'number') {
      const st = await RNFS.stat(mp);
      if (Number(st.size) !== item.remoteModelBytes) return false;
    }
    if (typeof item.remoteLabelsBytes === 'number') {
      const st = await RNFS.stat(lp);
      if (Number(st.size) !== item.remoteLabelsBytes) return false;
    }
  } catch {
    return false;
  }

  return true;
}

async function downloadToPath(params: {
  url: string;
  destPath: string;
  onProgress?: (ratio01: number) => void;
}) {
  const dir = params.destPath.split('/').slice(0, -1).join('/');
  await RNFS.mkdir(dir);
  const tmp = `${params.destPath}.partial-${Date.now()}`;

  const ret = RNFS.downloadFile({
    fromUrl: params.url,
    toFile: tmp,
    discretionary: true,
    progressInterval: 150,
    progressDivider: 1,
    progress: ev => {
      const total = Number(ev.contentLength);
      const done = Number(ev.bytesWritten);
      if (total > 0 && params.onProgress) params.onProgress(Math.min(1, done / total));
    },
  });

  const res = await ret.promise;
  if (res.statusCode && res.statusCode >= 400) {
    try {
      await RNFS.unlink(tmp);
    } catch {
      // ignore
    }
    throw new Error(`Download failed: HTTP ${res.statusCode}`);
  }

  // Atomic-ish replace
  try {
    const exists = await RNFS.exists(params.destPath);
    if (exists) await RNFS.unlink(params.destPath);
  } catch {
    // ignore
  }
  await RNFS.moveFile(tmp, params.destPath);
}

export function useYoloModel() {
  const [catalogState, setCatalogState] = useState<ModelCatalogState>('idle');
  const [catalogErrorMessage, setCatalogErrorMessage] = useState<string | null>(null);

  const [manifest, setManifest] = useState<RemoteModelManifestV1 | null>(null);
  const [activeModelId, setActiveModelId] = useState<string | null>(null);

  const [downloadProgressById, setDownloadProgressById] = useState<Record<string, number>>({});
  const [downloadRevision, setDownloadRevision] = useState(0);
  const [isClearingStorage, setIsClearingStorage] = useState(false);

  const [labels, setLabels] = useState<string[]>([]);
  const [model, setModel] = useState<TensorflowModel | undefined>(undefined);
  const [modelState, setModelState] = useState<'loading' | 'loaded' | 'error'>('loading');
  const [modelErrorMessage, setModelErrorMessage] = useState<string | null>(null);

  const isModelLoaded = modelState === 'loaded' && model != null;

  const manifestById = useMemo(() => {
    const m = new Map<string, RemoteYoloModel>();
    for (const it of manifest?.models ?? []) m.set(it.id, it);
    return m;
  }, [manifest]);

  const models = useMemo<ModelListItem[]>(() => {
    const out: ModelListItem[] = [];
    for (const rm of manifest?.models ?? []) {
      const it = normalizeRemoteModel(rm);
      it.isActive = activeModelId != null && it.id === activeModelId;
      out.push(it);
    }
    return out;
  }, [activeModelId, manifest]);

  const activeModelItem = useMemo(() => {
    if (activeModelId == null) return undefined;
    return models.find(m => m.id === activeModelId);
  }, [activeModelId, models]);

  const activeYoloParams = useMemo(() => {
    const m = activeModelItem;
    return {
      inputSize: m?.inputSize ?? YOLO.inputSize,
      confidenceThreshold: m?.confidenceThreshold ?? YOLO.confidenceThreshold,
      preNmsTopK: m?.preNmsTopK ?? YOLO.preNmsTopK,
      postNmsTopK: m?.postNmsTopK ?? YOLO.postNmsTopK,
    };
  }, [activeModelItem]);

  const loadTokenRef = useRef(0);

  const loadActiveModel = useCallback(async () => {
    const token = ++loadTokenRef.current;
    setModelState('loading');
    setModelErrorMessage(null);

    try {
      if (activeModelId == null) {
        if (loadTokenRef.current !== token) return;
        setModel(undefined);
        setLabels([]);
        setModelState('error');
        setModelErrorMessage('Нет выбранной модели. Открой "Модель" и выбери модель.');
        return;
      }

      const rm = manifestById.get(activeModelId);
      if (!rm) {
        throw new Error(
          `Активная модель "${activeModelId}" отсутствует в манифесте. Нажми "Обновить список моделей".`
        );
      }

      const mp = localModelPath(rm.id);
      const lp = localLabelsPath(rm.id);
      const [mExists, lExists] = await Promise.all([RNFS.exists(mp), RNFS.exists(lp)]);
      if (!mExists || !lExists) {
        throw new Error(
          `Модель "${rm.title}" не скачана. Открой выбор моделей и нажми "Скачать".`
        );
      }

      const [m, labs] = await Promise.all([
        loadTensorflowModel({ url: toFileUri(mp) }),
        loadLabelsFromFile(lp),
      ]);
      if (loadTokenRef.current !== token) return;
      setModel(m);
      setLabels(labs);
      setModelState('loaded');

      try {
        console.log('[tflite] delegate:', m.delegate);
        console.log('[tflite] inputs:', m.inputs);
        console.log('[tflite] outputs:', m.outputs);
      } catch {
        // ignore
      }
    } catch (e) {
      const msg = String(e);
      console.warn('[models] loadActiveModel failed', e);
      if (loadTokenRef.current !== token) return;
      setModel(undefined);
      setLabels([]);
      setModelState('error');
      setModelErrorMessage(msg);
    }
  }, [activeModelId, manifestById]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      await RNFS.mkdir(modelsRootDir());
      const [cachedManifest, persisted] = await Promise.all([
        readJsonFile<RemoteModelManifestV1>(manifestCachePath()),
        readJsonFile<PersistedState>(statePath()),
      ]);
      if (cancelled) return;
      if (cachedManifest && cachedManifest.version === 1) setManifest(cachedManifest);
      if (persisted && 'activeModelId' in persisted) {
        setActiveModelId(persisted.activeModelId ?? null);
      }
    })().catch(e => console.warn('[models] init failed', e));
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    // Перезагружаем модель при смене activeModelId или манифеста.
    loadActiveModel().catch(() => {
      // already handled
    });
  }, [loadActiveModel]);

  const refreshManifest = useCallback(async () => {
    try {
      setCatalogState('loading');
      setCatalogErrorMessage(null);

      const res = await fetch(MODELS.manifestUrl, {
        method: 'GET',
        headers: { Accept: 'application/json' },
      });
      if (!res.ok) {
        throw new Error(`Не удалось скачать манифест моделей: HTTP ${res.status}`);
      }

      const json = (await res.json()) as unknown;
      const m = json as Partial<RemoteModelManifestV1>;
      if (m.version !== 1 || !Array.isArray(m.models)) {
        throw new Error('Некорректный формат манифеста: ожидается { version: 1, models: [...] }.');
      }

      const normalized: RemoteModelManifestV1 = {
        version: 1,
        ...(typeof m.generatedAt === 'string' ? { generatedAt: m.generatedAt } : {}),
        models: (m.models as RemoteYoloModel[])
          .filter(x => x && typeof x.id === 'string' && typeof x.title === 'string')
          .filter(x => x.model && typeof x.model.url === 'string')
          .filter(x => x.labels && typeof x.labels.url === 'string'),
      };

      setManifest(normalized);
      await writeJsonFile(manifestCachePath(), normalized);
      setCatalogState('idle');
    } catch (e) {
      const msg = String(e);
      console.warn('[models] refreshManifest failed', e);
      setCatalogState('error');
      setCatalogErrorMessage(msg);
      Alert.alert('Не удалось обновить список моделей', msg);
    }
  }, []);

  const setActiveModel = useCallback(async (id: string) => {
    setActiveModelId(id);
    try {
      const st: PersistedState = { activeModelId: id, updatedAtIso: new Date().toISOString() };
      await writeJsonFile(statePath(), st);
    } catch (e) {
      console.warn('[models] persist state failed', e);
    }
  }, []);

  const clearStorage = useCallback(async () => {
    if (isClearingStorage) return;
    setIsClearingStorage(true);
    try {
      // Сначала сбрасываем активную модель, чтобы не пытаться грузить модель из директории, которую удаляем.
      setActiveModelId(null);
      setModel(undefined);
      setLabels([]);
      setModelState('loading');
      setModelErrorMessage(null);

      // Удаляем весь каталог хранилища моделей.
      const root = modelsRootDir();
      const exists = await RNFS.exists(root);
      if (exists) {
        await RNFS.unlink(root);
      }

      // Пересоздаём каталог и state.json с пустым выбором.
      await RNFS.mkdir(root);
      const st: PersistedState = {
        activeModelId: null,
        updatedAtIso: new Date().toISOString(),
      };
      await writeJsonFile(statePath(), st);

      // Сбрасываем кеш манифеста в памяти (его файл мы уже удалили).
      setManifest(null);
      setCatalogState('idle');
      setCatalogErrorMessage(null);
      setDownloadProgressById({});
      setDownloadRevision(v => v + 1);
    } catch (e) {
      Alert.alert('Не удалось очистить хранилище', String(e));
    } finally {
      setIsClearingStorage(false);
    }
  }, [isClearingStorage]);

  const ensureDownloaded = useCallback(
    async (id: string) => {
      try {
        const rm = manifestById.get(id);
        if (!rm) {
          throw new Error('Модель не найдена в манифесте. Нажми "Обновить список моделей".');
        }

        const item = normalizeRemoteModel(rm);
        const downloaded = await computeDownloadedFlag(item);
        if (downloaded) return;

        setDownloadProgressById(p => ({ ...p, [id]: 0 }));
        const dir = modelDir(id);
        await RNFS.mkdir(dir);

        await downloadToPath({
          url: rm.model.url,
          destPath: localModelPath(id),
          onProgress: r => setDownloadProgressById(p => ({ ...p, [id]: r * 0.85 })),
        });
        await downloadToPath({
          url: rm.labels.url,
          destPath: localLabelsPath(id),
          onProgress: r => setDownloadProgressById(p => ({ ...p, [id]: 0.85 + r * 0.15 })),
        });

        const ok = await computeDownloadedFlag(item);
        if (!ok) {
          throw new Error('Файл скачался, но проверка размера (bytes) не прошла. Проверь манифест.');
        }

        // Триггерим пересчёт downloaded-флагов для UI.
        setDownloadRevision(v => v + 1);
      } catch (e) {
        const msg = String(e);
        Alert.alert('Не удалось скачать модель', msg);
        throw e;
      } finally {
        setDownloadProgressById(p => {
          const next = { ...p };
          delete next[id];
          return next;
        });
      }
    },
    [manifestById]
  );

  const deleteDownloaded = useCallback(async (id: string) => {
    try {
      const dir = modelDir(id);
      const exists = await RNFS.exists(dir);
      if (!exists) return;
      await RNFS.unlink(dir);
      setDownloadRevision(v => v + 1);
    } catch (e) {
      Alert.alert('Не удалось удалить модель', String(e));
    }
  }, []);

  const [downloadedById, setDownloadedById] = useState<Record<string, boolean>>({});

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const pairs = await Promise.all(
        models.map(async m => {
          const ok = await computeDownloadedFlag(m);
          return [m.id, ok] as const;
        })
      );
      if (cancelled) return;
      const next: Record<string, boolean> = {};
      for (const [id, ok] of pairs) next[id] = ok;
      setDownloadedById(next);
    })().catch(e => console.warn('[models] computeDownloaded failed', e));
    return () => {
      cancelled = true;
    };
  }, [downloadRevision, models]);

  const modelsUi = useMemo<ModelListItem[]>(() => {
    return models.map(m => ({
      ...m,
      isDownloaded: downloadedById[m.id] ?? false,
    }));
  }, [downloadedById, models]);

  return {
    // active runtime
    model,
    labels,
    isModelLoaded,
    modelState,
    modelErrorMessage,
    activeModelId,
    activeModelItem,
    activeYoloParams,

    // catalog
    catalogState,
    catalogErrorMessage,
    models: modelsUi,
    downloadProgressById,
    isClearingStorage,

    // actions
    refreshManifest,
    setActiveModel,
    ensureDownloaded,
    deleteDownloaded,
    clearStorage,

    // small diagnostics
    manifestGeneratedAt: manifest?.generatedAt ?? null,
    manifestUrl: MODELS.manifestUrl,
    storageRoot: modelsRootDir(),
    platform: Platform.OS,
  };
}

