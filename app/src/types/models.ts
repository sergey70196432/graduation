export type RemoteAsset = {
  /**
   * Pre-signed URL или публичный URL до файла.
   */
  url: string;
  /**
   * Ожидаемый размер в байтах. Используем для проверки "докачалось ли".
   */
  bytes?: number;
  /**
   * Опционально: sha256 в hex. Сейчас мы его НЕ проверяем (без доп. зависимостей),
   * но полезно хранить в S3 для контроля версий/целостности.
   */
  sha256?: string;
};

export type RemoteYoloModel = {
  id: string;
  title: string;
  description?: string;

  /**
   * Размер входа модели (imgsz). Если не задан — берём дефолт из `YOLO.inputSize`.
   */
  inputSize?: number;

  /**
   * Пороги/лимиты можно переопределять на уровне модели.
   * Если не задано — берём дефолты из `YOLO`.
   */
  confidenceThreshold?: number;
  preNmsTopK?: number;
  postNmsTopK?: number;

  model: RemoteAsset; // .tflite
  labels: RemoteAsset; // labels.txt
};

export type RemoteModelManifestV1 = {
  version: 1;
  generatedAt?: string;
  models: RemoteYoloModel[];
};

export type ModelListItem = {
  id: string;
  title: string;
  description?: string;
  inputSize: number;
  confidenceThreshold: number;
  preNmsTopK: number;
  postNmsTopK: number;

  isActive: boolean;
  isDownloaded: boolean;
  localModelPath?: string;
  localLabelsPath?: string;
  remoteModelBytes?: number;
  remoteLabelsBytes?: number;
};

export type ModelCatalogState = 'idle' | 'loading' | 'error';

