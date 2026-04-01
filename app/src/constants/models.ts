import Config from 'react-native-config';

function readEnvString(name: string): string | null {
  const raw = (Config as Record<string, unknown> | undefined)?.[name];
  if (raw == null) return null;
  const s = String(raw).trim();
  return s.length > 0 ? s : null;
}

export const MODELS = {
  /**
   * URL на JSON-манифест в Object Storage.
   * Для Яндекс Object Storage сюда обычно попадает pre-signed URL или публичный https URL.
   *
   * Важно: манифест скачивается по нажатию "Обновить список моделей".
   */
  manifestUrl:
    readEnvString('MODELS_MANIFEST_URL') ??
    (() => {
      throw new Error(
        'MODELS_MANIFEST_URL is not set. Add it to app/.env (or .env.dev/.env.prod).'
      );
    })(),
  /**
   * Папка в DocumentDirectory, куда складываем скачанные модели и кеш манифеста.
   */
  deviceDirName: 'yolo-models',
} as const;

