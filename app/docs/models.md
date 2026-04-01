# Модели детектора знаков (YOLO), манифест и Object Storage

Этот документ описывает **модели детектора знаков** (YOLO TFLite), которые выбираются в попапе “Модель” и скачиваются по манифесту: `model.tflite` + `labels.txt`.

## Важно: в приложении нет bundled модели

В этой версии приложения **нет встроенной (bundled) модели**.
Это означает:

- при первом запуске модель **не выбрана**
- чтобы начать детекцию, нужно открыть попап “Модель”, обновить манифест и скачать/выбрать модель
- если модель не выбрана, на превью камеры будет затемняющий оверлей “Нет выбранной модели”

## Модели по сети (каталог моделей)

Приложение умеет:

- скачать JSON‑манифест моделей по URL
- показать список моделей
- скачать выбранную модель и `labels.txt`
- переключать активную модель

### Где задаётся URL манифеста

В `.env` (и при необходимости в `.env.dev` / `.env.prod`):

- `MODELS_MANIFEST_URL=https://<...>/manifest.json`

В коде это читается в `src/constants/models.ts` → `MODELS.manifestUrl`.

Важно: URL манифеста должен быть **стабильным** (публичный объект или backend endpoint).
Если сделать его pre‑signed — он истечёт и “Обновить” перестанет работать.

### Где хранятся файлы на устройстве

В `DocumentDirectory` (персистентно):

- `DocumentDirectory/yolo-models/manifest-cache.json` — кеш манифеста
- `DocumentDirectory/yolo-models/state.json` — активная модель
- `DocumentDirectory/yolo-models/<modelId>/model.tflite`
- `DocumentDirectory/yolo-models/<modelId>/labels.txt`

## Формат манифеста (JSON)

Пример:

```json
{
  "version": 1,
  "generatedAt": "2026-03-28T12:00:00.000Z",
  "models": [
    {
      "id": "yolo-signs-v1-640-nms",
      "title": "YOLO Signs 640 (NMS)",
      "description": "Пример",
      "inputSize": 640,
      "confidenceThreshold": 0.25,
      "model": { "url": "https://<...>/model.tflite", "bytes": 123456 },
      "labels": { "url": "https://<...>/labels.txt", "bytes": 1234 }
    }
  ]
}
```

Поля:

- `version`: сейчас `1`
- `generatedAt`: опционально (ISO)
- `models[]`:
  - `id`: уникальный ID (используется как имя папки в `DocumentDirectory`)
  - `title`: название в UI
  - `description`: опционально
  - `inputSize`: `imgsz` для letterbox (например 320/640/800)
  - `confidenceThreshold`: фильтрация по score на выходе модели
  - `model.url`: ссылка на `.tflite`
  - `labels.url`: ссылка на `labels.txt`
  - `bytes`: рекомендуется — приложение проверяет размер после скачивания
  - `sha256`: опционально (сейчас не проверяется, но полезно хранить для контроля целостности)

## Требование к TFLite модели: NMS внутри модели

В этой версии приложения предполагается, что TFLite отдаёт финальные детекции.
Типичный output shape:

- `Identity [1, 300, 6] float32`

где каждая детекция:

- `x1, y1, x2, y2, score, classId`

Именно под это написан декодер `decodeYoloV8DetectionsEmbeddedNms()` (см. `src/utils/yoloPostprocess.ts`).

## Как проверить shapes (inputs/outputs)

В приложении в логах после загрузки модели печатается:

- `[tflite] inputs: ...`
- `[tflite] outputs: ...`

А через Python можно проверить так:

```bash
pip install tensorflow
python - <<'PY'
import tensorflow as tf
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
print("Inputs:")
for d in interpreter.get_input_details():
    print(d["name"], d["shape"], d["dtype"])
print("Outputs:")
for d in interpreter.get_output_details():
    print(d["name"], d["shape"], d["dtype"])
PY
```

## Экспорт из Ultralytics (пример)

Команды зависят от твоей версии Ultralytics и того, как у тебя получился output `[1, N, 6]`.
Практический способ проверить:

1) экспортировать tflite
2) через Python `tf.lite.Interpreter` посмотреть shapes (inputs/outputs)

См. `docs/pipeline.md` и `docs/troubleshooting.md`.

## Yandex Cloud Object Storage (S3‑совместимое)

Рекомендуемая структура в бакете (пример):

- `models/manifest.json` (публичный)
- `models/yolo-signs-v1-640-nms/model.tflite`
- `models/yolo-signs-v1-640-nms/labels.txt`

В манифесте можно хранить:

- публичные ссылки
- или pre‑signed URL на `model.tflite` / `labels.txt`

Но `manifestUrl` лучше делать стабильным.

