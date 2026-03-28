# Offline YOLOv8 object detection (MVP)

Это учебный MVP: камера + офлайн-инференс YOLOv8n (TFLite) + отрисовка боксов поверх превью.

Главная цель — **простой и рабочий код**, который легко читать и менять под свою модель.

## Что умеет

- Превью камеры (Android + iOS)
- Start/Stop detection
- Switch camera (front/back)
- Офлайн-инференс TFLite на устройстве
- Отрисовка bounding boxes + класс + confidence
- FPS (по обновлениям детекций) и время инференса (только `model.runSync`)
- “Один кадр за раз”: если инференс идёт, новый кадр пропускается

## Стек

- **React Native CLI** (не Expo)
- **TypeScript**
- `react-native-vision-camera` — камера + frame processor
- `react-native-worklets-core` — worklets (нужно для frame processor)
- `vision-camera-resize-plugin` — быстрый resize кадра в worklet
- `react-native-fast-tflite` — TFLite inference через JSI (можно прямо в frame processor)
- `react-native-svg` — рисуем боксы поверх камеры

## Куда положить модель и labels

### Встроенная (bundled) модель

В приложении есть **fallback** модель, которая лежит в ассетах (чтобы всё работало даже без интернета):

- `app/assets/models/model_float16.tflite`
- `app/assets/models/labels.txt`

`labels.txt` — по одному классу в строке.

### Модели по сети (манифест + S3/Object Storage)

Приложение умеет скачивать модели по интернету и хранить их в **DocumentDirectory** (персистентно).

- **Где задаётся URL манифеста**: `src/constants/models.ts` → `MODELS.manifestUrl`
- **Где хранятся скачанные модели на устройстве**: `DocumentDirectory/yolo-models/`
  - `DocumentDirectory/yolo-models/manifest-cache.json` — кеш последнего манифеста
  - `DocumentDirectory/yolo-models/state.json` — выбранная модель
  - `DocumentDirectory/yolo-models/<modelId>/model.tflite`
  - `DocumentDirectory/yolo-models/<modelId>/labels.txt`

Манифест скачивается **по нажатию кнопки “Обновить”** в попапе выбора модели (кнопка “Модель” в колонке “Знаки”).

Важно: URL манифеста должен быть **стабильным** (например публичный объект или ваш backend-эндпоинт).
Если делать `manifestUrl` тоже pre-signed, он будет истекать, и приложение не сможет обновлять список моделей без обновления самой сборки.

#### Формат манифеста (JSON, UTF‑8)

```json
{
  "version": 1,
  "generatedAt": "2026-03-28T12:00:00.000Z",
  "models": [
    {
      "id": "yolo-signs-v1",
      "title": "YOLO Signs v1",
      "description": "Модель под мои дорожные знаки",
      "inputSize": 320,

      "confidenceThreshold": 0.25,
      "iouThreshold": 0.45,
      "preNmsTopK": 200,
      "postNmsTopK": 50,

      "model": {
        "url": "https://<presigned-url-to-model.tflite>",
        "bytes": 12345678,
        "sha256": "optional-hex-sha256"
      },
      "labels": {
        "url": "https://<presigned-url-to-labels.txt>",
        "bytes": 1234,
        "sha256": "optional-hex-sha256"
      }
    }
  ]
}
```

- `url`: **pre-signed** URL (или публичный URL) напрямую до файла
- `bytes`: рекомендуется указывать, приложение проверяет размер после скачивания
- `sha256`: опционально (сейчас не проверяется, но полезно хранить для контроля целостности/версий)

#### Как хранить файлы в Object Storage (Yandex Cloud)

Можно хранить как угодно, главное — чтобы в манифесте были прямые ссылки (pre-signed) на конкретные файлы.
Один из простых вариантов структуры:

- `models/manifest.json`
- `models/yolo-signs-v1/model.tflite`
- `models/yolo-signs-v1/labels.txt`
- `models/yolo-signs-v2/model.tflite`
- `models/yolo-signs-v2/labels.txt`

На стороне бэкенда/скрипта генерации манифеста ты:

- кладёшь файлы в бакет
- генерируешь pre-signed URL для `manifest.json`, `model.tflite`, `labels.txt`
- записываешь эти URL в манифест

## Экспорт модели из Ultralytics

Float32 (база для MVP):

```bash
yolo export model=best.pt format=tflite imgsz=320
```

Опционально int8 (после того как float32 заработал):

```bash
yolo export model=best.pt format=tflite imgsz=320 int8=True data=data.yaml
```

## Установка и запуск

Из корня репозитория:

```bash
cd app
npm install
npx pod-install
npx react-native run-android
npx react-native run-ios
```

## Важно про iOS и TFLite (New Architecture)

`react-native-fast-tflite` использует TurboModule метод `install()`. На iOS он корректно доступен, когда включена **New Architecture**.

В этом проекте New Architecture для iOS включена в `ios/Podfile`.

### Важно: фикс для RN 0.84+ (bridgeless) и `install() === false`

На React Native 0.84 (и похожих) iOS по умолчанию работает в режиме **bridgeless**. В этом режиме `RCTBridge currentBridge` — это `RCTBridgeProxy` (NSProxy), и прямой cast к `RCTCxxBridge` ломается.

Симптомы:

- в логах: `install() попытка ...: false`
- в UI: “Модель не загрузилась”

Чтобы это работало без переписывания нативки, в проект добавлен **автоматический патч** через `patch-package`, который правит `react-native-fast-tflite` так, чтобы он брал `runtime` через метод `runtime` у `currentBridge` (и это работает и для `RCTBridgeProxy`, и для `RCTCxxBridge`).

Это делается автоматически после `npm install` (скрипт `postinstall`).

Если ты менял зависимости или Podfile, сделай так:

```bash
cd app/ios
LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 pod install --repo-update
```

И потом в Xcode: **Product → Clean Build Folder**, и пересобери на устройство.

## Важные заметки (про выход модели)

YOLOv8 TFLite иногда отдаёт output в разных формах (shape), например:

- `[1, 84, 8400]`
- `[1, 8400, 84]`

В этом проекте shape логируется при загрузке модели:

- смотри логи: `[tflite] outputs: ... shape: ...`

Если детекции “нулевые” или боксы странные, чаще всего нужно поправить **одно место**:

- `src/utils/yoloPostprocess.ts` → функция `inferYoloOutputInfo(...)`

Там есть TODO на русском, где именно это делается.

## Ограничения MVP

- Внутри worklet делается и letterbox, и нормализация. Это не самый быстрый вариант, но понятный и рабочий.
- FPS считается по частоте обновления результата на JS стороне (мы специально обновляем UI не чаще ~12 раз/сек).
- Сейчас ожидается float32 output. Для int8 нужно добавить деквантизацию (есть TODO в одном месте).
