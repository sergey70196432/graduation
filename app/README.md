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

Пути фиксированные, чтобы было просто заменить файлы:

- `app/assets/models/best.tflite`
- `app/assets/models/labels.txt`

`labels.txt` — по одному классу в строке.

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
