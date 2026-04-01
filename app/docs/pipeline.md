# Архитектура пайплайна (кадр → TFLite → bbox)

## Термины

- **inputSize**: размер квадрата входа модели (например 640×640)
- **letterbox**: вписывание исходного кадра в квадрат `inputSize×inputSize` с сохранением пропорций и паддингами
- **NMS**: подавление пересекающихся боксов (в этом проекте встроено в модель)

## Camera mode (VisionCamera frame processor)

Высокоуровнево:

1) Берём `Frame` из `react-native-vision-camera`
2) Делаем resize до размеров, которые впишутся в `inputSize` (через `vision-camera-resize-plugin`)
3) Собираем float32 input (RGB, нормализация 0..1) и кладём в letterbox‑квадрат
4) Делаем `model.runSync([inputFloat])`
5) Декодируем выход `[x1,y1,x2,y2,score,classId]`
6) Переводим координаты из letterbox в координаты исходного кадра камеры
7) Отдаём детекции в UI (`DetectionOverlay`, колонка “Знаки”)

Ключевой файл: `src/hooks/useYoloDetector.ts`

## Video mode (thumbnail → decode → letterbox → tflite)

Высокоуровнево:

1) Берём кадр видео через `createThumbnail({ url, timeStamp, ... })`
2) Декодируем jpeg в RGBA
3) Строим letterboxed float32 input
4) `model.runSync([input])`
5) `decodeYoloV8DetectionsEmbeddedNms(...)`

Ключевой файл: `src/screens/VideoDetectionScreen.tsx`

## Декодирование output с встроенным NMS

Файл: `src/utils/yoloPostprocess.ts`

Ожидаем, что выход — это массив, где каждые 6 чисел:

- `x1, y1, x2, y2, score, classId`

Функция:

- фильтрует по `confidenceThreshold`
- чинит возможные инверсии `x2<x1` / `y2<y1`
- маппит координаты обратно из letterbox в исходный кадр

## Доп. стадия: классификация скорости (optional refinement)

После YOLO‑детекции может выполняться дополнительный классификатор, который уточняет значение скорости на знаках.

- код: `src/hooks/useYoloDetector.ts` (стадия `speed_cls`)
- модель: `src/hooks/useSpeedClassifierModel.ts`
- результат: в `Detection` появляются поля `refinedLabel` и `refinedConfidence`

Подробнее см. [docs/speed-classifier.md](./speed-classifier.md).

## Shapes (inputs/outputs)

Как проверить входы/выходы TFLite модели (shapes) — см. `docs/models.md` (раздел “Как проверить shapes (inputs/outputs)”).
