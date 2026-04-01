# Классификатор скоростных знаков (Speed classifier)

В приложении есть дополнительный модуль, который улучшает распознавание **значения** на скоростных знаках.

Идея:

- YOLO находит знак и его “базовый” класс (например `3.24_`*)
- затем отдельная маленькая модель‑классификатор уточняет **цифры скорости** и при успехе подменяет label на `refinedLabel`

## Где находится код

- загрузка speed‑модели: `src/hooks/useSpeedClassifierModel.ts`
- применение классификатора к детекциям: `src/hooks/useYoloDetector.ts` (стадия `speed_cls`)
- использование в UI:
  - overlay: `src/components/DetectionOverlay.tsx` показывает `refinedLabel ?? label`
  - колонка знаков: `src/screens/CameraDetectionScreen.tsx` учитывает `refinedLabel/refinedConfidence`

## Какие файлы модели используются

Сейчас классификатор скорости — **bundled** (лежит в ассетах приложения):

- `assets/models/speed_classifier/model_float32.tflite`
- `assets/models/speed_classifier/labels.txt`

`labels.txt` содержит классы в формате:

- `<base>_<value>`

Например: `3.24_40`, `3.24_60`, `3.24_110`, ...

## Как включается

В `CameraDetectionScreen` классификатор включён всегда, если модель загрузилась.

## Как работает (пайплайн)

Для каждой YOLO‑детекции:

1. Берём `base` из `d.label` (строка до `_`)
2. Проверяем, относится ли `base` к скоростным базам (`speedBases`, вычисляется из `labels.txt` классификатора)
3. Если это не speed‑класс — пропускаем детекцию без изменений
4. Если это speed‑класс:
  - берём ROI **из оригинального кадра камеры** (не из YOLO input tensor)
  - добавляем небольшой margin (`SPEED_CLS_MARGIN`)
  - crop+resize ROI в `SPEED_CLS_INPUT_SIZE × SPEED_CLS_INPUT_SIZE` через `vision-camera-resize-plugin`
  - нормализуем RGB (mean/std как в ImageNet)
  - делаем `speedModel.runSync([input])`
  - берём лучший класс (argmax) **только внутри той же базы** (это снижает путаницу между базами)
  - считаем `prob` через softmax
  - если `prob >= SPEED_CLS_MIN_CONF` → записываем:
    - `d.refinedLabel = speedLabels[idx]`
    - `d.refinedConfidence = prob`

## Какие есть настройки (в коде)

В `useYoloDetector.ts`:

- `SPEED_CLS_INPUT_SIZE`: размер входа классификатора (сейчас 128)
- `SPEED_CLS_MARGIN`: насколько расширяем bbox перед crop
- `SPEED_CLS_MIN_ROI_PX`: минимальный размер ROI (слишком маленькие знаки пропускаем)
- `SPEED_CLS_MIN_CONF`: минимальная уверенность, чтобы принять `refinedLabel`

## Ограничения и типичные проблемы

- **Перспектива/угол**: если знак снят под сильным углом, классификатор может путаться. Обычно лечится данными и аугментациями (perspective/affine) и/или более крупным inputSize классификатора.
- **Качество ROI**: если YOLO bbox обрезает цифры или знак маленький — классификатору сложно.
- **Производительность**: это дополнительный инференс на каждую speed‑детекцию; если speed‑знаков много, может повлиять на нагрев.

