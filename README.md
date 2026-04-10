# RoadSign Mobile Offline

Репозиторий дипломного проекта по офлайн-распознаванию дорожных знаков на мобильном устройстве.

Проект состоит из двух основных частей:

- мобильного приложения на React Native, которое запускает TFLite-модели на устройстве;
- Python-пайплайна для генерации данных, обучения моделей детекции и классификации скорости, оценки качества и экспорта моделей.

На практике система работает так:

1. детектор дорожных знаков находит знак на кадре;
2. для скоростных знаков дополнительный классификатор уточняет значение, например `3.24_40` или `3.24_60`;
3. мобильное приложение показывает bbox, подпись и связанные иконки знаков.

## Что есть в репозитории

```text
.
├── README.md
├── app/            # мобильное приложение React Native
├── make_dataset/   # генерация датасетов и подготовка ассетов
├── training/       # обучение, оценка и экспорт моделей
├── shared/         # общие ресурсы знаков: SVG, PNG, CSV, сплиты
├── datasets/       # сгенерированные версии датасетов
├── models/         # обученные и экспортированные модели
├── speed_test_roi/ # ROI-набор для отдельной оценки speed classifier
└── py_utils/       # небольшие общие Python-утилиты
```

## Основные модули

### `app/`

Мобильное приложение на React Native CLI для офлайн-детекции дорожных знаков.

Что умеет:

- работать с камерой в реальном времени;
- прогонять детекцию по видео;
- запускать TFLite-инференс на устройстве;
- загружать и переключать модели по JSON-манифесту;
- дополнительно уточнять значение скорости на speed-знаках через отдельный классификатор.

Ключевые документы:

- [`app/README.md`](app/README.md) — краткий вход в модуль;
- [`app/docs/README.md`](app/docs/README.md) — оглавление документации;
- [`app/docs/getting-started.md`](app/docs/getting-started.md) — установка и запуск;
- [`app/docs/pipeline.md`](app/docs/pipeline.md) — runtime-пайплайн детекции;
- [`app/docs/models.md`](app/docs/models.md) — формат моделей и манифеста;
- [`app/docs/speed-classifier.md`](app/docs/speed-classifier.md) — как встроен классификатор скорости.

### `make_dataset/`

Набор скриптов для подготовки обучающих данных.

Здесь находятся:

- генерация синтетического YOLO-датасета для детектора;
- генерация датасета для классификатора скорости;
- фоновые изображения, dashcam-кадры и вспомогательные инструменты подготовки ассетов.

Ключевые документы:

- [`make_dataset/README.md`](make_dataset/README.md) — обзор генераторов;
- [`make_dataset/README_yolo_signs_dataset.md`](make_dataset/README_yolo_signs_dataset.md) — датасет детекции;
- [`make_dataset/README_speed_classifier_dataset.md`](make_dataset/README_speed_classifier_dataset.md) — датасет классификатора скорости.

### `training/`

Скрипты для обучения, оценки и экспорта моделей.

Содержит два направления:

- `training/signs_detection/` — обучение YOLO-детектора дорожных знаков;
- `training/speed_classifier/` — обучение классификатора значения скорости, оценка и экспорт в TFLite.

Ключевые документы:

- [`training/signs_detection/README.md`](training/signs_detection/README.md);
- [`training/speed_classifier/README.md`](training/speed_classifier/README.md).

### `shared/`

Общий каталог знаков и графических ресурсов, которые используются и в генерации данных, и в приложении:

- SVG-шаблоны;
- CSV-описания классов;
- подготовленные PNG для speed classifier;
- разбиения составных знаков.

## Как читать проект с нуля

Если нужен быстрый маршрут по репозиторию:

1. начни с [`app/docs/overview.md`](app/docs/overview.md), чтобы понять пользовательский сценарий;
2. посмотри [`app/docs/pipeline.md`](app/docs/pipeline.md), чтобы увидеть runtime-пайплайн;
3. затем [`make_dataset/README.md`](make_dataset/README.md), чтобы понять, откуда берутся данные;
4. потом [`training/signs_detection/README.md`](training/signs_detection/README.md) и [`training/speed_classifier/README.md`](training/speed_classifier/README.md), чтобы понять обучение и экспорт.

## Типовой пайплайн разработки

### 1. Сгенерировать датасеты

- для детектора: `python make_dataset/generate_synth_yolo_dataset.py`
- для классификатора скорости: `python make_dataset/generate_speed_classifier_dataset.py`

Для генерации детекционного датасета папки `make_dataset/dashcam_frames` и `make_dataset/negative` нужно скачать отдельно с Яндекс.Диска и положить в `make_dataset/`.
Ссылка: 
- [dashcam_frames](https://disk.yandex.ru/d/VyC2czLyajINLw)
- [negative](https://disk.yandex.ru/d/gz2vv16C3n4BmA)

### 2. Обучить модели

- детектор: `python training/signs_detection/train.py`
- классификатор скорости: `python training/speed_classifier/train.py`

### 3. Оценить качество

- детектор и классификатор имеют отдельные eval-скрипты в `training/`;
- для классификатора скорости есть отдельная ROI-оценка через `eval_speed_classifier_on_rois.ipynb`.

### 4. Экспортировать модель и подключить в приложение

- классификатор скорости экспортируется через `training/speed_classifier/export_to_tflite.py`;
- мобильное приложение использует TFLite-модели и `labels.txt`, которые должны соответствовать друг другу по порядку классов;
- для детектора приложение ожидает TFLite-модель с NMS в графе и выходом формата `[1, N, 6]`.

## Важные замечания

- `datasets/` и `models/` в основном не хранятся в git целиком: это рабочие артефакты генерации и обучения.
- Корректность индексов классов для classifier-пайплайна завязана на `labels.txt`.
- Для speed classifier отдельно учитывается domain shift между синтетикой и реальными ROI, поэтому в пайплайне используются усиленные ROI-подобные аугментации и отдельная оценка на real ROI.
- Мобильное приложение в рамках проекта ориентировано прежде всего на iOS-сценарий.
