## Обучение детектора дорожных знаков (YOLOv8 → TFLite)

Этот модуль обучает/дообучает детектор дорожных знаков на базе **Ultralytics YOLOv8n** (предобучение на COCO).

### Артефакты и скрипты

- генерация датасета (YOLO): `make_dataset/generate_synth_yolo_dataset.py`
- обучение детектора: `training/signs_detection/train.py`
- выход обучения (Ultralytics): `models/signs_detection/<experiment>/...` (веса, логи, графики)

---

### Шаг 0. Подготовка окружения

Рекомендуется Python 3.10+ и виртуальное окружение.

Установка зависимостей:

```bash
python -m pip install -r training/signs_detection/requirements.txt
```

> Примечание: установка `torch/torchvision` зависит от платформы (CPU/CUDA/MPS). Если `pip` не ставит корректно, используйте инструкции с официального сайта PyTorch под вашу систему.

---

### Шаг 1. Сгенерировать датасет

Из корня репозитория:

```bash
python make_dataset/generate_synth_yolo_dataset.py
```

Скрипт создаст новую версию датасета вида `datasets/dataset_<n>/` и положит туда:

- `images/`, `labels/`
- `train.txt`, `val.txt`
- `dataset.yaml` (для Ultralytics)

---

### Шаг 2. Обучить (дообучить) модель

1) В `training/signs_detection/train.py` укажи путь к `dataset.yaml`:

- `DATASET_YAML = "datasets/dataset_1/dataset.yaml"` (пример; поставь актуальную версию)

2) Запуск обучения из корня:

```bash
python training/signs_detection/train.py
```

---

### Куда сохраняются результаты

- основной вывод Ultralytics: `models/signs_detection/<experiment>/...`
- дополнительно сохраняется стабильная копия лучшего чекпоинта и краткая сводка:
  - `models/signs_detection/signs_detector_best.pt`
  - `models/signs_detection/signs_detector_info.txt`

---

### Примечание про экспорт в TFLite для мобильного приложения

Мобильное приложение ожидает TFLite-модель детектора, где **NMS встроен в граф** и выход имеет форму детекций:

- `[1, N, 6] float32`: `x1, y1, x2, y2, score, classId`

Перед публикацией модели и добавлением в манифест нужно убедиться, что экспорт даёт совместимый output (см. `app/docs/models.md`, `app/docs/pipeline.md`).
