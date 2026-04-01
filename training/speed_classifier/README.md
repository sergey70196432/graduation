## Классификатор значения скорости (PyTorch → TFLite → React Native)

Этот модуль нужен, чтобы после детектора знаков (YOLO) уточнить **конкретное значение** на знаке скорости:
например не просто `3.24`, а `3.24_50`, `3.24_60`, ...

### Артефакты и скрипты

- `make_dataset/generate_speed_classifier_dataset.py`
  - генерирует датасет `datasets/speed_cls_v1/`
  - пишет `labels.txt` (порядок классов!)
- `training/speed_classifier/train.py`
  - обучает модель
  - каждый запуск складывает артефакты в новую папку `training/speed_classifier/runs/run_<n>/`
- `training/speed_classifier/eval.py`
  - делает confusion matrix
  - пишет `predictions_*.csv` (по каждому изображению: true/pred/conf/path)
- `training/speed_classifier/export_to_tflite.py`
  - экспорт PyTorch → ONNX → onnx2tf → TFLite
  - рядом кладёт `labels.txt`
- `app/assets/models/speed_classifier/`
  - итоговые `model_*.tflite` и `labels.txt` для приложения

---

### Шаг 0. Подготовка окружения

Нужен Python 3.10+ и зависимости для обучения:

```bash
pip install -r training/speed_classifier/requirements.txt
```

Для экспорта в TFLite зависимости отдельные (они более “капризные”):

```bash
pip install -r training/speed_classifier/requirements_convert.txt
```

> Если работаете с виртуальными окружениями (venv) **рекомендую** создавать отдельное окружение для классификатора, т.к. версии зависимостей могут напакастить всему остальному проекту

---

### Шаг 1. Сгенерировать датасет

Из корня репозитория:

```bash
python make_dataset/generate_speed_classifier_dataset.py
```

Скрипт возьмёт PNG-кропы знаков (прозрачный фон) и соберёт датасет в `datasets/speed_cls_v1/`.
Все параметры генерации (размер, аугментации, количество копий, bad-crops) находятся **вверху файла**.

На выходе будет:

- `datasets/speed_cls_v1/train/...`
- `datasets/speed_cls_v1/val/...` (если включено)
- `datasets/speed_cls_v1/test/...` (если включено)
- `datasets/speed_cls_v1/labels.txt`

Важно: `labels.txt` — это **источник истины для порядка классов**.

Скрипты обучения/оценки читают `labels.txt` и строят `class_to_idx` строго по нему.
Это защищает от рассинхрона (когда индексы классов “едут” из-за сортировки папок).

---

### Шаг 2. Обучить модель

```bash
python training/speed_classifier/train.py
```

Вверху `train.py` находятся параметры (epochs, batch size, lr, device, AMP, pretrained и т.п.).

#### Куда сохраняется обучение

Каждый запуск создаёт новую папку:

`training/speed_classifier/runs/run_1/`, `run_2/`, ...

Внутри:

- `config.json` — параметры запуска + список классов
- `metrics.jsonl` — метрики по эпохам
- `best.pt` — лучший чекпоинт (по val acc1), если есть val
- `last.pt` — последний чекпоинт
- `test_metrics.json` — финальная метрика на test (если test есть)

---

### Шаг 3. Оценить качество (confusion matrix + CSV)

```bash
python training/speed_classifier/eval.py
```

Параметры находятся вверху `eval.py`:

- `SPLIT`: `val` или `test`
- `CKPT_PATH`: путь к `best.pt` или `last.pt`
- `OUT_DIR`: куда писать артефакты. Если пустой — будет папка чекпоинта.

На выходе:

- `confusion_<split>.png`
- `confusions_<split>.txt` (самые частые ошибки)
- `predictions_<split>.csv` (по каждому файлу: true/pred/conf)

---

### Шаг 4. Экспорт в TFLite

```bash
python training/speed_classifier/export_to_tflite.py
```

Параметры вверху `export_to_tflite.py`:

- `CKPT_PATH`: путь к чекпоинту
- `OUT_DIR`: куда сохранять экспорт. Если пустой — будет `<папка_чекпоинта>/export`.
- `TFLITE_QUANT`: `fp16` или `fp32`

На выходе в папке export:

- `model_fp16.tflite` или `model_fp32.tflite`
- `labels.txt` (порядок классов из чекпоинта!)

---

### Как использовать в React Native

#### 1) Копируем ассеты

Скопируй в приложение:

- `model_*.tflite` → `app/assets/models/speed_classifier/model_float32.tflite` (или fp16)
- `labels.txt` → `app/assets/models/speed_classifier/labels.txt`

После изменения ассетов нужен **полный rebuild** приложения.

#### 2) Препроцессинг должен совпадать с обучением

На обучении в `dataset.py` используется:

- `ToTensor()` → диапазон 0..1
- `Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))`

Значит в приложении перед `model.runSync()` нужно делать то же самое:

\[
x = \frac{(u8/255) - mean}{std}
\]

Если этого не сделать — модель часто “угадывает цифру” (например 50), но путает базу/класс и падает confidence.

#### 3) Частая оптимизация

YOLO обычно хорошо различает “базу” знака (`3.24` vs `3.25`), поэтому можно ограничивать выбор
класса классификатора только этой базой и уточнять лишь скорость.


