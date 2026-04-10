## Генерация датасетов и ассетов (make_dataset)

Здесь есть несколько независимых инструментов. Подробности вынесены в отдельные документы:

- **Датасет для классификатора скорости (значение на знаке)**:
  - см. `make_dataset/README_speed_classifier_dataset.md`
  - основной скрипт: `make_dataset/generate_speed_classifier_dataset.py`
  - (опционально) генерация PNG разных размеров из SVG: `make_dataset/generate_speed_pngs.ipynb`

- **YOLO датасет детекции знаков (синтетика + внешние данные)**:
  - см. `make_dataset/README_yolo_signs_dataset.md`
  - основной скрипт: `make_dataset/generate_synth_yolo_dataset.py`

---

## Требования и установка (для make_dataset)

### Python
- **Python 3.10+** (рекомендуется; скрипты в репозитории в целом уже используют современные версии)

### Python-зависимости

Установить минимальные зависимости можно так:

```bash
python -m pip install -r make_dataset/requirements.txt
```

### Системные зависимости (только если вам нужен SVG → PNG)

Есть 2 варианта:
- **опционально**: `cairosvg` (Python-библиотека; может требовать системные Cairo зависимости)
- **системные утилиты**:
  - `rsvg-convert` (librsvg) или
  - `inkscape`

В ноутбуке `make_dataset/generate_speed_pngs.ipynb` используется `rsvg-convert`.

---

## Быстрый старт

### Speed classifier dataset

```bash
python -m pip install -r make_dataset/requirements.txt
python make_dataset/generate_speed_classifier_dataset.py
```

Подробности и вход/выход: `make_dataset/README_speed_classifier_dataset.md`.

### YOLO signs dataset

```bash
python -m pip install -r make_dataset/requirements.txt
python make_dataset/generate_synth_yolo_dataset.py
```

Перед запуском генератора детекционного датасета нужно отдельно скачать папки `make_dataset/dashcam_frames` и `make_dataset/negative` с Яндекс.Диска и распаковать их в `make_dataset/`.
Ссылка:
- [dashcam_frames](https://disk.yandex.ru/d/VyC2czLyajINLw)
- [negative](https://disk.yandex.ru/d/gz2vv16C3n4BmA)

Подробности и вход/выход: `make_dataset/README_yolo_signs_dataset.md`.
