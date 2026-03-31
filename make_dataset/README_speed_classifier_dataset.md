## Датасет для классификатора скорости (speed value classifier)

Этот датасет нужен, чтобы после детектора знаков (YOLO) уточнить **конкретное значение** на знаке скорости:
например не просто `3.24`, а `3.24_50`, `3.24_60`, ...

---

## Где что лежит

- **Скрипт генерации датасета**: `make_dataset/generate_speed_classifier_dataset.py`
- **(опционально) ноутбук SVG → PNG разных размеров**: `make_dataset/genefate_speed_pngs.ipynb`
- **Выходной датасет**: `datasets/speed_cls_v<n>/`
  - `train/<class>/*.png`
  - `val/<class>/*.png` (если включено)
  - `test/<class>/*.png` (если включено)
  - `labels.txt` — порядок классов (источник истины)
  - `stats.txt` — статистика генерации

---

## Требования

Установить зависимости:

```bash
python -m pip install -r make_dataset/requirements.txt
```

Если используете ноутбук `genefate_speed_pngs.ipynb`, нужен `rsvg-convert` (librsvg):
- macOS: `brew install librsvg`
- Ubuntu: `sudo apt-get install librsvg2-bin`

---

## Шаг 1 (опционально). Сгенерировать PNG разных размеров из SVG

Ноутбук: `make_dataset/genefate_speed_pngs.ipynb`

### Вход
- `shared/signs/speed_png/*.svg` с именами `<base>_<speed>.svg`, например `3.24_50.svg`.

### Выход
- `shared/signs/speed_png/pngs/<base>_<speed>/<base>_<speed>_<size>.png`

---

## Шаг 2. Сгенерировать датасет для классификатора

Запуск (из корня репозитория):

```bash
python make_dataset/generate_speed_classifier_dataset.py
```

### Вход
- PNG-кропы с прозрачностью: `shared/signs/speed_png/pngs/<class>/...`
  - где `<class>` это имя класса вида `3.24_50`, `3.24_60`, ...

### Выход
- `datasets/speed_cls_v<n>/...` (версия авто-инкрементится)

### Важно (очень)
- **Каждый запуск создаёт новую версию** `datasets/speed_cls_v1`, `speed_cls_v2`, ...
- `labels.txt` — это **источник истины порядка классов**.
  Он используется в обучении/оценке/экспорте, чтобы не было рассинхрона “индекс → строка”.

Все параметры генерации (сплиты, bad-crops, размеры, лимиты) находятся вверху файла
`make_dataset/generate_speed_classifier_dataset.py`.

