# Dev notes (патчи, SVG знаки, генераторы)

## patch-package

Проект использует `patch-package` для фиксов некоторых библиотек под RN 0.84+.

- патчи лежат в `app/patches/`
- применяются автоматически в `postinstall`

Команда:

```bash
cd app
npm install
```

## SVG знаки и реестр `signRegistry`

SVG файлы лежат вне `app/`:

- `../shared/signs/images/*.svg`

Чтобы Metro гарантированно забандлил все SVG и мы могли выбирать иконку по `label`,
генерируется файл‑реестр:

- `src/signs/signRegistry.tsx`

Скрипт генерации:

- `scripts/generateSignRegistry.mjs`

Запуск вручную:

```bash
cd app
npm run gen:signs
```

## Добавление нового знака (SVG)

1) добавить SVG в `shared/signs/images/`
2) запустить `npm run gen:signs`
3) убедиться, что label детекции совпадает с именем файла без `.svg`

