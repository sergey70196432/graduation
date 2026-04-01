# Offline YOLO TFLite (mobile)

Это мобильное приложение (React Native CLI) для офлайн‑детекции дорожных знаков:

- Camera (real‑time) и Video (test mode)
- TFLite inference на устройстве
- bbox overlay + список знаков (SVG)
- загрузка/переключение моделей по сетевому манифесту (кэш в `DocumentDirectory`)

**Важно:** приложение запускалось и тестировалось **только на iOS**. Android‑сборка/рантайм не гарантированы и могут потребовать дополнительных правок/настройки окружения.

## Документация (wiki)

Полная документация лежит в `docs/`.

- [Оглавление wiki](./docs/README.md)
- [Обзор и возможности](./docs/overview.md)
- [Установка и запуск](./docs/getting-started.md)
- [Экраны](./docs/screens.md)
- [Модели и манифест](./docs/models.md)
- [Архитектура пайплайна](./docs/pipeline.md)
- [Debug/Release сборки](./docs/build-modes.md)
- [Производительность](./docs/performance.md)
- [Troubleshooting](./docs/troubleshooting.md)
- [Dev notes](./docs/dev-notes.md)

## Quick start

```bash
cd app
npm install
# отредактируй .env: MODELS_MANIFEST_URL=...
npx pod-install
npm run ios
npm run android
```
