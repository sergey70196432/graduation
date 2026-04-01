# Быстрый старт (установка и запуск)

**Важно:** проект запускался и тестировался **только на iOS**. Инструкции для Android оставлены как ориентир, но сборка/запуск на Android могут потребовать дополнительных правок и настройки окружения.

## Требования

### Общие

- **Node.js**: см. `package.json → engines.node` (в проекте указано `>= 22.11.0`)
- **npm** (используется `package-lock.json`)

### iOS

- macOS + **Xcode**
- **CocoaPods**

### Android

- **Android Studio**
- Android SDK / NDK (ставятся через Android Studio)
- JDK совместимый с RN/Gradle (обычно 17)

## Установка зависимостей

Из папки `app/`:

```bash
cd app
npm install
```

## Настройка переменных окружения

В `.env` укажи URL манифеста моделей детектора:

- `MODELS_MANIFEST_URL=https://<...>/manifest.json`

См. подробнее: `docs/models.md`.

После `npm install` автоматически запускается `postinstall`:

- `patch-package` (патчи в `app/patches/`)
- генерация реестра SVG знаков (`npm run gen:signs`)

## iOS: установка pods

```bash
cd app
npx pod-install
```

Если есть проблемы с pods:

```bash
cd app/ios
LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 pod install --repo-update
```

## Запуск (Debug)

### Metro

```bash
cd app
npm start
```

### iOS

```bash
cd app
npm run ios
```

или через Xcode (Scheme → Run).

### Android

```bash
cd app
npm run android
```

## Проверка, что модель загрузилась

В логах после загрузки модели выводятся:

- `[tflite] delegate: ...`
- `[tflite] inputs: ...`
- `[tflite] outputs: ...`

Если модель не загрузилась — см. `docs/troubleshooting.md`.

