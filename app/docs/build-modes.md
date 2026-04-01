# Debug/Release сборки

## Как в JS понять Debug vs Release

В React Native есть глобальная константа:

- `__DEV__ === true` → debug/dev сборка (Metro)
- `__DEV__ === false` → release сборка

## Поведение приложения

В проекте используется правило:

- **Debug сборка**
  - bbox overlay (`DetectionOverlay`) рисуется всегда
  - кнопка “Debug” доступна
- **Release сборка**
  - кнопка “Debug” скрыта

Файл: `src/screens/CameraDetectionScreen.tsx`

## Как собрать Release

### iOS

- Release обычно делается через **Archive** в Xcode
- Debug — через **Run**

Если в приложении нет dev menu и `__DEV__` ведёт себя как `false`, вероятно ты запустил Release.

### Android

Классика:

- Debug: `npx react-native run-android`
- Release: `./gradlew assembleRelease` (из `android/`)

Подробности (если нужно) можно расширить под твой CI/подписи.

