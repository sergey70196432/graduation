const path = require('path');
const { getDefaultConfig, mergeConfig } = require('@react-native/metro-config');

/**
 * Metro configuration
 * https://reactnative.dev/docs/metro
 *
 * @type {import('@react-native/metro-config').MetroConfig}
 */
const defaultConfig = getDefaultConfig(__dirname);

// Когда мы импортируем файлы из `../shared/**`, Metro пытается резолвить зависимости (например `react`)
// относительно этой папки и поднимается вверх по дереву `node_modules`. В монорепо-структуре, где
// зависимости лежат в `app/node_modules`, это приводит к ошибке "Unable to resolve module react".
// Решение: явно указать Metro, где искать `node_modules` и как маппить модули.
const appNodeModules = path.resolve(__dirname, 'node_modules');
const extraNodeModules = new Proxy(
  {},
  {
    get: (_, name) => path.join(appNodeModules, String(name)),
  }
);

// Добавляем поддержку .tflite и .txt как ассетов (модель и labels лежат в assets/ и подключаются через require()).
// Добавляем поддержку .svg (импорт как React-компонент через react-native-svg-transformer)
const config = {
  transformer: {
    babelTransformerPath: require.resolve('react-native-svg-transformer'),
  },
  watchFolders: [path.resolve(__dirname, '../shared')],
  resolver: {
    assetExts: defaultConfig.resolver.assetExts.filter(ext => ext !== 'svg').concat(['tflite', 'txt']),
    sourceExts: [...defaultConfig.resolver.sourceExts, 'svg'],
    nodeModulesPaths: [appNodeModules],
    extraNodeModules,
  },
};

module.exports = mergeConfig(defaultConfig, config);
