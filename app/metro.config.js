const { getDefaultConfig, mergeConfig } = require('@react-native/metro-config');

/**
 * Metro configuration
 * https://reactnative.dev/docs/metro
 *
 * @type {import('@react-native/metro-config').MetroConfig}
 */
const defaultConfig = getDefaultConfig(__dirname);

// Добавляем поддержку .tflite и .txt как ассетов (модель и labels лежат в assets/ и подключаются через require()).
const config = {
  resolver: {
    assetExts: [...defaultConfig.resolver.assetExts, 'tflite', 'txt'],
  },
};

module.exports = mergeConfig(defaultConfig, config);
