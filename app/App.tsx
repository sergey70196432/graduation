import React, { useCallback, useState } from 'react';
import { StatusBar } from 'react-native';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { CameraDetectionScreen } from './src/screens/CameraDetectionScreen';
import { VideoDetectionScreen } from './src/screens/VideoDetectionScreen';

export default function App() {
  const [mode, setMode] = useState<'camera' | 'video'>('camera');

  const onOpenVideo = useCallback(() => setMode('video'), []);
  const onBackToCamera = useCallback(() => setMode('camera'), []);

  return (
    <SafeAreaProvider>
      <StatusBar barStyle="light-content" />
      {mode === 'camera' ? (
        <CameraDetectionScreen onOpenVideoTest={onOpenVideo} />
      ) : (
        <VideoDetectionScreen onBack={onBackToCamera} />
      )}
    </SafeAreaProvider>
  );
}
