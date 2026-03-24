import React from 'react';
import { StyleSheet, View } from 'react-native';
import Svg, { Rect, Text } from 'react-native-svg';
import type { Detection, FrameSize } from '../types/detection';

type Props = {
  detections: Detection[];
  // Размер кадра камеры, в координатах которого лежат bbox.
  frameSize: FrameSize | null;
  // Реальный размер вьюхи, в которую вписана камера (после layout).
  viewSize: { width: number; height: number };
};

function mapFrameToView(
  frame: FrameSize,
  view: { width: number; height: number }
) {
  // Камера у нас в режиме "contain", значит вписываем кадр целиком.
  const scale = Math.min(view.width / frame.width, view.height / frame.height);
  const scaledW = frame.width * scale;
  const scaledH = frame.height * scale;
  const offsetX = (view.width - scaledW) / 2;
  const offsetY = (view.height - scaledH) / 2;

  return { scale, offsetX, offsetY };
}

export function DetectionOverlay({ detections, frameSize, viewSize }: Props) {
  if (!frameSize) return null;
  if (viewSize.width <= 0 || viewSize.height <= 0) return null;

  const { scale, offsetX, offsetY } = mapFrameToView(frameSize, viewSize);

  return (
    <View pointerEvents="none" style={StyleSheet.absoluteFill}>
      <Svg width={viewSize.width} height={viewSize.height} style={styles.svg}>
        {detections.map((d, idx) => {
          const x = offsetX + d.bbox.x * scale;
          const y = offsetY + d.bbox.y * scale;
          const w = d.bbox.width * scale;
          const h = d.bbox.height * scale;

          const label = `${d.label} ${(d.confidence * 100).toFixed(0)}%`;
          const textY = Math.max(14, y - 6);

          return (
            <React.Fragment key={`${d.classId}-${idx}-${d.bbox.x.toFixed(0)}-${d.bbox.y.toFixed(0)}`}>
              <Rect
                x={x}
                y={y}
                width={w}
                height={h}
                stroke="rgba(0, 255, 170, 0.95)"
                strokeWidth={2}
                fill="rgba(0, 255, 170, 0.08)"
              />
              <Text
                x={x + 4}
                y={textY}
                fill="rgba(0, 255, 170, 0.95)"
                fontSize="12"
                fontWeight="700"
              >
                {label}
              </Text>
            </React.Fragment>
          );
        })}
      </Svg>
    </View>
  );
}

const styles = StyleSheet.create({
  svg: {
    position: 'absolute',
    left: 0,
    top: 0,
  },
});

