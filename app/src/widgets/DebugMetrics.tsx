import { ScrollView, StyleSheet, Text, View } from "react-native";
import { DetectorStats, SpeedModelState } from "../types/detection";

type Props = {
  stats: DetectorStats;
  isSpeedModelLoaded: boolean;
  speedModelState: SpeedModelState;
};

function Metric(props: { label: string; value: string }) {
  return (
    <View style={styles.metric}>
      <Text style={styles.metricLabel}>{props.label}</Text>
      <Text style={styles.metricValue}>{props.value}</Text>
    </View>
  );
}

export default function DebugMetrics({ stats, isSpeedModelLoaded, speedModelState }: Props) {
  return (
    <ScrollView
      style={styles.metrics}
    >
      <Metric label="FPS" value={stats.fps.toFixed(1)} />
      <Metric
        label="Total, ms"
        value={stats.lastTotalMs.toFixed(1)}
      />
      <Metric
        label="Inference, ms"
        value={stats.lastInferenceMs.toFixed(1)}
      />

      <Metric
        label="Resize, ms"
        value={stats.resizeMs.toFixed(1)}
      />
      <Metric
        label="Letterbox, ms"
        value={stats.letterboxMs.toFixed(1)}
      />
      <Metric
        label="Decode, ms"
        value={stats.decodeMs.toFixed(1)}
      />
      
      <Metric
        label="SpeedCls, ms"
        value={
          stats.lastSpeedClsRan
            ? stats.lastSpeedClsMs.toFixed(1)
            : '—'
        }
      />
      <Metric
        label="SpeedModel"
        value={isSpeedModelLoaded ? 'loaded' : speedModelState}
      />
      <Metric label="Objects" value={String(stats.lastNumDetections)} />
      <Metric label="Input" value={`${stats.inputSize}x${stats.inputSize}`} />
    </ScrollView>
  )
}

const styles = StyleSheet.create({
  metrics: {
    marginTop: 10,
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
  },
  metric: {
    minWidth: 100,
    paddingVertical: 8,
    paddingHorizontal: 10,
    borderRadius: 10,
    backgroundColor: 'rgba(255,255,255,0.06)',
  },
  metricLabel: {
    color: 'rgba(255,255,255,0.65)',
    fontSize: 12,
  },
  metricValue: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '700',
    marginTop: 2,
  },
})
