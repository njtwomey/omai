import { key, normal } from 'omai-engine/random'
import type { Run, TraceEvent } from 'omai-engine/trace'
import { Chart, ControlPanel, FigureFrame, NumberField, Toggle, useColourMode } from 'omai-render/react'
import { metricChart } from 'omai-render/trace'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Training curves from a run',
  question: 'How do I plot the metrics an algorithm recorded, smoothed?',
  explain:
    'Iterative algorithms record a `Run` of trace events. `metricChart(run, names, { smoothing, log })` draws each metric against its step; smoothing is a debiased exponential moving average (as TensorBoard’s) with the raw values left faint underneath.',
}

// A recorded run: a noisy, decaying training loss and a validation loss, as metric events.
const noise = Array.from(normal(key(2), [600]).data)
const events: TraceEvent[] = Array.from({ length: 300 }, (_, t): TraceEvent[] => [
  {
    type: 'metric',
    runId: 'r',
    step: t,
    name: 'train loss',
    value: 2 * Math.exp(-t / 60) + 0.1 + 0.08 * (noise[t] ?? 0),
  },
  {
    type: 'metric',
    runId: 'r',
    step: t,
    name: 'valid loss',
    value: 2 * Math.exp(-t / 80) + 0.25 + 0.05 * (noise[300 + t] ?? 0),
  },
]).flat()
const run: Run = { schema: 'omai.run/1', events }

export default function TrainingCurves(): ReactNode {
  const mode = useColourMode()
  const [smoothing, setSmoothing] = useState(0.8)
  const [log, setLog] = useState(false)
  const option = useMemo(() => metricChart(run, undefined, { mode, smoothing, log }), [mode, smoothing, log])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <NumberField label="smoothing" value={smoothing} onChange={setSmoothing} min={0} max={0.99} />
        <Toggle label="log scale" value={log} onChange={setLog} />
      </ControlPanel>
      <FigureFrame filename="training-curves">
        <Chart option={option} />
      </FigureFrame>
    </div>
  )
}
