import { cartesian, matrix } from 'omai-render/charts'
import { figure } from 'omai-render/figure'
import { Chart, ControlPanel, FigureFrame, NumberField, Toggle, useColourMode } from 'omai-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Annotated heatmap',
  question: 'How do I draw a table of numbers as coloured cells with their values written in?',
  explain:
    "`matrix(z, { rowLabels, colLabels, annotate })` is a layer of labelled cells coloured on a ramp, each with its value; give `centre` for a diverging ramp. Here: rainfall by month and city, toy numbers, with the ramp centred on each city's average when you ask for anomalies.",
}

const FIGURE = figure({ width: 760, height: 380 })
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
const CITIES = ['Galway', 'London', 'Madrid', 'Mumbai', 'Sydney']
// Toy monthly rainfall (mm): a base, a seasonal swing and its phase per city.
const SHAPE: readonly [number, number, number][] = [
  [110, 40, 0],
  [55, 10, 1],
  [35, 25, 0.5],
  [200, 300, 6.5],
  [100, 30, 3],
]

export default function AnnotatedHeatmap(): ReactNode {
  const mode = useColourMode()
  const [anomalies, setAnomalies] = useState(false)
  const [scale, setScale] = useState(1)
  const option = useMemo(() => {
    const z = SHAPE.map(([base, swing, phase]) =>
      MONTHS.map((_, m) => Math.max(0, Math.round(scale * (base + swing * Math.cos(((m - phase) * Math.PI) / 6))))),
    )
    const shown = anomalies ? z.map((row) => row.map((v) => v - row.reduce((a, b) => a + b, 0) / row.length)) : z
    return cartesian(
      [
        matrix(shown, {
          rowLabels: CITIES,
          colLabels: MONTHS,
          annotate: true,
          format: (v) => Math.round(v).toString(),
          mode,
          ...(anomalies ? { centre: 0 } : {}),
          label: anomalies ? 'mm from the city average' : 'mm',
        }),
      ],
      { mode },
    )
  }, [anomalies, scale, mode])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <Toggle label="anomalies (diverging)" value={anomalies} onChange={setAnomalies} />
        <NumberField label="wetter" value={scale} onChange={setScale} min={0.25} max={4} step={0.25} log />
      </ControlPanel>
      <FigureFrame figure={FIGURE} filename="annotated-heatmap">
        <Chart option={option} figure={FIGURE} />
      </FigureFrame>
    </div>
  )
}
