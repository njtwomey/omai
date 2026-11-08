import { cartesian, points } from 'omai-render/charts'
import { figure } from 'omai-render/figure'
import { Chart, ControlPanel, FigureFrame, NumberField, useColourMode } from 'omai-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Bifurcation diagram',
  question: "How do I show how a system's long-run behaviour changes with a parameter?",
  explain:
    'For each value of the parameter, iterate the map past its transient and plot the values it keeps visiting, as tiny `points`. The logistic map x → r x (1 − x) settles to one value, then splits into two, four, eight (period doubling) and tips into chaos near r ≈ 3.57, with windows of order inside it. Narrow the range to zoom in on a window.',
}

const FIGURE = figure({ width: 860, height: 460 })

export default function BifurcationDiagram(): ReactNode {
  const mode = useColourMode()
  const [low, setLow] = useState(2.8)
  const [high, setHigh] = useState(4)
  const [columns, setColumns] = useState(600)
  const option = useMemo(() => {
    const rs: number[] = []
    const xs: number[] = []
    const a = Math.min(low, high - 0.001)
    for (let c = 0; c < columns; c++) {
      const r = a + ((high - a) * c) / (columns - 1)
      let x = 0.5
      for (let i = 0; i < 400; i++) x = r * x * (1 - x)
      for (let i = 0; i < 120; i++) {
        x = r * x * (1 - x)
        rs.push(r)
        xs.push(x)
      }
    }
    return cartesian([points([{ x: rs, y: xs, name: 'visited', size: 1, opacity: 0.35 }], { mode })], {
      mode,
      x: { label: 'r', limits: [a, high] },
      y: { label: 'x', limits: [0, 1] },
    })
  }, [low, high, columns, mode])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={3}>
        <NumberField label="r from" value={low} onChange={setLow} min={1} max={3.99} step={0.01} />
        <NumberField label="r to" value={high} onChange={setHigh} min={1.01} max={4} step={0.01} />
        <NumberField label="columns" value={columns} onChange={setColumns} min={50} max={2000} integer log />
      </ControlPanel>
      <FigureFrame figure={FIGURE} filename="bifurcation-diagram">
        <Chart option={option} figure={FIGURE} />
      </FigureFrame>
    </div>
  )
}
