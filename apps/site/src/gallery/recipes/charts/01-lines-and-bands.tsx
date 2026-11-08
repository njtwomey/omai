import { band, cartesian, lines, points } from 'omai-render/charts'
import { Chart, ControlPanel, FigureFrame, NumberField, Toggle, useColourMode } from 'omai-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Lines and bands',
  question: 'How do I draw curves with an uncertainty band and observations on top?',
  explain:
    'Every chart builder returns a layer; `cartesian(layers, options)` draws layers on one pair of axes in the figure. `band` fills between two curves as one polygon, `lines` draws curves (dashes, steps, areas), and `points` scatters observations.',
}

const x = Array.from({ length: 81 }, (_, i) => i / 8)
const observed = { x: [1, 2.5, 4, 5.5, 7, 8.5], y: [0.9, 0.5, -0.8, -0.6, 0.7, 0.8] }

export default function LinesAndBands(): ReactNode {
  const mode = useColourMode()
  const [width, setWidth] = useState(0.25)
  const [growth, setGrowth] = useState(0.03)
  const [showPoints, setShowPoints] = useState(true)
  const option = useMemo(() => {
    const spread = x.map((v) => width + growth * v)
    return cartesian(
      [
        band(
          [
            {
              x,
              lower: x.map((v, i) => Math.sin(v) - (spread[i] ?? 0)),
              upper: x.map((v, i) => Math.sin(v) + (spread[i] ?? 0)),
            },
          ],
          {
            mode,
          },
        ),
        lines(
          [
            { x, y: x.map(Math.sin), name: 'mean' },
            { x, y: x.map((v) => Math.cos(v) / 2), name: 'cos / 2', dash: 'dashed' },
          ],
          { mode },
        ),
        ...(showPoints ? [points([{ ...observed, name: 'observed', colour: 'tab3' }], { mode })] : []),
      ],
      { mode, x: { label: 'time' }, y: { label: 'value', limits: [-2.2, 2.2] }, tooltip: 'axis' },
    )
  }, [mode, width, growth, showPoints])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={3}>
        <NumberField label="band half-width" value={width} onChange={setWidth} min={0} max={1} />
        <NumberField label="growth per unit time" value={growth} onChange={setGrowth} min={0} max={0.15} />
        <Toggle label="observations" value={showPoints} onChange={setShowPoints} />
      </ControlPanel>
      <FigureFrame filename="lines-and-bands">
        <Chart option={option} />
      </FigureFrame>
    </div>
  )
}
