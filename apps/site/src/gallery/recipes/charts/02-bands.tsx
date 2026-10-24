import { type ReactNode, useMemo, useState } from 'react'
import { band, cartesian, lines } from 'whitebox-render/charts'
import { figure } from 'whitebox-render/figure'
import { Chart, ControlPanel, NumberField, useColourMode } from 'whitebox-render/react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Bands',
  question: 'How do I shade the region between two curves, such as an interval around a mean?',
  explain:
    '`band([{ x, lower, upper }])` is a layer of filled polygons between two curves, exact through crossings; `lines` is a layer of curves. `cartesian(layers, options)` draws any layers on one pair of axes, in order, so a band sits under its mean.',
}

// The fixed figure the chart fills (the builders' default size).
const FIGURE = figure()

export default function Bands(): ReactNode {
  const mode = useColourMode()
  const [k, setK] = useState(2)
  const option = useMemo(() => {
    const x = Array.from({ length: 200 }, (_, i) => (i / 199) * 10)
    const mean = x.map((t) => Math.sin(t) + 0.1 * t)
    const sd = x.map((t) => 0.15 + 0.05 * t)
    return cartesian(
      [
        band(
          [
            {
              x,
              lower: mean.map((m, i) => m - k * (sd[i] ?? 0)),
              upper: mean.map((m, i) => m + k * (sd[i] ?? 0)),
              name: `± ${k} sd`,
            },
          ],
          { mode },
        ),
        lines([{ x, y: mean, name: 'mean' }], { mode }),
      ],
      { mode, x: { label: 'x' }, y: { label: 'y', limits: [-4, 6] } },
    )
  }, [k, mode])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel>
        <NumberField label="half-width in standard deviations" value={k} onChange={setK} min={0} max={3} step={0.1} />
      </ControlPanel>
      <Chart option={option} figure={FIGURE} />
    </div>
  )
}
