import { type ReactNode, useMemo, useState } from 'react'
import { lineChart } from 'whitebox-render/charts'
import { figure } from 'whitebox-render/figure'
import { Chart, ControlPanel, NumberField, useColourMode } from 'whitebox-render/react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Line chart',
  question: 'How do I draw a few named curves on one pair of axes?',
  explain:
    '`lineChart(curves, options)` returns an ECharts option and `Chart` draws it at the figure’s fixed size, in the page’s colour mode. Each curve is `{ x, y, name }` (plus `colour`, `dash`, `step`, `area`); curves take the tab10 colours in order, and a legend appears once there is more than one name.',
}

// The fixed figure the chart fills (the builders' default size).
const FIGURE = figure()

export default function LineChart(): ReactNode {
  const mode = useColourMode()
  const [frequency, setFrequency] = useState(1)
  const option = useMemo(() => {
    const x = Array.from({ length: 300 }, (_, i) => (i / 299) * 4 * Math.PI)
    return lineChart(
      [
        { x, y: x.map((t) => Math.sin(frequency * t)), name: 'sin' },
        { x, y: x.map((t) => Math.cos(frequency * t)), name: 'cos' },
        { x, y: x.map((t) => Math.exp(-t / 4) * Math.sin(frequency * t)), name: 'damped', dash: 'dashed' },
      ],
      { mode, x: { label: 't' }, y: { label: 'y', limits: [-1.2, 1.2] } },
    )
  }, [frequency, mode])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel>
        <NumberField label="frequency" value={frequency} onChange={setFrequency} min={0.25} max={4} log />
      </ControlPanel>
      <Chart option={option} figure={FIGURE} />
    </div>
  )
}
