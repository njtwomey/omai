import { cartesian, heatmap } from 'mdml-render/charts'
import { figure } from 'mdml-render/figure'
import { Chart, Choice, ControlPanel, FigureFrame, NumberField, useColourMode } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Sequential and diverging ramps',
  question: 'When do I use a diverging colour ramp, and how do I centre it?',
  explain:
    "A sequential ramp (viridis) suits magnitudes; a diverging ramp (red to blue) suits signed values about a meaningful middle, set with `centre`. Move the centre off zero and the same field reads differently: colour should put its midpoint where the data's is.",
}

const FIGURE = figure({ width: 640, height: 520 })

export default function DivergingRamps(): ReactNode {
  const mode = useColourMode()
  const [ramp, setRamp] = useState<'viridis' | 'rdbu' | 'coolwarm'>('rdbu')
  const [centre, setCentre] = useState(0)
  const option = useMemo(() => {
    const n = 60
    const g = Array.from({ length: n }, (_, i) => -3 + (6 * i) / (n - 1))
    const z = g.map((y) => g.map((x) => Math.sin(x) * Math.cos(y) + 0.2 * x))
    return cartesian(
      [heatmap(z, { x: g, y: g, mode, ramp, ...(ramp === 'viridis' ? {} : { centre }), label: 'value' })],
      { mode, aspect: 'equal', x: { label: 'x' }, y: { label: 'y' } },
    )
  }, [ramp, centre, mode])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <Choice label="ramp" value={ramp} onChange={setRamp} options={['viridis', 'rdbu', 'coolwarm'] as const} />
        <NumberField label="centre" value={centre} onChange={setCentre} min={-1} max={1} step={0.05} />
      </ControlPanel>
      <FigureFrame figure={FIGURE} filename="diverging-ramps">
        <Chart option={option} figure={FIGURE} />
      </FigureFrame>
    </div>
  )
}
