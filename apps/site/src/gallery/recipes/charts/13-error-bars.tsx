import { key, normal } from 'mdml-engine/random'
import { toArray } from 'mdml-engine/tensor'
import { cartesian, errorBars, lines } from 'mdml-render/charts'
import { figure } from 'mdml-render/figure'
import { Chart, ControlPanel, FigureFrame, NumberField, SeedInput, useColourMode } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Error bars in x and y',
  question: 'How do I draw measurements with uncertainty in both coordinates?',
  explain:
    '`errorBars(points)` draws each point with a horizontal bar of ± its x error and a vertical bar of ± its y error, capped. Toy measurements of a spring: extension against load, both read off with error, beside the true line.',
}

const FIGURE = figure({ width: 680, height: 440 })

export default function ErrorBars(): ReactNode {
  const mode = useColourMode()
  const [xErr, setXErr] = useState(0.15)
  const [yErr, setYErr] = useState(0.4)
  const [seed, setSeed] = useState(0)
  const option = useMemo(() => {
    const z = toArray(normal(key(seed), [24])) as number[]
    const data = Array.from({ length: 12 }, (_, i) => {
      const load = 0.5 + i * 0.75
      return { x: load + xErr * 0.6 * (z[2 * i] ?? 0), y: 1.6 * load + yErr * 0.6 * (z[2 * i + 1] ?? 0), xErr, yErr }
    })
    return cartesian(
      [
        lines([{ x: [0, 9.5], y: [0, 15.2], name: 'true line', dash: 'dashed', colour: 'muted' }], { mode }),
        errorBars(data, { mode, name: 'measured' }),
      ],
      { mode, x: { label: 'load (N)', limits: [0, 9.5] }, y: { label: 'extension (cm)' } },
    )
  }, [xErr, yErr, seed, mode])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={3}>
        <NumberField label="x error" value={xErr} onChange={setXErr} min={0} max={1} step={0.05} />
        <NumberField label="y error" value={yErr} onChange={setYErr} min={0} max={2} step={0.05} />
        <SeedInput value={seed} onChange={setSeed} />
      </ControlPanel>
      <FigureFrame figure={FIGURE} filename="error-bars">
        <Chart option={option} figure={FIGURE} />
      </FigureFrame>
    </div>
  )
}
