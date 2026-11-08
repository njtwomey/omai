import { normal } from 'omai-engine/dists'
import { key } from 'omai-engine/random'
import { toArray } from 'omai-engine/tensor'
import { residualChart } from 'omai-render/charts'
import { figure } from 'omai-render/figure'
import { Chart, ControlPanel, FigureFrame, NumberField, SeedInput, useColourMode } from 'omai-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Residual plot',
  question: 'How do I check a fit by its residuals?',
  explain:
    '`residualChart(fitted, residuals)` scatters residuals against fitted values about zero. A straight line fitted to data with some curvature leaves a smile in the residuals; noise that grows with x leaves a funnel. With neither, the residuals are a shapeless band.',
}

const FIGURE = figure({ width: 640, height: 400 })

export default function ResidualPlot(): ReactNode {
  const mode = useColourMode()
  const [curvature, setCurvature] = useState(0.6)
  const [funnel, setFunnel] = useState(0)
  const [seed, setSeed] = useState(0)
  const option = useMemo(() => {
    const n = 200
    const x = Array.from({ length: n }, (_, i) => -2 + (4 * i) / (n - 1))
    const z = toArray(normal(0, 1).sample(key(seed), [n])) as number[]
    const y = x.map((v, i) => 1 + 0.8 * v + curvature * v * v + (0.3 + funnel * (v + 2)) * (z[i] ?? 0))
    // The least-squares line y = a + b x.
    const mx = x.reduce((a, b) => a + b, 0) / n
    const my = y.reduce((a, b) => a + b, 0) / n
    const b = x.reduce((s, v, i) => s + (v - mx) * ((y[i] ?? 0) - my), 0) / x.reduce((s, v) => s + (v - mx) ** 2, 0)
    const a = my - b * mx
    const fitted = x.map((v) => a + b * v)
    return residualChart(
      fitted,
      y.map((v, i) => v - (fitted[i] ?? 0)),
      { mode },
    )
  }, [curvature, funnel, seed, mode])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={3}>
        <NumberField label="curvature" value={curvature} onChange={setCurvature} min={-1} max={1} step={0.05} />
        <NumberField label="noise growth" value={funnel} onChange={setFunnel} min={0} max={1} step={0.05} />
        <SeedInput value={seed} onChange={setSeed} />
      </ControlPanel>
      <FigureFrame figure={FIGURE} filename="residual-plot">
        <Chart option={option} figure={FIGURE} />
      </FigureFrame>
    </div>
  )
}
