import { key, normal } from 'mdml-engine/random'
import { histogram, points, subplots } from 'mdml-render/charts'
import { figure, withMarginals } from 'mdml-render/figure'
import { Chart, ControlPanel, FigureFrame, NumberField, useColourMode } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Marginal distributions',
  question: 'How do I put histograms on the edges of a scatter plot, touching it?',
  explain:
    '`withMarginals()` is a grid spec with a `main` cell and `top` and `right` cells that share its x and y axes with zero gaps. `subplots(grid, panels)` gives shared axes one range and hides inner tick labels; hover is linked across them.',
}

const fig = figure({ width: 520, aspect: 1.1 })
const n = 400
const z = Array.from(normal(key(3), [2 * n]).data)

export default function Marginals(): ReactNode {
  const mode = useColourMode()
  const [rho, setRho] = useState(0.7)
  const [bins, setBins] = useState(24)
  const option = useMemo(() => {
    const x = z.slice(0, n)
    const y = x.map((a, i) => rho * a + Math.sqrt(1 - rho * rho) * (z[n + i] ?? 0))
    return subplots(
      withMarginals({ ratio: 0.22 }),
      {
        main: {
          layers: [points([{ x, y, size: 5, opacity: 0.6, outline: false }], { mode })],
          options: { x: { label: 'x', limits: [-4, 4] }, y: { label: 'y', limits: [-4, 4] } },
        },
        top: {
          layers: [histogram(x, { bins, range: [-4, 4], mode })],
          options: { y: { show: false, splitLine: false } },
        },
        right: {
          layers: [histogram(y, { bins, range: [-4, 4], horizontal: true, mode })],
          options: { x: { show: false, splitLine: false } },
        },
      },
      { mode, figure: fig },
    )
  }, [mode, rho, bins])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <NumberField label="correlation ρ" value={rho} onChange={setRho} min={-0.99} max={0.99} />
        <NumberField label="bins" value={bins} onChange={setBins} min={4} max={60} integer />
      </ControlPanel>
      <FigureFrame figure={fig} filename="marginals">
        <Chart option={option} />
      </FigureFrame>
    </div>
  )
}
