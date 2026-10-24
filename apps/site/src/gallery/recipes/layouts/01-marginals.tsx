import { type ReactNode, useMemo } from 'react'
import { histogram, points, subplots } from 'whitebox-render/charts'
import { figure, withMarginals } from 'whitebox-render/figure'
import { Chart, FigureFrame, useColourMode } from 'whitebox-render/react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Marginal distributions',
  question: 'How do I put histograms on the edges of a scatter plot, touching it?',
  explain:
    '`withMarginals()` is a grid spec with a `main` cell and `top` and `right` cells that share its x and y axes with zero gaps. `subplots(grid, panels)` gives shared axes one range and hides inner tick labels; hover is linked across them.',
}

const fig = figure({ width: 520, aspect: 1.1 })
// A deterministic correlated cloud.
const n = 300
const u = Array.from({ length: n }, (_, i) => Math.sqrt(-2 * Math.log((i + 0.5) / n)) * Math.cos(i * 2.39996))
const v = Array.from({ length: n }, (_, i) => Math.sqrt(-2 * Math.log((i + 0.5) / n)) * Math.sin(i * 2.39996))
const x = u
const y = u.map((a, i) => 0.7 * a + 0.5 * (v[i] ?? 0))

export default function Marginals(): ReactNode {
  const mode = useColourMode()
  const option = useMemo(
    () =>
      subplots(
        withMarginals({ ratio: 0.22 }),
        {
          main: {
            layers: [points([{ x, y, size: 5, opacity: 0.6 }], { mode })],
            options: { x: { label: 'x' }, y: { label: 'y' } },
          },
          top: { layers: [histogram(x, { bins: 24, mode })], options: { y: { show: false } } },
          right: { layers: [histogram(y, { bins: 24, horizontal: true, mode })], options: { x: { show: false } } },
        },
        { mode, figure: fig },
      ),
    [mode],
  )
  return (
    <FigureFrame figure={fig} filename="marginals">
      <Chart option={option} />
    </FigureFrame>
  )
}
