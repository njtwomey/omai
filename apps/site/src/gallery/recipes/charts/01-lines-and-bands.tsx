import { type ReactNode, useMemo } from 'react'
import { band, cartesian, lines, points } from 'whitebox-render/charts'
import { Chart, FigureFrame, useColourMode } from 'whitebox-render/react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Lines and bands',
  question: 'How do I draw curves with an uncertainty band and observations on top?',
  explain:
    'Every chart builder returns a layer; `cartesian(layers, options)` draws layers on one pair of axes in the figure. `band` fills between two curves as one polygon, `lines` draws curves (dashes, steps, areas), and `points` scatters observations.',
}

const x = Array.from({ length: 81 }, (_, i) => i / 8)

export default function LinesAndBands(): ReactNode {
  const mode = useColourMode()
  const option = useMemo(
    () =>
      cartesian(
        [
          band(
            [
              {
                x,
                lower: x.map((v) => Math.sin(v) - 0.25 - 0.03 * v),
                upper: x.map((v) => Math.sin(v) + 0.25 + 0.03 * v),
              },
            ],
            { mode },
          ),
          lines(
            [
              { x, y: x.map(Math.sin), name: 'mean' },
              { x, y: x.map((v) => Math.cos(v) / 2), name: 'cos / 2', dash: 'dashed' },
            ],
            { mode },
          ),
          points(
            [{ x: [1, 2.5, 4, 5.5, 7, 8.5], y: [0.9, 0.5, -0.8, -0.6, 0.7, 0.8], name: 'observed', colour: 'tab3' }],
            { mode },
          ),
        ],
        { mode, x: { label: 'time' }, y: { label: 'value' }, tooltip: 'axis' },
      ),
    [mode],
  )
  return (
    <FigureFrame filename="lines-and-bands">
      <Chart option={option} />
    </FigureFrame>
  )
}
