import { type ReactNode, useMemo } from 'react'
import { cartesian, fieldArrows, streamlines, vectorField } from 'whitebox-render/charts'
import { figure } from 'whitebox-render/figure'
import { Chart, FigureFrame, useColourMode } from 'whitebox-render/react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Vector fields',
  question: 'How do I draw a flow: arrows on a grid and streamlines through it?',
  explain:
    "`fieldArrows(field, { x, y, n })` samples a field as arrows of equal length (or `length: 'magnitude'`), coloured by magnitude in `vectorField`. `streamlines(paths)` draws trajectories, such as those from the engine’s `dynamics.streamline`, with an arrow head showing the direction.",
}

const fig = figure({ width: 480, aspect: 1.15 })
// A damped rotation: dx = -0.2x - y, dy = x - 0.2y.
const field = (x: number, y: number): [number, number] => [-0.2 * x - y, x - 0.2 * y]
function trajectory(x0: number, y0: number): [number, number][] {
  const out: [number, number][] = [[x0, y0]]
  let [x, y] = [x0, y0]
  for (let i = 0; i < 300; i++) {
    const [u, v] = field(x, y)
    x += 0.02 * u
    y += 0.02 * v
    out.push([x, y])
  }
  return out
}

export default function VectorFieldRecipe(): ReactNode {
  const mode = useColourMode()
  const option = useMemo(
    () =>
      cartesian(
        [
          vectorField(fieldArrows(field, { x: [-2, 2], y: [-2, 2], n: 15 }), { mode }),
          streamlines([trajectory(1.8, 0), trajectory(-1.8, 0.5), trajectory(0, -1.9)], { mode, colour: 'fg' }),
        ],
        { mode, figure: fig, aspect: 'equal', x: { limits: [-2.2, 2.2] }, y: { limits: [-2.2, 2.2] } },
      ),
    [mode],
  )
  return (
    <FigureFrame figure={fig} filename="vector-field">
      <Chart option={option} />
    </FigureFrame>
  )
}
