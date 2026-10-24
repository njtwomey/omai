import { type ReactNode, useMemo, useState } from 'react'
import { cartesian, fieldArrows, streamlines, vectorField } from 'whitebox-render/charts'
import { figure } from 'whitebox-render/figure'
import { Chart, ControlPanel, FigureFrame, NumberField, Toggle, useColourMode } from 'whitebox-render/react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Vector fields',
  question: 'How do I draw a flow: arrows on a grid and streamlines through it?',
  explain:
    "`fieldArrows(field, { x, y, n })` samples a field as arrows of equal length (or `length: 'magnitude'`), coloured by magnitude in `vectorField`. `streamlines(paths)` draws trajectories, such as those from the engine’s `dynamics.streamline`, with an arrow head showing the direction.",
}

const fig = figure({ width: 480, aspect: 1.15 })
// A damped rotation: dx = -d x - y, dy = x - d y.
type Field = (x: number, y: number) => [number, number]
const rotation =
  (d: number): Field =>
  (x, y) => [-d * x - y, x - d * y]
function trajectory(field: Field, x0: number, y0: number): [number, number][] {
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
  const [damping, setDamping] = useState(0.2)
  const [byMagnitude, setByMagnitude] = useState(false)
  const option = useMemo(() => {
    const field = rotation(damping)
    return cartesian(
      [
        vectorField(fieldArrows(field, { x: [-2, 2], y: [-2, 2], n: 15, length: byMagnitude ? 'magnitude' : 'unit' }), {
          mode,
        }),
        streamlines([trajectory(field, 1.8, 0), trajectory(field, -1.8, 0.5), trajectory(field, 0, -1.9)], {
          mode,
          colour: 'fg',
        }),
      ],
      { mode, figure: fig, aspect: 'equal', x: { limits: [-2.2, 2.2] }, y: { limits: [-2.2, 2.2] } },
    )
  }, [mode, damping, byMagnitude])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <NumberField label="damping" value={damping} onChange={setDamping} min={-0.3} max={1.5} />
        <Toggle label="length by magnitude" value={byMagnitude} onChange={setByMagnitude} />
      </ControlPanel>
      <FigureFrame figure={fig} filename="vector-field">
        <Chart option={option} />
      </FigureFrame>
    </div>
  )
}
