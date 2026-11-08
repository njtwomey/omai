import { cartesian, parametric, points } from 'omai-render/charts'
import { figure } from 'omai-render/figure'
import { Chart, Choice, ControlPanel, FigureFrame, useColourMode } from 'omai-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Box aspect and data aspect',
  question: 'How do I keep a circle round in a wide figure?',
  explain:
    "The box aspect is the figure’s shape (here 2:1); the data aspect relates data units on the two axes. With `aspect: 'equal'` one unit is as long on x as on y, so the x range widens to fill the box and the unit circle stays round. `auto` stretches the data to the box.",
}

export default function BoxAndDataAspect(): ReactNode {
  const mode = useColourMode()
  const [box, setBox] = useState<'2:1' | '1:1' | '16:9'>('2:1')
  const [aspect, setAspect] = useState<'equal' | 'auto'>('equal')
  const fig = useMemo(() => figure({ mode, aspect: box }), [box, mode])
  const option = useMemo(
    () =>
      cartesian(
        [
          parametric(Math.cos, Math.sin, [0, 2 * Math.PI], { mode, name: 'unit circle' }),
          points([{ x: [0, 1, 0], y: [0, 0, 1], labels: ['(0, 0)', '(1, 0)', '(0, 1)'] }], { mode, colourOffset: 1 }),
        ],
        { mode, figure: fig, aspect, x: { limits: [-1.2, 1.2] }, y: { limits: [-1.2, 1.2] } },
      ),
    [fig, aspect, mode],
  )
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <Choice label="box aspect" value={box} options={['2:1', '1:1', '16:9'] as const} onChange={setBox} />
        <Choice label="data aspect" value={aspect} options={['equal', 'auto'] as const} onChange={setAspect} />
      </ControlPanel>
      <FigureFrame figure={fig}>
        <Chart option={option} />
      </FigureFrame>
    </div>
  )
}
