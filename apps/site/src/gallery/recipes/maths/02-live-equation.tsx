import { cartesian, fn, points } from 'omai-render/charts'
import { figure } from 'omai-render/figure'
import { live, tex } from 'omai-render/maths'
import { Chart, ControlPanel, Equation, FigureFrame, NumberField, useColourMode } from 'omai-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Live equations',
  question: 'How do I bind an equation to the parameters a figure is drawn with?',
  explain:
    'The `tex` tag writes TeX with interpolated slots for values; `live(value, { digits, strong })` formats and highlights a value. `Equation` re-typesets as the controls move, so the equation and the curve change together.',
}

const fig = figure({ width: 560, aspect: 1.8 })

export default function LiveEquation(): ReactNode {
  const mode = useColourMode()
  const [a, setA] = useState(0.5)
  const [b, setB] = useState(-1)
  const [c, setC] = useState(0.25)
  const option = useMemo(
    () =>
      cartesian(
        [
          fn((x) => a * x * x + b * x + c, [-4, 4], { mode, name: 'f' }),
          ...(a !== 0
            ? [points([{ x: [-b / (2 * a)], y: [c - (b * b) / (4 * a)], name: 'vertex', colour: 'tab3' }], { mode })]
            : []),
        ],
        { mode, figure: fig, y: { limits: [-6, 10] } },
      ),
    [a, b, c, mode],
  )
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={3}>
        <NumberField label="a" value={a} onChange={setA} min={-2} max={2} />
        <NumberField label="b" value={b} onChange={setB} min={-4} max={4} />
        <NumberField label="c" value={c} onChange={setC} min={-4} max={4} />
      </ControlPanel>
      <div className="text-lg">
        <Equation>
          {tex`f(x) = ${live(a, { digits: 3 })}\,x^2 + (${live(b, { digits: 3 })})\,x + ${live(c, { digits: 3 })}, \qquad x^\star = -\frac{b}{2a} = ${live(a === 0 ? Number.NaN : -b / (2 * a), { digits: 3, strong: true })}`}
        </Equation>
      </div>
      <FigureFrame figure={fig} filename="live-equation">
        <Chart option={option} />
      </FigureFrame>
    </div>
  )
}
