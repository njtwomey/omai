import { cartesian, fieldArrows, lines, vectorField } from 'mdml-render/charts'
import { figure } from 'mdml-render/figure'
import { Chart, Choice, ControlPanel, FigureFrame, NumberField, useColourMode } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Phase portrait',
  question: 'How do I draw the flow of a two-dimensional system with some of its trajectories?',
  explain:
    'The direction field from `fieldArrows(f, { x, y })` drawn by `vectorField`, with trajectories traced from a ring of starting points (Runge-Kutta steps) as `lines`. The damped pendulum spirals into its rest points; with no damping its orbits close; the Van der Pol oscillator settles on a limit cycle from inside and out.',
}

const FIGURE = figure({ width: 600, height: 560 })
type System = 'pendulum' | 'van der Pol'

function trace(
  f: (x: number, y: number) => readonly [number, number],
  x0: number,
  y0: number,
): { x: number[]; y: number[] } {
  const h = 0.02
  const xs = [x0]
  const ys = [y0]
  let x = x0
  let y = y0
  for (let i = 0; i < 1500; i++) {
    const [a1, b1] = f(x, y)
    const [a2, b2] = f(x + (h / 2) * a1, y + (h / 2) * b1)
    const [a3, b3] = f(x + (h / 2) * a2, y + (h / 2) * b2)
    const [a4, b4] = f(x + h * a3, y + h * b3)
    x += (h / 6) * (a1 + 2 * a2 + 2 * a3 + a4)
    y += (h / 6) * (b1 + 2 * b2 + 2 * b3 + b4)
    if (!(Math.abs(x) < 10 && Math.abs(y) < 10)) break
    xs.push(x)
    ys.push(y)
  }
  return { x: xs, y: ys }
}

export default function PhasePortrait(): ReactNode {
  const mode = useColourMode()
  const [system, setSystem] = useState<System>('pendulum')
  const [parameter, setParameter] = useState(0.3)
  const option = useMemo(() => {
    const f =
      system === 'pendulum'
        ? (x: number, y: number): readonly [number, number] => [y, -Math.sin(x) - parameter * y]
        : (x: number, y: number): readonly [number, number] => [y, parameter * 4 * (1 - x * x) * y - x]
    const range = { x: [-4, 4] as [number, number], y: [-3.5, 3.5] as [number, number] }
    const starts = Array.from({ length: 10 }, (_, k): [number, number] => [
      3 * Math.cos((2 * Math.PI * k) / 10),
      2.8 * Math.sin((2 * Math.PI * k) / 10),
    ])
    return cartesian(
      [
        vectorField(fieldArrows(f, { ...range, n: 21 }), { mode, colour: 'muted' }),
        lines(
          starts.map(([x0, y0], k) => ({ ...trace(f, x0, y0), name: `__orbit${k}`, width: 1.5 })),
          { mode },
        ),
      ],
      {
        mode,
        aspect: 'equal',
        x: { label: system === 'pendulum' ? 'angle' : 'x', limits: range.x },
        y: { label: system === 'pendulum' ? 'angular velocity' : 'dx/dt', limits: range.y },
      },
    )
  }, [system, parameter, mode])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <Choice label="system" value={system} onChange={setSystem} options={['pendulum', 'van der Pol'] as System[]} />
        <NumberField
          label={system === 'pendulum' ? 'damping' : 'nonlinearity'}
          value={parameter}
          onChange={setParameter}
          min={0}
          max={1.5}
          step={0.05}
        />
      </ControlPanel>
      <FigureFrame figure={FIGURE} filename="phase-portrait">
        <Chart option={option} figure={FIGURE} />
      </FigureFrame>
    </div>
  )
}
