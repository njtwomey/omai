import { cartesian } from 'mdml-render/charts'
import { figure } from 'mdml-render/figure'
import { Chart, ControlPanel, FigureFrame, NumberField, useColourMode } from 'mdml-render/react'
import { optimisationPaths } from 'mdml-render/trace'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Optimisation paths',
  question: 'How do I draw an optimiser’s path over its objective?',
  explain:
    '`optimisationPaths(f, paths, { x, y })` samples the objective (log-scaled, filled and contoured) and draws each path with a hollow start and a filled end. Here gradient descent with and without momentum on Rosenbrock’s valley; `parameterPath(run, name)` reads such a path from a recorded run.',
}

const fig = figure({ width: 600, aspect: 1.35 })
const f = (x: number, y: number): number => (1 - x) ** 2 + 10 * (y - x * x) ** 2
const grad = (x: number, y: number): [number, number] => [-2 * (1 - x) - 40 * x * (y - x * x), 20 * (y - x * x)]

function descend(rate: number, momentum: number, steps: number): [number, number][] {
  let p: [number, number] = [-1.5, 1.8]
  let v: [number, number] = [0, 0]
  const out: [number, number][] = [p]
  for (let i = 0; i < steps; i++) {
    const [gx, gy] = grad(p[0], p[1])
    v = [momentum * v[0] - rate * gx, momentum * v[1] - rate * gy]
    p = [p[0] + v[0], p[1] + v[1]]
    if (!p.every(Number.isFinite) || Math.abs(p[0]) > 10 || Math.abs(p[1]) > 10) break
    out.push(p)
  }
  return out
}

export default function OptimisationPaths(): ReactNode {
  const mode = useColourMode()
  const [rate, setRate] = useState(0.01)
  const [momentum, setMomentum] = useState(0.8)
  const [steps, setSteps] = useState(150)
  const option = useMemo(
    () =>
      cartesian(
        optimisationPaths(
          f,
          [
            { name: 'gradient descent', points: descend(rate, 0, steps) },
            { name: `momentum ${momentum}`, points: descend(rate, momentum, steps) },
          ],
          { x: [-2, 2], y: [-1, 3], mode },
        ),
        { mode, figure: fig, x: { limits: [-2, 2] }, y: { limits: [-1, 3] } },
      ),
    [mode, rate, momentum, steps],
  )
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={3}>
        <NumberField label="learning rate" value={rate} onChange={setRate} min={0.0005} max={0.03} log />
        <NumberField label="momentum" value={momentum} onChange={setMomentum} min={0} max={0.98} />
        <NumberField label="steps" value={steps} onChange={setSteps} min={10} max={1000} integer />
      </ControlPanel>
      <FigureFrame figure={fig} filename="optimisation-paths">
        <Chart option={option} />
      </FigureFrame>
    </div>
  )
}
