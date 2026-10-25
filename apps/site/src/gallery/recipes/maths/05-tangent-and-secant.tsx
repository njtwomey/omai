import { cartesian, fn, points, secantLine, tangentLine } from 'mdml-render/charts'
import { figure } from 'mdml-render/figure'
import { Chart, ControlPanel, FigureFrame, NumberField, useColourMode } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Tangent and secant',
  question: 'How do I show a derivative as the limit of secant slopes?',
  explain:
    "`secantLine(f, a, a + h)` draws the chord through two points of a curve and `tangentLine(f, a, { derivative })` the tangent at one. Shrink h and the secant's slope, shown in the title, closes in on the derivative.",
}

const FIGURE = figure({ width: 760, height: 420 })
const f = (x: number): number => 0.25 * x ** 3 - x + 1
const df = (x: number): number => 0.75 * x * x - 1

export default function TangentAndSecant(): ReactNode {
  const mode = useColourMode()
  const [a, setA] = useState(1)
  const [h, setH] = useState(1.5)
  const option = useMemo(() => {
    const slope = (f(a + h) - f(a)) / h
    return cartesian(
      [
        fn(f, [-3, 3], { mode, name: 'f' }),
        secantLine(f, a, a + h, { mode, extend: 1.5, name: 'secant', colourOffset: 1 }),
        tangentLine(f, a, { mode, derivative: df(a), span: 3, name: 'tangent', colourOffset: 2 }),
        points([{ x: [a, a + h], y: [f(a), f(a + h)], name: 'points', size: 8 }], { mode, colourOffset: 1 }),
      ],
      {
        mode,
        title: `secant slope ${slope.toFixed(4)} · derivative ${df(a).toFixed(4)}`,
        x: { label: 'x', limits: [-3, 3] },
        y: { label: 'f(x)', limits: [-3, 5] },
      },
    )
  }, [a, h, mode])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <NumberField label="a" value={a} onChange={setA} min={-2.5} max={2.5} step={0.05} />
        <NumberField label="h" value={h} onChange={setH} min={0.001} max={2} step={0.001} log />
      </ControlPanel>
      <FigureFrame figure={FIGURE} filename="tangent-and-secant">
        <Chart option={option} figure={FIGURE} />
      </FigureFrame>
    </div>
  )
}
