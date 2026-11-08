import { cartesian, fn, riemann } from 'omai-render/charts'
import { figure } from 'omai-render/figure'
import { Chart, Choice, ControlPanel, FigureFrame, NumberField, useColourMode } from 'omai-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Riemann sums',
  question: 'How do I show an integral as the limit of rectangles?',
  explain:
    '`riemann(f, [a, b], n, { rule })` draws the left, right, midpoint or trapezoid approximation as shaded pieces and returns its sum. The exact integral of sin(x) + 1.2 over [0, 2π] is 2.4π ≈ 7.54; the title shows the error, which falls like 1/n for left and right sums and 1/n² for midpoint and trapezoid.',
}

const FIGURE = figure({ width: 760, height: 400 })
type Rule = 'left' | 'right' | 'midpoint' | 'trapezoid'
const f = (x: number): number => Math.sin(x) + 1.2 + 0.3 * Math.sin(3 * x)
const EXACT = 2 * Math.PI * 1.2

export default function RiemannSums(): ReactNode {
  const mode = useColourMode()
  const [n, setN] = useState(8)
  const [rule, setRule] = useState<Rule>('left')
  const option = useMemo(() => {
    const r = riemann(f, [0, 2 * Math.PI], n, { rule, mode })
    return cartesian([r, fn(f, [0, 2 * Math.PI], { mode, name: 'f' })], {
      mode,
      title: `${rule} sum with ${n} pieces: ${r.sum.toFixed(4)} (error ${(r.sum - EXACT).toExponential(2)})`,
      x: { label: 'x', limits: [0, 2 * Math.PI] },
      y: { label: 'f(x)', limits: [0, 2.6] },
    })
  }, [n, rule, mode])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <NumberField label="pieces" value={n} onChange={setN} min={1} max={200} integer log />
        <Choice
          label="rule"
          value={rule}
          onChange={setRule}
          options={['left', 'right', 'midpoint', 'trapezoid'] as Rule[]}
        />
      </ControlPanel>
      <FigureFrame figure={FIGURE} filename="riemann-sums">
        <Chart option={option} figure={FIGURE} />
      </FigureFrame>
    </div>
  )
}
