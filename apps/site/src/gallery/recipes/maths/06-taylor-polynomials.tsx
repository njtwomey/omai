import { cartesian, fn, taylor } from 'omai-render/charts'
import { figure } from 'omai-render/figure'
import { Chart, Choice, ControlPanel, FigureFrame, NumberField, useColourMode } from 'omai-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Taylor polynomials',
  question: 'How do I show a function approximated by its Taylor polynomials?',
  explain:
    '`taylor(derivatives, x0, range)` draws the polynomial whose derivatives at x0 are the ones given. For sin and exp they repeat or stay put, so any order is easy; raise the order and the polynomial hugs the function over a wider interval. For log(1 + x) the series stops converging beyond |x| = 1 however many terms you take.',
}

const FIGURE = figure({ width: 760, height: 420 })
type Name = 'sin' | 'exp' | 'log(1 + x)'
const FUNCS: Record<Name, { f: (x: number) => number; d: (k: number, x0: number) => number; y: [number, number] }> = {
  sin: {
    f: Math.sin,
    d: (k, x0) => [Math.sin, Math.cos, (x: number) => -Math.sin(x), (x: number) => -Math.cos(x)][k % 4]?.(x0) ?? 0,
    y: [-2, 2],
  },
  exp: { f: Math.exp, d: (_k, x0) => Math.exp(x0), y: [-1, 8] },
  'log(1 + x)': {
    f: (x) => (x > -1 ? Math.log(1 + x) : Number.NaN),
    d: (k, x0) => (k === 0 ? Math.log(1 + x0) : ((k % 2 === 1 ? 1 : -1) * factorial(k - 1)) / (1 + x0) ** k),
    y: [-3, 2],
  },
}
function factorial(k: number): number {
  let p = 1
  for (let i = 2; i <= k; i++) p *= i
  return p
}

export default function TaylorPolynomials(): ReactNode {
  const mode = useColourMode()
  const [name, setName] = useState<Name>('sin')
  const [order, setOrder] = useState(5)
  const [x0, setX0] = useState(0)
  const option = useMemo(() => {
    const g = FUNCS[name]
    const ds = Array.from({ length: order + 1 }, (_, k) => g.d(k, x0))
    return cartesian(
      [
        fn(g.f, [-6, 6], { mode, name, n: 600 }),
        taylor(ds, x0, [-6, 6], { mode, n: 600, name: `order ${order} at ${x0}`, colourOffset: 1 }),
      ],
      { mode, x: { label: 'x', limits: [-6, 6] }, y: { label: 'y', limits: g.y } },
    )
  }, [name, order, x0, mode])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={3}>
        <Choice label="function" value={name} onChange={setName} options={Object.keys(FUNCS) as Name[]} />
        <NumberField label="order" value={order} onChange={setOrder} min={0} max={30} integer />
        <NumberField label="centre x0" value={x0} onChange={setX0} min={-0.5} max={3} step={0.25} />
      </ControlPanel>
      <FigureFrame figure={FIGURE} filename="taylor-polynomials">
        <Chart option={option} figure={FIGURE} />
      </FigureFrame>
    </div>
  )
}
