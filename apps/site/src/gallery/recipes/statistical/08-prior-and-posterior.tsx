import { beta } from 'mdml-engine/dists'
import { cartesian, priorPosterior } from 'mdml-render/charts'
import { figure } from 'mdml-render/figure'
import { Chart, ControlPanel, FigureFrame, NumberField, useColourMode } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Prior and posterior',
  question: 'How do I draw a prior beside the posterior the data turn it into?',
  explain:
    '`priorPosterior(prior, posterior)` draws both densities on one axis, the prior dashed. For a coin with a Beta(a, b) prior, h heads and t tails give a Beta(a + h, b + t) posterior: few flips leave the prior visible, many flips wash it out.',
}

const FIGURE = figure({ width: 640, height: 380 })

export default function PriorPosterior(): ReactNode {
  const mode = useColourMode()
  const [a, setA] = useState(2)
  const [b, setB] = useState(2)
  const [heads, setHeads] = useState(7)
  const [tails, setTails] = useState(3)
  const option = useMemo(
    () =>
      cartesian([priorPosterior(beta(a, b), beta(a + heads, b + tails), { mode, range: [0, 1], n: 400 })], {
        mode,
        x: { label: 'chance of heads', limits: [0, 1] },
        y: { label: 'density' },
      }),
    [a, b, heads, tails, mode],
  )
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={4}>
        <NumberField label="prior a" value={a} onChange={setA} min={0.5} max={50} step={0.5} log />
        <NumberField label="prior b" value={b} onChange={setB} min={0.5} max={50} step={0.5} log />
        <NumberField label="heads" value={heads} onChange={setHeads} min={0} max={1000} integer log />
        <NumberField label="tails" value={tails} onChange={setTails} min={0} max={1000} integer log />
      </ControlPanel>
      <FigureFrame figure={FIGURE} filename="prior-and-posterior">
        <Chart option={option} figure={FIGURE} />
      </FigureFrame>
    </div>
  )
}
