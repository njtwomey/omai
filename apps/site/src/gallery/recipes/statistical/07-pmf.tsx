import { binomial, poisson } from 'omai-engine/dists'
import { cartesian, pmf } from 'omai-render/charts'
import { figure } from 'omai-render/figure'
import { Chart, Choice, ControlPanel, FigureFrame, NumberField, useColourMode } from 'omai-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Probability mass functions',
  question: 'How do I draw a discrete distribution as stems?',
  explain:
    "`pmf(dist, { range })` draws a discrete distribution's probabilities as stems at the integers. A binomial with many trials and a small chance looks like a Poisson with the same mean: set the trials high and watch the two agree.",
}

const FIGURE = figure({ width: 640, height: 380 })

export default function Pmf(): ReactNode {
  const mode = useColourMode()
  const [family, setFamily] = useState<'binomial' | 'poisson' | 'both'>('both')
  const [trials, setTrials] = useState(20)
  const [mean, setMean] = useState(4)
  const option = useMemo(() => {
    const top = Math.max(12, Math.ceil(mean + 4 * Math.sqrt(mean) + 2))
    const range: [number, number] = [0, top]
    const layers = []
    if (family !== 'poisson')
      layers.push(
        pmf(binomial(trials, Math.min(1, mean / trials)), {
          range,
          mode,
          name: `binomial(${trials}, ${(mean / trials).toFixed(3)})`,
        }),
      )
    if (family !== 'binomial')
      layers.push(pmf(poisson(mean), { range, mode, name: `Poisson(${mean})`, colourOffset: 1 }))
    return cartesian(layers, { mode, x: { label: 'k', limits: [-0.5, top + 0.5] }, y: { label: 'P(K = k)' } })
  }, [family, trials, mean, mode])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={3}>
        <Choice label="show" value={family} onChange={setFamily} options={['binomial', 'poisson', 'both'] as const} />
        <NumberField label="trials" value={trials} onChange={setTrials} min={Math.ceil(mean)} max={500} integer log />
        <NumberField label="mean" value={mean} onChange={setMean} min={0.5} max={20} step={0.5} />
      </ControlPanel>
      <FigureFrame figure={FIGURE} filename="pmf">
        <Chart option={option} figure={FIGURE} />
      </FigureFrame>
    </div>
  )
}
