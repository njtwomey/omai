import { normal } from 'omai-engine/dists'
import { key } from 'omai-engine/random'
import { toArray } from 'omai-engine/tensor'
import { cartesian, ecdfCurve, lines } from 'omai-render/charts'
import { figure } from 'omai-render/figure'
import { Chart, ControlPanel, FigureFrame, NumberField, SeedInput, useColourMode } from 'omai-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Empirical CDF',
  question: 'How do I draw the empirical distribution function of a sample against the true one?',
  explain:
    '`ecdfCurve(samples)` is the step function that rises by 1/n at each observation. Drawn with the true CDF as a line, it closes in on it as the sample grows (the Glivenko–Cantelli theorem).',
}

// The fixed figure the chart fills (the builders' default size).
const FIGURE = figure()

export default function Ecdf(): ReactNode {
  const mode = useColourMode()
  const [n, setN] = useState(30)
  const [seed, setSeed] = useState(0)
  const option = useMemo(() => {
    const dist = normal(0, 1)
    const sample = toArray(dist.sample(key(seed), [n])) as number[]
    const x = Array.from({ length: 200 }, (_, i) => -4 + (8 * i) / 199)
    const cdf = toArray(dist.cdf(x)) as number[]
    return cartesian(
      [
        ecdfCurve(sample, { mode, name: `ECDF (n = ${n})` }),
        lines([{ x, y: cdf, name: 'true CDF', dash: 'dashed' }], { mode, colourOffset: 1 }),
      ],
      {
        mode,
        x: { label: 'x', limits: [-4, 4] },
        y: { label: 'F(x)', limits: [0, 1] },
      },
    )
  }, [n, seed, mode])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <NumberField label="sample size" value={n} onChange={setN} min={5} max={5000} integer log />
        <SeedInput value={seed} onChange={setSeed} />
      </ControlPanel>
      <FigureFrame figure={FIGURE} filename="ecdf">
        <Chart option={option} />
      </FigureFrame>
    </div>
  )
}
