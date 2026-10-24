import { type ReactNode, useMemo, useState } from 'react'
import { normal as normalDist } from 'whitebox-engine/dists'
import { key, normal } from 'whitebox-engine/random'
import { cartesian, density, histogram, kde, rug } from 'whitebox-render/charts'
import { Chart, ControlPanel, FigureFrame, NumberField, Select, useColourMode } from 'whitebox-render/react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Histogram, KDE and density',
  question: 'How do I compare a sample with a density?',
  explain:
    '`histogram(values, { bins, density: true })` bins with the engine’s `stats.histogram` (counts, densities, rules such as `fd` and `sturges`, or a count); `kde` draws a Gaussian kernel density estimate, `density(dist)` any engine `Distribution`, and `rug` marks each observation. Larger samples settle onto the density.',
}

const pool = Array.from(normal(key(11), [5000]).data)
const rules = ['auto', 'fd', 'sturges', 'scott', '10', '40'] as const

export default function HistogramDensity(): ReactNode {
  const mode = useColourMode()
  const [n, setN] = useState(200)
  const [rule, setRule] = useState<(typeof rules)[number]>('fd')
  const option = useMemo(() => {
    const sample = pool.slice(0, n)
    const bins = /^\d+$/.test(rule) ? Number(rule) : (rule as 'auto' | 'fd' | 'sturges' | 'scott')
    return cartesian(
      [
        histogram(sample, { bins, density: true, mode, name: 'histogram' }),
        kde(sample, { mode, name: 'KDE', colourOffset: 1 }),
        density(normalDist(0, 1), { mode, name: 'N(0, 1)', colour: 'fg', dash: 'dashed' }),
        rug(sample.slice(0, 300), { mode }),
      ],
      { mode, x: { label: 'x', limits: [-4, 4] }, y: { label: 'density', limits: [0, 0.65] } },
    )
  }, [mode, n, rule])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <NumberField label="sample size" value={n} onChange={setN} min={10} max={5000} integer log />
        <Select label="bins" value={rule} options={rules} onChange={setRule} />
      </ControlPanel>
      <FigureFrame filename="histogram">
        <Chart option={option} />
      </FigureFrame>
    </div>
  )
}
