import { type ReactNode, useMemo, useState } from 'react'
import { normal } from 'whitebox-engine/dists'
import { key } from 'whitebox-engine/random'
import { toArray } from 'whitebox-engine/tensor'
import { cartesian, density, histogram } from 'whitebox-render/charts'
import { figure } from 'whitebox-render/figure'
import { Chart, ControlPanel, NumberField, SeedInput, Toggle, useColourMode } from 'whitebox-render/react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Histogram',
  question: 'How do I draw a histogram, and compare it with the density it was drawn from?',
  explain:
    '`histogram(values, { bins })` bins with whitebox-engine’s `stats.histogram` (a count, or a rule such as `"fd"`), drawn as adjoining bars on a value axis; `density: true` scales it to integrate to one, so it can share axes with `density(dist)`.',
}

// The fixed figure the chart fills (the builders' default size).
const FIGURE = figure()

export default function Histogram(): ReactNode {
  const mode = useColourMode()
  const [n, setN] = useState(300)
  const [bins, setBins] = useState(20)
  const [seed, setSeed] = useState(0)
  const [overlay, setOverlay] = useState(true)
  const option = useMemo(() => {
    const dist = normal(0, 1)
    const values = toArray(dist.sample(key(seed), [n])) as number[]
    const layers = [histogram(values, { bins, density: true, mode, name: 'sample' })]
    if (overlay) layers.push(density(dist, { mode, name: 'N(0, 1)', colourOffset: 1, range: [-4, 4] }))
    return cartesian(layers, { mode, x: { label: 'x', limits: [-4, 4] }, y: { label: 'density' } })
  }, [n, bins, seed, overlay, mode])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={4}>
        <NumberField label="sample size" value={n} onChange={setN} min={10} max={10000} integer log />
        <NumberField label="bins" value={bins} onChange={setBins} min={2} max={100} integer />
        <SeedInput value={seed} onChange={setSeed} />
        <Toggle label="true density" value={overlay} onChange={setOverlay} />
      </ControlPanel>
      <Chart option={option} figure={FIGURE} />
    </div>
  )
}
