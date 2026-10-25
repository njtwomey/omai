import { normal } from 'mdml-engine/dists'
import { key } from 'mdml-engine/random'
import { toArray } from 'mdml-engine/tensor'
import { intervalChart } from 'mdml-render/charts'
import { figure } from 'mdml-render/figure'
import { Chart, ControlPanel, FigureFrame, NumberField, SeedInput, useColourMode } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Forest plot',
  question: 'How do I show several estimates with their confidence intervals?',
  explain:
    '`intervalChart(items, { reference })` draws each estimate as a point with its interval as a bar, one row per item, and a reference line (no effect). Here eight sites each estimate the same effect from their own sample; smaller samples give wider intervals, and about one in twenty 95% intervals misses the truth.',
}

const FIGURE = figure({ width: 640, height: 400 })
const SITES = ['Aberdeen', 'Bristol', 'Cork', 'Dundee', 'Exeter', 'Galway', 'Leeds', 'York']

export default function ForestPlot(): ReactNode {
  const mode = useColourMode()
  const [effect, setEffect] = useState(0.4)
  const [size, setSize] = useState(60)
  const [seed, setSeed] = useState(0)
  const option = useMemo(() => {
    const items = SITES.map((name, i) => {
      const n = Math.max(5, Math.round(size * (0.4 + (1.4 * i) / SITES.length)))
      const x = toArray(normal(effect, 1).sample(key(seed * 100 + i), [n])) as number[]
      const mean = x.reduce((a, b) => a + b, 0) / n
      const sd = Math.sqrt(x.reduce((a, b) => a + (b - mean) ** 2, 0) / (n - 1))
      const half = (1.96 * sd) / Math.sqrt(n)
      return {
        name: `${name} (n = ${n})`,
        value: { value: mean, interval: [mean - half, mean + half] as [number, number], level: 0.95 },
      }
    })
    return intervalChart(items, { mode, reference: 0, x: { label: 'effect' } })
  }, [effect, size, seed, mode])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={3}>
        <NumberField label="true effect" value={effect} onChange={setEffect} min={-1} max={1} step={0.05} />
        <NumberField label="typical sample" value={size} onChange={setSize} min={10} max={1000} integer log />
        <SeedInput value={seed} onChange={setSeed} />
      </ControlPanel>
      <FigureFrame figure={FIGURE} filename="forest-plot">
        <Chart option={option} figure={FIGURE} />
      </FigureFrame>
    </div>
  )
}
