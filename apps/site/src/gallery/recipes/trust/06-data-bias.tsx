import { dataBias, type GroupDisparity } from 'omai-engine/exposition'
import { ControlPanel, ExpositionView, NumberField, SeedInput } from 'omai-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'
import { uniforms } from '../../toys/synthetic.ts'

export const recipe: Recipe = {
  title: 'Bias in data',
  question:
    'How do I show where unfairness comes from before any model is trained: who is in the data, how they were labelled, what stands in for them?',
  explain:
    'A `dataBias` record looks at the data alone: each group’s share of the rows against its share of the population, its base rate of the positive label, how many labels were flipped, how well the other features predict the group (proxies), and how much a model would amplify the base-rate gap. `DataBiasView` draws these in one figure. The toy data has a minority that is under-represented, has a lower base rate, and has labels flipped against it at the rate you set; a proxy feature leaks its group by the strength you choose.',
  shows: ['dataBias', 'DataBiasView'],
}

export default function DataBiasRecipe(): ReactNode {
  const [share, setShare] = useState(0.15)
  const [flip, setFlip] = useState(0.1)
  const [proxy, setProxy] = useState(0.75)
  const [seed, setSeed] = useState(1)
  const record = useMemo(() => {
    const n = 4000
    const u = uniforms(seed, 3 * n)
    const group = Array.from({ length: n }, (_, i) => ((u[i] ?? 0) < share ? 1 : 0))
    const truth = group.map((g, i) => ((u[n + i] ?? 0) < (g === 1 ? 0.35 : 0.45) ? 1 : 0))
    const recorded = truth.map((y, i) => {
      const r = u[2 * n + i] ?? 0
      if (y === 1 && group[i] === 1 && r < flip * 2) return 0
      if (y === 0 && group[i] === 0 && r < flip / 2) return 1
      return y
    })
    const counts = [0, 1].map((g) => group.filter((v) => v === g).length)
    const base = [0, 1].map(
      (g) => recorded.filter((y, i) => group[i] === g && y === 1).length / Math.max(1, counts[g] ?? 1),
    )
    const overall = recorded.reduce<number>((a, b) => a + b, 0) / n
    const flipShare = (g: number, from: number): number => {
      const clean = truth.flatMap((y, i) => (group[i] === g && y === from ? [i] : []))
      return clean.filter((i) => recorded[i] !== from).length / Math.max(1, clean.length)
    }
    const flipsDown = [0, 1].map((g) => flipShare(g, 1))
    const flipsUp = [0, 1].map((g) => flipShare(g, 0))
    const gap = (base[0] ?? 0) - (base[1] ?? 0)
    const disparities: GroupDisparity[] = [
      { name: 'base rate difference', value: gap, ideal: 0, kind: 'difference' },
      { name: 'base rate ratio', value: (base[1] ?? 0) / Math.max(1e-9, base[0] ?? 1), ideal: 1, kind: 'ratio' },
    ]
    const auc = [0.5 + 0.45 * proxy, 0.5 + 0.2 * proxy, 0.52]
    return dataBias({
      attribute: 'group',
      groups: ['majority', 'minority'],
      reference: 0,
      counts,
      shares: counts.map((c) => c / n),
      populationShares: [0.7, 0.3],
      baseRates: base,
      overallBaseRate: overall,
      disparities,
      labelFlips: { positiveToNegative: flipsDown, negativeToPositive: flipsUp },
      proxies: {
        features: ['postcode', 'job title', 'height'],
        auc,
        dependence: auc.map((a) => 2 * (a - 0.5)),
        combinedAuc: Math.min(0.99, (auc[0] ?? 0.5) + 0.05),
      },
      amplification: { selectionRates: [(base[0] ?? 0) * 1.1, (base[1] ?? 0) * 0.8], value: 0.5 * gap },
    })
  }, [share, flip, proxy, seed])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={4}>
        <NumberField
          label="minority share of the rows"
          value={share}
          onChange={setShare}
          min={0.02}
          max={0.5}
          step={0.01}
        />
        <NumberField label="labels flipped" value={flip} onChange={setFlip} min={0} max={0.4} step={0.02} />
        <NumberField label="proxy strength" value={proxy} onChange={setProxy} min={0} max={1} step={0.05} />
        <SeedInput value={seed} onChange={setSeed} />
      </ControlPanel>
      <ExpositionView record={record} title="Bias in the data before the model" />
    </div>
  )
}
