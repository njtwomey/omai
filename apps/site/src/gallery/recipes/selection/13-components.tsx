import { componentsRecord } from 'omai-engine/exposition'
import { eigh } from 'omai-engine/numerics/linalg'
import { tensor, toArray } from 'omai-engine/tensor'
import { ControlPanel, ExpositionView, NumberField, SeedInput } from 'omai-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'
import { gaussianRows } from '../../toys/synthetic.ts'

export const recipe: Recipe = {
  title: 'Components',
  question: 'How do I show the directions a decomposition found and how much each explains?',
  explain:
    'A `components` record holds the components (each a weighting of the original features), the variance each explains, and optionally the rows’ scores. `ComponentsView` draws the rows’ scores on the first two components with the features as arrows (a biplot), the explained variance as bars with their running total, and the loadings as a heatmap; read a component by which features it combines and with which sign. The data is six measurements driven by two hidden factors (the first three features by one, the last three by the other) plus noise, so two components should explain most of the variance and split the features in two groups. Raise the noise and the rest fills in.',
  shows: ['components', 'ComponentsView'],
}

const FEATURES = ['height', 'arm', 'leg', 'income', 'rent', 'savings']

export default function ComponentsRecipe(): ReactNode {
  const [noise, setNoise] = useState(0.5)
  const [n, setN] = useState(200)
  const [seed, setSeed] = useState(1)
  const record = useMemo(() => {
    const z = gaussianRows(seed, n, 8)
    const rows = z.map((r) => {
      const [a = 0, b = 0, ...e] = r
      return [a, 0.9 * a, 0.8 * a, b, 0.85 * b, 0.7 * b].map((v, j) => v + noise * (e[j] ?? 0))
    })
    const d = FEATURES.length
    const mean = FEATURES.map((_, j) => rows.reduce((s, r) => s + (r[j] ?? 0), 0) / n)
    const centred = rows.map((r) => r.map((v, j) => v - (mean[j] ?? 0)))
    const cov = FEATURES.map((_, i) =>
      FEATURES.map((_, j) => centred.reduce((s, r) => s + (r[i] ?? 0) * (r[j] ?? 0), 0) / (n - 1)),
    )
    const e = eigh(tensor(cov))
    const values = toArray(e.eigenvalues) as number[]
    const vectors = toArray(e.eigenvectors) as number[][]
    const order = values.map((_, i) => d - 1 - i)
    const variance = order.map((i) => values[i] ?? 0)
    const total = variance.reduce((s, v) => s + v, 0)
    const components = order.slice(0, 4).map((c) => FEATURES.map((_, j) => vectors[j]?.[c] ?? 0))
    return componentsRecord({
      method: 'PCA',
      features: FEATURES,
      components,
      space: 'features',
      explainedVariance: variance.slice(0, 4),
      explainedVarianceRatio: variance.slice(0, 4).map((v) => v / total),
      scores: centred.map((r) => components.map((c) => c.reduce((s, w, j) => s + w * (r[j] ?? 0), 0))),
    })
  }, [noise, n, seed])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={3}>
        <NumberField label="noise" value={noise} onChange={setNoise} min={0.05} max={2} step={0.05} />
        <NumberField label="rows" value={n} onChange={setN} min={20} max={1000} step={20} integer />
        <SeedInput value={seed} onChange={setSeed} />
      </ControlPanel>
      <ExpositionView record={record} title="Directions of most variance" />
    </div>
  )
}
