import { recordsOf } from 'mdml-engine/exposition'
import { defineEstimator } from 'mdml-engine/learn'
import { explainSeries, type SeriesMethod } from 'mdml-engine/methods/explain'
import { asTensor, tensor, toFlat } from 'mdml-engine/tensor'
import { AttributionSeriesView, Choice, ControlPanel, NumberField } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../../gallery/recipe.ts'

export const recipe: Recipe = {
  title: 'Attributions along a series',
  question: 'How do I show which stretch of a series a model’s output depends on?',
  explain:
    '`explainSeries` explains a model of whole series by removing stretches of one: occlusion slides a window of steps to the series’ mean, ablation removes consecutive windows, and SHAP gives the windows’ Shapley values (shared among their steps). The model here scores the peak in the second half against the first half’s level, so the attributions gather where the peak is. `AttributionSeriesView` colours each step by its attribution and draws them as bars beneath.',
}

// region
const T = 48
const series = (peak: number): number[] =>
  Array.from({ length: T }, (_, t) => Math.sin(t / 4) * 0.3 + 2 * Math.exp(-((t - peak) ** 2) / 8))
const peakModel = defineEstimator(
  {
    name: 'peak score',
    kind: 'regressor',
    fit: () => ({}),
    methods: () => ({
      predict: (x) => {
        const t = asTensor(x)
        const [m = 0, d = 0] = t.shape
        const v = toFlat(t)
        return tensor(
          Array.from({ length: m }, (_, i) => {
            const r = v.slice(i * d, (i + 1) * d)
            return Math.max(...r.slice(d / 2)) - r.slice(0, d / 2).reduce((a, b) => a + b, 0) / (d / 2)
          }),
        )
      },
    }),
  },
  {},
).fit({ x: [series(30)], y: [0] })
const methods: SeriesMethod[] = ['occlusion', 'ablation', 'shap']

export default function SeriesRecipe(): ReactNode {
  const [peak, setPeak] = useState(34)
  const [method, setMethod] = useState<SeriesMethod>('occlusion')
  const record = useMemo(() => {
    const data = { x: [series(peak)] }
    return recordsOf(explainSeries(peakModel, data, 0, { method, window: 4 }).exposition, 'featureAttributions')[0]
  }, [peak, method])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <NumberField label="peak at" value={peak} onChange={setPeak} min={2} max={46} integer />
        <Choice label="explainer" value={method} onChange={setMethod} options={methods} />
      </ControlPanel>
      {record !== undefined && <AttributionSeriesView record={record} />}
    </div>
  )
}
// endregion
