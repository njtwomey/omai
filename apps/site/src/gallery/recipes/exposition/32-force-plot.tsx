import { diabetes } from 'mdml-engine/datasets'
import { recordsOf } from 'mdml-engine/exposition'
import { randomForestRegressor } from 'mdml-engine/methods/ensemble'
import { type AttributionMethod, explainPrediction } from 'mdml-engine/methods/explain'
import { key } from 'mdml-engine/random'
import { Choice, ControlPanel, ForcePlotView, NumberField } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Force plot',
  question: 'How do I show the forces behind one prediction on a single line?',
  explain:
    '`ForcePlotView` draws a `featureAttributions` row as shap’s force plot: features that raise the prediction are red arrows ending at f(x) from the left, those that lower it blue arrows from the right, each as long as its attribution; the base value E[f(x)] is marked where the forces balance. Tree SHAP, partition SHAP (Owen values over a correlation clustering of the features) and conditional SHAP (missing features drawn given the present ones) give different splits of the same prediction.',
}

// region
const data = diabetes()
const forest = randomForestRegressor({ nEstimators: 30, maxDepth: 5 }).fit(data, { key: key(3) })
const methods: AttributionMethod[] = ['tree', 'partition', 'conditional']

export default function ForcePlotRecipe(): ReactNode {
  const [row, setRow] = useState(12)
  const [method, setMethod] = useState<AttributionMethod>('tree')
  const record = useMemo(
    () =>
      recordsOf(
        explainPrediction(forest, data, row, { method, background: 25, nSamples: 30, key: key(1) }).exposition,
        'featureAttributions',
      )[0],
    [row, method],
  )
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <NumberField label="patient (row)" value={row} onChange={setRow} min={0} max={441} integer />
        <Choice label="explainer" value={method} onChange={setMethod} options={methods} />
      </ControlPanel>
      {record !== undefined && <ForcePlotView record={record} />}
    </div>
  )
}
// endregion
