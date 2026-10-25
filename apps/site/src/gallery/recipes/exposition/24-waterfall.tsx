import { type ReactNode, useMemo, useState } from 'react'
import { diabetes } from 'whitebox-engine/datasets'
import { recordsOf } from 'whitebox-engine/exposition'
import { randomForestRegressor } from 'whitebox-engine/methods/ensemble'
import { type AttributionMethod, explainPrediction } from 'whitebox-engine/methods/explain'
import { key } from 'whitebox-engine/random'
import { Choice, ControlPanel, NumberField, WaterfallView } from 'whitebox-render/react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Explaining one prediction',
  question: 'How do I show why a model made one prediction?',
  explain:
    '`explainPrediction` returns a `featureAttributions` record: the expected output, how far each feature pushed this row’s prediction up or down, and the row’s values. `WaterfallView` starts at E[f(x)] and adds the features in turn, largest at the top, to end at f(x). Tree SHAP is exact for the forest; the exact, kernel and sampling explainers treat it as a black box against 50 background rows (the same Shapley values, by other means), and LIME fits a local linear model whose waterfall ends at its own prediction.',
}

// region
const data = diabetes()
const forest = randomForestRegressor({ nEstimators: 30, maxDepth: 5 }).fit(data, { key: key(3) })
const methods: AttributionMethod[] = ['tree', 'exact', 'kernel', 'sampling', 'lime']

export default function WaterfallRecipe(): ReactNode {
  const [row, setRow] = useState(12)
  const [method, setMethod] = useState<AttributionMethod>('tree')
  const record = useMemo(
    () => recordsOf(explainPrediction(forest, data, row, { method, key: key(1) }).exposition, 'featureAttributions')[0],
    [row, method],
  )
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <NumberField label="patient (row)" value={row} onChange={setRow} min={0} max={441} integer />
        <Choice label="explainer" value={method} onChange={setMethod} options={methods} />
      </ControlPanel>
      {record !== undefined && <WaterfallView record={record} />}
    </div>
  )
}
// endregion
