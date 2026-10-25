import { type ReactNode, useMemo, useState } from 'react'
import { diabetes } from 'whitebox-engine/datasets'
import { recordsOf } from 'whitebox-engine/exposition'
import { takeData } from 'whitebox-engine/learn'
import { randomForestRegressor } from 'whitebox-engine/methods/ensemble'
import { evaluateExplanations } from 'whitebox-engine/methods/explain'
import { key } from 'whitebox-engine/random'
import { ControlPanel, CurvesView, ExplanationQualityView, NumberField } from 'whitebox-render/react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Evaluating explanations',
  question: 'How do I show which of several explainers deserves trust on a model?',
  explain:
    '`evaluateExplanations` explains the same rows with Tree SHAP, sampling SHAP and LIME and scores each as Quantus does: deletion and insertion areas (remove the most relevant features first; a faithful explanation makes the prediction move fast), faithfulness correlation, sensitivity to small perturbations, sparseness and the completeness gap. `ExplanationQualityView` gives each metric a panel, its better direction and winner in the title, and the explainers’ rank agreement as a matrix; `CurvesView` draws the mean deletion and insertion curves.',
}

// region
const data = takeData(
  diabetes(),
  Array.from({ length: 200 }, (_, i) => i),
)
const forest = randomForestRegressor({ nEstimators: 20, maxDepth: 5 }).fit(data, { key: key(3) })

export default function ExplanationQualityRecipe(): ReactNode {
  const [rows, setRows] = useState(8)
  const result = useMemo(
    () =>
      evaluateExplanations(forest, data, {
        explainers: ['tree', 'sampling', 'lime'],
        rows,
        background: 20,
        key: key(1),
      }),
    [rows],
  )
  const quality = recordsOf(result.exposition, 'explanationQuality')[0]
  const curves = recordsOf(result.exposition, 'curves')[0]
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={1}>
        <NumberField label="rows explained" value={rows} onChange={setRows} min={2} max={30} integer />
      </ControlPanel>
      {quality !== undefined && <ExplanationQualityView record={quality} />}
      {curves !== undefined && <CurvesView record={curves} />}
    </div>
  )
}
// endregion
