import { type ReactNode, useMemo, useState } from 'react'
import { diabetes } from 'whitebox-engine/datasets'
import { recordsOf } from 'whitebox-engine/exposition'
import { kFold } from 'whitebox-engine/learn/model-selection'
import { standardScaler } from 'whitebox-engine/learn/preprocessing'
import { lassoCV } from 'whitebox-engine/methods/linear'
import { ControlPanel, CrossValidationView, NumberField } from 'whitebox-render/react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Cross-validation curve',
  question: 'How do I show how cross-validation chose a parameter?',
  explain:
    'The lasso’s `crossValidationCurve` record holds every fold’s error along the penalty grid. `CrossValidationView` draws the folds faintly, their mean with a band of one standard error, the best penalty and the one-standard-error choice (the largest penalty whose error is under the dotted line). More folds narrow the band.',
}

// region
const raw = diabetes()
const data = { ...raw, x: standardScaler().fit(raw).transform(raw.x) }

export default function CrossValidationRecipe(): ReactNode {
  const [folds, setFolds] = useState(5)
  const curve = useMemo(() => {
    const fitted = lassoCV({ nAlphas: 40, cv: kFold(folds) }).fit(data)
    return recordsOf(fitted.exposition, 'crossValidationCurve')[0]
  }, [folds])
  if (curve === undefined) return null
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <NumberField label="folds" value={folds} onChange={setFolds} min={2} max={20} integer />
      </ControlPanel>
      <CrossValidationView record={curve} scoreLabel="validation MSE" inverse />
    </div>
  )
}
// endregion
