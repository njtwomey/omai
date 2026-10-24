import { type ReactNode, useMemo, useState } from 'react'
import { iris } from 'whitebox-engine/datasets'
import { gaussianNB } from 'whitebox-engine/methods/naive-bayes'
import { toArray } from 'whitebox-engine/tensor'
import { ControlPanel, LikelihoodView, NumberField } from 'whitebox-render/react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Likelihood table',
  question: 'How do I show why naive Bayes chose a class for a row?',
  explain:
    'Naive Bayes adds a log prior and one log-likelihood term per feature for every class; its `explain(row)` returns those terms as a `likelihoodTable` record. `LikelihoodView` draws them as a matrix beside each class’s total; the largest total is the prediction. Step through the rows and watch which feature decides.',
}

// region
const data = iris()
const rows = toArray(data.x) as number[][]
const fitted = gaussianNB().fit(data)

export default function LikelihoodRecipe(): ReactNode {
  const [row, setRow] = useState(70)
  const table = useMemo(() => fitted.explain(rows[row] ?? [], data.featureNames, data.classes), [row])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <NumberField label="row" value={row} onChange={setRow} min={0} max={rows.length - 1} integer />
      </ControlPanel>
      <LikelihoodView record={table} />
    </div>
  )
}
// endregion
