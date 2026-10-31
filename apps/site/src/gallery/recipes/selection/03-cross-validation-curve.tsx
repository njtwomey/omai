import { cvPath } from 'mdml-engine/linear'
import { ControlPanel, ExpositionView, NumberField, SeedInput } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'
import { sparseRegression } from '../../toys/synthetic.ts'

export const recipe: Recipe = {
  title: 'Cross-validation curve',
  question: 'How do I choose a penalty from held-out error, and how sure is that choice?',
  explain:
    'A `crossValidationCurve` record has the mean score at each value of one parameter with its standard error, the score of every fold, the best value and the largest value within one standard error of it. `CrossValidationView` draws the curve with its band, every fold faint, and marks both choices. `cvPath` cross-validates a lasso path on the sparse toy problem. The one-standard-error rule picks a sparser model than the minimum whenever the curve is flat near its bottom; more noise flattens it.',
  shows: ['crossValidationCurve', 'CrossValidationView'],
}

export default function CrossValidationCurveRecipe(): ReactNode {
  const [noise, setNoise] = useState(2)
  const [n, setN] = useState(100)
  const [seed, setSeed] = useState(1)
  const record = useMemo(() => cvPath(sparseRegression(seed, n, 8, noise)).curve, [noise, n, seed])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={3}>
        <NumberField label="noise" value={noise} onChange={setNoise} min={0} max={8} step={0.5} />
        <NumberField label="rows" value={n} onChange={setN} min={30} max={500} step={10} integer />
        <SeedInput value={seed} onChange={setSeed} />
      </ControlPanel>
      <ExpositionView record={record} title="Held-out error along the penalty" />
    </div>
  )
}
