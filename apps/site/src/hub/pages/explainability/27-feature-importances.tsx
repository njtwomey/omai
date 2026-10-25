import { trainTestSplit } from 'mdml-engine/data'
import { breastCancer } from 'mdml-engine/datasets'
import { recordsOf } from 'mdml-engine/exposition'
import { randomForestClassifier } from 'mdml-engine/methods/ensemble'
import { featureImportance } from 'mdml-engine/methods/explain'
import { key } from 'mdml-engine/random'
import { Choice, ControlPanel, ImportanceView, NumberField } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../../gallery/recipe.ts'

export const recipe: Recipe = {
  title: 'Feature importances',
  question: 'How do I show which features a model relies on, with their uncertainty?',
  explain:
    '`featureImportance` shuffles one feature at a time and records the drop in the score (permutation importance), every repeat kept, in an `importances` record; `ImportanceView` draws the mean with a whisker of one standard deviation and each repeat as a dot. Measured on held-out rows it shows what generalises; the forest’s impurity importances come from the training splits alone.',
}

// region
const { train, test } = trainTestSplit(breastCancer(), { testSize: 0.3, key: key(0), stratify: true })
const forest = randomForestClassifier({ nEstimators: 40, maxDepth: 6 }).fit(train, { key: key(2) })
type Measure = 'permutation (held out)' | 'permutation (training)' | 'impurity'

export default function ImportanceRecipe(): ReactNode {
  const [measure, setMeasure] = useState<Measure>('permutation (held out)')
  const [repeats, setRepeats] = useState(10)
  const record = useMemo(() => {
    const r =
      measure === 'impurity'
        ? featureImportance(forest, train, { method: 'impurity', on: 'training' })
        : featureImportance(forest, measure === 'permutation (held out)' ? test : train, {
            nRepeats: repeats,
            key: key(4),
          })
    return recordsOf(r.exposition, 'importances')[0]
  }, [measure, repeats])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <Choice
          label="measure"
          value={measure}
          onChange={setMeasure}
          options={['permutation (held out)', 'permutation (training)', 'impurity'] as Measure[]}
        />
        <NumberField label="repeats" value={repeats} onChange={setRepeats} min={1} max={30} integer />
      </ControlPanel>
      {record !== undefined && <ImportanceView record={record} maxFeatures={12} />}
    </div>
  )
}
// endregion
