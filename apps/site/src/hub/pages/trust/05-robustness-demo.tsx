import { generate } from 'mdml-engine/data'
import { makeClassification } from 'mdml-engine/datasets'
import { recordsOf } from 'mdml-engine/exposition'
import type { Estimator } from 'mdml-engine/learn'
import { randomForestClassifier } from 'mdml-engine/methods/ensemble'
import { logisticRegression } from 'mdml-engine/methods/linear'
import { robustnessStudy } from 'mdml-engine/methods/trust'
import { key } from 'mdml-engine/random'
import { Choice, ControlPanel, CurvesView } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../../gallery/recipe.ts'

export const recipe: Recipe = {
  title: 'Robustness to noise and missing values',
  question: 'How gracefully does a classifier degrade when its inputs get worse?',
  explain:
    '`robustnessStudy(model, data)` adds Gaussian noise to the inputs at growing scales (in each feature’s standard deviations), to every feature at once and to one feature at a time, and blanks growing shares of values (filled with the feature’s mean). The single-feature curves show which features the model leans on: noise in a feature it ignores costs nothing. A forest and a linear model lean on features differently.',
}

type Model = 'logistic regression' | 'random forest'
const MODELS: Readonly<Record<Model, () => Estimator>> = {
  'logistic regression': () => logisticRegression(),
  'random forest': () => randomForestClassifier({ nEstimators: 30, maxDepth: 6 }),
}

export default function RobustnessRecipe(): ReactNode {
  const [model, setModel] = useState<Model>('logistic regression')
  const data = useMemo(
    () =>
      generate(makeClassification, {
        key: key(9),
        knobs: { n: 800, nFeatures: 6, nInformative: 3, nRedundant: 0, shuffle: false },
      }).data,
    [],
  )
  const records = useMemo(() => {
    const fitted = MODELS[model]().fit(data, { key: key(1) })
    return recordsOf(robustnessStudy(fitted, data, { key: key(2) }).exposition, 'curves')
  }, [data, model])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={1}>
        <Choice
          label="model"
          value={model}
          onChange={setModel}
          options={['logistic regression', 'random forest'] as Model[]}
        />
      </ControlPanel>
      <div className="flex flex-wrap gap-4">
        {records.map((r) => (
          <CurvesView key={r.xLabel} record={r} play={false} />
        ))}
      </div>
    </div>
  )
}
