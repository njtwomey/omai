import { diabetes } from 'mdml-engine/datasets'
import { recordsOf } from 'mdml-engine/exposition'
import { gradientBoostingRegressor, randomForestRegressor } from 'mdml-engine/methods/ensemble'
import { explainDataset } from 'mdml-engine/methods/explain'
import { key } from 'mdml-engine/random'
import {
  AttributionBarView,
  BeeswarmView,
  Choice,
  ControlPanel,
  DependenceScatterView,
  Select,
} from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Attribution summary',
  question: 'How do I show what drives a model’s predictions across a whole dataset?',
  explain:
    '`explainDataset` explains every row at once (Tree SHAP here), in one `featureAttributions` record. `BeeswarmView` puts a point per row on each feature’s line at its attribution, coloured by the feature’s value: high bmi (red) pushes the predicted progression up. `AttributionBarView` averages the magnitudes into a global importance, and `DependenceScatterView` plots one feature’s values against its attributions, coloured by the feature it interacts with most.',
}

// region
const data = diabetes()
const names = data.featureNames
type Model = 'random forest' | 'gradient boosting'

export default function BeeswarmRecipe(): ReactNode {
  const [model, setModel] = useState<Model>('gradient boosting')
  const [feature, setFeature] = useState(2)
  const record = useMemo(() => {
    const est =
      model === 'random forest'
        ? randomForestRegressor({ nEstimators: 30, maxDepth: 5 })
        : gradientBoostingRegressor({ nEstimators: 60, maxDepth: 3, learningRate: 0.1 })
    const fitted = est.fit(data, { key: key(3) })
    return recordsOf(explainDataset(fitted, data, { maxRows: 442 }).exposition, 'featureAttributions')[0]
  }, [model])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <Choice
          label="model"
          value={model}
          onChange={setModel}
          options={['gradient boosting', 'random forest'] as Model[]}
        />
        <Select
          label="dependence of"
          value={feature}
          onChange={setFeature}
          options={names.map((n, j) => ({ value: j, label: n }))}
        />
      </ControlPanel>
      {record !== undefined && (
        <>
          <BeeswarmView record={record} />
          <AttributionBarView record={record} />
          <DependenceScatterView record={record} feature={feature} />
        </>
      )}
    </div>
  )
}
// endregion
