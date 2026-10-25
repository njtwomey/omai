import { type ReactNode, useMemo, useState } from 'react'
import { diabetes } from 'whitebox-engine/datasets'
import { recordsOf } from 'whitebox-engine/exposition'
import { randomForestRegressor } from 'whitebox-engine/methods/ensemble'
import { aleEffects, dependencePlot } from 'whitebox-engine/methods/explain'
import { key } from 'whitebox-engine/random'
import { Choice, ControlPanel, DependenceSurfaceView, DependenceView, Select } from 'whitebox-render/react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Partial dependence, ICE and ALE',
  question: 'How do I show how a model’s prediction changes with one feature, and with a pair?',
  explain:
    '`dependencePlot` sweeps a feature over a grid in every row and averages the predictions (partial dependence), keeping each row’s own curve (ICE), centred or differentiated on request; `aleEffects` accumulates the local changes within quantile bins instead, which stays honest when features are correlated. Both return a `dependence` record: `DependenceView` draws the average over the ICE curves and the data’s rug, and `DependenceSurfaceView` the pair’s joint effect (the two-way PDP, or the second-order ALE’s pure interaction).',
}

// region
const data = diabetes()
const forest = randomForestRegressor({ nEstimators: 30, maxDepth: 5 }).fit(data, { key: key(3) })
type Method = 'partial dependence' | 'ALE'
type Transform = 'none' | 'centred' | 'derivative'

export default function DependenceRecipe(): ReactNode {
  const [method, setMethod] = useState<Method>('partial dependence')
  const [transform, setTransform] = useState<Transform>('none')
  const [feature, setFeature] = useState(2)
  const record = useMemo(() => {
    const r =
      method === 'ALE'
        ? aleEffects(forest, data, { features: [feature], pairs: [[2, 8]], bins: 10 })
        : dependencePlot(forest, data, {
            features: [feature],
            pairs: [[2, 8]],
            transform,
            gridResolution: 25,
            maxRows: 80,
          })
    return recordsOf(r.exposition, 'dependence')[0]
  }, [method, transform, feature])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={3}>
        <Choice
          label="method"
          value={method}
          onChange={setMethod}
          options={['partial dependence', 'ALE'] as Method[]}
        />
        <Choice
          label="ICE curves"
          value={transform}
          onChange={setTransform}
          options={['none', 'centred', 'derivative'] as Transform[]}
        />
        <Select
          label="feature"
          value={feature}
          onChange={setFeature}
          options={data.featureNames.map((n, j) => ({ value: j, label: n }))}
        />
      </ControlPanel>
      {record !== undefined && (
        <>
          <DependenceView record={record} />
          <DependenceSurfaceView record={record} />
        </>
      )}
    </div>
  )
}
// endregion
