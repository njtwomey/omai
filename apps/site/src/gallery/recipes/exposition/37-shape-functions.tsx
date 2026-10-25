import { type ReactNode, useMemo, useState } from 'react'
import { diabetes } from 'whitebox-engine/datasets'
import { recordsOf } from 'whitebox-engine/exposition'
import { explainableBoostingMachine } from 'whitebox-engine/methods/explain'
import { ControlPanel, DependenceSurfaceView, DependenceView, ImportanceView, NumberField } from 'whitebox-render/react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Shape functions of a glass-box model',
  question: 'How do I show a model that is its own explanation?',
  explain:
    '`explainableBoostingMachine` fits an EBM: one shape function per feature, learnt by cycling through the features and boosting each a little at a time, then a few pairwise terms. Its `dependence` record (method `shape function`) is drawn by `DependenceView` as steps, one per bin, and its strongest pair by `DependenceSurfaceView`; the `importances` record ranks the terms by their mean absolute contribution. Nothing is approximated: the prediction is the intercept plus these terms.',
}

// region
const data = diabetes()

export default function ShapeFunctionRecipe(): ReactNode {
  const [pairs, setPairs] = useState(1)
  const result = useMemo(
    () => explainableBoostingMachine(data, { rounds: 120, learningRate: 0.1, bins: 24, interactions: pairs }),
    [pairs],
  )
  const shapes = recordsOf(result.exposition, 'dependence')[0]
  const terms = recordsOf(result.exposition, 'importances')[0]
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={1}>
        <NumberField label="pairwise terms" value={pairs} onChange={setPairs} min={0} max={3} integer />
      </ControlPanel>
      {shapes !== undefined && <DependenceView record={shapes} all />}
      {shapes?.surfaces !== undefined && <DependenceSurfaceView record={shapes} />}
      {terms !== undefined && <ImportanceView record={terms} />}
    </div>
  )
}
// endregion
