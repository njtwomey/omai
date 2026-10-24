import { type ReactNode, useMemo, useState } from 'react'
import { breastCancer } from 'whitebox-engine/datasets'
import { recordsOf } from 'whitebox-engine/exposition'
import { decisionTreeClassifier } from 'whitebox-engine/methods/trees'
import { ControlPanel, NumberField, PruningPathView } from 'whitebox-render/react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Pruning path',
  question: 'How do I show what cost-complexity pruning trades away?',
  explain:
    'Every tree records its cost-complexity pruning path as a `pruningPath` record: the effective alphas at which subtrees are cut, with the total leaf impurity and the number of leaves after each cut. `PruningPathView` draws both; move alpha to see how many leaves a pruned tree keeps and what impurity it accepts.',
}

// region
const data = breastCancer()

export default function PruningPathRecipe(): ReactNode {
  const [alpha, setAlpha] = useState(0.01)
  const path = useMemo(() => recordsOf(decisionTreeClassifier().fit(data).exposition, 'pruningPath')[0], [])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <NumberField label="alpha" value={alpha} onChange={setAlpha} min={0} max={0.1} />
      </ControlPanel>
      {path !== undefined && <PruningPathView record={path} selected={alpha} />}
    </div>
  )
}
// endregion
