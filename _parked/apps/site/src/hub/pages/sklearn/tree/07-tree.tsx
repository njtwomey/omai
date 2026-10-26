import { iris } from 'mdml-engine/datasets'
import { recordsOf } from 'mdml-engine/exposition'
import { type ClassificationCriterion, decisionTreeClassifier } from 'mdml-engine/methods/trees'
import { Choice, ControlPanel, NumberField, TreeView } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../../../gallery/recipe.ts'

export const recipe: Recipe = {
  title: 'Decision tree',
  question: 'How do I show how a decision tree was grown, split by split?',
  explain:
    'A decision tree returns a `tree` record: every node’s rows, class counts, impurity, chosen split and every candidate split it evaluated, with the order of growth. `TreeView` draws the tree (each card’s bar is its class proportions); click a node to see its candidate splits, impurity decrease against threshold per feature, and play the growth.',
}

// region
const data = iris()

export default function TreeRecipe(): ReactNode {
  const [maxDepth, setMaxDepth] = useState(3)
  const [criterion, setCriterion] = useState<ClassificationCriterion>('gini')
  const tree = useMemo(
    () => recordsOf(decisionTreeClassifier({ maxDepth, criterion }).fit(data).exposition, 'tree')[0],
    [maxDepth, criterion],
  )
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <NumberField label="maximum depth" value={maxDepth} onChange={setMaxDepth} min={1} max={6} integer />
        <Choice
          label="criterion"
          value={criterion}
          onChange={setCriterion}
          options={['gini', 'entropy'] as ClassificationCriterion[]}
        />
      </ControlPanel>
      {tree !== undefined && <TreeView record={tree} />}
    </div>
  )
}
// endregion
