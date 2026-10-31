import { recordsOf } from 'mdml-engine/exposition'
import { decisionTreeClassifier } from 'mdml-engine/trees'
import { Choice, ControlPanel, ExpositionView, NumberField, SeedInput } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'
import { crescents } from '../../toys/synthetic.ts'

export const recipe: Recipe = {
  title: 'Decision tree',
  question: 'How do I show how a tree splits the data, node by node?',
  explain:
    'A `tree` record keeps every node with the rows that reached it, its class counts and impurity, the split chosen and the other candidates it beat. `TreeView` draws the tree as a diagram with the class mix in each node and a bar on each edge for the rows it carries, and lets the nodes be inspected in the order they were grown. The fit is `decisionTreeClassifier(...).fit(...)`, whose `exposition` holds the record. Depth and the criterion change the shape; label noise makes it grow branches that only fit the noise.',
  shows: ['tree', 'TreeView'],
}

export default function DecisionTreeRecipe(): ReactNode {
  const [depth, setDepth] = useState(3)
  const [criterion, setCriterion] = useState<'gini' | 'entropy'>('gini')
  const [flip, setFlip] = useState(0.05)
  const [seed, setSeed] = useState(1)
  const record = useMemo(() => {
    const fitted = decisionTreeClassifier({ maxDepth: depth, criterion }).fit(crescents(seed, 120, flip))
    return recordsOf(fitted.exposition, 'tree')[0]
  }, [depth, criterion, flip, seed])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={4}>
        <NumberField label="depth" value={depth} onChange={setDepth} min={1} max={6} integer />
        <Choice label="criterion" value={criterion} options={['gini', 'entropy'] as const} onChange={setCriterion} />
        <NumberField label="labels flipped" value={flip} onChange={setFlip} min={0} max={0.3} step={0.01} />
        <SeedInput value={seed} onChange={setSeed} />
      </ControlPanel>
      {record !== undefined && <ExpositionView record={record} title="A tree on two crescents" />}
    </div>
  )
}
