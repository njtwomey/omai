import { recordsOf } from 'mdml-engine/exposition'
import { decisionTreeClassifier } from 'mdml-engine/trees'
import { ControlPanel, ExpositionView, NumberField, SeedInput } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'
import { crescents } from '../../toys/synthetic.ts'

export const recipe: Recipe = {
  title: 'Pruning path',
  question: 'How do I show what is lost, tree by tree, when the weakest branches are cut?',
  explain:
    'A `pruningPath` record lists the effective complexity penalties at which the weakest link of a grown tree is pruned, with the total impurity and the leaf count of each pruned tree. `PruningPathView` draws impurity and leaves against the penalty as steps: the early cuts cost almost nothing, and past a knee each one costs more. The tree is grown deep on noisy crescents, so most of its leaves are noise and the path shows how many can go for free.',
  shows: ['pruningPath', 'PruningPathView'],
}

export default function PruningPathRecipe(): ReactNode {
  const [depth, setDepth] = useState(10)
  const [flip, setFlip] = useState(0.12)
  const [seed, setSeed] = useState(1)
  const record = useMemo(
    () =>
      recordsOf(
        decisionTreeClassifier({ maxDepth: depth }).fit(crescents(seed, 200, flip)).exposition,
        'pruningPath',
      )[0],
    [depth, flip, seed],
  )
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={3}>
        <NumberField label="depth grown" value={depth} onChange={setDepth} min={2} max={16} integer />
        <NumberField label="labels flipped" value={flip} onChange={setFlip} min={0} max={0.4} step={0.02} />
        <SeedInput value={seed} onChange={setSeed} />
      </ControlPanel>
      {record !== undefined && <ExpositionView record={record} title="Cutting the weakest branches" />}
    </div>
  )
}
