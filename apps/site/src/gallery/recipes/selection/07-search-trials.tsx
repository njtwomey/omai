import { recordsOf } from 'mdml-engine/exposition'
import { gridSearchCV } from 'mdml-engine/learn/model-selection'
import { decisionTreeClassifier } from 'mdml-engine/trees'
import { ControlPanel, ExpositionView, NumberField, SeedInput } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'
import { crescents } from '../../toys/synthetic.ts'

export const recipe: Recipe = {
  title: 'Search trials',
  question: 'How do I show every setting a search tried and how each scored?',
  explain:
    'A `searchTrials` record has one entry per candidate: its parameters, its score, per-step scores for those that report them, and whether it completed, was pruned or failed. `SearchTrialsView` draws the trials by score against each parameter, the best marked, and a parallel view of the parameters across trials. `gridSearchCV` tries every depth and leaf size for a tree on noisy crescents, cross-validated. Raise the noise and the best setting moves to a smaller tree.',
  shows: ['searchTrials', 'SearchTrialsView'],
}

export default function SearchTrialsRecipe(): ReactNode {
  const [flip, setFlip] = useState(0.1)
  const [n, setN] = useState(200)
  const [seed, setSeed] = useState(1)
  const record = useMemo(() => {
    const r = gridSearchCV(
      decisionTreeClassifier(),
      { maxDepth: [1, 2, 3, 4, 6, 8, 12], minSamplesLeaf: [1, 3, 8, 20] },
      crescents(seed, n, flip),
      { cv: 3 },
    )
    return recordsOf(r.exposition, 'searchTrials')[0]
  }, [flip, n, seed])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={3}>
        <NumberField label="labels flipped" value={flip} onChange={setFlip} min={0} max={0.4} step={0.02} />
        <NumberField label="rows" value={n} onChange={setN} min={60} max={600} step={20} integer />
        <SeedInput value={seed} onChange={setSeed} />
      </ControlPanel>
      {record !== undefined && <ExpositionView record={record} title="Twenty-eight settings tried" />}
    </div>
  )
}
