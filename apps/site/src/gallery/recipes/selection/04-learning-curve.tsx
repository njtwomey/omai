import { recordsOf } from 'mdml-engine/exposition'
import { learningCurve } from 'mdml-engine/learn/model-selection'
import { decisionTreeClassifier } from 'mdml-engine/trees'
import { ControlPanel, ExpositionView, NumberField, SeedInput } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'
import { crescents } from '../../toys/synthetic.ts'

export const recipe: Recipe = {
  title: 'Learning curve',
  question: 'How do I show whether a model needs more data or a different model?',
  explain:
    'A `learningCurve` record is the training score and the validation score at growing training-set sizes, with their spread across folds. `LearningCurveView` draws both with bands: a gap that stays wide is variance (more data helps, or a simpler model), two curves that meet low are bias (a richer model is needed). `learningCurve` cross-validates a decision tree on two noisy crescents. A deep tree memorises (train 1, validation far below); a stump underfits (both low and close).',
  shows: ['learningCurve', 'LearningCurveView'],
}

export default function LearningCurveRecipe(): ReactNode {
  const [depth, setDepth] = useState(8)
  const [flip, setFlip] = useState(0.1)
  const [seed, setSeed] = useState(1)
  const record = useMemo(() => {
    const r = learningCurve(decisionTreeClassifier({ maxDepth: depth }), crescents(seed, 400, flip), {
      cv: 5,
      trainSizes: [0.1, 0.2, 0.35, 0.5, 0.7, 1],
    })
    return recordsOf(r.exposition, 'learningCurve')[0]
  }, [depth, flip, seed])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={3}>
        <NumberField label="tree depth" value={depth} onChange={setDepth} min={1} max={14} integer />
        <NumberField label="labels flipped" value={flip} onChange={setFlip} min={0} max={0.4} step={0.02} />
        <SeedInput value={seed} onChange={setSeed} />
      </ControlPanel>
      {record !== undefined && <ExpositionView record={record} title="Score against training size" />}
    </div>
  )
}
