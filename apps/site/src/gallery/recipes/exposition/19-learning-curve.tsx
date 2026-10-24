import { type ReactNode, useMemo, useState } from 'react'
import { breastCancer } from 'whitebox-engine/datasets'
import { recordsOf } from 'whitebox-engine/exposition'
import { learningCurve } from 'whitebox-engine/learn/model-selection'
import { decisionTreeClassifier } from 'whitebox-engine/methods/trees'
import { key } from 'whitebox-engine/random'
import { ControlPanel, LearningCurveView, NumberField } from 'whitebox-render/react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Learning curve',
  question: 'How do I show whether more data would help a model?',
  explain:
    'A learning curve scores a model on its training rows and on held-out rows as the training set grows; `learningCurve` returns the cross-validated scores as a `learningCurve` record in its `exposition`, and `LearningCurveView` draws both with their spread. A deep tree fits its training rows perfectly and the gap closes slowly (variance); a stump scores the same on both (bias).',
}

// region
const data = breastCancer()

export default function LearningCurveRecipe(): ReactNode {
  const [depth, setDepth] = useState(6)
  const record = useMemo(() => {
    const r = learningCurve(decisionTreeClassifier({ maxDepth: depth }), data, {
      cv: 5,
      trainSizes: [0.1, 0.2, 0.35, 0.5, 0.7, 1],
      shuffle: true,
      key: key(2),
    })
    return recordsOf(r.exposition, 'learningCurve')[0]
  }, [depth])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <NumberField label="tree depth" value={depth} onChange={setDepth} min={1} max={12} integer />
      </ControlPanel>
      {record !== undefined && <LearningCurveView record={record} />}
    </div>
  )
}
// endregion
