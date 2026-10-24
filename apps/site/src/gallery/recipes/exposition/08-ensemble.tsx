import { type ReactNode, useMemo, useState } from 'react'
import { generate } from 'whitebox-engine/data'
import { moons } from 'whitebox-engine/datasets'
import { recordsOf } from 'whitebox-engine/exposition'
import { adaBoostClassifier, gradientBoostingClassifier } from 'whitebox-engine/methods/ensemble'
import { key } from 'whitebox-engine/random'
import { toArray } from 'whitebox-engine/tensor'
import { Choice, ControlPanel, EnsembleView, NumberField } from 'whitebox-render/react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Ensemble stages',
  question: 'How do I show an ensemble being built, member by member?',
  explain:
    'Boosting returns an `ensembleStages` record: each member’s weight, the training score after each stage, and what each member was fitted to. `EnsembleView` plays it: for AdaBoost the rows the ensemble still gets wrong grow as their sample weights rise; for gradient boosting the rows are coloured by the residuals the next tree fits.',
}

// region
const { data } = generate(moons, { knobs: { n: 140, noise: 0.25 }, key: key(5) })
const rows = toArray(data.x) as number[][]
const labels = Array.from(data.y.data)

type Method = 'AdaBoost' | 'gradient boosting'

export default function EnsembleRecipe(): ReactNode {
  const [method, setMethod] = useState<Method>('AdaBoost')
  const [members, setMembers] = useState(30)
  const [rate, setRate] = useState(0.5)
  const stages = useMemo(() => {
    const est =
      method === 'AdaBoost'
        ? adaBoostClassifier({ nEstimators: members, learningRate: rate })
        : gradientBoostingClassifier({ nEstimators: members, learningRate: rate, maxDepth: 2 })
    return recordsOf(est.fit(data).exposition, 'ensembleStages')[0]
  }, [method, members, rate])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={3}>
        <Choice
          label="method"
          value={method}
          onChange={setMethod}
          options={['AdaBoost', 'gradient boosting'] as Method[]}
        />
        <NumberField label="members" value={members} onChange={setMembers} min={1} max={100} integer />
        <NumberField label="learning rate" value={rate} onChange={setRate} min={0.01} max={2} log />
      </ControlPanel>
      {stages !== undefined && (
        <EnsembleView
          record={stages}
          points={rows}
          labels={labels}
          scoreLabel={method === 'AdaBoost' ? 'training accuracy' : 'training log loss'}
        />
      )}
    </div>
  )
}
// endregion
