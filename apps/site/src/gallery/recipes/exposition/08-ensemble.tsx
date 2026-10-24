import { type ReactNode, useMemo, useState } from 'react'
import { generate } from 'whitebox-engine/data'
import { moons } from 'whitebox-engine/datasets'
import { recordsOf } from 'whitebox-engine/exposition'
import {
  adaBoostClassifier,
  gradientBoostingClassifier,
  randomForestClassifier,
} from 'whitebox-engine/methods/ensemble'
import { key } from 'whitebox-engine/random'
import { Choice, ControlPanel, EnsembleView, NumberField } from 'whitebox-render/react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Ensemble stages',
  question: 'How do I show an ensemble being built, member by member?',
  explain:
    'Boosting and forests return an `ensembleStages` record: each member’s weight, the training score after each stage (named, with its direction), what each member was fitted to, and the decision surface after up to 40 stages. `EnsembleView` draws from the record alone and plays the stages: the regions of the ensemble so far, the rows sized by what the newest member was fitted to (AdaBoost’s sample weights grow on the rows it gets wrong; gradient boosting’s residuals), the misclassified rows ringed, beside the scores and the member weights.',
}

// region
const { data } = generate(moons, { knobs: { n: 140, noise: 0.25 }, key: key(5) })
type Method = 'AdaBoost' | 'gradient boosting' | 'random forest'

export default function EnsembleRecipe(): ReactNode {
  const [method, setMethod] = useState<Method>('AdaBoost')
  const [members, setMembers] = useState(30)
  const [rate, setRate] = useState(0.5)
  const stages = useMemo(() => {
    const est =
      method === 'AdaBoost'
        ? adaBoostClassifier({ nEstimators: members, learningRate: rate })
        : method === 'gradient boosting'
          ? gradientBoostingClassifier({ nEstimators: members, learningRate: rate, maxDepth: 2 })
          : randomForestClassifier({ nEstimators: members, maxDepth: 6 })
    return recordsOf(est.fit(data, { key: key(1) }).exposition, 'ensembleStages')[0]
  }, [method, members, rate])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={3}>
        <Choice
          label="method"
          value={method}
          onChange={setMethod}
          options={['AdaBoost', 'gradient boosting', 'random forest'] as Method[]}
        />
        <NumberField label="members" value={members} onChange={setMembers} min={1} max={100} integer />
        <NumberField label="learning rate" value={rate} onChange={setRate} min={0.01} max={2} log />
      </ControlPanel>
      {stages !== undefined && <EnsembleView record={stages} />}
    </div>
  )
}
// endregion
