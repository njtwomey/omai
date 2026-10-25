import { type ReactNode, useMemo, useState } from 'react'
import { generate } from 'whitebox-engine/data'
import { sineCurve } from 'whitebox-engine/datasets'
import { recordsOf } from 'whitebox-engine/exposition'
import { kNeighborsRegressor } from 'whitebox-engine/methods/neighbours'
import { decisionTreeRegressor } from 'whitebox-engine/methods/trees'
import { ensembleUncertainty } from 'whitebox-engine/methods/trust'
import { key } from 'whitebox-engine/random'
import { Choice, ControlPanel, CurvesView, NumberField } from 'whitebox-render/react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Epistemic and aleatoric uncertainty',
  question: 'Which uncertainty would more data remove, and which would it not?',
  explain:
    '`ensembleUncertainty(estimator, data, { members })` fits a bootstrap ensemble. Where the members disagree the model is unsure of itself (epistemic uncertainty): little data, or none beyond the edges of the data. The out-of-bag residuals, less that disagreement, estimate the noise (aleatoric uncertainty), which no amount of data removes. More rows narrow the epistemic band inside the data but not outside it; more noise widens the total band everywhere.',
}

type Model = 'k nearest neighbours' | 'regression tree'

export default function UncertaintyRecipe(): ReactNode {
  const [n, setN] = useState(80)
  const [noise, setNoise] = useState(0.2)
  const [members, setMembers] = useState(20)
  const [model, setModel] = useState<Model>('regression tree')
  const data = useMemo(() => generate(sineCurve, { key: key(7), knobs: { n, noise } }).data, [n, noise])
  const record = useMemo(() => {
    const est =
      model === 'regression tree' ? decisionTreeRegressor({ maxDepth: 5 }) : kNeighborsRegressor({ nNeighbors: 7 })
    return recordsOf(ensembleUncertainty(est, data, { members, key: key(1) }).exposition, 'curves')[0]
  }, [data, members, model])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={4}>
        <NumberField label="rows" value={n} onChange={setN} min={10} max={1000} step={10} />
        <NumberField label="noise" value={noise} onChange={setNoise} min={0} max={1} step={0.05} />
        <NumberField label="members" value={members} onChange={setMembers} min={2} max={60} step={1} />
        <Choice
          label="model"
          value={model}
          onChange={setModel}
          options={['regression tree', 'k nearest neighbours'] as Model[]}
        />
      </ControlPanel>
      {record !== undefined && <CurvesView record={record} play={false} />}
    </div>
  )
}
