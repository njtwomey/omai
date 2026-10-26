import { oldFaithful } from 'mdml-engine/datasets'
import { recordsOf } from 'mdml-engine/exposition'
import { standardScaler } from 'mdml-engine/learn/preprocessing'
import { type CovarianceType, gaussianMixture } from 'mdml-engine/methods/mixture'
import { key } from 'mdml-engine/random'
import { toArray } from 'mdml-engine/tensor'
import { Choice, ControlPanel, MixturePlayer, NumberField } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../../../gallery/recipe.ts'

export const recipe: Recipe = {
  title: 'Mixture steps',
  question: 'How do I play back the EM steps of a Gaussian mixture?',
  explain:
    'A Gaussian mixture returns a `mixtureSteps` record: weights, means and covariances after every EM step, with the log likelihood (which never falls). `MixturePlayer` plays it on the Old Faithful eruptions: each component’s one and two standard-deviation ellipses, the regions where each component is the most responsible, the rows coloured the same way, and the weights, all from the record.',
}

// region
const raw = oldFaithful().data
const rows = toArray(standardScaler().fit(raw).transform(raw.x)) as number[][]

export default function MixtureStepsRecipe(): ReactNode {
  const [components, setComponents] = useState(2)
  const [type, setType] = useState<CovarianceType>('full')
  const [seed, setSeed] = useState(1)
  const steps = useMemo(() => {
    const fitted = gaussianMixture({ nComponents: components, covarianceType: type, init: 'random', tol: 1e-5 }).fit(
      { x: rows },
      { key: key(seed) },
    )
    return recordsOf(fitted.exposition, 'mixtureSteps')[0]
  }, [components, type, seed])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={3}>
        <NumberField label="components" value={components} onChange={setComponents} min={1} max={8} integer />
        <Choice
          label="covariance"
          value={type}
          onChange={setType}
          options={['full', 'tied', 'diag', 'spherical'] as CovarianceType[]}
        />
        <NumberField label="seed" value={seed} onChange={setSeed} min={0} max={99} integer />
      </ControlPanel>
      {steps !== undefined && <MixturePlayer record={steps} />}
    </div>
  )
}
// endregion
