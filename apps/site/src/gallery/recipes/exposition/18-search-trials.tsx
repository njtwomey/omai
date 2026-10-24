import { type ReactNode, useMemo, useState } from 'react'
import { breastCancer } from 'whitebox-engine/datasets'
import { searchTrials } from 'whitebox-engine/exposition'
import { gridSearchCV, randomizedSearchCV } from 'whitebox-engine/learn/model-selection'
import { standardScaler } from 'whitebox-engine/learn/preprocessing'
import { kNeighborsClassifier } from 'whitebox-engine/methods/neighbours'
import { key } from 'whitebox-engine/random'
import { Choice, ControlPanel, NumberField, SearchTrialsView } from 'whitebox-render/react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Search trials',
  question: 'How do I show every trial of a hyperparameter search?',
  explain:
    'A `searchTrials` record holds every trial of a search with its value, its per-fold (or per-step) values and whether it completed. Here a cross-validated grid or random search over k-NN’s neighbours and Minkowski order becomes one; `SearchTrialsView` draws the values in order with the best so far, the folds faintly behind each trial, and the value against each parameter.',
}

// region
const raw = breastCancer()
const data = { ...raw, x: standardScaler().fit(raw).transform(raw.x) }
const neighbours = Array.from({ length: 30 }, (_, i) => i + 1)

type Method = 'grid' | 'random'

export default function SearchTrialsRecipe(): ReactNode {
  const [method, setMethod] = useState<Method>('random')
  const [budget, setBudget] = useState(20)
  const record = useMemo(() => {
    const est = kNeighborsClassifier({ explainRows: 0 })
    const space = { nNeighbors: neighbours, p: [1, 2] }
    const result =
      method === 'grid'
        ? gridSearchCV(est, { nNeighbors: neighbours.filter((_, i) => i < budget / 2), p: [1, 2] }, data, { cv: 5 })
        : randomizedSearchCV(est, space, data, { cv: 5, nIter: budget, key: key(1) })
    const mean = Object.values(result.meanTestScore)[0] ?? []
    const folds = Object.values(result.splitTestScores)[0] ?? []
    return searchTrials({
      method,
      parameters: ['nNeighbors', 'p'],
      trials: result.params.map((p, i) => ({
        params: { nNeighbors: Number(p['nNeighbors']), p: Number(p['p']) },
        value: mean[i] ?? Number.NaN,
        intermediate: [...(folds[i] ?? [])],
        state: 'complete' as const,
      })),
      higherIsBetter: true,
      best: result.bestIndex,
    })
  }, [method, budget])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <Choice label="search" value={method} onChange={setMethod} options={['grid', 'random'] as Method[]} />
        <NumberField label="trials" value={budget} onChange={setBudget} min={2} max={60} integer />
      </ControlPanel>
      <SearchTrialsView record={record} label="accuracy (5-fold)" />
    </div>
  )
}
// endregion
