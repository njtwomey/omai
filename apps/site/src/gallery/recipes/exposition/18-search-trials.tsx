import { breastCancer } from 'mdml-engine/datasets'
import { recordsOf } from 'mdml-engine/exposition'
import { gridSearchCV, randomizedSearchCV } from 'mdml-engine/learn/model-selection'
import { standardScaler } from 'mdml-engine/learn/preprocessing'
import { kNeighborsClassifier } from 'mdml-engine/methods/neighbours'
import { key } from 'mdml-engine/random'
import { Choice, ControlPanel, NumberField, SearchTrialsView } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Search trials',
  question: 'How do I show every trial of a hyperparameter search?',
  explain:
    'A `searchTrials` record holds every trial of a search with its value, its per-fold (or per-step) values and whether it completed. `gridSearchCV` and `randomizedSearchCV` return one in their `exposition` (with the scorer’s name), here over k-NN’s neighbours and Minkowski order; `SearchTrialsView` draws the values in order with the best so far, the folds faintly behind each trial, and the value against each parameter.',
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
    return recordsOf(result.exposition, 'searchTrials')[0]
  }, [method, budget])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <Choice label="search" value={method} onChange={setMethod} options={['grid', 'random'] as Method[]} />
        <NumberField label="trials" value={budget} onChange={setBudget} min={2} max={60} integer />
      </ControlPanel>
      {record !== undefined && <SearchTrialsView record={record} />}
    </div>
  )
}
// endregion
