import { type ReactNode, useMemo, useState } from 'react'
import { generate } from 'whitebox-engine/data'
import { moons } from 'whitebox-engine/datasets'
import { recordsOf } from 'whitebox-engine/exposition'
import { kNeighborsClassifier } from 'whitebox-engine/methods/neighbours'
import { key } from 'whitebox-engine/random'
import { toArray } from 'whitebox-engine/tensor'
import { ControlPanel, NeighbourView, NumberField } from 'whitebox-render/react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Neighbours',
  question: 'How do I show which neighbours a k-nearest-neighbour model consulted?',
  explain:
    'k-NN records the neighbours of the first training rows (each left out of its own list) as a `neighbours` record: indices and distances, nearest first. `NeighbourView` joins the query to its neighbours, draws the radius that holds them, and charts their distances.',
}

// region
const { data } = generate(moons, { knobs: { n: 150, noise: 0.2 }, key: key(12) })
const rows = toArray(data.x) as number[][]
const labels = Array.from(data.y.data)

export default function NeighboursRecipe(): ReactNode {
  const [k, setK] = useState(7)
  const [query, setQuery] = useState(3)
  const record = useMemo(
    () => recordsOf(kNeighborsClassifier({ nNeighbors: k, explainRows: 40 }).fit(data).exposition, 'neighbours')[0],
    [k],
  )
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <NumberField label="neighbours k" value={k} onChange={setK} min={1} max={30} integer />
        <NumberField label="query row" value={query} onChange={setQuery} min={0} max={39} integer />
      </ControlPanel>
      {record !== undefined && <NeighbourView record={record} points={rows} labels={labels} query={query} />}
    </div>
  )
}
// endregion
