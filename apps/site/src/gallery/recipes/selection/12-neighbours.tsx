import { neighbourRecord } from 'mdml-engine/exposition'
import { ControlPanel, ExpositionView, NumberField, SeedInput } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'
import { blobs } from '../../toys/synthetic.ts'

export const recipe: Recipe = {
  title: 'Neighbours',
  question: 'How do I show which training rows a prediction leans on?',
  explain:
    'A `neighbours` record lists, for each query, the training rows nearest to it with their distances, plus the rows and the queries themselves. `NeighbourView` draws the rows with a query ringed and joined to each of its neighbours, a circle out to the farthest, and the distances as bars by neighbour rank, coloured by the neighbour’s class; a selector steps through the queries. Move the number of neighbours and the circle around a query grows to take in more rows, and a query between two blobs begins to draw from both.',
  shows: ['neighbours', 'NeighbourView'],
}

export default function NeighboursRecipe(): ReactNode {
  const [k, setK] = useState(5)
  const [seed, setSeed] = useState(1)
  const [spread, setSpread] = useState(0.8)
  const record = useMemo(() => {
    const { points, labels } = blobs(seed, 3, 25, spread)
    const queries = [
      [0, 0],
      [2.5, 1.5],
      [-3, -2],
    ]
    const order = queries.map((q) => {
      const d = points.map((p, i) => ({ i, d: Math.hypot((p[0] ?? 0) - (q[0] ?? 0), (p[1] ?? 0) - (q[1] ?? 0)) }))
      return d.sort((a, b) => a.d - b.d).slice(0, k)
    })
    return neighbourRecord({
      indices: order.map((o) => o.map((e) => e.i)),
      distances: order.map((o) => o.map((e) => e.d)),
      metric: 'euclidean',
      queries,
      rows: { points, features: ['x', 'y'], labels, classes: ['blob 0', 'blob 1', 'blob 2'] },
    })
  }, [k, seed, spread])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={3}>
        <NumberField label="neighbours k" value={k} onChange={setK} min={1} max={15} integer />
        <NumberField label="blob spread" value={spread} onChange={setSpread} min={0.3} max={1.6} step={0.1} />
        <SeedInput value={seed} onChange={setSeed} />
      </ControlPanel>
      <ExpositionView
        record={record}
        points={record.rows?.points ?? []}
        labels={record.rows?.labels ?? []}
        title="Three queries and their neighbours"
      />
    </div>
  )
}
