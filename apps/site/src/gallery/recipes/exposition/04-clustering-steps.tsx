import { type ReactNode, useMemo, useState } from 'react'
import { generate } from 'whitebox-engine/data'
import { blobs, moons } from 'whitebox-engine/datasets'
import { recordsOf } from 'whitebox-engine/exposition'
import { dbscan, kmeans } from 'whitebox-engine/methods/cluster'
import { key } from 'whitebox-engine/random'
import { toArray } from 'whitebox-engine/tensor'
import { Choice, ClusteringPlayer, ControlPanel, NumberField } from 'whitebox-render/react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Clustering steps',
  question: 'How do I play back how a clustering method assigned the rows?',
  explain:
    'Clustering methods return a `clusteringSteps` record: the assignment after every iteration, the rows clustered, and the centres and objective for k-means. `ClusteringPlayer` draws from the record alone and plays it: the centres walk to their clusters, their nearest-centre regions shifting, while the inertia falls. DBSCAN’s steps grow each cluster from a core row; core rows are large, border rows small, noise a grey cross.',
}

// region
const blobRows = toArray(
  generate(blobs, { knobs: { n: 240, centers: 4, clusterStd: 1.6 }, key: key(7) }).data.x,
) as number[][]
const moonRows = toArray(generate(moons, { knobs: { n: 240, noise: 0.09 }, key: key(2) }).data.x) as number[][]

type Method = 'k-means' | 'DBSCAN'

export default function ClusteringStepsRecipe(): ReactNode {
  const [method, setMethod] = useState<Method>('k-means')
  const [k, setK] = useState(4)
  const [seed, setSeed] = useState(0)
  const [eps, setEps] = useState(0.18)
  const steps = useMemo(() => {
    const fitted =
      method === 'k-means'
        ? kmeans({ nClusters: k, init: 'random' }).fit({ x: blobRows }, { key: key(seed) })
        : dbscan({ eps, minSamples: 5 }).fit({ x: moonRows })
    return recordsOf(fitted.exposition, 'clusteringSteps')[0]
  }, [method, k, seed, eps])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={3}>
        <Choice label="method" value={method} onChange={setMethod} options={['k-means', 'DBSCAN']} />
        {method === 'k-means' ? (
          <>
            <NumberField label="clusters k" value={k} onChange={setK} min={1} max={10} integer />
            <NumberField label="seed" value={seed} onChange={setSeed} min={0} max={99} integer />
          </>
        ) : (
          <NumberField label="radius ε" value={eps} onChange={setEps} min={0.03} max={1} log />
        )}
      </ControlPanel>
      {steps !== undefined && <ClusteringPlayer record={steps} />}
    </div>
  )
}
// endregion
