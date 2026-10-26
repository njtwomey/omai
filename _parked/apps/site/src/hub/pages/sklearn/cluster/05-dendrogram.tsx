import { generate } from 'mdml-engine/data'
import { blobs } from 'mdml-engine/datasets'
import { recordsOf } from 'mdml-engine/exposition'
import { agglomerativeClustering, type Linkage } from 'mdml-engine/methods/cluster'
import { key } from 'mdml-engine/random'
import { toArray } from 'mdml-engine/tensor'
import { Choice, ControlPanel, DendrogramView, NumberField } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../../../gallery/recipe.ts'

export const recipe: Recipe = {
  title: 'Dendrogram',
  question: 'How do I show a hierarchical clustering and where it is cut?',
  explain:
    'Agglomerative clustering records its merges (scipy’s linkage rows) in its `clusteringSteps` record. `DendrogramView` draws them as a dendrogram cut into the asked number of clusters, links below the cut in their cluster’s colour, beside the rows coloured the same way. Single linkage chains; Ward keeps clusters compact.',
}

// region
const rows = toArray(
  generate(blobs, { knobs: { n: 60, centers: 4, clusterStd: 1.3 }, key: key(11) }).data.x,
) as number[][]

export default function DendrogramRecipe(): ReactNode {
  const [linkage, setLinkage] = useState<Linkage>('ward')
  const [clusters, setClusters] = useState(4)
  const steps = useMemo(
    () =>
      recordsOf(agglomerativeClustering({ linkage, nClusters: 2 }).fit({ x: rows }).exposition, 'clusteringSteps')[0],
    [linkage],
  )
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <Choice
          label="linkage"
          value={linkage}
          onChange={setLinkage}
          options={['ward', 'single', 'complete', 'average'] as Linkage[]}
        />
        <NumberField label="clusters" value={clusters} onChange={setClusters} min={1} max={12} integer />
      </ControlPanel>
      {steps !== undefined && <DendrogramView record={steps} clusters={clusters} />}
    </div>
  )
}
// endregion
