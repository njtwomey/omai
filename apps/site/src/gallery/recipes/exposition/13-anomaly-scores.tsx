import { type ReactNode, useMemo, useState } from 'react'
import { generate } from 'whitebox-engine/data'
import { blobs } from 'whitebox-engine/datasets'
import { recordsOf } from 'whitebox-engine/exposition'
import { isolationForest, localOutlierFactor } from 'whitebox-engine/methods/anomaly'
import { key, uniform } from 'whitebox-engine/random'
import { toArray } from 'whitebox-engine/tensor'
import { AnomalyView, Choice, ControlPanel, NumberField } from 'whitebox-render/react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Anomaly scores',
  question: 'How do I show which rows a detector flags, and why?',
  explain:
    'Anomaly detectors return an `anomalyScores` record: a score per row, the threshold that flags outliers, and per-row evidence (an isolation forest’s mean path length, LOF’s local reachability density). `AnomalyView` draws the scores’ histogram with the threshold, the evidence against the score, and the rows coloured by score with the flagged ones ringed. Raise the contamination to flag more.',
}

// region
const inliers = toArray(
  generate(blobs, { knobs: { n: 200, centers: 2, clusterStd: 0.9 }, key: key(8) }).data.x,
) as number[][]
const outliers = toArray(uniform(key(9), [16, 2], { low: -12, high: 12 })) as number[][]
const rows = [...inliers, ...outliers]

type Method = 'isolation forest' | 'LOF'

export default function AnomalyRecipe(): ReactNode {
  const [method, setMethod] = useState<Method>('isolation forest')
  const [contamination, setContamination] = useState(0.08)
  const record = useMemo(() => {
    const est =
      method === 'isolation forest'
        ? isolationForest({ contamination, nEstimators: 100 })
        : localOutlierFactor({ contamination, nNeighbors: 20 })
    return recordsOf(est.fit({ x: rows }, { key: key(0) }).exposition, 'anomalyScores')[0]
  }, [method, contamination])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <Choice
          label="detector"
          value={method}
          onChange={setMethod}
          options={['isolation forest', 'LOF'] as Method[]}
        />
        <NumberField label="contamination" value={contamination} onChange={setContamination} min={0.01} max={0.3} />
      </ControlPanel>
      {record !== undefined && <AnomalyView record={record} />}
    </div>
  )
}
// endregion
