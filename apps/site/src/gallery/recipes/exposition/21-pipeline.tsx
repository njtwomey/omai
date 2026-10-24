import { type ReactNode, useMemo, useState } from 'react'
import { wine } from 'whitebox-engine/datasets'
import { pipelineStages } from 'whitebox-engine/exposition'
import { standardScaler } from 'whitebox-engine/learn/preprocessing'
import { pca } from 'whitebox-engine/methods/decomposition'
import { logisticRegression } from 'whitebox-engine/methods/linear'
import { toArray } from 'whitebox-engine/tensor'
import { ControlPanel, NumberField, PipelineView } from 'whitebox-render/react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Pipeline',
  question: 'How do I show what each stage of a pipeline did to the data?',
  explain:
    'A `pipelineStages` record holds, per stage, the shape it takes and gives, what it learned and a sample of its output. Here a scaler, PCA and a logistic regression are fitted in turn on the wines and recorded; `PipelineView` draws one card per stage. Fewer components squeeze 13 columns into fewer before the classifier sees them.',
}

// region
const data = wine()
const sample = (x: unknown): number[][] => (toArray(x as never) as number[][]).slice(0, 4)

export default function PipelineRecipe(): ReactNode {
  const [k, setK] = useState(3)
  const record = useMemo(() => {
    const scaler = standardScaler().fit(data)
    const scaled = scaler.transform(data.x)
    const reducer = pca({ nComponents: k }).fit({ ...data, x: scaled })
    const reduced = reducer.transform(scaled)
    const model = logisticRegression({ C: 1 }).fit({ ...data, x: reduced })
    const n = data.x.shape[0] ?? 0
    const d = data.featureNames.length
    const pcs = Array.from({ length: k }, (_, i) => `pc${i + 1}`)
    return pipelineStages({
      stages: [
        {
          name: 'scale',
          kind: 'transformer',
          inputColumns: data.featureNames,
          outputColumns: data.featureNames,
          inputSample: sample(data.x),
          outputSample: sample(scaled),
          inputShape: [n, d],
          outputShape: [n, d],
          learned: [
            { name: 'mean', value: scaler.params.mean },
            { name: 'scale', value: scaler.params.scale },
          ],
        },
        {
          name: 'pca',
          kind: 'transformer',
          inputColumns: data.featureNames,
          outputColumns: pcs,
          inputSample: sample(scaled),
          outputSample: sample(reduced),
          inputShape: [n, d],
          outputShape: [n, k],
          learned: [
            { name: 'variance kept', value: reducer.params.explainedVarianceRatio.reduce((a, b) => a + b, 0) },
            { name: 'ratios', value: reducer.params.explainedVarianceRatio },
          ],
        },
        {
          name: 'logistic',
          kind: 'estimator',
          inputColumns: pcs,
          outputColumns: data.classes.map((c) => `p(${c})`),
          inputSample: sample(reduced),
          outputSample: sample(model.predictProba?.(reduced)),
          inputShape: [n, k],
          outputShape: [n, data.classes.length],
          learned: [
            { name: 'accuracy', value: model.score?.({ ...data, x: reduced }) ?? Number.NaN },
            ...model.params.coef.map((c, i) => ({ name: `coef ${data.classes[i] ?? i}`, value: c })),
          ],
        },
      ],
    })
  }, [k])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <NumberField label="components" value={k} onChange={setK} min={1} max={13} integer />
      </ControlPanel>
      <PipelineView record={record} />
    </div>
  )
}
// endregion
