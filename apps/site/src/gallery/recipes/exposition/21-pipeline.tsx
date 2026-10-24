import { type ReactNode, useMemo, useState } from 'react'
import { wine } from 'whitebox-engine/datasets'
import { recordsOf } from 'whitebox-engine/exposition'
import { pipeline } from 'whitebox-engine/learn'
import { standardScaler } from 'whitebox-engine/learn/preprocessing'
import { pca } from 'whitebox-engine/methods/decomposition'
import { logisticRegression } from 'whitebox-engine/methods/linear'
import { ControlPanel, NumberField, PipelineView } from 'whitebox-render/react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Pipeline',
  question: 'How do I show what each stage of a pipeline did to the data?',
  explain:
    'A fitted `pipeline` returns a `pipelineStages` record in its `exposition`: per stage, the shape it takes and gives, what it learned and the first rows of its output (then the stages’ own records). Here a scaler, PCA and a logistic regression on the wines; `PipelineView` draws one card per stage. Fewer components squeeze 13 columns into fewer before the classifier sees them.',
}

// region
const data = wine()

export default function PipelineRecipe(): ReactNode {
  const [k, setK] = useState(3)
  const record = useMemo(() => {
    const fitted = pipeline([
      ['scale', standardScaler()],
      ['pca', pca({ nComponents: k })],
      ['logistic', logisticRegression({ C: 1 })],
    ]).fit(data)
    return recordsOf(fitted.exposition, 'pipelineStages')[0]
  }, [k])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <NumberField label="components" value={k} onChange={setK} min={1} max={13} integer />
      </ControlPanel>
      {record !== undefined && <PipelineView record={record} />}
    </div>
  )
}
// endregion
