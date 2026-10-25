import { wine } from 'mdml-engine/datasets'
import { recordsOf } from 'mdml-engine/exposition'
import { standardScaler } from 'mdml-engine/learn/preprocessing'
import { pca } from 'mdml-engine/methods/decomposition'
import { ComponentsView, ControlPanel, NumberField, Toggle } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Components',
  question: 'How do I show what a decomposition found: scores, loadings and variance?',
  explain:
    'PCA (and factor analysis, ICA, NMF, PLS) returns a `components` record. `ComponentsView` draws the biplot (the wines in the plane of two components, coloured by cultivar, each feature’s loading an arrow), the scree with the cumulative share of variance, and the loadings. Without standardising, the largest-scaled feature (proline) takes the first component alone.',
}

// region
const raw = wine()
const scaled = { ...raw, x: standardScaler().fit(raw).transform(raw.x) }
const labels = Array.from(raw.y.data)

export default function ComponentsRecipe(): ReactNode {
  const [standardise, setStandardise] = useState(true)
  const [second, setSecond] = useState(2)
  const record = useMemo(
    () => recordsOf(pca({ nComponents: 6 }).fit(standardise ? scaled : raw).exposition, 'components')[0],
    [standardise],
  )
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <Toggle label="standardise features" value={standardise} onChange={setStandardise} />
        <NumberField label="biplot’s second component" value={second} onChange={setSecond} min={2} max={6} integer />
      </ControlPanel>
      {record !== undefined && <ComponentsView record={record} labels={labels} pair={[0, second - 1]} />}
    </div>
  )
}
// endregion
