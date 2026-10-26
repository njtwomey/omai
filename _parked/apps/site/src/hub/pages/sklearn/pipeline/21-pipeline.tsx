import { generate } from 'mdml-engine/data'
import { moons } from 'mdml-engine/datasets'
import { recordsOf } from 'mdml-engine/exposition'
import { type NamedStep, pipeline } from 'mdml-engine/learn'
import { polynomialFeatures, standardScaler } from 'mdml-engine/learn/preprocessing'
import { pca } from 'mdml-engine/methods/decomposition'
import { logisticRegression } from 'mdml-engine/methods/linear'
import { key } from 'mdml-engine/random'
import { ControlPanel, NumberField, PipelineView, Toggle } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../../../gallery/recipe.ts'

export const recipe: Recipe = {
  title: 'Pipeline',
  question: 'How do I show what each stage of a pipeline did to the data?',
  explain:
    'A fitted `pipeline` returns a `pipelineStages` record in its `exposition`: per stage, the shape it takes and gives, its output rows (up to 500) with their classes and each column’s mean, standard deviation, minimum and maximum, and its fitted parameters; on the final estimator, the whole pipeline’s `decisionSurface` over the original inputs with its training loss. `PipelineView` draws a card per stage and the selected stage’s output in the form that suits it. Here the two moons go through polynomial features (a table: too many columns to plot), standardisation (the same table, every standard deviation now 1, beside the bars of the scales before), PCA (the rows on the first two components with the variance each keeps) and a logistic regression: its probability over the two raw features with the 0.5 boundary, the misclassified rows ringed and the mean log loss. A log loss is a property of a labelled point, so there is no true loss surface away from the data; switched to “expected loss”, the panel shades the log loss the model itself expects at each point (the entropy of its predicted probability, highest along the boundary) and sizes each training row by its own log loss, so confident mistakes stand out. Raise the degree to bend the boundary round the moons; turn off standardisation to see PCA chase the large-scale cubic columns; lower C to flatten the probabilities.',
}

// region
const { data } = generate(moons, { knobs: { n: 200, noise: 0.2 }, key: key(3) })

/** The number of polynomial features of two inputs up to a degree (without the bias column). */
const featuresAt = (degree: number): number => ((degree + 1) * (degree + 2)) / 2 - 1

export default function PipelineRecipe(): ReactNode {
  const [degree, setDegree] = useState(3)
  const [standardise, setStandardise] = useState(true)
  const [components, setComponents] = useState(4)
  const [c, setC] = useState(1)
  const k = Math.min(components, featuresAt(degree))
  const record = useMemo(() => {
    const steps: NamedStep[] = [
      ['poly', polynomialFeatures({ degree, includeBias: false })],
      ...(standardise ? [['scale', standardScaler()] as const] : []),
      ...(k > 0 ? [['pca', pca({ nComponents: k })] as const] : []),
      ['logistic', logisticRegression({ C: c })],
    ]
    return recordsOf(pipeline(steps).fit(data).exposition, 'pipelineStages')[0]
  }, [degree, standardise, k, c])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <NumberField label="polynomial degree" value={degree} onChange={setDegree} min={1} max={6} integer />
        <Toggle label="standardise" value={standardise} onChange={setStandardise} />
        <NumberField
          label={`PCA components (0 for none, at most ${featuresAt(degree)})`}
          value={k}
          onChange={setComponents}
          min={0}
          max={featuresAt(degree)}
          integer
        />
        <NumberField label="C (inverse penalty)" value={c} onChange={setC} min={0.001} max={100} log />
      </ControlPanel>
      {record !== undefined && <PipelineView record={record} />}
    </div>
  )
}
// endregion
