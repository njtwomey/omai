import { generate } from 'mdml-engine/data'
import { moons } from 'mdml-engine/datasets'
import { type DecisionSurface, recordsOf } from 'mdml-engine/exposition'
import { logisticRegression, perceptron } from 'mdml-engine/methods/linear'
import { gaussianMixture } from 'mdml-engine/methods/mixture'
import { kNeighborsClassifier } from 'mdml-engine/methods/neighbours'
import { key } from 'mdml-engine/random'
import { Choice, ControlPanel, DecisionSurfaceView, NumberField } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Decision surface',
  question: 'How do I show what a model predicts over two features?',
  explain:
    'Methods with two features return a `decisionSurface` record: predictions and probabilities on a grid, the training rows, and, for iterative trainers, the surface at steps through training. `DecisionSurfaceView` shades each class’s region by the model’s confidence, draws the boundary and the 0.25 and 0.75 probability contours, and puts the training rows on top; when the record has steps it plays them beside the training objective (logistic regression’s IRLS iterations; the perceptron’s updates, ringing the row that caused each), opening on the fitted model. For a density (the Gaussian mixture) it draws the log density instead.',
}

// region
const { data } = generate(moons, { knobs: { n: 160, noise: 0.25 }, key: key(3) })

type Model = 'logistic' | 'perceptron' | 'k-NN' | 'mixture'

function surfaceOf(model: Model, setting: number): DecisionSurface | undefined {
  const fitted =
    model === 'logistic'
      ? logisticRegression({ C: setting }).fit(data)
      : model === 'perceptron'
        ? perceptron({ eta0: setting }).fit(data, { key: key(1) })
        : model === 'k-NN'
          ? kNeighborsClassifier({ nNeighbors: setting, explainRows: 0 }).fit(data)
          : gaussianMixture({ nComponents: setting }).fit(data, { key: key(0) })
  return recordsOf(fitted.exposition, 'decisionSurface')[0]
}

const SETTINGS: Record<Model, { label: string; min: number; max: number; log?: boolean }> = {
  logistic: { label: 'C (inverse penalty)', min: 0.001, max: 100, log: true },
  perceptron: { label: 'learning rate', min: 0.01, max: 10, log: true },
  'k-NN': { label: 'neighbours k', min: 1, max: 60 },
  mixture: { label: 'components', min: 1, max: 10 },
}

export default function DecisionSurfaceRecipe(): ReactNode {
  const [model, setModel] = useState<Model>('logistic')
  const [settings, setSettings] = useState(() => ({
    logistic: 1,
    perceptron: 1,
    'k-NN': 15,
    mixture: 4,
  }))
  const s = SETTINGS[model]
  const setting = settings[model]
  const surface = useMemo(() => surfaceOf(model, setting), [model, setting])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <Choice label="model" value={model} onChange={setModel} options={Object.keys(SETTINGS) as Model[]} />
        <NumberField
          label={s.label}
          value={setting}
          onChange={(v) => setSettings((old) => ({ ...old, [model]: v }))}
          min={s.min}
          max={s.max}
          {...(s.log === true ? { log: true } : { integer: true })}
        />
      </ControlPanel>
      {surface !== undefined && <DecisionSurfaceView record={surface} aspect="equal" />}
    </div>
  )
}
// endregion
