import { type ReactNode, useMemo, useState } from 'react'
import { generate } from 'whitebox-engine/data'
import { moons } from 'whitebox-engine/datasets'
import { type DecisionSurface, recordsOf } from 'whitebox-engine/exposition'
import { logisticRegression } from 'whitebox-engine/methods/linear'
import { gaussianMixture } from 'whitebox-engine/methods/mixture'
import { kNeighborsClassifier } from 'whitebox-engine/methods/neighbours'
import { key } from 'whitebox-engine/random'
import { toArray } from 'whitebox-engine/tensor'
import { Choice, ControlPanel, DecisionSurfaceView, NumberField } from 'whitebox-render/react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Decision surface',
  question: 'How do I show what a model predicts over two features?',
  explain:
    'Methods with two features return a `decisionSurface` record: predictions and probabilities on a grid. `DecisionSurfaceView` shades each class’s region by the model’s confidence, draws the boundary and the 0.25 and 0.75 probability contours, and puts the training rows on top; for a density (the Gaussian mixture) it draws the log density instead.',
}

// region
const { data } = generate(moons, { knobs: { n: 160, noise: 0.25 }, key: key(3) })
const rows = toArray(data.x) as number[][]
const labels = Array.from(data.y.data)

type Model = 'logistic' | 'k-NN' | 'mixture'

function surfaceOf(model: Model, setting: number): DecisionSurface | undefined {
  const fitted =
    model === 'logistic'
      ? logisticRegression({ C: setting }).fit(data)
      : model === 'k-NN'
        ? kNeighborsClassifier({ nNeighbors: setting, explainRows: 0 }).fit(data)
        : gaussianMixture({ nComponents: setting }).fit(data, { key: key(0) })
  return recordsOf(fitted.exposition, 'decisionSurface')[0]
}

const SETTINGS: Record<Model, { label: string; min: number; max: number; log?: boolean }> = {
  logistic: { label: 'C (inverse penalty)', min: 0.001, max: 100, log: true },
  'k-NN': { label: 'neighbours k', min: 1, max: 60 },
  mixture: { label: 'components', min: 1, max: 10 },
}

export default function DecisionSurfaceRecipe(): ReactNode {
  const [model, setModel] = useState<Model>('k-NN')
  const [settings, setSettings] = useState(() => ({
    logistic: 1,
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
      {surface !== undefined && <DecisionSurfaceView record={surface} points={rows} labels={labels} aspect="equal" />}
    </div>
  )
}
// endregion
