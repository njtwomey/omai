import { type ReactNode, useMemo, useState } from 'react'
import { generate } from 'whitebox-engine/data'
import { makeClassification } from 'whitebox-engine/datasets'
import { recordsOf } from 'whitebox-engine/exposition'
import type { Estimator } from 'whitebox-engine/learn'
import { randomForestClassifier } from 'whitebox-engine/methods/ensemble'
import { logisticRegression } from 'whitebox-engine/methods/linear'
import { gaussianNB } from 'whitebox-engine/methods/naive-bayes'
import { calibrateClassifier } from 'whitebox-engine/methods/trust'
import { key } from 'whitebox-engine/random'
import { CalibrationView, Choice, ControlPanel, NumberField, Toggle } from 'whitebox-render/react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Reliability diagram',
  question: 'How do I show whether a classifier’s probabilities can be trusted, and repair them?',
  explain:
    'A `calibration` record holds reliability curves: forecasts binned, each bin’s observed frequency against its mean forecast, with the expected (ECE) and maximum calibration error. `calibrateClassifier(model, data)` fits Platt scaling, isotonic regression and temperature scaling on half the rows and measures every curve on the other half; `CalibrationView` draws them against the diagonal of perfect calibration, with the forecasts per bin beneath. Naive Bayes with redundant features is overconfident (its curve is an S); logistic regression is close to calibrated already; a forest is underconfident near 0 and 1. Equal-mass bins put the same number of forecasts in each.',
}

type Model = 'naive Bayes' | 'logistic regression' | 'random forest'
const MODELS: Readonly<Record<Model, () => Estimator>> = {
  'naive Bayes': () => gaussianNB(),
  'logistic regression': () => logisticRegression(),
  'random forest': () => randomForestClassifier({ nEstimators: 30, maxDepth: 6 }),
}

export default function ReliabilityRecipe(): ReactNode {
  const [model, setModel] = useState<Model>('naive Bayes')
  const [nBins, setNBins] = useState(10)
  const [quantile, setQuantile] = useState(false)
  const data = useMemo(
    () =>
      generate(makeClassification, {
        key: key(4),
        knobs: { n: 2000, nFeatures: 12, nInformative: 3, nRedundant: 6, classSep: 0.8 },
      }).data,
    [],
  )
  const fitted = useMemo(() => MODELS[model]().fit(data, { key: key(1) }), [model, data])
  const record = useMemo(
    () =>
      recordsOf(
        calibrateClassifier(fitted, data, { nBins, strategy: quantile ? 'quantile' : 'uniform', key: key(2) })
          .exposition,
        'calibration',
      )[0],
    [fitted, data, nBins, quantile],
  )
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={3}>
        <Choice
          label="model"
          value={model}
          onChange={setModel}
          options={['naive Bayes', 'logistic regression', 'random forest'] as Model[]}
        />
        <NumberField label="bins" value={nBins} onChange={setNBins} min={3} max={30} step={1} />
        <Toggle label="equal-mass bins" value={quantile} onChange={setQuantile} />
      </ControlPanel>
      {record !== undefined && <CalibrationView record={record} />}
    </div>
  )
}
