import { sgdRegressor as sgdRegressorEstimator } from 'whitebox-engine/methods/linear'
import { defineEntry, type HubEntry } from '../../entry/entry.ts'
import { fitEstimator } from '../../entry/run.ts'

/** SGD regression (sklearn.linear_model.SGDRegressor). */
export const sgdRegressor: HubEntry = defineEntry({
  id: 'online/sgd-regressor',
  title: 'SGD regressor',
  summary: 'Least squares (or Huber, or epsilon-insensitive) regression learned one row at a time.',
  collection: 'online',
  tags: ['online learning', 'regression', 'linear', 'stochastic gradient descent'],
  datasets: ['regressionTabular', 'regression1d'],
  dataset: 'makeRegression',
  ...fitEstimator(sgdRegressorEstimator({ maxIter: 30 })),
  views: [
    { record: 'learningCurve', title: 'Progressive and training squared error' },
    { record: 'coefficientTable', title: 'Coefficients over the samples seen', area: 'side' },
  ],
  methods: [{ file: 'methods/linear/online.ts', name: 'sgdRegressor' }],
})
