import { linearRegression as linearRegressionMethod } from 'whitebox-engine/methods/linear'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** Linear regression: ordinary least squares with standard errors, t tests and intervals. */
export const linearRegression: HubEntry = defineEntry({
  id: 'sklearn/linear-model/linear-regression',
  title: 'Linear regression',
  summary: 'Ordinary least squares with standard errors, t tests and intervals.',
  collection: 'sklearn/linear-model',
  tags: ['regression', 'linear'],
  datasets: ['regressionTabular', 'regression1d'],
  ...fitEstimator(linearRegressionMethod()),
  views: [{ record: 'coefficientTable', title: 'Coefficients' }],
  methods: [{ file: 'methods/linear/regression.ts', name: 'linearRegression' }],
})
