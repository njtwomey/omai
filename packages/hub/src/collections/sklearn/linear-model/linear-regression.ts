/**
 * Hub entry for ordinary least squares linear regression with standard errors, t tests and confidence intervals, on
 * tabular regression data with a few informative features and one-dimensional regression curves.
 */

import { linearRegression as linearRegressionMethod } from 'mdml-engine/methods/linear'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** Linear regression: ordinary least squares with standard errors, t tests and intervals. */
export const linearRegression: HubEntry = defineEntry({
  id: 'sklearn/linear-model/linear-regression',
  title: 'Linear regression',
  summary:
    'Ordinary least squares with standard errors and confidence intervals: the informative coefficients stand clear of zero, the noise ones straddle it.',
  collection: 'sklearn/linear-model',
  tags: ['regression', 'linear'],
  datasets: ['regressionTabular', 'regression1d'],
  dataset: 'sparseUncorrelated',
  ...fitEstimator(linearRegressionMethod()),
  views: [{ record: 'coefficientTable', title: 'Coefficients' }],
  methods: [{ file: 'methods/linear/regression.ts', name: 'linearRegression' }],
})
