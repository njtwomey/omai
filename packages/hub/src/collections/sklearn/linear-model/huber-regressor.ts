/**
 * Hub entry for Huber regression, squared loss near the fit and absolute loss far from it so outliers count less, on a
 * line with outliers and tabular regression data.
 */

import { huberRegressor as huberRegressorMethod } from 'whitebox-engine/methods/linear'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** Huber regression: squared loss near the fit, absolute loss far from it: robust to outliers. */
export const huberRegressor: HubEntry = defineEntry({
  id: 'sklearn/linear-model/huber-regressor',
  title: 'Huber regression',
  summary: 'Squared loss near the fit, absolute loss far from it: robust to outliers.',
  collection: 'sklearn/linear-model',
  tags: ['regression', 'linear', 'robust'],
  datasets: ['regression1d', 'regressionTabular'],
  dataset: 'linearOutliers',
  ...fitEstimator(huberRegressorMethod(), { surface: true }),
  views: [
    { record: 'decisionSurface', title: 'Fitted function' },
    { record: 'anomalyScores', title: 'Anomaly scores', area: 'side' },
    { record: 'coefficientTable', title: 'Coefficients', area: 'side' },
  ],
  methods: [{ file: 'methods/linear/robust.ts', name: 'huberRegressor' }],
})
