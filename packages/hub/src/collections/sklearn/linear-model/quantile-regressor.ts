/**
 * Hub entry for quantile regression, linear fits to a conditional quantile by the pinball loss, on a line with outliers
 * and tabular regression data.
 */

import { quantileRegressor as quantileRegressorMethod } from 'mdml-engine/methods/linear'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** Quantile regression: linear fits to a conditional quantile by the pinball loss. */
export const quantileRegressor: HubEntry = defineEntry({
  id: 'sklearn/linear-model/quantile-regressor',
  title: 'Quantile regression',
  summary: 'Linear fits to a conditional quantile by the pinball loss.',
  collection: 'sklearn/linear-model',
  tags: ['regression', 'linear', 'quantiles'],
  datasets: ['regression1d', 'regressionTabular'],
  dataset: 'linearOutliers',
  ...fitEstimator(quantileRegressorMethod({ alpha: 0 }), { surface: true }),
  views: [
    { record: 'decisionSurface', title: 'Fitted function' },
    { record: 'coefficientTable', title: 'Coefficients', area: 'side' },
  ],
  methods: [{ file: 'methods/linear/robust.ts', name: 'quantileRegressor' }],
})
