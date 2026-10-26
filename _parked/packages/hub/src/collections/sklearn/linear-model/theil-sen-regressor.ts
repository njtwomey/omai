/**
 * Hub entry for the Theil-Sen regressor, the spatial median of the fits to many subsets, on a line with outliers and
 * tabular regression data.
 */

import { theilSenRegressor as theilSenRegressorMethod } from 'mdml-engine/methods/linear'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** Theil–Sen: the spatial median of the fits to many subsets: robust slopes. */
export const theilSenRegressor: HubEntry = defineEntry({
  id: 'sklearn/linear-model/theil-sen-regressor',
  title: 'Theil–Sen',
  summary: 'The spatial median of the fits to many subsets: robust slopes.',
  collection: 'sklearn/linear-model',
  tags: ['regression', 'linear', 'robust'],
  datasets: ['regression1d', 'regressionTabular'],
  dataset: 'linearOutliers',
  ...fitEstimator(theilSenRegressorMethod(), { surface: true }),
  views: [
    { record: 'decisionSurface', title: 'Fitted function' },
    { record: 'coefficientTable', title: 'Coefficients', area: 'side' },
  ],
  methods: [{ file: 'methods/linear/robust.ts', name: 'theilSenRegressor' }],
})
