/**
 * Hub entry for least angle regression (LARS), moving the coefficients along the direction equiangular to the most
 * correlated features, on tabular regression data with a few informative features and one-dimensional regression
 * curves.
 */

import { lars as larsMethod } from 'mdml-engine/methods/linear'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** Least angle regression: move the coefficients along the direction equiangular to the most correlated features. */
export const lars: HubEntry = defineEntry({
  id: 'sklearn/linear-model/lars',
  title: 'Least angle regression',
  summary: 'Move the coefficients along the direction equiangular to the most correlated features.',
  collection: 'sklearn/linear-model',
  tags: ['regression', 'linear', 'sparsity', 'paths'],
  datasets: ['regressionTabular', 'regression1d'],
  dataset: 'sparseUncorrelated',
  ...fitEstimator(larsMethod()),
  views: [
    { record: 'regularisationPath', title: 'LARS path' },
    { record: 'coefficientTable', title: 'Coefficients', area: 'side' },
  ],
  methods: [{ file: 'methods/linear/lars.ts', name: 'lars' }],
})
