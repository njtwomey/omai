/**
 * Hub entry for the lasso solved by LARS, giving the whole lasso path exactly, on tabular regression data with a few
 * informative features and one-dimensional regression curves.
 */

import { lassoLars as lassoLarsMethod } from 'whitebox-engine/methods/linear'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** Lasso (LARS): the whole lasso path, exactly, by least angle regression with drops. */
export const lassoLars: HubEntry = defineEntry({
  id: 'sklearn/linear-model/lasso-lars',
  title: 'Lasso (LARS)',
  summary: 'The whole lasso path, exactly, by least angle regression with drops.',
  collection: 'sklearn/linear-model',
  tags: ['regression', 'linear', 'sparsity', 'paths'],
  datasets: ['regressionTabular', 'regression1d'],
  dataset: 'sparseUncorrelated',
  ...fitEstimator(lassoLarsMethod({ alpha: 0.1 })),
  views: [
    { record: 'regularisationPath', title: 'Lasso path' },
    { record: 'coefficientTable', title: 'Coefficients', area: 'side' },
  ],
  methods: [{ file: 'methods/linear/lars.ts', name: 'lassoLars' }],
})
