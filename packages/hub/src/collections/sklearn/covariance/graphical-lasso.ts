/**
 * Hub entry for the graphical lasso, a sparse inverse covariance showing which features are conditionally independent,
 * on tabular classification data and tabular regression data.
 */

import { graphicalLassoEstimator } from 'whitebox-engine/methods/covariance'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** Graphical lasso: a sparse inverse covariance: which features are conditionally independent. */
export const graphicalLasso: HubEntry = defineEntry({
  id: 'sklearn/covariance/graphical-lasso',
  title: 'Graphical lasso',
  summary: 'A sparse inverse covariance: which features are conditionally independent.',
  collection: 'sklearn/covariance',
  tags: ['covariance', 'sparsity', 'graphs'],
  datasets: ['classificationTabular', 'regressionTabular'],
  ...fitEstimator(graphicalLassoEstimator(), { unsupervised: true }),
  views: [
    { record: 'regularisationPath', title: 'Regularisation path' },
    { record: 'anomalyScores', title: 'Anomaly scores', area: 'side' },
  ],
  methods: [{ file: 'methods/covariance/graphical-lasso.ts', name: 'graphicalLassoEstimator' }],
})
