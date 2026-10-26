/**
 * Hub entry for sparse Gaussian process regression through a few inducing points (VFE, FITC and DTC), on
 * one-dimensional regression curves and tabular regression data.
 */

import { sparseGpRegressor as sparseGpRegressorMethod } from 'mdml-engine/methods/gp'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** Sparse Gaussian process: a Gaussian process through a few inducing points (VFE, FITC, DTC): the cost of a handful of basis functions. */
export const sparseGpRegressor: HubEntry = defineEntry({
  id: 'sklearn/gaussian-process/sparse-gp-regressor',
  title: 'Sparse Gaussian process',
  summary:
    'A Gaussian process through a few inducing points (VFE, FITC, DTC): the cost of a handful of basis functions.',
  collection: 'sklearn/gaussian-process',
  tags: ['regression', 'bayesian', 'kernels', 'scalable'],
  datasets: ['regression1d', 'regressionTabular'],
  maxRows: 200,
  ...fitEstimator(sparseGpRegressorMethod(), { surface: true }),
  views: [
    { record: 'posteriorGrid', title: 'Posterior' },
    { record: 'coefficientTable', title: 'Hyperparameters', area: 'side' },
  ],
  methods: [{ file: 'methods/gp/sparse.ts', name: 'sparseGpRegressor' }],
})
