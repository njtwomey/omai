/**
 * Hub entry for kernel ridge regression, ridge regression in a kernel's feature space (the Gaussian process posterior
 * mean), on one-dimensional regression curves and tabular regression data.
 */

import { kernelRidge as kernelRidgeMethod } from 'whitebox-engine/methods/kernel-methods'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** Kernel ridge regression: ridge regression in a kernel’s feature space: the Gaussian process posterior mean. */
export const kernelRidge: HubEntry = defineEntry({
  id: 'sklearn/kernel-ridge/kernel-ridge',
  title: 'Kernel ridge regression',
  summary: 'Ridge regression in a kernel’s feature space: the Gaussian process posterior mean.',
  collection: 'sklearn/kernel-ridge',
  tags: ['regression', 'kernels'],
  datasets: ['regression1d', 'regressionTabular'],
  maxRows: 200,
  ...fitEstimator(kernelRidgeMethod({ kernel: 'rbf', gamma: 10, alpha: 0.1 }), { surface: true }),
  views: [
    { record: 'posteriorGrid', title: 'Fit' },
    { record: 'crossValidationCurve', title: 'Cross-validation', area: 'side' },
  ],
  methods: [{ file: 'methods/kernel-methods/kernel-ridge.ts', name: 'kernelRidge' }],
})
