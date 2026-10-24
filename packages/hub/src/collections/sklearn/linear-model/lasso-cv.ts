import { lassoCV } from 'whitebox-engine/methods/linear'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** Lasso (cross-validated): the lasso with its penalty chosen by k-fold cross-validation along the path. */
export const lassoCv: HubEntry = defineEntry({
  id: 'sklearn/linear-model/lasso-cv',
  title: 'Lasso (cross-validated)',
  summary: 'The lasso with its penalty chosen by k-fold cross-validation along the path.',
  collection: 'sklearn/linear-model',
  tags: ['regression', 'linear', 'sparsity', 'cross-validation'],
  datasets: ['regressionTabular', 'regression1d'],
  ...fitEstimator(lassoCV()),
  views: [
    { record: 'regularisationPath', title: 'Regularisation path' },
    { record: 'crossValidationCurve', title: 'Cross-validation', area: 'side' },
    { record: 'coefficientTable', title: 'Coefficients', area: 'side' },
  ],
  methods: [{ file: 'methods/linear/regression.ts', name: 'lassoCV' }],
})
