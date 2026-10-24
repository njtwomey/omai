import { lasso as lassoEstimator } from 'whitebox-engine/methods/linear'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** The lasso (sklearn.linear_model.Lasso): coordinate descent, its regularisation path and the chosen penalty. */
export const lasso: HubEntry = defineEntry({
  id: 'sklearn/linear-model/lasso',
  title: 'Lasso',
  summary: 'L1-penalised least squares: coefficients enter one by one as the penalty falls.',
  collection: 'sklearn/linear-model',
  tags: ['regression', 'linear', 'sparsity', 'regularisation'],
  datasets: ['regressionTabular', 'regression1d'],
  ...fitEstimator(lassoEstimator({ alpha: 0.1 })),
  views: [
    { record: 'regularisationPath', title: 'Regularisation path' },
    { record: 'coefficientTable', title: 'Coefficients', area: 'side' },
  ],
  methods: [{ file: 'methods/linear/regression.ts', name: 'lasso' }],
})
