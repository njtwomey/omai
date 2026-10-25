/**
 * Hub entry for orthogonal matching pursuit, greedy sparse regression that adds the feature most correlated with the
 * residual and refits, on tabular regression data with a few informative features and one-dimensional regression
 * curves.
 */

import { orthogonalMatchingPursuit as orthogonalMatchingPursuitMethod } from 'mdml-engine/methods/linear'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** Orthogonal matching pursuit: greedy sparse regression: add the feature most correlated with the residual, refit, repeat. */
export const orthogonalMatchingPursuit: HubEntry = defineEntry({
  id: 'sklearn/linear-model/orthogonal-matching-pursuit',
  title: 'Orthogonal matching pursuit',
  summary: 'Greedy sparse regression: add the feature most correlated with the residual, refit, repeat.',
  collection: 'sklearn/linear-model',
  tags: ['regression', 'linear', 'sparsity', 'greedy'],
  datasets: ['regressionTabular', 'regression1d'],
  dataset: 'sparseUncorrelated',
  ...fitEstimator(orthogonalMatchingPursuitMethod({ nNonzeroCoefs: 6 })),
  views: [
    { record: 'regularisationPath', title: 'Regularisation path' },
    { record: 'coefficientTable', title: 'Coefficients', area: 'side' },
  ],
  methods: [{ file: 'methods/linear/regression.ts', name: 'orthogonalMatchingPursuit' }],
})
