import { orthogonalMatchingPursuit as orthogonalMatchingPursuitMethod } from 'whitebox-engine/methods/linear'
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
  ...fitEstimator(orthogonalMatchingPursuitMethod()),
  views: [
    { record: 'regularisationPath', title: 'Regularisation path' },
    { record: 'coefficientTable', title: 'Coefficients', area: 'side' },
  ],
  methods: [{ file: 'methods/linear/regression.ts', name: 'orthogonalMatchingPursuit' }],
})
