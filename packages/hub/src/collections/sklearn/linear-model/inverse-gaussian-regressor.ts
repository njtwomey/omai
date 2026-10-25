/**
 * Hub entry for inverse Gaussian regression, a GLM for positive targets whose variance grows as the cube of the mean,
 * on one-dimensional GLM responses and tabular data.
 */

import { inverseGaussianRegressor as inverseGaussianRegressorMethod } from 'whitebox-engine/methods/linear'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** Inverse Gaussian regression: a GLM for positive targets whose variance grows as the cube of the mean. */
export const inverseGaussianRegressor: HubEntry = defineEntry({
  id: 'sklearn/linear-model/inverse-gaussian-regressor',
  title: 'Inverse Gaussian regression',
  summary: 'A GLM for positive targets whose variance grows as the cube of the mean.',
  collection: 'sklearn/linear-model',
  tags: ['regression', 'glm'],
  datasets: ['glm1d', 'regressionTabular'],
  dataset: 'inverseGaussianResponses',
  ...fitEstimator(inverseGaussianRegressorMethod({ alpha: 0 }), { surface: true }),
  views: [
    { record: 'decisionSurface', title: 'Fitted mean' },
    { record: 'coefficientTable', title: 'Coefficients', area: 'side' },
  ],
  methods: [{ file: 'methods/linear/glm.ts', name: 'inverseGaussianRegressor' }],
})
