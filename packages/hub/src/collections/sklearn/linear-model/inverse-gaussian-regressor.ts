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
  datasets: ['regressionTabular'],
  ...fitEstimator(inverseGaussianRegressorMethod()),
  views: [{ record: 'coefficientTable', title: 'Coefficients' }],
  methods: [{ file: 'methods/linear/glm.ts', name: 'inverseGaussianRegressor' }],
})
