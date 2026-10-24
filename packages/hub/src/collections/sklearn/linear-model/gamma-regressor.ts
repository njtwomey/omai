import { gammaRegressor as gammaRegressorMethod } from 'whitebox-engine/methods/linear'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** Gamma regression: a log-linear model for positive, right-skewed targets. */
export const gammaRegressor: HubEntry = defineEntry({
  id: 'sklearn/linear-model/gamma-regressor',
  title: 'Gamma regression',
  summary: 'A log-linear model for positive, right-skewed targets.',
  collection: 'sklearn/linear-model',
  tags: ['regression', 'glm'],
  datasets: ['regressionTabular'],
  ...fitEstimator(gammaRegressorMethod()),
  views: [{ record: 'coefficientTable', title: 'Coefficients' }],
  methods: [{ file: 'methods/linear/glm.ts', name: 'gammaRegressor' }],
})
