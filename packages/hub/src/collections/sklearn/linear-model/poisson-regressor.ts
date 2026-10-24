import { poissonRegressor as poissonRegressorMethod } from 'whitebox-engine/methods/linear'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** Poisson regression: a log-linear model for counts, fitted by IRLS. */
export const poissonRegressor: HubEntry = defineEntry({
  id: 'sklearn/linear-model/poisson-regressor',
  title: 'Poisson regression',
  summary: 'A log-linear model for counts, fitted by IRLS.',
  collection: 'sklearn/linear-model',
  tags: ['regression', 'glm', 'counts'],
  datasets: ['regressionTabular'],
  ...fitEstimator(poissonRegressorMethod()),
  views: [{ record: 'coefficientTable', title: 'Coefficients' }],
  methods: [{ file: 'methods/linear/glm.ts', name: 'poissonRegressor' }],
})
