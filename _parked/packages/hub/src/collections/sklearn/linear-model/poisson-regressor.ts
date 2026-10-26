/**
 * Hub entry for Poisson regression, a log-linear model for counts fitted by IRLS, on one-dimensional GLM responses and
 * tabular data.
 */

import { poissonRegressor as poissonRegressorMethod } from 'mdml-engine/methods/linear'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** Poisson regression: a log-linear model for counts, fitted by IRLS. */
export const poissonRegressor: HubEntry = defineEntry({
  id: 'sklearn/linear-model/poisson-regressor',
  title: 'Poisson regression',
  summary: 'A log-linear model for counts, fitted by IRLS.',
  collection: 'sklearn/linear-model',
  tags: ['regression', 'glm', 'counts'],
  datasets: ['glm1d', 'regressionTabular'],
  dataset: 'poissonCounts',
  ...fitEstimator(poissonRegressorMethod({ alpha: 0 }), { surface: true }),
  views: [
    { record: 'decisionSurface', title: 'Fitted mean' },
    { record: 'coefficientTable', title: 'Coefficients', area: 'side' },
  ],
  methods: [{ file: 'methods/linear/glm.ts', name: 'poissonRegressor' }],
})
