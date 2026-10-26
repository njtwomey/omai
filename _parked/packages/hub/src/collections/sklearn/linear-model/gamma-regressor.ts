/**
 * Hub entry for gamma regression, a log-linear GLM for positive, right-skewed targets, on one-dimensional GLM responses
 * and tabular data.
 */

import { gammaRegressor as gammaRegressorMethod } from 'mdml-engine/methods/linear'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** Gamma regression: a log-linear model for positive, right-skewed targets. */
export const gammaRegressor: HubEntry = defineEntry({
  id: 'sklearn/linear-model/gamma-regressor',
  title: 'Gamma regression',
  summary: 'A log-linear model for positive, right-skewed targets.',
  collection: 'sklearn/linear-model',
  tags: ['regression', 'glm'],
  datasets: ['glm1d', 'regressionTabular'],
  dataset: 'gammaResponses',
  ...fitEstimator(gammaRegressorMethod({ alpha: 0 }), { surface: true }),
  views: [
    { record: 'decisionSurface', title: 'Fitted mean' },
    { record: 'coefficientTable', title: 'Coefficients', area: 'side' },
  ],
  methods: [{ file: 'methods/linear/glm.ts', name: 'gammaRegressor' }],
})
