/**
 * Hub entry for negative binomial regression, a count model with more variance than the Poisson allows, on
 * overdispersed counts and tabular data.
 */

import { negativeBinomialRegressor as negativeBinomialRegressorMethod } from 'mdml-engine/methods/linear'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** Negative binomial regression: a count model with more variance than the Poisson allows. */
export const negativeBinomialRegressor: HubEntry = defineEntry({
  id: 'sklearn/linear-model/negative-binomial-regressor',
  title: 'Negative binomial regression',
  summary: 'A count model with more variance than the Poisson allows.',
  collection: 'sklearn/linear-model',
  tags: ['regression', 'glm', 'counts'],
  datasets: ['glm1d', 'regressionTabular'],
  dataset: 'overdispersedCounts',
  ...fitEstimator(negativeBinomialRegressorMethod({ alpha: 0 }), { surface: true }),
  views: [
    { record: 'decisionSurface', title: 'Fitted mean' },
    { record: 'coefficientTable', title: 'Coefficients', area: 'side' },
  ],
  methods: [{ file: 'methods/linear/glm.ts', name: 'negativeBinomialRegressor' }],
})
