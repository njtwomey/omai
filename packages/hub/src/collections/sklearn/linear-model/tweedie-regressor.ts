import { tweedieRegressor as tweedieRegressorMethod } from 'whitebox-engine/methods/linear'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** Tweedie regression: one GLM family spanning normal, Poisson, gamma and inverse Gaussian by its power. */
export const tweedieRegressor: HubEntry = defineEntry({
  id: 'sklearn/linear-model/tweedie-regressor',
  title: 'Tweedie regression',
  summary: 'One GLM family spanning normal, Poisson, gamma and inverse Gaussian by its power.',
  collection: 'sklearn/linear-model',
  tags: ['regression', 'glm'],
  datasets: ['glm1d', 'regressionTabular'],
  dataset: 'tweedieResponses',
  ...fitEstimator(tweedieRegressorMethod({ alpha: 0 }), { surface: true }),
  views: [
    { record: 'decisionSurface', title: 'Fitted mean' },
    { record: 'coefficientTable', title: 'Coefficients', area: 'side' },
  ],
  methods: [{ file: 'methods/linear/glm.ts', name: 'tweedieRegressor' }],
})
