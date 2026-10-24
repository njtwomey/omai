import { negativeBinomialRegressor as negativeBinomialRegressorMethod } from 'whitebox-engine/methods/linear'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** Negative binomial regression: a count model with more variance than the Poisson allows. */
export const negativeBinomialRegressor: HubEntry = defineEntry({
  id: 'sklearn/linear-model/negative-binomial-regressor',
  title: 'Negative binomial regression',
  summary: 'A count model with more variance than the Poisson allows.',
  collection: 'sklearn/linear-model',
  tags: ['regression', 'glm', 'counts'],
  datasets: ['regressionTabular'],
  ...fitEstimator(negativeBinomialRegressorMethod()),
  views: [{ record: 'coefficientTable', title: 'Coefficients' }],
  methods: [{ file: 'methods/linear/glm.ts', name: 'negativeBinomialRegressor' }],
})
