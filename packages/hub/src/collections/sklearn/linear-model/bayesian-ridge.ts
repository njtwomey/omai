import { bayesianRidge as bayesianRidgeMethod } from 'whitebox-engine/methods/linear'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** Bayesian ridge: ridge whose penalty and noise level are learned by evidence maximisation. */
export const bayesianRidge: HubEntry = defineEntry({
  id: 'sklearn/linear-model/bayesian-ridge',
  title: 'Bayesian ridge',
  summary: 'Ridge whose penalty and noise level are learned by evidence maximisation.',
  collection: 'sklearn/linear-model',
  tags: ['regression', 'linear', 'bayesian'],
  datasets: ['regressionTabular', 'regression1d'],
  ...fitEstimator(bayesianRidgeMethod()),
  views: [
    { record: 'hyperparameterSurface', title: 'Evidence surface' },
    { record: 'coefficientTable', title: 'Coefficients', area: 'side' },
  ],
  methods: [{ file: 'methods/linear/bayesian.ts', name: 'bayesianRidge' }],
})
