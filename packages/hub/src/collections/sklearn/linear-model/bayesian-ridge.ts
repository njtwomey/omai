/**
 * Hub entry for Bayesian ridge regression, its penalty and noise level learned by maximising the evidence, on tabular
 * regression data and one-dimensional regression curves.
 */

import { bayesianRidge as bayesianRidgeMethod } from 'whitebox-engine/methods/linear'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** Bayesian ridge: ridge whose penalty and noise level are learned by evidence maximisation. */
export const bayesianRidge: HubEntry = defineEntry({
  id: 'sklearn/linear-model/bayesian-ridge',
  title: 'Bayesian ridge',
  summary:
    'Ridge whose penalty and noise level are learned by maximising the evidence: the path climbs the evidence surface to its peak.',
  collection: 'sklearn/linear-model',
  tags: ['regression', 'linear', 'bayesian'],
  datasets: ['regressionTabular', 'regression1d'],
  ...fitEstimator(bayesianRidgeMethod(), { exclude: ['maxIter'] }),
  views: [
    { record: 'hyperparameterSurface', title: 'Evidence surface' },
    { record: 'coefficientTable', title: 'Coefficients', area: 'side' },
  ],
  methods: [{ file: 'methods/linear/bayesian.ts', name: 'bayesianRidge' }],
})
