import { ardRegression as ardRegressionMethod } from 'whitebox-engine/methods/linear'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** ARD regression: automatic relevance determination: one prior precision per coefficient prunes features. */
export const ardRegression: HubEntry = defineEntry({
  id: 'sklearn/linear-model/ard-regression',
  title: 'ARD regression',
  summary: 'Automatic relevance determination: one prior precision per coefficient prunes features.',
  collection: 'sklearn/linear-model',
  tags: ['regression', 'linear', 'bayesian', 'sparsity'],
  datasets: ['regressionTabular', 'regression1d'],
  ...fitEstimator(ardRegressionMethod()),
  views: [{ record: 'coefficientTable', title: 'Coefficients' }],
  methods: [{ file: 'methods/linear/bayesian.ts', name: 'ardRegression' }],
})
