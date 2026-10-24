import { elasticNet as elasticNetMethod } from 'whitebox-engine/methods/linear'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** Elastic net: a mix of the lasso’s L1 and ridge’s L2 penalties, solved by coordinate descent. */
export const elasticNet: HubEntry = defineEntry({
  id: 'sklearn/linear-model/elastic-net',
  title: 'Elastic net',
  summary: 'A mix of the lasso’s L1 and ridge’s L2 penalties, solved by coordinate descent.',
  collection: 'sklearn/linear-model',
  tags: ['regression', 'linear', 'sparsity', 'regularisation'],
  datasets: ['regressionTabular', 'regression1d'],
  dataset: 'sparseUncorrelated',
  ...fitEstimator(elasticNetMethod({ alpha: 0.1 })),
  views: [
    { record: 'regularisationPath', title: 'Regularisation path' },
    { record: 'coefficientTable', title: 'Coefficients', area: 'side' },
  ],
  methods: [{ file: 'methods/linear/regression.ts', name: 'elasticNet' }],
})
