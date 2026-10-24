import { ridge as ridgeMethod } from 'whitebox-engine/methods/linear'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** Ridge: least squares with an L2 penalty: coefficients shrink smoothly towards zero. */
export const ridge: HubEntry = defineEntry({
  id: 'sklearn/linear-model/ridge',
  title: 'Ridge',
  summary: 'Least squares with an L2 penalty: coefficients shrink smoothly towards zero.',
  collection: 'sklearn/linear-model',
  tags: ['regression', 'linear', 'regularisation'],
  datasets: ['regressionTabular', 'regression1d'],
  ...fitEstimator(ridgeMethod({ alpha: 1 })),
  views: [
    { record: 'regularisationPath', title: 'Regularisation path' },
    { record: 'coefficientTable', title: 'Coefficients', area: 'side' },
  ],
  methods: [{ file: 'methods/linear/regression.ts', name: 'ridge' }],
})
