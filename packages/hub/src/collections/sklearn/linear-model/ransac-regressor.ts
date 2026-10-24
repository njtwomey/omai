import { ransacRegressor as ransacRegressorMethod } from 'whitebox-engine/methods/linear'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** RANSAC: fit to random minimal subsets and keep the model with the most inliers. */
export const ransacRegressor: HubEntry = defineEntry({
  id: 'sklearn/linear-model/ransac-regressor',
  title: 'RANSAC',
  summary: 'Fit to random minimal subsets and keep the model with the most inliers.',
  collection: 'sklearn/linear-model',
  tags: ['regression', 'linear', 'robust'],
  datasets: ['regression1d', 'regressionTabular'],
  dataset: 'linearOutliers',
  ...fitEstimator(ransacRegressorMethod(), { surface: true }),
  views: [
    { record: 'decisionSurface', title: 'Fitted function' },
    { record: 'anomalyScores', title: 'Anomaly scores', area: 'side' },
    { record: 'coefficientTable', title: 'Coefficients', area: 'side' },
  ],
  methods: [{ file: 'methods/linear/robust.ts', name: 'ransacRegressor' }],
})
