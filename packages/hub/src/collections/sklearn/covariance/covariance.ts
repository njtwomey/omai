import { covarianceEstimator } from 'whitebox-engine/methods/covariance'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** Covariance estimation: empirical and shrunk (Ledoit–Wolf, OAS) covariance, with Mahalanobis distances. */
export const covariance: HubEntry = defineEntry({
  id: 'sklearn/covariance/covariance',
  title: 'Covariance estimation',
  summary: 'Empirical and shrunk (Ledoit–Wolf, OAS) covariance, with Mahalanobis distances.',
  collection: 'sklearn/covariance',
  tags: ['covariance', 'shrinkage'],
  datasets: ['anomaly2d', 'density2d'],
  ...fitEstimator(covarianceEstimator(), { unsupervised: true }),
  views: [{ record: 'anomalyScores', title: 'Anomaly scores' }],
  methods: [{ file: 'methods/covariance/shrinkage.ts', name: 'covarianceEstimator' }],
})
