/**
 * Hub entry for the minimum covariance determinant, a robust covariance from the most concentrated half of the data, on
 * two-dimensional data with outliers.
 */

import { minCovDet as minCovDetMethod } from 'whitebox-engine/methods/covariance'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** Minimum covariance determinant: a robust covariance from the most concentrated half of the data. */
export const minCovDet: HubEntry = defineEntry({
  id: 'sklearn/covariance/min-cov-det',
  title: 'Minimum covariance determinant',
  summary: 'A robust covariance from the most concentrated half of the data.',
  collection: 'sklearn/covariance',
  tags: ['covariance', 'robust'],
  datasets: ['anomaly2d'],
  ...fitEstimator(minCovDetMethod(), { unsupervised: true }),
  views: [{ record: 'anomalyScores', title: 'Anomaly scores' }],
  methods: [{ file: 'methods/covariance/robust.ts', name: 'minCovDet' }],
})
