/**
 * Hub entry for the elliptic envelope, flagging points far from a robust Gaussian fit as outliers, on two-dimensional
 * data with outliers.
 */

import { ellipticEnvelope as ellipticEnvelopeMethod } from 'whitebox-engine/methods/anomaly'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** Elliptic envelope: outliers as points far from a robust Gaussian fit. */
export const ellipticEnvelope: HubEntry = defineEntry({
  id: 'sklearn/covariance/elliptic-envelope',
  title: 'Elliptic envelope',
  summary: 'Outliers as points far from a robust Gaussian fit.',
  collection: 'sklearn/covariance',
  tags: ['anomaly detection', 'covariance', 'robust'],
  datasets: ['anomaly2d'],
  ...fitEstimator(ellipticEnvelopeMethod(), { unsupervised: true }),
  views: [
    { record: 'anomalyScores', title: 'Anomaly scores' },
    { record: 'decisionSurface', title: 'Score surface', area: 'side' },
  ],
  methods: [{ file: 'methods/anomaly/statistical.ts', name: 'ellipticEnvelope' }],
})
