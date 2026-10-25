/**
 * Hub entry for the z-score outlier detector, flagging points many standard deviations from the mean, on
 * two-dimensional data with outliers.
 */

import { zScoreDetector } from 'whitebox-engine/methods/anomaly'
import { defineEntry, type HubEntry } from '../../entry/entry.ts'
import { fitEstimator } from '../../entry/run.ts'

/** z-score detector: outliers as points many standard deviations from the mean. */
export const zScore: HubEntry = defineEntry({
  id: 'pyod/z-score',
  title: 'z-score detector',
  summary: 'Outliers as points many standard deviations from the mean.',
  collection: 'pyod',
  tags: ['anomaly detection', 'statistical'],
  datasets: ['anomaly2d'],
  ...fitEstimator(zScoreDetector(), { unsupervised: true }),
  views: [
    { record: 'anomalyScores', title: 'Anomaly scores' },
    { record: 'decisionSurface', title: 'Score surface', area: 'side' },
  ],
  methods: [{ file: 'methods/anomaly/statistical.ts', name: 'zScoreDetector' }],
})
