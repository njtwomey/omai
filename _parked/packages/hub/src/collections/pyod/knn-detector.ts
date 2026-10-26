/**
 * Hub entry for the k-nearest-neighbour outlier detector, scoring each point by its distance to its k-th nearest
 * neighbour, on two-dimensional data with outliers.
 */

import { knnDetector as knnDetectorMethod } from 'mdml-engine/methods/anomaly'
import { defineEntry, type HubEntry } from '../../entry/entry.ts'
import { fitEstimator } from '../../entry/run.ts'

/** k-NN detector: outlier scores from the distance to the k-th nearest neighbour. */
export const knnDetector: HubEntry = defineEntry({
  id: 'pyod/knn-detector',
  title: 'k-NN detector',
  summary: 'Outlier scores from the distance to the k-th nearest neighbour.',
  collection: 'pyod',
  tags: ['anomaly detection', 'neighbours'],
  datasets: ['anomaly2d'],
  ...fitEstimator(knnDetectorMethod(), { unsupervised: true }),
  views: [
    { record: 'anomalyScores', title: 'Anomaly scores' },
    { record: 'decisionSurface', title: 'Score surface', area: 'side' },
  ],
  methods: [{ file: 'methods/anomaly/neighbours.ts', name: 'knnDetector' }],
})
