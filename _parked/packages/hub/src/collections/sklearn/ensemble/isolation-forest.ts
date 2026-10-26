/**
 * Hub entry for the isolation forest, which scores anomalies by how few random splits isolate them, on two-dimensional
 * data with outliers.
 */

import { isolationForest as isolationForestMethod } from 'mdml-engine/methods/anomaly'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** Isolation forest: anomalies are isolated by fewer random splits than normal points. */
export const isolationForest: HubEntry = defineEntry({
  id: 'sklearn/ensemble/isolation-forest',
  title: 'Isolation forest',
  summary: 'Anomalies are isolated by fewer random splits than normal points.',
  collection: 'sklearn/ensemble',
  tags: ['anomaly detection', 'ensembles', 'trees'],
  datasets: ['anomaly2d'],
  ...fitEstimator(isolationForestMethod(), { unsupervised: true }),
  views: [
    { record: 'anomalyScores', title: 'Anomaly scores' },
    { record: 'decisionSurface', title: 'Score surface', area: 'side' },
  ],
  methods: [{ file: 'methods/anomaly/isolation.ts', name: 'isolationForest' }],
})
