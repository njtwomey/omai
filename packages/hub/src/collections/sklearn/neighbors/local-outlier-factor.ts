/**
 * Hub entry for the local outlier factor, how much sparser a point's neighbourhood is than its neighbours', on
 * two-dimensional data with outliers.
 */

import { localOutlierFactor as localOutlierFactorMethod } from 'whitebox-engine/methods/anomaly'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** Local outlier factor: how much sparser a point’s neighbourhood is than its neighbours’. */
export const localOutlierFactor: HubEntry = defineEntry({
  id: 'sklearn/neighbors/local-outlier-factor',
  title: 'Local outlier factor',
  summary: 'How much sparser a point’s neighbourhood is than its neighbours’.',
  collection: 'sklearn/neighbors',
  tags: ['anomaly detection', 'neighbours'],
  datasets: ['anomaly2d'],
  ...fitEstimator(localOutlierFactorMethod(), { unsupervised: true }),
  views: [
    { record: 'anomalyScores', title: 'Anomaly scores' },
    { record: 'neighbours', title: 'Neighbours', area: 'side' },
    { record: 'decisionSurface', title: 'Score surface', area: 'side' },
  ],
  methods: [{ file: 'methods/anomaly/neighbours.ts', name: 'localOutlierFactor' }],
})
