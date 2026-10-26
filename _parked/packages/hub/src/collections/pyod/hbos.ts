/**
 * Hub entry for HBOS, histogram-based outlier scores that add up each feature's rarity, on two-dimensional data with
 * outliers.
 */

import { hbos as hbosMethod } from 'mdml-engine/methods/anomaly'
import { defineEntry, type HubEntry } from '../../entry/entry.ts'
import { fitEstimator } from '../../entry/run.ts'

/** HBOS: histogram-based outlier scores: rarity in each feature’s histogram, added. */
export const hbos: HubEntry = defineEntry({
  id: 'pyod/hbos',
  title: 'HBOS',
  summary: 'Histogram-based outlier scores: rarity in each feature’s histogram, added.',
  collection: 'pyod',
  tags: ['anomaly detection', 'histograms'],
  datasets: ['anomaly2d'],
  ...fitEstimator(hbosMethod(), { unsupervised: true }),
  views: [
    { record: 'anomalyScores', title: 'Anomaly scores' },
    { record: 'decisionSurface', title: 'Score surface', area: 'side' },
  ],
  methods: [{ file: 'methods/anomaly/statistical.ts', name: 'hbos' }],
})
