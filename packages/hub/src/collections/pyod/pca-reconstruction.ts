import { pcaReconstructionDetector } from 'whitebox-engine/methods/anomaly'
import { defineEntry, type HubEntry } from '../../entry/entry.ts'
import { fitEstimator } from '../../entry/run.ts'

/** PCA reconstruction error: outliers as points the leading components reconstruct badly. */
export const pcaReconstruction: HubEntry = defineEntry({
  id: 'pyod/pca-reconstruction',
  title: 'PCA reconstruction error',
  summary: 'Outliers as points the leading components reconstruct badly.',
  collection: 'pyod',
  tags: ['anomaly detection', 'decomposition'],
  datasets: ['anomaly2d'],
  ...fitEstimator(pcaReconstructionDetector(), { unsupervised: true }),
  views: [
    { record: 'anomalyScores', title: 'Anomaly scores' },
    { record: 'decisionSurface', title: 'Score surface', area: 'side' },
  ],
  methods: [{ file: 'methods/anomaly/statistical.ts', name: 'pcaReconstructionDetector' }],
})
