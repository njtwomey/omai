import { svdd as svddMethod } from 'whitebox-engine/methods/svm'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** Support vector data description: the smallest sphere in feature space that holds most of the data. */
export const svdd: HubEntry = defineEntry({
  id: 'sklearn/svm/svdd',
  title: 'Support vector data description',
  summary: 'The smallest sphere in feature space that holds most of the data.',
  collection: 'sklearn/svm',
  tags: ['anomaly detection', 'kernels'],
  datasets: ['anomaly2d'],
  ...fitEstimator(svddMethod(), { unsupervised: true }),
  views: [
    { record: 'margin', title: 'Margin and support vectors' },
    { record: 'anomalyScores', title: 'Anomaly scores', area: 'side' },
    { record: 'decisionSurface', title: 'Score surface', area: 'side' },
  ],
  methods: [{ file: 'methods/svm/one-class.ts', name: 'svdd' }],
})
