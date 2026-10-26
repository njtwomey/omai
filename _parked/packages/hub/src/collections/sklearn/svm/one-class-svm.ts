/**
 * Hub entry for the one-class SVM, a boundary around the bulk of the data in a kernel's feature space, on
 * two-dimensional data with outliers.
 */

import { oneClassSvm as oneClassSvmMethod } from 'mdml-engine/methods/svm'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** One-class SVM: a boundary around the bulk of the data in a kernel’s feature space. */
export const oneClassSvm: HubEntry = defineEntry({
  id: 'sklearn/svm/one-class-svm',
  title: 'One-class SVM',
  summary: 'A boundary around the bulk of the data in a kernel’s feature space.',
  collection: 'sklearn/svm',
  tags: ['anomaly detection', 'kernels'],
  datasets: ['anomaly2d'],
  ...fitEstimator(oneClassSvmMethod(), { unsupervised: true }),
  views: [
    { record: 'margin', title: 'Margin and support vectors' },
    { record: 'anomalyScores', title: 'Anomaly scores', area: 'side' },
    { record: 'decisionSurface', title: 'Score surface', area: 'side' },
  ],
  methods: [{ file: 'methods/svm/one-class.ts', name: 'oneClassSvm' }],
})
