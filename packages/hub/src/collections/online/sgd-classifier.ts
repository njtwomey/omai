/**
 * Hub entry for SGD classification (as scikit-learn's `SGDClassifier`), a linear classifier learned one row at a time,
 * on two-dimensional classification data and tabular classification data.
 */

import { sgdClassifier as sgdClassifierEstimator } from 'whitebox-engine/methods/linear'
import { defineEntry, type HubEntry } from '../../entry/entry.ts'
import { fitEstimator } from '../../entry/run.ts'

/** SGD classification (sklearn.linear_model.SGDClassifier): a linear model learned one row at a time. */
export const sgdClassifier: HubEntry = defineEntry({
  id: 'online/sgd-classifier',
  title: 'SGD classifier',
  summary: 'A linear classifier learned one row at a time, its step size shrinking on a schedule.',
  collection: 'online',
  tags: ['online learning', 'classification', 'linear', 'stochastic gradient descent'],
  datasets: ['classification2d', 'classificationTabular'],
  dataset: 'blobs',
  ...fitEstimator(sgdClassifierEstimator({ maxIter: 30 })),
  views: [
    { record: 'learningCurve', title: 'Progressive and training error' },
    { record: 'decisionSurface', title: 'Decision surface', area: 'side' },
    { record: 'coefficientTable', title: 'Coefficients over the samples seen', area: 'below' },
  ],
  methods: [{ file: 'methods/linear/online.ts', name: 'sgdClassifier' }],
  references: ['Bottou (2010), Large-scale machine learning with stochastic gradient descent'],
})
