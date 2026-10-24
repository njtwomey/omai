import { kNeighborsClassifier, kNeighborsRegressor } from 'whitebox-engine/methods/neighbours'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEither } from '../../../entry/run.ts'

/** k nearest neighbours: predict from the k closest training points: a vote or an average. */
export const kNearest: HubEntry = defineEntry({
  id: 'sklearn/neighbors/k-nearest',
  title: 'k nearest neighbours',
  summary: 'Predict from the k closest training points: a vote or an average.',
  collection: 'sklearn/neighbors',
  tags: ['classification', 'regression', 'neighbours', 'nonparametric'],
  datasets: ['classification2d', 'regression1d', 'classificationTabular', 'regressionTabular'],
  ...fitEither(kNeighborsClassifier({ nNeighbors: 5 }), kNeighborsRegressor({ nNeighbors: 5 }), { surface: true }),
  views: [
    { record: 'decisionSurface', title: 'Predictions' },
    { record: 'neighbours', title: 'Neighbours', area: 'side' },
  ],
  methods: [
    { file: 'methods/neighbours/knn.ts', name: 'kNeighborsClassifier' },
    { file: 'methods/neighbours/knn.ts', name: 'kNeighborsRegressor' },
  ],
})
