import { radiusNeighborsClassifier, radiusNeighborsRegressor } from 'whitebox-engine/methods/neighbours'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEither } from '../../../entry/run.ts'

/** Radius neighbours: predict from every training point within a fixed radius. */
export const radiusNeighbors: HubEntry = defineEntry({
  id: 'sklearn/neighbors/radius-neighbors',
  title: 'Radius neighbours',
  summary: 'Predict from every training point within a fixed radius.',
  collection: 'sklearn/neighbors',
  tags: ['classification', 'regression', 'neighbours', 'nonparametric'],
  datasets: ['classification2d', 'regression1d', 'classificationTabular', 'regressionTabular'],
  ...fitEither(radiusNeighborsClassifier(), radiusNeighborsRegressor(), { surface: true }),
  views: [
    { record: 'decisionSurface', title: 'Predictions' },
    { record: 'neighbours', title: 'Neighbours', area: 'side' },
  ],
  methods: [
    { file: 'methods/neighbours/knn.ts', name: 'radiusNeighborsClassifier' },
    { file: 'methods/neighbours/knn.ts', name: 'radiusNeighborsRegressor' },
  ],
})
