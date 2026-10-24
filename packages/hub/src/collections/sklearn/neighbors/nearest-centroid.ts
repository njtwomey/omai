import { nearestCentroid as nearestCentroidMethod } from 'whitebox-engine/methods/neighbours'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** Nearest centroid: classify by the closest class mean. */
export const nearestCentroid: HubEntry = defineEntry({
  id: 'sklearn/neighbors/nearest-centroid',
  title: 'Nearest centroid',
  summary: 'Classify by the closest class mean.',
  collection: 'sklearn/neighbors',
  tags: ['classification', 'neighbours', 'centroids'],
  datasets: ['classification2d', 'classificationTabular'],
  ...fitEstimator(nearestCentroidMethod(), { surface: true }),
  views: [
    { record: 'decisionSurface', title: 'Decision surface' },
    { record: 'coefficientTable', title: 'Coefficients', area: 'side' },
  ],
  methods: [{ file: 'methods/neighbours/centroid.ts', name: 'nearestCentroid' }],
})
