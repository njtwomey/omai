/**
 * Hub entry for k-means (as scikit-learn's `KMeans`), Lloyd's iterations from k-means++ seeds, on two-dimensional
 * clustering datasets.
 */

import { kmeans as kmeansEstimator } from 'mdml-engine/methods/cluster'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** k-means (sklearn.cluster.KMeans): Lloyd's iterations from k-means++ seeds. */
export const kmeans: HubEntry = defineEntry({
  id: 'sklearn/cluster/kmeans',
  title: 'k-means',
  summary: 'Alternate assigning points to the nearest centre and moving each centre to its points’ mean.',
  collection: 'sklearn/cluster',
  tags: ['clustering', 'centroids', 'iterative'],
  datasets: ['clustering2d'],
  ...fitEstimator(kmeansEstimator({ nClusters: 3 }), { unsupervised: true, exclude: ['tol'] }),
  views: [
    { record: 'clusteringSteps', title: 'Lloyd iterations' },
    { record: 'decisionSurface', title: 'Cells of the centres', area: 'side' },
  ],
  methods: [{ file: 'methods/cluster/kmeans.ts', name: 'kmeans' }],
})
