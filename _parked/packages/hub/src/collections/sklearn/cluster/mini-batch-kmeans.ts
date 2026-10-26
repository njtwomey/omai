/**
 * Hub entry for mini-batch k-means, its centres nudged by small random batches, on two-dimensional clustering datasets.
 */

import { miniBatchKMeans } from 'mdml-engine/methods/cluster'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** Mini-batch k-means: k-means with centres nudged by small random batches. */
export const miniBatchKmeans: HubEntry = defineEntry({
  id: 'sklearn/cluster/mini-batch-kmeans',
  title: 'Mini-batch k-means',
  summary: 'k-means with centres nudged by small random batches.',
  collection: 'sklearn/cluster',
  tags: ['clustering', 'centroids', 'online'],
  datasets: ['clustering2d'],
  ...fitEstimator(miniBatchKMeans({ nClusters: 3 }), { unsupervised: true }),
  views: [{ record: 'clusteringSteps', title: 'Clustering steps' }],
  methods: [{ file: 'methods/cluster/kmeans.ts', name: 'miniBatchKMeans' }],
})
