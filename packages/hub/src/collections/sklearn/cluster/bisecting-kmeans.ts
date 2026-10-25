/**
 * Hub entry for bisecting k-means, splitting the largest cluster in two until there are k, on two-dimensional
 * clustering datasets.
 */

import { bisectingKMeans } from 'whitebox-engine/methods/cluster'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** Bisecting k-means: split the largest cluster in two with k-means, until there are k. */
export const bisectingKmeans: HubEntry = defineEntry({
  id: 'sklearn/cluster/bisecting-kmeans',
  title: 'Bisecting k-means',
  summary: 'Split the largest cluster in two with k-means, until there are k.',
  collection: 'sklearn/cluster',
  tags: ['clustering', 'centroids', 'hierarchical'],
  datasets: ['clustering2d'],
  ...fitEstimator(bisectingKMeans({ nClusters: 3 }), { unsupervised: true }),
  views: [{ record: 'clusteringSteps', title: 'Clustering steps' }],
  methods: [{ file: 'methods/cluster/kmeans.ts', name: 'bisectingKMeans' }],
})
