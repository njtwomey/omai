/**
 * Hub entry for k-medoids, clusters centred on actual data points found by BUILD and SWAP, on two-dimensional
 * clustering datasets.
 */

import { kMedoids as kMedoidsMethod } from 'mdml-engine/methods/cluster'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** k-medoids: clusters around actual data points, by BUILD and SWAP. */
export const kMedoids: HubEntry = defineEntry({
  id: 'sklearn/cluster/k-medoids',
  title: 'k-medoids',
  summary: 'Clusters around actual data points, by BUILD and SWAP.',
  collection: 'sklearn/cluster',
  tags: ['clustering', 'medoids'],
  datasets: ['clustering2d'],
  ...fitEstimator(kMedoidsMethod({ nClusters: 3 }), { unsupervised: true }),
  views: [{ record: 'clusteringSteps', title: 'Clustering steps' }],
  methods: [{ file: 'methods/cluster/medoids.ts', name: 'kMedoids' }],
})
