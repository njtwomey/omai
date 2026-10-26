/**
 * Hub entry for spectral clustering, clustering the leading eigenvectors of a similarity graph's Laplacian, on
 * two-dimensional clustering datasets such as the two moons.
 */

import { spectralClustering as spectralClusteringMethod } from 'mdml-engine/methods/cluster'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** Spectral clustering: cluster the leading eigenvectors of a similarity graph’s Laplacian. */
export const spectralClustering: HubEntry = defineEntry({
  id: 'sklearn/cluster/spectral-clustering',
  title: 'Spectral clustering',
  summary:
    'Cluster the leading eigenvectors of a similarity graph’s Laplacian: it follows the two moons where k-means cuts straight across.',
  collection: 'sklearn/cluster',
  tags: ['clustering', 'graphs', 'spectral'],
  datasets: ['clustering2d'],
  dataset: 'moons',
  ...fitEstimator(spectralClusteringMethod({ gamma: 20 }), { unsupervised: true }),
  views: [{ record: 'clusteringSteps', title: 'Clustering steps' }],
  methods: [{ file: 'methods/cluster/spectral.ts', name: 'spectralClustering' }],
})
