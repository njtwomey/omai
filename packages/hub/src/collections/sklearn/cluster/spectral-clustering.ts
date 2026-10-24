import { spectralClustering as spectralClusteringMethod } from 'whitebox-engine/methods/cluster'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** Spectral clustering: cluster the leading eigenvectors of a similarity graph’s Laplacian. */
export const spectralClustering: HubEntry = defineEntry({
  id: 'sklearn/cluster/spectral-clustering',
  title: 'Spectral clustering',
  summary: 'Cluster the leading eigenvectors of a similarity graph’s Laplacian.',
  collection: 'sklearn/cluster',
  tags: ['clustering', 'graphs', 'spectral'],
  datasets: ['clustering2d'],
  dataset: 'moons',
  ...fitEstimator(spectralClusteringMethod(), { unsupervised: true }),
  views: [{ record: 'clusteringSteps', title: 'Clustering steps' }],
  methods: [{ file: 'methods/cluster/spectral.ts', name: 'spectralClustering' }],
})
