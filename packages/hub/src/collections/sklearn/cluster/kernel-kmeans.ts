import { kernelKMeans } from 'whitebox-engine/methods/cluster'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** Kernel k-means: k-means in a kernel’s feature space: non-convex clusters. */
export const kernelKmeans: HubEntry = defineEntry({
  id: 'sklearn/cluster/kernel-kmeans',
  title: 'Kernel k-means',
  summary: 'k-means in a kernel’s feature space: non-convex clusters.',
  collection: 'sklearn/cluster',
  tags: ['clustering', 'kernels'],
  datasets: ['clustering2d'],
  dataset: 'circles',
  ...fitEstimator(kernelKMeans({ nClusters: 2 }), { unsupervised: true }),
  views: [{ record: 'clusteringSteps', title: 'Clustering steps' }],
  methods: [{ file: 'methods/cluster/kmeans.ts', name: 'kernelKMeans' }],
})
