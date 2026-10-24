import { copKMeans } from 'whitebox-engine/methods/cluster'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** COP k-means: k-means that respects must-link and cannot-link constraints. */
export const copKmeans: HubEntry = defineEntry({
  id: 'sklearn/cluster/cop-kmeans',
  title: 'COP k-means',
  summary: 'k-means that respects must-link and cannot-link constraints.',
  collection: 'sklearn/cluster',
  tags: ['clustering', 'centroids', 'constraints'],
  datasets: ['clustering2d'],
  ...fitEstimator(copKMeans(), { unsupervised: true }),
  views: [{ record: 'clusteringSteps', title: 'Clustering steps' }],
  methods: [{ file: 'methods/cluster/constrained.ts', name: 'copKMeans' }],
})
