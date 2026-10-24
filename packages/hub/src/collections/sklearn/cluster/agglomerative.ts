import { agglomerativeClustering } from 'whitebox-engine/methods/cluster'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** Agglomerative clustering: merge the two closest clusters, again and again: a dendrogram. */
export const agglomerative: HubEntry = defineEntry({
  id: 'sklearn/cluster/agglomerative',
  title: 'Agglomerative clustering',
  summary: 'Merge the two closest clusters, again and again: the dendrogram, cut where three clusters remain.',
  collection: 'sklearn/cluster',
  tags: ['clustering', 'hierarchical'],
  datasets: ['clustering2d'],
  ...fitEstimator(agglomerativeClustering({ nClusters: 3 }), { unsupervised: true }),
  views: [{ record: 'clusteringSteps', title: 'Clustering steps' }],
  methods: [{ file: 'methods/cluster/hierarchical.ts', name: 'agglomerativeClustering' }],
})
