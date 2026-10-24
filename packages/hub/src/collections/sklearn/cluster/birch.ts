import { birch as birchMethod } from 'whitebox-engine/methods/cluster'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** BIRCH: summarise the data in a tree of clustering features, then cluster the leaves. */
export const birch: HubEntry = defineEntry({
  id: 'sklearn/cluster/birch',
  title: 'BIRCH',
  summary: 'Summarise the data in a tree of clustering features, then cluster the leaves.',
  collection: 'sklearn/cluster',
  tags: ['clustering', 'hierarchical', 'online'],
  datasets: ['clustering2d'],
  ...fitEstimator(birchMethod(), { unsupervised: true }),
  views: [{ record: 'clusteringSteps', title: 'Clustering steps' }],
  methods: [{ file: 'methods/cluster/exemplar.ts', name: 'birch' }],
})
