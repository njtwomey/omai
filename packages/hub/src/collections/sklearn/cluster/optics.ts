import { optics as opticsMethod } from 'whitebox-engine/methods/cluster'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** OPTICS: an ordering of the points by reachability: every DBSCAN radius at once. */
export const optics: HubEntry = defineEntry({
  id: 'sklearn/cluster/optics',
  title: 'OPTICS',
  summary: 'An ordering of the points by reachability: every DBSCAN radius at once.',
  collection: 'sklearn/cluster',
  tags: ['clustering', 'density'],
  datasets: ['clustering2d'],
  ...fitEstimator(opticsMethod(), { unsupervised: true }),
  views: [{ record: 'clusteringSteps', title: 'Clustering steps' }],
  methods: [{ file: 'methods/cluster/density.ts', name: 'optics' }],
})
