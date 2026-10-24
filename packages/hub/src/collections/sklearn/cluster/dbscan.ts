import { dbscan as dbscanMethod } from 'whitebox-engine/methods/cluster'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** DBSCAN: clusters as dense regions: core points, the points they reach, and noise. */
export const dbscan: HubEntry = defineEntry({
  id: 'sklearn/cluster/dbscan',
  title: 'DBSCAN',
  summary: 'Clusters as dense regions: core points, the points they reach, and noise.',
  collection: 'sklearn/cluster',
  tags: ['clustering', 'density'],
  datasets: ['clustering2d'],
  dataset: 'moons',
  ...fitEstimator(dbscanMethod(), { unsupervised: true }),
  views: [{ record: 'clusteringSteps', title: 'Clustering steps' }],
  methods: [{ file: 'methods/cluster/density.ts', name: 'dbscan' }],
})
