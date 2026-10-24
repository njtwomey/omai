import { fuzzyCMeans as fuzzyCMeansMethod } from 'whitebox-engine/methods/cluster'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** Fuzzy c-means: every point belongs to every cluster, to a degree. */
export const fuzzyCMeans: HubEntry = defineEntry({
  id: 'sklearn/cluster/fuzzy-c-means',
  title: 'Fuzzy c-means',
  summary: 'Every point belongs to every cluster, to a degree.',
  collection: 'sklearn/cluster',
  tags: ['clustering', 'centroids', 'soft'],
  datasets: ['clustering2d'],
  ...fitEstimator(fuzzyCMeansMethod(), { unsupervised: true }),
  views: [{ record: 'clusteringSteps', title: 'Clustering steps' }],
  methods: [{ file: 'methods/cluster/fuzzy.ts', name: 'fuzzyCMeans' }],
})
