import { isomap as isomapMethod } from 'whitebox-engine/methods/manifold'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** Isomap: classical scaling of geodesic distances along a neighbour graph. */
export const isomap: HubEntry = defineEntry({
  id: 'sklearn/manifold/isomap',
  title: 'Isomap',
  summary: 'Classical scaling of geodesic distances along a neighbour graph.',
  collection: 'sklearn/manifold',
  tags: ['manifold learning', 'embedding', 'graphs'],
  datasets: ['manifold3d'],
  ...fitEstimator(isomapMethod(), { unsupervised: true }),
  views: [{ record: 'embeddingSteps', title: 'Embedding' }],
  methods: [{ file: 'methods/manifold/isomap.ts', name: 'isomap' }],
})
