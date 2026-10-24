import { umap as umapMethod } from 'whitebox-engine/methods/manifold'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** UMAP: lay out a fuzzy neighbour graph of the data by cross-entropy. */
export const umap: HubEntry = defineEntry({
  id: 'sklearn/manifold/umap',
  title: 'UMAP',
  summary: 'Lay out a fuzzy neighbour graph of the data by cross-entropy.',
  collection: 'sklearn/manifold',
  tags: ['manifold learning', 'embedding', 'graphs'],
  datasets: ['manifold3d', 'classificationTabular'],
  ...fitEstimator(umapMethod(), { unsupervised: true }),
  views: [{ record: 'embeddingSteps', title: 'Embedding' }],
  methods: [{ file: 'methods/manifold/umap.ts', name: 'umap' }],
})
