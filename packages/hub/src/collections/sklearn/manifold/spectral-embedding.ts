import { spectralEmbedding as spectralEmbeddingMethod } from 'whitebox-engine/methods/manifold'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** Spectral embedding: the bottom eigenvectors of a neighbour graph’s Laplacian (Laplacian eigenmaps). */
export const spectralEmbedding: HubEntry = defineEntry({
  id: 'sklearn/manifold/spectral-embedding',
  title: 'Spectral embedding',
  summary: 'The bottom eigenvectors of a neighbour graph’s Laplacian (Laplacian eigenmaps).',
  collection: 'sklearn/manifold',
  tags: ['manifold learning', 'embedding', 'spectral'],
  datasets: ['manifold3d', 'classificationTabular'],
  ...fitEstimator(spectralEmbeddingMethod(), { unsupervised: true }),
  views: [{ record: 'embeddingSteps', title: 'Embedding' }],
  methods: [{ file: 'methods/manifold/spectral.ts', name: 'spectralEmbedding' }],
})
