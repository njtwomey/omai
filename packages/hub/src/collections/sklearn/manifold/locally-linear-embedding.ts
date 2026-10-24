import { locallyLinearEmbedding as locallyLinearEmbeddingMethod } from 'whitebox-engine/methods/manifold'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** Locally linear embedding: keep each point’s reconstruction from its neighbours. */
export const locallyLinearEmbedding: HubEntry = defineEntry({
  id: 'sklearn/manifold/locally-linear-embedding',
  title: 'Locally linear embedding',
  summary: 'Keep each point’s reconstruction from its neighbours.',
  collection: 'sklearn/manifold',
  tags: ['manifold learning', 'embedding'],
  datasets: ['manifold3d', 'classificationTabular'],
  ...fitEstimator(locallyLinearEmbeddingMethod(), { unsupervised: true }),
  views: [{ record: 'embeddingSteps', title: 'Embedding' }],
  methods: [{ file: 'methods/manifold/lle.ts', name: 'locallyLinearEmbedding' }],
})
