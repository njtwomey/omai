/**
 * Hub entry for spectral embedding (Laplacian eigenmaps), the bottom eigenvectors of a neighbour graph's Laplacian, on
 * three-dimensional manifolds such as the Swiss roll and tabular data.
 */

import { spectralEmbedding as spectralEmbeddingMethod } from 'mdml-engine/methods/manifold'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'
import { rowColours } from '../../../entry/slots.ts'

/** Spectral embedding: the bottom eigenvectors of a neighbour graph’s Laplacian (Laplacian eigenmaps). */
export const spectralEmbedding: HubEntry = defineEntry({
  id: 'sklearn/manifold/spectral-embedding',
  title: 'Spectral embedding',
  summary:
    'The bottom eigenvectors of a neighbour graph’s Laplacian (Laplacian eigenmaps): neighbours on the roll stay neighbours.',
  collection: 'sklearn/manifold',
  tags: ['manifold learning', 'embedding', 'spectral'],
  datasets: ['manifold3d', 'classificationTabular'],
  knobs: { n: 300 },
  ...fitEstimator(spectralEmbeddingMethod(), { unsupervised: true }),
  views: [{ record: 'embeddingSteps', title: 'Embedding', view: 'EmbeddingPlayer', props: (_, c) => rowColours(c) }],
  methods: [{ file: 'methods/manifold/spectral.ts', name: 'spectralEmbedding' }],
})
