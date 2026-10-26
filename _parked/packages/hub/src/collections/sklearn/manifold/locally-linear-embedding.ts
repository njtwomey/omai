/**
 * Hub entry for locally linear embedding, which keeps each point's reconstruction from its neighbours, on
 * three-dimensional manifolds such as the Swiss roll and tabular data.
 */

import { locallyLinearEmbedding as locallyLinearEmbeddingMethod } from 'mdml-engine/methods/manifold'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'
import { rowColours } from '../../../entry/slots.ts'

/** Locally linear embedding: keep each point’s reconstruction from its neighbours. */
export const locallyLinearEmbedding: HubEntry = defineEntry({
  id: 'sklearn/manifold/locally-linear-embedding',
  title: 'Locally linear embedding',
  summary: 'Keep each point’s reconstruction from its neighbours: the S curve flattens with its colours in order.',
  collection: 'sklearn/manifold',
  tags: ['manifold learning', 'embedding'],
  datasets: ['manifold3d', 'classificationTabular'],
  dataset: 'sCurve',
  knobs: { n: 300 },
  ...fitEstimator(locallyLinearEmbeddingMethod({ nNeighbors: 10 }), { unsupervised: true }),
  views: [{ record: 'embeddingSteps', title: 'Embedding', view: 'EmbeddingPlayer', props: (_, c) => rowColours(c) }],
  methods: [{ file: 'methods/manifold/lle.ts', name: 'locallyLinearEmbedding' }],
})
