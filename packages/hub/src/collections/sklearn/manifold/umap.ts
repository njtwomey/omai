/**
 * Hub entry for UMAP, which lays out a fuzzy neighbour graph of the data by cross-entropy, on three-dimensional
 * manifolds such as the Swiss roll and tabular data, starting from the handwritten digits.
 */

import { umap as umapMethod } from 'whitebox-engine/methods/manifold'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'
import { rowColours } from '../../../entry/slots.ts'

/** UMAP: lay out a fuzzy neighbour graph of the data by cross-entropy. */
export const umap: HubEntry = defineEntry({
  id: 'sklearn/manifold/umap',
  title: 'UMAP',
  summary:
    'Lay out a fuzzy neighbour graph of the data by cross-entropy: the handwritten digits fall into one cluster per digit.',
  collection: 'sklearn/manifold',
  tags: ['manifold learning', 'embedding', 'graphs'],
  datasets: ['classificationTabular', 'manifold3d'],
  dataset: 'digits',
  maxRows: 150,
  ...fitEstimator(umapMethod(), { unsupervised: true }),
  views: [{ record: 'embeddingSteps', title: 'Embedding', view: 'EmbeddingPlayer', props: (_, c) => rowColours(c) }],
  methods: [{ file: 'methods/manifold/umap.ts', name: 'umap' }],
})
