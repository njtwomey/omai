import { tsne as tsneMethod } from 'whitebox-engine/methods/manifold'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'
import { rowColours } from '../../../entry/slots.ts'

/** t-SNE: match neighbour probabilities in the plane to those in the data, with heavy tails. */
export const tsne: HubEntry = defineEntry({
  id: 'sklearn/manifold/tsne',
  title: 't-SNE',
  summary:
    'Match neighbour probabilities in the plane to those in the data, with heavy tails: the classes pull apart into clusters.',
  collection: 'sklearn/manifold',
  tags: ['manifold learning', 'embedding', 'visualisation'],
  datasets: ['classificationTabular', 'manifold3d'],
  maxRows: 200,
  ...fitEstimator(tsneMethod({ maxIter: 350, perplexity: 8 }), { unsupervised: true }),
  views: [{ record: 'embeddingSteps', title: 'Embedding', view: 'EmbeddingPlayer', props: (_, c) => rowColours(c) }],
  methods: [{ file: 'methods/manifold/tsne.ts', name: 'tsne' }],
})
