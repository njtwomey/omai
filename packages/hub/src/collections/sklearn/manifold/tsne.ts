import { tsne as tsneMethod } from 'whitebox-engine/methods/manifold'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** t-SNE: match neighbour probabilities in the plane to those in the data, with heavy tails. */
export const tsne: HubEntry = defineEntry({
  id: 'sklearn/manifold/tsne',
  title: 't-SNE',
  summary: 'Match neighbour probabilities in the plane to those in the data, with heavy tails.',
  collection: 'sklearn/manifold',
  tags: ['manifold learning', 'embedding', 'visualisation'],
  datasets: ['manifold3d', 'classificationTabular'],
  ...fitEstimator(tsneMethod(), { unsupervised: true }),
  views: [{ record: 'embeddingSteps', title: 'Embedding' }],
  methods: [{ file: 'methods/manifold/tsne.ts', name: 'tsne' }],
})
