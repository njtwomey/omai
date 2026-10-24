import { mds as mdsMethod } from 'whitebox-engine/methods/manifold'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** Multidimensional scaling: place points so their distances match the data’s (SMACOF stress). */
export const mds: HubEntry = defineEntry({
  id: 'sklearn/manifold/mds',
  title: 'Multidimensional scaling',
  summary: 'Place points so their distances match the data’s (SMACOF stress).',
  collection: 'sklearn/manifold',
  tags: ['manifold learning', 'embedding'],
  datasets: ['manifold3d', 'classificationTabular'],
  ...fitEstimator(mdsMethod(), { unsupervised: true }),
  views: [{ record: 'embeddingSteps', title: 'Embedding' }],
  methods: [{ file: 'methods/manifold/mds.ts', name: 'mds' }],
})
