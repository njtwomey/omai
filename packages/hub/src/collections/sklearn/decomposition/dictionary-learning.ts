/**
 * Hub entry for dictionary learning, sparse codes over a learned dictionary of atoms, on three-dimensional manifolds
 * such as the Swiss roll and tabular data.
 */

import { dictionaryLearning as dictionaryLearningMethod } from 'mdml-engine/methods/decomposition'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** Dictionary learning: sparse codes over a learned dictionary of atoms. */
export const dictionaryLearning: HubEntry = defineEntry({
  id: 'sklearn/decomposition/dictionary-learning',
  title: 'Dictionary learning',
  summary: 'Sparse codes over a learned dictionary of atoms.',
  collection: 'sklearn/decomposition',
  tags: ['decomposition', 'sparsity'],
  datasets: ['classificationTabular', 'manifold3d'],
  ...fitEstimator(dictionaryLearningMethod({ nComponents: 2 }), { unsupervised: true }),
  views: [{ record: 'components', title: 'Components' }],
  methods: [{ file: 'methods/decomposition/svd.ts', name: 'dictionaryLearning' }],
})
