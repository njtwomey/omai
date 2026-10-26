/**
 * Hub entry for categorical naive Bayes, classes as independent categorical features, on tabular classification data
 * such as the handwritten digits.
 */

import { categoricalNB } from 'mdml-engine/methods/naive-bayes'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** Categorical naive Bayes: classes as independent categorical features. */
export const categoricalNb: HubEntry = defineEntry({
  id: 'sklearn/naive-bayes/categorical-nb',
  title: 'Categorical naive Bayes',
  summary: 'Classes as independent categorical features.',
  collection: 'sklearn/naive-bayes',
  tags: ['classification', 'probabilistic'],
  datasets: ['classificationTabular'],
  dataset: 'digits',
  ...fitEstimator(categoricalNB()),
  views: [{ record: 'likelihoodTable', title: 'Likelihoods' }],
  methods: [{ file: 'methods/naive-bayes/naive-bayes.ts', name: 'categoricalNB' }],
})
