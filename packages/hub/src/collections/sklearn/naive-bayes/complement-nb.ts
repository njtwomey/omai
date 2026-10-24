import { complementNB } from 'whitebox-engine/methods/naive-bayes'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** Complement naive Bayes: multinomial naive Bayes fitted to each class’s complement, for skewed classes. */
export const complementNb: HubEntry = defineEntry({
  id: 'sklearn/naive-bayes/complement-nb',
  title: 'Complement naive Bayes',
  summary: 'Multinomial naive Bayes fitted to each class’s complement, for skewed classes.',
  collection: 'sklearn/naive-bayes',
  tags: ['classification', 'probabilistic', 'counts'],
  datasets: ['classificationTabular'],
  dataset: 'digits',
  ...fitEstimator(complementNB()),
  views: [{ record: 'likelihoodTable', title: 'Likelihoods' }],
  methods: [{ file: 'methods/naive-bayes/naive-bayes.ts', name: 'complementNB' }],
})
