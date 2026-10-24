import { bernoulliNB } from 'whitebox-engine/methods/naive-bayes'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** Bernoulli naive Bayes: classes as independent binary features (thresholded). */
export const bernoulliNb: HubEntry = defineEntry({
  id: 'sklearn/naive-bayes/bernoulli-nb',
  title: 'Bernoulli naive Bayes',
  summary: 'Classes as independent binary features (thresholded).',
  collection: 'sklearn/naive-bayes',
  tags: ['classification', 'probabilistic'],
  datasets: ['classificationTabular', 'classification2d'],
  dataset: 'digits',
  ...fitEstimator(bernoulliNB()),
  views: [{ record: 'likelihoodTable', title: 'Likelihoods' }],
  methods: [{ file: 'methods/naive-bayes/naive-bayes.ts', name: 'bernoulliNB' }],
})
