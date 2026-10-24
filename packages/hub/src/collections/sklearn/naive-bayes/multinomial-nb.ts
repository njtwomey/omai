import { multinomialNB } from 'whitebox-engine/methods/naive-bayes'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** Multinomial naive Bayes: classes as multinomials over count features. */
export const multinomialNb: HubEntry = defineEntry({
  id: 'sklearn/naive-bayes/multinomial-nb',
  title: 'Multinomial naive Bayes',
  summary: 'Classes as multinomials over count features.',
  collection: 'sklearn/naive-bayes',
  tags: ['classification', 'probabilistic', 'counts'],
  datasets: ['classificationTabular'],
  dataset: 'digits',
  ...fitEstimator(multinomialNB()),
  views: [{ record: 'likelihoodTable', title: 'Likelihoods' }],
  methods: [{ file: 'methods/naive-bayes/naive-bayes.ts', name: 'multinomialNB' }],
})
