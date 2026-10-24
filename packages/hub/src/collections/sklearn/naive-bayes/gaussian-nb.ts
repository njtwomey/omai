import { gaussianNB } from 'whitebox-engine/methods/naive-bayes'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** Gaussian naive Bayes: classes as products of independent normal features. */
export const gaussianNb: HubEntry = defineEntry({
  id: 'sklearn/naive-bayes/gaussian-nb',
  title: 'Gaussian naive Bayes',
  summary: 'Classes as products of independent normal features.',
  collection: 'sklearn/naive-bayes',
  tags: ['classification', 'probabilistic', 'generative'],
  datasets: ['classification2d', 'classificationTabular'],
  ...fitEstimator(gaussianNB(), { surface: true }),
  views: [
    { record: 'decisionSurface', title: 'Decision surface' },
    { record: 'likelihoodTable', title: 'Likelihoods', area: 'side' },
  ],
  methods: [{ file: 'methods/naive-bayes/naive-bayes.ts', name: 'gaussianNB' }],
})
