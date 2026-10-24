import { gaussianNB } from 'whitebox-engine/methods/naive-bayes'
import { defineEntry, type HubEntry } from '../../entry/entry.ts'
import { studyEntry } from './_study.ts'

/** Expected error reduction (lite): buy the label that would most lower the expected error on the pool. */
export const expectedErrorReduction: HubEntry = defineEntry({
  id: 'active-learning/expected-error-reduction',
  title: 'Expected error reduction',
  summary: 'For a few candidates, refit naive Bayes with each possible label; buy the one that most lowers error.',
  collection: 'active-learning',
  tags: ['active learning', 'classification', 'decision theory', 'label budget'],
  datasets: ['classification2d'],
  dataset: 'blobs',
  maxRows: 150,
  ...studyEntry({
    strategies: ['expected-error-reduction', 'ranked-batch'],
    budget: 12,
    candidates: 10,
    estimator: gaussianNB(),
  }),
  views: [
    { record: 'learningCurve', title: 'Accuracy by labels bought (against random)' },
    { record: 'clusteringSteps', title: 'Labels bought, round by round', area: 'side' },
  ],
  methods: [
    { file: 'methods/active-learning/studies.ts', name: 'activeLearningStudy' },
    { file: 'lib/learn/active-learning.ts', name: 'expectedErrorReduction' },
  ],
  references: [
    'Roy and McCallum (2001), Toward optimal active learning through sampling estimation of error reduction',
  ],
})
