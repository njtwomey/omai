import { defineEntry, type HubEntry } from '../../entry/entry.ts'
import { studyEntry } from './_study.ts'

/** Query by committee: ask for the labels a bagged committee disagrees about most. */
export const queryByCommittee: HubEntry = defineEntry({
  id: 'active-learning/query-by-committee',
  title: 'Query by committee',
  summary: 'A committee of models fitted on resamples votes; buy the label of the point they disagree about most.',
  collection: 'active-learning',
  tags: ['active learning', 'classification', 'committee', 'label budget'],
  datasets: ['classification2d', 'classificationTabular'],
  dataset: 'blobs',
  ...studyEntry({ strategies: ['vote-entropy', 'consensus-entropy', 'max-disagreement'], budget: 15 }),
  views: [
    { record: 'learningCurve', title: 'Accuracy by labels bought (against random)' },
    { record: 'clusteringSteps', title: 'Labels bought, round by round', area: 'side' },
    { record: 'decisionSurface', title: 'Final decision surface', area: 'below' },
  ],
  methods: [
    { file: 'methods/active-learning/studies.ts', name: 'activeLearningStudy' },
    { file: 'lib/learn/active-learning.ts', name: 'voteEntropySampling' },
  ],
  references: ['Seung, Opper and Sompolinsky (1992), Query by committee'],
})
