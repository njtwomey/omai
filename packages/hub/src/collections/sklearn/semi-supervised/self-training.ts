/**
 * Hub entry for self-training (as scikit-learn's `SelfTrainingClassifier`), a classifier labelling its most confident
 * unlabelled points round by round, on two-dimensional classification data.
 */

import { selfTraining as selfTrainingEstimator } from 'whitebox-engine/methods/semi-supervised'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { withHiddenLabels } from './_hidden.ts'

/** Self-training (sklearn.semi_supervised.SelfTrainingClassifier): a classifier labels its most confident points. */
export const selfTraining: HubEntry = defineEntry({
  id: 'sklearn/semi-supervised/self-training',
  title: 'Self-training',
  summary:
    'A classifier fitted on a few labels gives its most confident unlabelled points labels, then refits: round by round, mistakes made early are kept.',
  collection: 'sklearn/semi-supervised',
  tags: ['semi-supervised', 'classification', 'pseudo-labelling', 'wrapper'],
  datasets: ['classification2d'],
  dataset: 'moons',
  ...withHiddenLabels(selfTrainingEstimator({ criterion: 'kBest', kBest: 30, maxIter: 7 })),
  views: [
    { record: 'clusteringSteps', title: 'Pseudo-labels per round' },
    { record: 'learningCurve', title: 'Labelled fraction and accuracy', area: 'side' },
    { record: 'decisionSurface', title: 'Final decision surface', area: 'below' },
  ],
  methods: [{ file: 'methods/semi-supervised/self-training.ts', name: 'selfTraining' }],
  references: ['Yarowsky (1995), Unsupervised word sense disambiguation rivaling supervised methods'],
})
