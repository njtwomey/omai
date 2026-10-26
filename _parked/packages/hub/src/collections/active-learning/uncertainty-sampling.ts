/**
 * Hub entry for uncertainty sampling, the active-learning strategy that asks for the labels the model is least sure of
 * (least confidence, smallest margin or highest entropy), on two-dimensional classification data and tabular
 * classification data.
 */

import { defineEntry, type HubEntry } from '../../entry/entry.ts'
import { studyEntry } from './_study.ts'

/** Uncertainty sampling: ask for the labels the model is least sure of. */
export const uncertaintySampling: HubEntry = defineEntry({
  id: 'active-learning/uncertainty-sampling',
  title: 'Uncertainty sampling',
  summary: 'Buy the label of the point the model is least sure about: least confident, smallest margin or entropy.',
  collection: 'active-learning',
  tags: ['active learning', 'classification', 'label budget'],
  datasets: ['classification2d', 'classificationTabular'],
  dataset: 'blobs',
  ...studyEntry({ strategies: ['margin', 'least-confident', 'entropy', 'random'] }),
  views: [
    { record: 'learningCurve', title: 'Accuracy by labels bought (against random)' },
    { record: 'clusteringSteps', title: 'Labels bought, round by round', area: 'side' },
    { record: 'decisionSurface', title: 'Final decision surface', area: 'below' },
  ],
  methods: [
    { file: 'methods/active-learning/studies.ts', name: 'activeLearningStudy' },
    { file: 'lib/learn/active-learning.ts', name: 'uncertaintySampling' },
  ],
  references: ['Settles (2009), Active learning literature survey', 'Lewis and Gale (1994)'],
})
