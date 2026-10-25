/**
 * Hub entry for one-vs-one multiclass classification, one binary classifier per pair of classes and then a vote, on
 * two-dimensional classification data and tabular classification data.
 */

import { oneVsOneClassifier } from 'whitebox-engine/methods/multiclass'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** One-vs-one: one binary classifier per pair of classes, then a vote. */
export const oneVsOne: HubEntry = defineEntry({
  id: 'sklearn/multiclass/one-vs-one',
  title: 'One-vs-one',
  summary: 'One binary classifier per pair of classes, then a vote.',
  collection: 'sklearn/multiclass',
  tags: ['classification', 'multiclass', 'reductions'],
  datasets: ['classification2d', 'classificationTabular'],
  dataset: 'blobs',
  ...fitEstimator(oneVsOneClassifier(), { surface: true }),
  views: [
    { record: 'decisionSurface', title: 'Decision surface' },
    { record: 'ensembleStages', title: 'Binary problems', area: 'side' },
  ],
  methods: [{ file: 'methods/multiclass/reductions.ts', name: 'oneVsOneClassifier' }],
})
