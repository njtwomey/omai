/**
 * Hub entry for one-vs-rest multiclass classification, one binary classifier per class with the most confident winning,
 * on two-dimensional classification data and tabular classification data.
 */

import { oneVsRestClassifier } from 'whitebox-engine/methods/multiclass'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** One-vs-rest: one binary classifier per class against the rest; the most confident wins. */
export const oneVsRest: HubEntry = defineEntry({
  id: 'sklearn/multiclass/one-vs-rest',
  title: 'One-vs-rest',
  summary: 'One binary classifier per class against the rest; the most confident wins.',
  collection: 'sklearn/multiclass',
  tags: ['classification', 'multiclass', 'reductions'],
  datasets: ['classification2d', 'classificationTabular'],
  dataset: 'blobs',
  ...fitEstimator(oneVsRestClassifier(), { surface: true }),
  views: [
    { record: 'decisionSurface', title: 'Decision surface' },
    { record: 'ensembleStages', title: 'Binary problems', area: 'side' },
  ],
  methods: [{ file: 'methods/multiclass/reductions.ts', name: 'oneVsRestClassifier' }],
})
