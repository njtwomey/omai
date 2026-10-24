import { outputCodeClassifier } from 'whitebox-engine/methods/multiclass'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** Error-correcting output codes: each class a binary code word; classify by the nearest code. */
export const outputCodes: HubEntry = defineEntry({
  id: 'sklearn/multiclass/output-codes',
  title: 'Error-correcting output codes',
  summary: 'Each class a binary code word; classify by the nearest code.',
  collection: 'sklearn/multiclass',
  tags: ['classification', 'multiclass', 'reductions'],
  datasets: ['classification2d', 'classificationTabular'],
  dataset: 'blobs',
  ...fitEstimator(outputCodeClassifier(), { surface: true }),
  views: [
    { record: 'decisionSurface', title: 'Decision surface' },
    { record: 'ensembleStages', title: 'Binary problems', area: 'side' },
  ],
  methods: [{ file: 'methods/multiclass/reductions.ts', name: 'outputCodeClassifier' }],
})
