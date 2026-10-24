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
  datasets: ['classificationTabular', 'classification2d'],
  ...fitEstimator(oneVsOneClassifier(), { surface: true }),
  views: [
    { record: 'decisionSurface', title: 'Decision surface' },
    { record: 'ensembleStages', title: 'Binary problems', area: 'side' },
  ],
  methods: [{ file: 'methods/multiclass/reductions.ts', name: 'oneVsOneClassifier' }],
})
