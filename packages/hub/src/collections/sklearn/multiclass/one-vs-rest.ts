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
  datasets: ['classificationTabular', 'classification2d'],
  ...fitEstimator(oneVsRestClassifier(), { surface: true }),
  views: [
    { record: 'decisionSurface', title: 'Decision surface' },
    { record: 'ensembleStages', title: 'Binary problems', area: 'side' },
  ],
  methods: [{ file: 'methods/multiclass/reductions.ts', name: 'oneVsRestClassifier' }],
})
