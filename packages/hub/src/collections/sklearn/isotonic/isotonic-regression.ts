import { isotonicRegression as isotonicRegressionMethod } from 'whitebox-engine/methods/linear'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** Isotonic regression: the best monotone fit, by pooling adjacent violators. */
export const isotonicRegression: HubEntry = defineEntry({
  id: 'sklearn/isotonic/isotonic-regression',
  title: 'Isotonic regression',
  summary: 'The best monotone fit, by pooling adjacent violators.',
  collection: 'sklearn/isotonic',
  tags: ['regression', 'monotone', 'nonparametric'],
  datasets: ['regression1d'],
  dataset: 'stepCurve',
  ...fitEstimator(isotonicRegressionMethod(), { surface: true }),
  views: [
    { record: 'decisionSurface', title: 'Fitted function' },
    { record: 'posteriorGrid', title: 'Fit', area: 'side' },
  ],
  methods: [{ file: 'methods/linear/isotonic.ts', name: 'isotonicRegression' }],
})
