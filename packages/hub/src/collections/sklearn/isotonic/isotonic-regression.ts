/**
 * Hub entry for isotonic regression, the best monotone fit found by pooling adjacent violators, on one-dimensional
 * regression curves.
 */

import { isotonicRegression as isotonicRegressionMethod } from 'mdml-engine/methods/linear'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** Isotonic regression: the best monotone fit, by pooling adjacent violators. */
export const isotonicRegression: HubEntry = defineEntry({
  id: 'sklearn/isotonic/isotonic-regression',
  title: 'Isotonic regression',
  summary: 'The best monotone fit, by pooling adjacent violators: a staircase that never steps down.',
  collection: 'sklearn/isotonic',
  tags: ['regression', 'monotone', 'nonparametric'],
  datasets: ['regression1d'],
  dataset: 'stepCurve',
  knobs: { steps: 3 },
  ...fitEstimator(isotonicRegressionMethod(), { surface: true }),
  views: [{ record: 'posteriorGrid', title: 'Fit' }],
  methods: [{ file: 'methods/linear/isotonic.ts', name: 'isotonicRegression' }],
})
