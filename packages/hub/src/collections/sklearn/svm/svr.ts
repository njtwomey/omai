/**
 * Hub entry for support vector regression, a tube of width epsilon around a kernel fit with the points outside it as
 * support vectors, on one-dimensional regression curves and tabular regression data.
 */

import { svr as svrMethod } from 'whitebox-engine/methods/svm'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** Support vector regression: a tube of width epsilon around a kernel fit; only points outside it are support vectors. */
export const svr: HubEntry = defineEntry({
  id: 'sklearn/svm/svr',
  title: 'Support vector regression',
  summary: 'A tube of width epsilon around a kernel fit; only points outside it are support vectors.',
  collection: 'sklearn/svm',
  tags: ['regression', 'kernels', 'margins'],
  datasets: ['regression1d', 'regressionTabular'],
  ...fitEstimator(svrMethod(), { surface: true }),
  views: [
    { record: 'decisionSurface', title: 'Fitted function' },
    { record: 'margin', title: 'Margin and support vectors', area: 'side' },
  ],
  methods: [{ file: 'methods/svm/regressor.ts', name: 'svr' }],
})
