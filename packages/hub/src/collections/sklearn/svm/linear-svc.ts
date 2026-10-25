/**
 * Hub entry for the linear support vector classifier solved by dual coordinate descent, on two-dimensional
 * classification data and tabular classification data.
 */

import { linearSvc as linearSvcMethod } from 'whitebox-engine/methods/svm'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** Linear SVC: a linear support vector machine by dual coordinate descent. */
export const linearSvc: HubEntry = defineEntry({
  id: 'sklearn/svm/linear-svc',
  title: 'Linear SVC',
  summary: 'A linear support vector machine by dual coordinate descent.',
  collection: 'sklearn/svm',
  tags: ['classification', 'linear', 'margins'],
  datasets: ['classification2d', 'classificationTabular'],
  ...fitEstimator(linearSvcMethod(), { surface: true }),
  views: [
    { record: 'margin', title: 'Margin and support vectors' },
    { record: 'decisionSurface', title: 'Decision surface', area: 'side' },
    { record: 'coefficientTable', title: 'Coefficients', area: 'side' },
  ],
  methods: [{ file: 'methods/svm/linear.ts', name: 'linearSvc' }],
})
