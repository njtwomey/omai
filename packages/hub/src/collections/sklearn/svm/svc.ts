import { svc as svcMethod } from 'whitebox-engine/methods/svm'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** Support vector classifier: the widest margin between classes, in a kernel’s feature space, solved by SMO. */
export const svc: HubEntry = defineEntry({
  id: 'sklearn/svm/svc',
  title: 'Support vector classifier',
  summary: 'The widest margin between classes, in a kernel’s feature space, solved by SMO.',
  collection: 'sklearn/svm',
  tags: ['classification', 'kernels', 'margins'],
  datasets: ['classification2d', 'classificationTabular'],
  ...fitEstimator(svcMethod(), { surface: true }),
  views: [
    { record: 'decisionSurface', title: 'Decision surface' },
    { record: 'margin', title: 'Margin and support vectors', area: 'side' },
  ],
  methods: [{ file: 'methods/svm/classifier.ts', name: 'svc' }],
})
