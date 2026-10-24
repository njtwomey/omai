import { onlinePerceptron } from 'whitebox-engine/methods/linear'
import { defineEntry, type HubEntry } from '../../entry/entry.ts'
import { fitEstimator } from '../../entry/run.ts'

/** Rosenblatt's perceptron run online: every mistake moves the boundary towards the misclassified row. */
export const perceptron: HubEntry = defineEntry({
  id: 'online/perceptron',
  title: 'Online perceptron',
  summary: 'Every mistake adds the misclassified row to the weights; on separable data the mistakes stop.',
  collection: 'online',
  tags: ['online learning', 'classification', 'linear', 'mistake bound'],
  datasets: ['classification2d', 'classificationTabular'],
  dataset: 'blobs',
  ...fitEstimator(onlinePerceptron({ maxIter: 20 })),
  views: [
    { record: 'learningCurve', title: 'Mistakes as rows arrive' },
    { record: 'decisionSurface', title: 'Decision surface', area: 'side' },
    { record: 'curves', title: 'Coefficients as rows arrive', area: 'below' },
  ],
  methods: [{ file: 'methods/linear/online.ts', name: 'onlinePerceptron' }],
  references: ['Rosenblatt (1958), The perceptron'],
})
