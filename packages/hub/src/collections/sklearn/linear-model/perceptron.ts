import { perceptron as perceptronMethod } from 'whitebox-engine/methods/linear'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** Perceptron: rosenblatt’s mistake-driven updates of a linear separator. */
export const perceptron: HubEntry = defineEntry({
  id: 'sklearn/linear-model/perceptron',
  title: 'Perceptron',
  summary: 'Rosenblatt’s mistake-driven updates of a linear separator.',
  collection: 'sklearn/linear-model',
  tags: ['classification', 'linear', 'online'],
  datasets: ['classification2d', 'classificationTabular'],
  ...fitEstimator(perceptronMethod(), { surface: true }),
  views: [
    { record: 'decisionSurface', title: 'Decision surface' },
    { record: 'coefficientTable', title: 'Coefficients', area: 'side' },
  ],
  methods: [{ file: 'methods/linear/classification.ts', name: 'perceptron' }],
})
