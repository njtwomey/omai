import { logisticRegression as logisticRegressionMethod } from 'whitebox-engine/methods/linear'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** Logistic regression: a linear model of the log odds, fitted by maximum likelihood. */
export const logisticRegression: HubEntry = defineEntry({
  id: 'sklearn/linear-model/logistic-regression',
  title: 'Logistic regression',
  summary: 'A linear model of the log odds, fitted by maximum likelihood.',
  collection: 'sklearn/linear-model',
  tags: ['classification', 'linear', 'probabilistic'],
  datasets: ['classification2d', 'classificationTabular'],
  ...fitEstimator(logisticRegressionMethod(), { surface: true }),
  views: [
    { record: 'decisionSurface', title: 'Decision surface' },
    { record: 'coefficientTable', title: 'Coefficients', area: 'side' },
  ],
  methods: [{ file: 'methods/linear/classification.ts', name: 'logisticRegression' }],
})
