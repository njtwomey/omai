/**
 * Hub entry for the ridge classifier, ridge regression on plus-or-minus-one targets classifying by the sign, on
 * two-dimensional classification data and tabular classification data.
 */

import { ridgeClassifier as ridgeClassifierMethod } from 'whitebox-engine/methods/linear'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** Ridge classifier: ridge regression on ±1 targets, classifying by the sign. */
export const ridgeClassifier: HubEntry = defineEntry({
  id: 'sklearn/linear-model/ridge-classifier',
  title: 'Ridge classifier',
  summary: 'Ridge regression on ±1 targets, classifying by the sign.',
  collection: 'sklearn/linear-model',
  tags: ['classification', 'linear'],
  datasets: ['classification2d', 'classificationTabular'],
  ...fitEstimator(ridgeClassifierMethod(), { surface: true }),
  views: [
    { record: 'decisionSurface', title: 'Decision surface' },
    { record: 'coefficientTable', title: 'Coefficients', area: 'side' },
  ],
  methods: [{ file: 'methods/linear/classification.ts', name: 'ridgeClassifier' }],
})
