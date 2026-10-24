import { ridgeCV } from 'whitebox-engine/methods/linear'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** Ridge (cross-validated): ridge with its penalty chosen by cross-validation along the path. */
export const ridgeCv: HubEntry = defineEntry({
  id: 'sklearn/linear-model/ridge-cv',
  title: 'Ridge (cross-validated)',
  summary: 'Ridge with its penalty chosen by cross-validation along the path.',
  collection: 'sklearn/linear-model',
  tags: ['regression', 'linear', 'cross-validation'],
  datasets: ['regressionTabular', 'regression1d'],
  // A log grid of 30 penalties, so the path and the cross-validation curve are curves rather than sklearn's three points.
  ...fitEstimator(ridgeCV({ alphas: Array.from({ length: 30 }, (_, i) => 10 ** (-3 + (7 * i) / 29)) })),
  views: [
    { record: 'regularisationPath', title: 'Regularisation path' },
    { record: 'crossValidationCurve', title: 'Cross-validation', area: 'side' },
    { record: 'coefficientTable', title: 'Coefficients', area: 'side' },
  ],
  methods: [{ file: 'methods/linear/regression.ts', name: 'ridgeCV' }],
})
