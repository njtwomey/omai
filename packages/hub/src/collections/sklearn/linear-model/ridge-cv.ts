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
  ...fitEstimator(ridgeCV()),
  views: [
    { record: 'regularisationPath', title: 'Regularisation path' },
    { record: 'crossValidationCurve', title: 'Cross-validation', area: 'side' },
    { record: 'coefficientTable', title: 'Coefficients', area: 'side' },
  ],
  methods: [{ file: 'methods/linear/regression.ts', name: 'ridgeCV' }],
})
