/**
 * Hub entry for the elastic net with its penalty chosen by cross-validation, on tabular regression data with a few
 * informative features and one-dimensional regression curves.
 */

import { elasticNetCV } from 'mdml-engine/methods/linear'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** Elastic net (cross-validated): the elastic net with its penalty chosen by cross-validation. */
export const elasticNetCv: HubEntry = defineEntry({
  id: 'sklearn/linear-model/elastic-net-cv',
  title: 'Elastic net (cross-validated)',
  summary: 'The elastic net with its penalty chosen by cross-validation.',
  collection: 'sklearn/linear-model',
  tags: ['regression', 'linear', 'cross-validation'],
  datasets: ['regressionTabular', 'regression1d'],
  dataset: 'sparseUncorrelated',
  ...fitEstimator(elasticNetCV()),
  views: [
    { record: 'regularisationPath', title: 'Regularisation path' },
    { record: 'crossValidationCurve', title: 'Cross-validation', area: 'side' },
    { record: 'coefficientTable', title: 'Coefficients', area: 'side' },
  ],
  methods: [{ file: 'methods/linear/regression.ts', name: 'elasticNetCV' }],
})
