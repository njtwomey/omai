/**
 * Hub entry for gradient boosting, each new tree fitted to the gradient of the loss of the ensemble so far, on
 * two-dimensional and tabular classification and regression data.
 */

import { gradientBoostingClassifier, gradientBoostingRegressor } from 'mdml-engine/methods/ensemble'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEither } from '../../../entry/run.ts'

/** Gradient boosting: fit each new tree to the gradient of the loss of the ensemble so far. */
export const gradientBoosting: HubEntry = defineEntry({
  id: 'sklearn/ensemble/gradient-boosting',
  title: 'Gradient boosting',
  summary: 'Fit each new tree to the gradient of the loss of the ensemble so far.',
  collection: 'sklearn/ensemble',
  tags: ['classification', 'regression', 'ensembles', 'boosting'],
  datasets: ['classification2d', 'regression1d', 'classificationTabular', 'regressionTabular'],
  ...fitEither(gradientBoostingClassifier({ nEstimators: 50 }), gradientBoostingRegressor({ nEstimators: 50 }), {
    surface: true,
  }),
  views: [
    { record: 'decisionSurface', title: 'Predictions' },
    { record: 'ensembleStages', title: 'Stages', area: 'side' },
    { record: 'tree', title: 'A member tree', area: 'below' },
  ],
  methods: [
    { file: 'methods/ensemble/gradient-boosting.ts', name: 'gradientBoostingClassifier' },
    { file: 'methods/ensemble/gradient-boosting.ts', name: 'gradientBoostingRegressor' },
  ],
})
