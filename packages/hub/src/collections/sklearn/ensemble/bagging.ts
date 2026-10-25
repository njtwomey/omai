/**
 * Hub entry for bagging (bootstrap aggregation), the same learner fitted to resampled data and averaged, on
 * two-dimensional and tabular classification and regression data.
 */

import { baggingClassifier, baggingRegressor } from 'whitebox-engine/methods/ensemble'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEither } from '../../../entry/run.ts'

/** Bagging: bootstrap aggregation: the same learner on resampled data, averaged. */
export const bagging: HubEntry = defineEntry({
  id: 'sklearn/ensemble/bagging',
  title: 'Bagging',
  summary: 'Bootstrap aggregation: the same learner on resampled data, averaged.',
  collection: 'sklearn/ensemble',
  tags: ['classification', 'regression', 'ensembles', 'bagging'],
  datasets: ['classification2d', 'regression1d', 'classificationTabular', 'regressionTabular'],
  ...fitEither(baggingClassifier({ nEstimators: 20 }), baggingRegressor({ nEstimators: 20 }), { surface: true }),
  views: [
    { record: 'decisionSurface', title: 'Predictions' },
    { record: 'ensembleStages', title: 'Stages', area: 'side' },
  ],
  methods: [
    { file: 'methods/ensemble/bagging.ts', name: 'baggingClassifier' },
    { file: 'methods/ensemble/bagging.ts', name: 'baggingRegressor' },
  ],
})
