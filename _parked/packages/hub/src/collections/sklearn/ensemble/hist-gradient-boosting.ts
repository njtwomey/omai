/**
 * Hub entry for histogram gradient boosting, gradient boosting on binned features with trees grown leaf by leaf, on
 * two-dimensional and tabular classification and regression data.
 */

import { histGradientBoostingClassifier, histGradientBoostingRegressor } from 'mdml-engine/methods/ensemble'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEither } from '../../../entry/run.ts'

/** Histogram gradient boosting: gradient boosting on binned features, grown leaf by leaf. */
export const histGradientBoosting: HubEntry = defineEntry({
  id: 'sklearn/ensemble/hist-gradient-boosting',
  title: 'Histogram gradient boosting',
  summary: 'Gradient boosting on binned features, grown leaf by leaf.',
  collection: 'sklearn/ensemble',
  tags: ['classification', 'regression', 'ensembles', 'boosting'],
  datasets: ['classification2d', 'regression1d', 'classificationTabular', 'regressionTabular'],
  ...fitEither(histGradientBoostingClassifier({ maxIter: 50 }), histGradientBoostingRegressor({ maxIter: 50 }), {
    surface: true,
  }),
  views: [
    { record: 'decisionSurface', title: 'Predictions' },
    { record: 'ensembleStages', title: 'Stages', area: 'side' },
  ],
  methods: [
    { file: 'methods/ensemble/histogram-boosting.ts', name: 'histGradientBoostingClassifier' },
    { file: 'methods/ensemble/histogram-boosting.ts', name: 'histGradientBoostingRegressor' },
  ],
})
