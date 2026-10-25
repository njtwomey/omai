/**
 * Hub entry for the random forest, many deep trees on bootstrap samples and random feature subsets, averaged, on
 * two-dimensional and tabular classification and regression data.
 */

import { randomForestClassifier, randomForestRegressor } from 'mdml-engine/methods/ensemble'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEither } from '../../../entry/run.ts'

/** Random forest: many deep trees on bootstrap samples and random feature subsets, averaged. */
export const randomForest: HubEntry = defineEntry({
  id: 'sklearn/ensemble/random-forest',
  title: 'Random forest',
  summary: 'Many deep trees on bootstrap samples and random feature subsets, averaged.',
  collection: 'sklearn/ensemble',
  tags: ['classification', 'regression', 'ensembles', 'trees', 'bagging'],
  datasets: ['classification2d', 'regression1d', 'classificationTabular', 'regressionTabular'],
  ...fitEither(randomForestClassifier({ nEstimators: 25 }), randomForestRegressor({ nEstimators: 25 }), {
    surface: true,
  }),
  views: [
    { record: 'decisionSurface', title: 'Predictions' },
    { record: 'ensembleStages', title: 'Stages', area: 'side' },
    { record: 'tree', title: 'A member tree', area: 'below' },
  ],
  methods: [
    { file: 'methods/ensemble/forest.ts', name: 'randomForestClassifier' },
    { file: 'methods/ensemble/forest.ts', name: 'randomForestRegressor' },
  ],
})
