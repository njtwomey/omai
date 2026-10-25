/**
 * Hub entry for AdaBoost, which reweights the examples the ensemble gets wrong and adds another weak learner, on
 * two-dimensional and tabular classification and regression data.
 */

import { adaBoostClassifier, adaBoostRegressor } from 'mdml-engine/methods/ensemble'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEither } from '../../../entry/run.ts'

/** AdaBoost: reweight the examples the ensemble gets wrong and add another weak learner. */
export const adaboost: HubEntry = defineEntry({
  id: 'sklearn/ensemble/adaboost',
  title: 'AdaBoost',
  summary: 'Reweight the examples the ensemble gets wrong and add another weak learner.',
  collection: 'sklearn/ensemble',
  tags: ['classification', 'regression', 'ensembles', 'boosting'],
  datasets: ['classification2d', 'regression1d', 'classificationTabular', 'regressionTabular'],
  ...fitEither(adaBoostClassifier({ nEstimators: 30 }), adaBoostRegressor({ nEstimators: 30 }), { surface: true }),
  views: [
    { record: 'decisionSurface', title: 'Predictions' },
    { record: 'ensembleStages', title: 'Stages', area: 'side' },
    { record: 'tree', title: 'A member tree', area: 'side' },
  ],
  methods: [
    { file: 'methods/ensemble/adaboost.ts', name: 'adaBoostClassifier' },
    { file: 'methods/ensemble/adaboost.ts', name: 'adaBoostRegressor' },
  ],
})
