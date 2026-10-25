/**
 * Hub entry for extra trees, a forest of extremely randomised trees, on two-dimensional and tabular classification and
 * regression data.
 */

import { extraTreesClassifier, extraTreesRegressor } from 'mdml-engine/methods/ensemble'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEither } from '../../../entry/run.ts'

/** Extra trees: a forest of extremely randomised trees. */
export const extraTrees: HubEntry = defineEntry({
  id: 'sklearn/ensemble/extra-trees',
  title: 'Extra trees',
  summary: 'A forest of extremely randomised trees.',
  collection: 'sklearn/ensemble',
  tags: ['classification', 'regression', 'ensembles', 'trees'],
  datasets: ['classification2d', 'regression1d', 'classificationTabular', 'regressionTabular'],
  ...fitEither(extraTreesClassifier({ nEstimators: 25 }), extraTreesRegressor({ nEstimators: 25 }), { surface: true }),
  views: [
    { record: 'decisionSurface', title: 'Predictions' },
    { record: 'ensembleStages', title: 'Stages', area: 'side' },
    { record: 'tree', title: 'A member tree', area: 'below' },
  ],
  methods: [
    { file: 'methods/ensemble/forest.ts', name: 'extraTreesClassifier' },
    { file: 'methods/ensemble/forest.ts', name: 'extraTreesRegressor' },
  ],
})
