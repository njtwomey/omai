/**
 * Hub entry for stacking, a tree, k nearest neighbours and a linear model combined by a final model fitted to their
 * out-of-fold predictions, on two-dimensional classification data and one-dimensional regression curves.
 */

import { stackingClassifier, stackingRegressor } from 'mdml-engine/methods/ensemble'
import { logisticRegression, ridge } from 'mdml-engine/methods/linear'
import { kNeighborsClassifier, kNeighborsRegressor } from 'mdml-engine/methods/neighbours'
import { decisionTreeClassifier, decisionTreeRegressor } from 'mdml-engine/methods/trees'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEither } from '../../../entry/run.ts'

/** Stacking: a tree, k nearest neighbours and a linear model, combined by a model fitted to their out-of-fold outputs. */
export const stacking: HubEntry = defineEntry({
  id: 'sklearn/ensemble/stacking',
  title: 'Stacking',
  summary: 'A final model learns how to combine the members’ out-of-fold predictions.',
  collection: 'sklearn/ensemble',
  tags: ['classification', 'regression', 'ensembles'],
  datasets: ['classification2d', 'regression1d'],
  ...fitEither(
    stackingClassifier({
      estimators: [
        ['tree', decisionTreeClassifier({ maxDepth: 4 })],
        ['knn', kNeighborsClassifier({ nNeighbors: 7 })],
        ['logistic', logisticRegression()],
      ],
    }),
    stackingRegressor({
      estimators: [
        ['tree', decisionTreeRegressor({ maxDepth: 4 })],
        ['knn', kNeighborsRegressor({ nNeighbors: 7 })],
        ['ridge', ridge()],
      ],
    }),
    { surface: true },
  ),
  views: [
    { record: 'decisionSurface', title: 'Predictions' },
    { record: 'ensembleStages', title: 'Members', area: 'side' },
  ],
  methods: [
    { file: 'methods/ensemble/stacking.ts', name: 'stackingClassifier' },
    { file: 'methods/ensemble/stacking.ts', name: 'stackingRegressor' },
  ],
})
