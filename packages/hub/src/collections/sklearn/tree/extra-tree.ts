/**
 * Hub entry for the extremely randomised tree, whose split thresholds are drawn at random rather than searched, on
 * two-dimensional and tabular classification and regression data.
 */

import { extraTreeClassifier, extraTreeRegressor } from 'mdml-engine/methods/trees'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEither } from '../../../entry/run.ts'

/** Extremely randomised tree: a tree whose split thresholds are drawn at random, not searched. */
export const extraTree: HubEntry = defineEntry({
  id: 'sklearn/tree/extra-tree',
  title: 'Extremely randomised tree',
  summary: 'A tree whose split thresholds are drawn at random, not searched.',
  collection: 'sklearn/tree',
  tags: ['classification', 'regression', 'trees', 'randomised'],
  datasets: ['classification2d', 'regression1d', 'classificationTabular', 'regressionTabular'],
  ...fitEither(extraTreeClassifier({ maxDepth: 6 }), extraTreeRegressor({ maxDepth: 6 }), { surface: true }),
  views: [
    { record: 'tree', title: 'The tree' },
    { record: 'decisionSurface', title: 'Predictions', area: 'side' },
    { record: 'pruningPath', title: 'Cost-complexity pruning', area: 'side' },
  ],
  methods: [
    { file: 'methods/trees/grow.ts', name: 'extraTreeClassifier' },
    { file: 'methods/trees/grow.ts', name: 'extraTreeRegressor' },
  ],
})
