import { decisionTreeClassifier, decisionTreeRegressor } from 'whitebox-engine/methods/trees'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEither } from '../../../entry/run.ts'

/** Decision tree: greedy recursive splits, each the one that most reduces impurity. */
export const decisionTree: HubEntry = defineEntry({
  id: 'sklearn/tree/decision-tree',
  title: 'Decision tree',
  summary: 'Greedy recursive splits, each the one that most reduces impurity.',
  collection: 'sklearn/tree',
  tags: ['classification', 'regression', 'trees'],
  datasets: ['classification2d', 'regression1d', 'classificationTabular', 'regressionTabular'],
  ...fitEither(decisionTreeClassifier({ maxDepth: 4 }), decisionTreeRegressor({ maxDepth: 4 }), { surface: true }),
  views: [
    { record: 'tree', title: 'The tree' },
    { record: 'decisionSurface', title: 'Predictions', area: 'side' },
    { record: 'pruningPath', title: 'Cost-complexity pruning', area: 'side' },
  ],
  methods: [
    { file: 'methods/trees/grow.ts', name: 'decisionTreeClassifier' },
    { file: 'methods/trees/grow.ts', name: 'decisionTreeRegressor' },
  ],
})
