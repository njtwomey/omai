import { votingClassifier, votingRegressor } from 'whitebox-engine/methods/ensemble'
import { logisticRegression, ridge } from 'whitebox-engine/methods/linear'
import { kNeighborsClassifier, kNeighborsRegressor } from 'whitebox-engine/methods/neighbours'
import { decisionTreeClassifier, decisionTreeRegressor } from 'whitebox-engine/methods/trees'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEither } from '../../../entry/run.ts'

/** Voting: a tree, k nearest neighbours and a linear model, combined by a vote or an average. */
export const voting: HubEntry = defineEntry({
  id: 'sklearn/ensemble/voting',
  title: 'Voting',
  summary: 'Different models side by side, combined by a (weighted) vote or an average of their outputs.',
  collection: 'sklearn/ensemble',
  tags: ['classification', 'regression', 'ensembles'],
  datasets: ['classification2d', 'regression1d'],
  ...fitEither(
    votingClassifier({
      estimators: [
        ['tree', decisionTreeClassifier({ maxDepth: 4 })],
        ['knn', kNeighborsClassifier({ nNeighbors: 7 })],
        ['logistic', logisticRegression()],
      ],
      voting: 'soft',
    }),
    votingRegressor({
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
    { file: 'methods/ensemble/stacking.ts', name: 'votingClassifier' },
    { file: 'methods/ensemble/stacking.ts', name: 'votingRegressor' },
  ],
})
