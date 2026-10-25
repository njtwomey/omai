// What the explainability entries share (private to collections/explainability): the model a demo explains, fitted
// to the chosen dataset as a classifier or a regressor.
import type { Dataset } from 'whitebox-engine/data'
import { type Fitted, type LearnData, matrixView } from 'whitebox-engine/learn'
import {
  gradientBoostingClassifier,
  gradientBoostingRegressor,
  randomForestClassifier,
  randomForestRegressor,
} from 'whitebox-engine/methods/ensemble'
import { kNeighborsClassifier, kNeighborsRegressor } from 'whitebox-engine/methods/neighbours'
import type { Key } from 'whitebox-engine/random'
import { choice, int, type Space } from 'whitebox-engine/space'
import { learnDataOf } from '../../entry/run.ts'

/** The models the explainability demos fit. */
export type ExplainedModel = 'random forest' | 'gradient boosting' | 'k-nearest neighbours'

/** The model control, over the models given. */
export function modelControl(models: readonly ExplainedModel[], fallback: ExplainedModel): Space {
  return choice([...models], { default: fallback, label: 'model' })
}

/** The row control: which row of the data to explain (clamped to the data). */
export const rowControl: Space = int(0, 400, { default: 0, label: 'row explained' })

/** Fit the named model to a dataset: a classifier for classification data, a regressor otherwise. */
export function fitModel(
  data: Dataset | undefined,
  model: ExplainedModel,
  key: Key,
): { readonly fitted: Fitted; readonly learn: LearnData; readonly n: number } {
  const learn = learnDataOf(data)
  const classify = data?.kind === 'classification'
  const est =
    model === 'gradient boosting'
      ? classify
        ? gradientBoostingClassifier({ nEstimators: 40, maxDepth: 3 })
        : gradientBoostingRegressor({ nEstimators: 60, maxDepth: 3 })
      : model === 'k-nearest neighbours'
        ? classify
          ? kNeighborsClassifier({ nNeighbors: 10 })
          : kNeighborsRegressor({ nNeighbors: 10 })
        : classify
          ? randomForestClassifier({ nEstimators: 30, maxDepth: 6 })
          : randomForestRegressor({ nEstimators: 30, maxDepth: 6 })
  const fitted = est.fit(learn, { key, explain: false })
  return { fitted, learn, n: matrixView(learn.x, 'fitModel', { allowNaN: true }).n }
}
