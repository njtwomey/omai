import { gaussianNB, kNeighborsClassifier, learnGolden, ridge } from '../../../testing/learn.ts'
import { type Tensor, toFlat } from '../../tensor/index.ts'
import { pipeline } from '../compose.ts'
import { requireVerb } from '../estimator.ts'
import { polynomialFeatures, standardScaler } from '../preprocessing/index.ts'
import { crossValidate, crossValPredict, learningCurve, validationCurve } from './cross-validation.ts'
import { evaluateCandidates, gridSearchCV } from './search.ts'
import { kFold, stratifiedKFold } from './splitters.ts'

const poly = () =>
  pipeline([
    ['poly', polynomialFeatures()],
    ['scale', standardScaler()],
    ['ridge', ridge()],
  ])

learnGolden('cross-validation against sklearn.model_selection', 'learn/model_selection/cross_validation', {
  crossValidateRidge: ([x, y]) => {
    const r = crossValidate(
      ridge({ alpha: 1 }),
      { x: x as Tensor, y: y as Tensor },
      {
        cv: kFold(4),
        scoring: ['r2', 'neg_mean_squared_error'],
        returnTrainScore: true,
      },
    )
    return {
      testR2: r.testScores['r2'] ?? [],
      trainR2: r.trainScores?.['r2'] ?? [],
      testMse: r.testScores['neg_mean_squared_error'] ?? [],
      trainMse: r.trainScores?.['neg_mean_squared_error'] ?? [],
    }
  },
  crossValidateNb: ([x, y]) => {
    const r = crossValidate(
      gaussianNB(),
      { x: x as Tensor, y: y as Tensor },
      {
        cv: 3,
        scoring: ['accuracy', 'neg_log_loss', 'roc_auc', 'neg_brier_score'],
      },
    )
    return {
      accuracy: r.testScores['accuracy'] ?? [],
      logLoss: r.testScores['neg_log_loss'] ?? [],
      rocAuc: r.testScores['roc_auc'] ?? [],
      brier: r.testScores['neg_brier_score'] ?? [],
    }
  },
  crossValPredictRidge: ([x, y]) => ({
    predict: crossValPredict(ridge({ alpha: 1 }), { x: x as Tensor, y: y as Tensor }, { cv: kFold(5) }),
  }),
  crossValPredictProba: ([x, y]) => ({
    proba: crossValPredict(
      gaussianNB(),
      { x: x as Tensor, y: y as Tensor },
      { cv: stratifiedKFold(4), method: 'predictProba' },
    ),
  }),
  learningCurve: ([x, y]) => {
    const r = learningCurve(
      ridge({ alpha: 1 }),
      { x: x as Tensor, y: y as Tensor },
      { trainSizes: [0.2, 0.5, 1], cv: kFold(4), scoring: 'r2' },
    )
    return { sizes: [...r.values], train: r.trainScores, test: r.testScores }
  },
  validationCurve: ([x, y]) => {
    const r = validationCurve(ridge(), { x: x as Tensor, y: y as Tensor }, 'alpha', [0.01, 1, 100], {
      cv: kFold(3),
      scoring: 'neg_mean_absolute_error',
    })
    return { train: r.trainScores, test: r.testScores }
  },
})

learnGolden('searches against sklearn.model_selection', 'learn/model_selection/search', {
  gridPipeline: ([x, y, z]) => {
    const r = gridSearchCV(
      poly(),
      { 'poly.degree': [1, 2, 3], 'ridge.alpha': [0.1, 1, 10] },
      { x: x as Tensor, y: y as Tensor },
      {
        cv: kFold(4),
        scoring: 'neg_mean_squared_error',
        returnTrainScore: true,
      },
    )
    const best = r.bestEstimator
    if (best === undefined) throw new Error('no refit')
    return {
      mean: r.meanTestScore['neg_mean_squared_error'] ?? [],
      std: r.stdTestScore['neg_mean_squared_error'] ?? [],
      rank: r.rankTestScore['neg_mean_squared_error'] ?? [],
      splits: r.splitTestScores['neg_mean_squared_error'] ?? [],
      meanTrain: r.meanTrainScore?.['neg_mean_squared_error'] ?? [],
      bestIndex: r.bestIndex,
      predict: requireVerb(best, 'predict', 'test')(z as Tensor),
    }
  },
  gridKnn: ([x, y]) => {
    const r = gridSearchCV(
      kNeighborsClassifier(),
      { k: [1, 3, 5, 7, 9] },
      { x: x as Tensor, y: y as Tensor },
      {
        cv: stratifiedKFold(3),
        scoring: 'accuracy',
      },
    )
    return { mean: r.meanTestScore['accuracy'] ?? [], rank: r.rankTestScore['accuracy'] ?? [], bestIndex: r.bestIndex }
  },
  randomizedPipeline: ([x, y, degrees, alphas]) => {
    const a = toFlat(alphas as Tensor)
    const candidates = toFlat(degrees as Tensor).map((d, i) => ({ 'poly.degree': d, 'ridge.alpha': a[i] }))
    const r = evaluateCandidates(
      poly(),
      candidates,
      { x: x as Tensor, y: y as Tensor },
      { cv: kFold(3), scoring: 'r2', refit: false },
    )
    return { mean: r.meanTestScore['r2'] ?? [], rank: r.rankTestScore['r2'] ?? [] }
  },
})
