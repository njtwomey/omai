import { sliceAxis, type Tensor } from '../../lib/tensor/index.ts'
import { learnGolden, num, str } from '../../testing/learn.ts'
import {
  type LogisticRegressionHyper,
  logisticRegression,
  probitRegression,
  ridgeClassifier,
} from './classification.ts'

type In = readonly [Tensor, Tensor]

learnGolden('linear classifiers against scikit-learn and statsmodels', 'methods/linear/classification', {
  logistic: (inputs, p) => {
    const [x, y] = inputs as In
    const penalty = str(p, 'penalty') as LogisticRegressionHyper['penalty']
    const f = logisticRegression({
      C: num(p, 'C'),
      penalty,
      l1Ratio: num(p, 'l1Ratio'),
      tol: 1e-12,
      maxIter: 100000,
    }).fit({ x, y })
    const binary = f.params.coef.length === 1
    return {
      coef: binary ? (f.params.coef[0] ?? []) : f.params.coef,
      intercept: f.params.intercept,
      proba: f.predictProba?.(sliceAxis(x, 0, 0, 5)) as Tensor,
    }
  },
  ridgeClassifier: (inputs) => {
    const [x, y] = inputs as In
    const f = ridgeClassifier({ alpha: 2 }).fit({ x, y })
    return { coef: f.params.coef, intercept: f.params.intercept }
  },
  probit: (inputs) => {
    const [x, y] = inputs as In
    const f = probitRegression().fit({ x, y })
    const table = f.exposition.find((r) => r.kind === 'coefficientTable')
    return {
      params: [...(f.params.coef[0] ?? []), f.params.intercept[0] ?? 0],
      bse: table?.kind === 'coefficientTable' ? (table.standardErrors ?? []) : [],
    }
  },
})
