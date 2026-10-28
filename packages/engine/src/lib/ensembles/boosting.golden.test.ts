import { learnGolden, num, str } from '../../testing/learn.ts'
import { sliceAxis, type Tensor, toFlat } from '../tensor/index.ts'
import { decisionTreeClassifier } from '../trees/decision-tree.ts'
import { adaBoostClassifier } from './adaboost.ts'
import {
  type ClassificationLoss,
  gradientBoostingClassifier,
  gradientBoostingRegressor,
  type RegressionLoss,
} from './gradient-boosting.ts'
import { histGradientBoostingClassifier, histGradientBoostingRegressor } from './histogram-boosting.ts'

type In = readonly [Tensor, Tensor]

const stagesOf = (decisions: readonly Float64Array[], width: number): number[][] =>
  decisions.flatMap((d) =>
    Array.from({ length: d.length / width }, (_, i) => {
      const row = Array.from(d.subarray(i * width, (i + 1) * width))
      const mean = width > 1 ? row.reduce((a, b) => a + b, 0) / width : 0
      return row.map((v) => v - mean)
    }),
  )

learnGolden('gradient boosting against scikit-learn', 'ensembles/gradient_boosting', {
  regressor: (inputs, p) => {
    const [x, y] = inputs as In
    const f = gradientBoostingRegressor({
      loss: str(p, 'loss') as RegressionLoss,
      alpha: num(p, 'alpha'),
      nEstimators: 15,
      maxDepth: 2,
      learningRate: 0.3,
    }).fit({ x, y })
    const head = sliceAxis(x, 0, 0, 8)
    return {
      predict: f.predict(head),
      staged: f.stagedDecisionFunction(head).map((d) => Array.from(d)),
    }
  },
  classifier: (inputs, p) => {
    const [x, y] = inputs as In
    const f = gradientBoostingClassifier({
      loss: str(p, 'loss') as ClassificationLoss,
      nEstimators: 10,
      maxDepth: 2,
      learningRate: 0.5,
    }).fit({ x, y })
    const head = sliceAxis(x, 0, 0, 8)
    const k = f.params.init.length
    if (str(p, 'loss') === 'exponential')
      return { proba: f.predictProba?.(head) as Tensor, decision: toFlat(f.decisionFunction(head)).map((v) => [v]) }
    return { proba: f.predictProba?.(head) as Tensor, staged: stagesOf(f.stagedDecisionFunction(head), k) }
  },
})

learnGolden('AdaBoost (SAMME) against scikit-learn', 'ensembles/adaboost', {
  classifier: (inputs, p) => {
    const [x, y] = inputs as In
    const est = adaBoostClassifier({
      estimator: decisionTreeClassifier({ maxDepth: num(p, 'maxDepth'), record: 0 }),
      nEstimators: num(p, 'nEstimators'),
      learningRate: num(p, 'learningRate'),
    })
    const s = est.session({ x, y }, { mode: 'eager' })
    const f = s.fittedAt(s.t)
    const head = sliceAxis(x, 0, 0, 6)
    const rows = (t: Tensor): number[][] => {
      const d = toFlat(t)
      const width = d.length / 6
      return Array.from({ length: 6 }, (_, i) => Array.from(d.slice(i * width, (i + 1) * width)))
    }
    const staged: number[][] = []
    for (let t = 1; t <= s.t; t++) staged.push(...rows(s.fittedAt(t).decisionFunction?.(head) as Tensor))
    return {
      weights: f.params.weights,
      errors: [...s.state.errors],
      decision: rows(f.decisionFunction?.(head) as Tensor),
      proba: f.predictProba?.(head) as Tensor,
      staged,
      predict: f.predict(head),
    }
  },
})

learnGolden('histogram gradient boosting against scikit-learn', 'ensembles/histogram_boosting', {
  regressor: (inputs, p) => {
    const [x, y] = inputs as In
    const f = histGradientBoostingRegressor({
      loss: str(p, 'loss') as RegressionLoss,
      l2: num(p, 'l2'),
      maxBins: num(p, 'maxBins'),
      nEstimators: 12,
      learningRate: 0.2,
      maxLeafNodes: 6,
      minSamplesLeaf: 5,
    }).fit({ x, y })
    const head = sliceAxis(x, 0, 0, 8)
    const link = (v: number): number => (str(p, 'loss') === 'poisson' ? Math.exp(v) : v)
    return { predict: f.predict(head), staged: f.stagedDecisionFunction(head).map((d) => Array.from(d, link)) }
  },
  classifier: (inputs) => {
    const [x, y] = inputs as In
    const f = histGradientBoostingClassifier({
      nEstimators: 12,
      learningRate: 0.2,
      maxLeafNodes: 6,
      minSamplesLeaf: 5,
    }).fit({ x, y })
    const head = sliceAxis(x, 0, 0, 8)
    return {
      proba: f.predictProba?.(head) as Tensor,
      staged: stagesOf(f.stagedDecisionFunction(head), f.params.init.length),
    }
  },
})
