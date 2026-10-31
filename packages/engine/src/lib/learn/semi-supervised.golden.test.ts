import { learnGolden, logisticRegression, num, str } from '../../testing/learn.ts'
import type { Tensor } from '../tensor/index.ts'
import { gaussianNaiveBayes } from './naive-bayes.ts'
import { labelPropagation, labelSpreading, selfTraining } from './semi-supervised.ts'

const graph = (p: Readonly<Record<string, unknown>>): Parameters<typeof labelPropagation>[0] => ({
  kernel: str(p, 'kernel') as 'rbf' | 'knn',
  ...(p['gamma'] === undefined ? {} : { gamma: num(p, 'gamma') }),
  ...(p['n_neighbors'] === undefined ? {} : { nNeighbors: num(p, 'n_neighbors') }),
  maxIter: num(p, 'maxIter'),
  tol: num(p, 'tol'),
})

learnGolden('semi-supervised learning against sklearn.semi_supervised', 'learn/semi_supervised', {
  propagation: ([x, y, q], p) => {
    const f = labelPropagation(graph(p)).fit({ x: x as Tensor, y: y as Tensor })
    return {
      distributions: [...f.labelDistributions.map((r) => [...r])],
      transduction: [...f.transduction],
      proba: f.predictProba(q as Tensor),
      predict: f.predict(q as Tensor),
    }
  },
  spreading: ([x, y, q], p) => {
    const f = labelSpreading({ ...graph(p), alpha: num(p, 'alpha') }).fit({ x: x as Tensor, y: y as Tensor })
    return {
      distributions: [...f.labelDistributions.map((r) => [...r])],
      transduction: [...f.transduction],
      proba: f.predictProba(q as Tensor),
      predict: f.predict(q as Tensor),
    }
  },
  selftraining: ([x, y, q], p) => {
    const crit = str(p, 'criterion') as 'threshold' | 'k_best'
    const f = selfTraining(str(p, 'base') === 'logistic' ? logisticRegression({ C: 1 }) : gaussianNaiveBayes(), {
      criterion: crit,
      maxIter: num(p, 'maxIter'),
      ...(p['threshold'] === undefined ? {} : { threshold: num(p, 'threshold') }),
      ...(p['kBest'] === undefined ? {} : { kBest: num(p, 'kBest') }),
    }).fit({ x: x as Tensor, y: y as Tensor })
    return {
      transduction: [...f.transduction],
      iter: [...f.labelledIter],
      proba: f.predictProba?.(q as Tensor) ?? [],
      predict: f.predict(q as Tensor),
    }
  },
})
