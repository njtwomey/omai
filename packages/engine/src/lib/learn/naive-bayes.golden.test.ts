import { learnGolden, num, str } from '../../testing/learn.ts'
import type { Tensor } from '../tensor/index.ts'
import { aode, type NaiveBayesFamily, naiveBayes, treeAugmentedNaiveBayes } from './naive-bayes.ts'

learnGolden('naive Bayes against sklearn.naive_bayes', 'learn/naive_bayes', {
  nb: ([x, y, q, w], p) => {
    const family = str(p, 'family') as NaiveBayesFamily
    const est = naiveBayes({
      family,
      alpha: num(p, 'alpha', 1),
      varSmoothing: num(p, 'varSmoothing', 1e-9),
      binarize: num(p, 'binarize', 0),
      norm: p['norm'] === true,
      fitPrior: p['fitPrior'] !== false,
      ...(p['minCategories'] === undefined ? {} : { minCategories: num(p, 'minCategories') }),
    })
    const f = est.fit({ x: x as Tensor, y: y as Tensor, ...(w === undefined ? {} : { weights: w as Tensor }) })
    const prior = Array.from(f.classLogPrior)
    return { proba: f.predictProba(q as Tensor), predict: f.predict(q as Tensor), logPrior: prior }
  },
  tan: ([x, y, q], p) => {
    const f = treeAugmentedNaiveBayes({ alpha: num(p, 'alpha'), root: num(p, 'root') }).fit({
      x: x as Tensor,
      y: y as Tensor,
    })
    return {
      edges: f.edges.map((e) => [e.parent, e.child]),
      weights: f.edges.map((e) => e.weight),
      proba: f.predictProba(q as Tensor),
    }
  },
  aode: ([x, y, q], p) => {
    const f = aode({ alpha: num(p, 'alpha'), minCount: num(p, 'minCount') }).fit({ x: x as Tensor, y: y as Tensor })
    return { proba: f.predictProba(q as Tensor) }
  },
})
