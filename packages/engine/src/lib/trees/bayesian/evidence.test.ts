import { describe, expect, it } from 'vitest'
import { lgamma } from '../../numerics/special/index.ts'
import { item, sum, toFlat } from '../../tensor/index.ts'
import { dirichletLeaf, normalGammaLeaf, regressionPrior } from './leaf.ts'
import { treePrior } from './prior.ts'
import { quantileRules, scoreTree, treeSpace } from './space.ts'
import { bayesData, bayesLeaf, bayesSplit, treeLogEvidence } from './tree.ts'

const evidenceOf = (model: ReturnType<typeof normalGammaLeaf>, y: number[]): number =>
  item(model.logEvidence(sum(model.statistics(y), 0)))

describe('the exact evidence of a leaf', () => {
  it('Beta-Bernoulli equals the product of sequential predictive probabilities', () => {
    const model = dirichletLeaf(2, 1)
    const labels = [1, 0, 0, 1, 1, 1, 0]
    let logP = 0
    const counts: number[] = [0, 0]
    for (const y of labels) {
      const n = (counts[0] ?? 0) + (counts[1] ?? 0)
      logP += Math.log(((counts[y] ?? 0) + 1) / (n + 2))
      counts[y] = (counts[y] ?? 0) + 1
    }
    expect(evidenceOf(model, labels)).toBeCloseTo(logP, 12)
  })

  it('Dirichlet with three classes equals the sequential predictive product', () => {
    const model = dirichletLeaf(3, 0.5)
    const labels = [0, 2, 2, 1, 2, 0, 0, 0]
    let logP = 0
    const counts: number[] = [0, 0, 0]
    labels.forEach((y, t) => {
      logP += Math.log(((counts[y] ?? 0) + 0.5) / (t + 1.5))
      counts[y] = (counts[y] ?? 0) + 1
    })
    expect(evidenceOf(model, labels)).toBeCloseTo(logP, 12)
  })

  it('Normal-Gamma equals the product of sequential Student-t predictive densities', () => {
    const prior = { m0: 0.5, kappa0: 2, alpha0: 3, beta0: 1.5 }
    const model = normalGammaLeaf(prior)
    const y = [1.2, -0.3, 0.8, 2.1, 0.4]
    let { kappa0: kappa, alpha0: alpha, beta0: beta, m0: m } = prior
    let logP = 0
    for (const v of y) {
      const df = 2 * alpha
      const scale2 = (beta * (kappa + 1)) / (alpha * kappa)
      const z = (v - m) ** 2 / scale2
      logP +=
        item(lgamma((df + 1) / 2)) -
        item(lgamma(df / 2)) -
        0.5 * Math.log(df * Math.PI * scale2) -
        ((df + 1) / 2) * Math.log(1 + z / df)
      const next = kappa + 1
      beta += (kappa * (v - m) ** 2) / (2 * next)
      m = (kappa * m + v) / next
      kappa = next
      alpha += 0.5
    }
    expect(evidenceOf(model, y)).toBeCloseTo(logP, 10)
  })

  it('Normal-Gamma equals a direct numerical integral over the mean and the precision', () => {
    const prior = { m0: 0, kappa0: 1, alpha0: 2, beta0: 1 }
    const model = normalGammaLeaf(prior)
    const y = [0.7, -0.2, 1.1]
    const lg = (x: number): number => item(lgamma(x))
    let total = 0
    const dtau = 0.01
    const dmu = 0.02
    for (let tau = dtau / 2; tau < 40; tau += dtau) {
      const logGamma =
        prior.alpha0 * Math.log(prior.beta0) - lg(prior.alpha0) + (prior.alpha0 - 1) * Math.log(tau) - prior.beta0 * tau
      for (let mu = -8; mu < 8; mu += dmu) {
        const logPrior =
          0.5 * Math.log((prior.kappa0 * tau) / (2 * Math.PI)) - 0.5 * prior.kappa0 * tau * (mu - prior.m0) ** 2
        const logLike = y.reduce((a, v) => a + 0.5 * Math.log(tau / (2 * Math.PI)) - 0.5 * tau * (v - mu) ** 2, 0)
        total += Math.exp(logGamma + logPrior + logLike) * dtau * dmu
      }
    }
    expect(evidenceOf(model, y)).toBeCloseTo(Math.log(total), 3)
  })

  it('an empty leaf has evidence one', () => {
    expect(evidenceOf(normalGammaLeaf({ m0: 0, kappa0: 1, alpha0: 2, beta0: 1 }), [])).toBeCloseTo(0, 12)
  })
})

describe('the evidence of a tree', () => {
  it('is the sum of its leaves, and a clean step prefers the split', () => {
    const xs = [[0], [1], [2], [3], [4], [5]]
    const y = [0, 0.1, -0.1, 5, 5.1, 4.9]
    const model = normalGammaLeaf(regressionPrior(y))
    const data = bayesData(xs, y, model)
    const split = bayesSplit(0, 2.5, bayesLeaf, bayesLeaf)
    const direct = evidenceOf(model, y.slice(0, 3)) + evidenceOf(model, y.slice(3))
    expect(treeLogEvidence(data, split)).toBeCloseTo(direct, 12)
    expect(treeLogEvidence(data, split)).toBeGreaterThan(treeLogEvidence(data, bayesLeaf))
  })

  it('scores a tree in a space with the exact prior', () => {
    const xs = [[0], [1], [2], [3], [4], [5]]
    const y = [0, 0.1, -0.1, 5, 5.1, 4.9]
    const model = normalGammaLeaf(regressionPrior(y))
    const prior = treePrior({ alpha: 0.95, maxDepth: 2, minLeaf: 1 })
    const space = treeSpace(bayesData(xs, y, model), prior, quantileRules(xs, 10))
    const rules = space.rules.thresholds.length
    expect(rules).toBe(5)
    // The stump: it does not split, with probability 1 - alpha.
    expect(Math.exp(scoreTree(space, bayesLeaf).logPrior)).toBeCloseTo(1 - 0.95, 12)
    // One split at the root: alpha, one rule out of five, both children leaves at depth 1 (0.95 / 4 split probability).
    const one = bayesSplit(0, 2.5, bayesLeaf, bayesLeaf)
    const p1 = (0.95 / 5) * (1 - 0.95 / 4) ** 2
    expect(Math.exp(scoreTree(space, one).logPrior)).toBeCloseTo(p1, 12)
    expect(toFlat(model.statistics(y)).length).toBe(y.length * 3)
  })
})
