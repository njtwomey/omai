import { describe, expect, it } from 'vitest'
import { expectConformant } from '../../testing/check.ts'
import { jacfwd } from '../autodiff/index.ts'
import { meanOf, varianceOf } from '../base/index.ts'
import { normal, uniform } from '../dists/index.ts'
import { foldIn, key } from '../random/index.ts'
import { item, sliceAxis, stack, sum, tensor, toFlat, type Value } from '../tensor/index.ts'
import { timeGrid, valuesAt } from './paths.ts'
import {
  betaProcess,
  chineseRestaurant,
  dirichletProcess,
  expectedDishes,
  expectedTables,
  gammaProcess,
  gammaSubordinator,
  indianBuffet,
  partitionLogProbability,
  polyaUrn,
  sampleFromMeasure,
  stickBreaking,
} from './random-measures.ts'

/** 2000 independent stick-breaking draws stacked `[2000, 5]`, differentiable in alpha. */
function stickBreakingMany(a: Value): Value {
  return stack(
    Array.from({ length: 2000 }, (_, i) => stickBreaking(foldIn(key(2), i), a, 5)),
    0,
  )
}

describe('Dirichlet and Pitman-Yor processes', () => {
  it('stick-breaking weights have E[w_k] = (1/(1+alpha)) (alpha/(1+alpha))^{k-1}', () => {
    const alpha = 2
    const runs = Array.from({ length: 2000 }, (_, i) => toFlat(stickBreaking(foldIn(key(1), i), alpha, 5)))
    for (let k = 0; k < 3; k++) {
      const w = runs.map((r) => r[k] ?? 0)
      const want = (1 / (1 + alpha)) * (alpha / (1 + alpha)) ** k
      expect(Math.abs(meanOf(w) - want)).toBeLessThan(0.02)
    }
    // beta draws use rejection, so check the implicit gradient by its mean: d E[w_1] / d alpha = -1 / (1 + alpha)^2
    const grads = toFlat(jacfwd((a: Value) => sliceAxis(stickBreakingMany(a), 1, 0, 1))(tensor(alpha)))
    expect(Math.abs(meanOf(grads) + 1 / 9)).toBeLessThan(4 * Math.sqrt(varianceOf(grads) / grads.length))
  })

  it('DP draws are discrete probability measures centred on the base', () => {
    const dp = dirichletProcess(key(3), 5, normal(1, 1), 300)
    expect(item(sum(dp.weights))).toBeGreaterThan(0.99)
    const x = toFlat(sampleFromMeasure(key(4), dp, 500))
    expect(new Set(x).size).toBeLessThan(100)
    const py = dirichletProcess(key(5), 1, normal(0, 1), 50, { discount: 0.5 })
    expect(py.weights.shape).toEqual([50])
  })

  it('the Chinese restaurant opens the expected number of tables', () => {
    for (const [alpha, d] of [
      [1, 0],
      [2, 0.4],
    ] as const) {
      const k = Array.from(
        { length: 300 },
        (_, i) => chineseRestaurant(foldIn(key(6), i), alpha, 50, { discount: d }).sizes.length,
      )
      const want = item(expectedTables(alpha, 50, { discount: d }))
      expect(Math.abs(meanOf(k) - want)).toBeLessThan(0.1 * want + 0.3)
    }
    const r = chineseRestaurant(key(7), 1, 30)
    expect(r.tables[0]).toBe(0)
    expect(Array.from(r.sizes).reduce((a, b) => a + b, 0)).toBe(30)
  })

  it('the partition probabilities of n = 3 sum to one (Ewens and Pitman-Yor)', () => {
    for (const d of [0, 0.3]) {
      // partitions of {1, 2, 3}: one {3}, three {2, 1}, one {1, 1, 1}
      const p = (sizes: number[]): number => Math.exp(item(partitionLogProbability(sizes, 1.7, { discount: d })))
      expect(p([3]) + 3 * p([2, 1]) + p([1, 1, 1])).toBeCloseTo(1, 12)
    }
    expectConformant((a: Value) => partitionLogProbability([2, 1, 1], a), [tensor(0.9)], { vmap: false })
    expect(() => partitionLogProbability([0], 1)).toThrow()
  })

  it('the Polya urn draws ties at the Dirichlet rate', () => {
    const runs = Array.from({ length: 300 }, (_, i) => new Set(polyaUrn(foldIn(key(8), i), 2, normal(0, 1), 40)).size)
    const want = item(expectedTables(2, 40))
    expect(Math.abs(meanOf(runs) - want)).toBeLessThan(0.1 * want)
  })
})

describe('Indian buffet, beta and gamma processes', () => {
  it('the Indian buffet serves alpha H_n dishes and Poisson(alpha) dishes per customer', () => {
    const k = Array.from({ length: 300 }, (_, i) => indianBuffet(foldIn(key(9), i), 2, 10).shape[1] ?? 0)
    expect(Math.abs(meanOf(k) - expectedDishes(2, 10))).toBeLessThan(0.1 * expectedDishes(2, 10))
    // each customer takes Poisson(alpha) dishes: E[total] = n alpha
    const totals = Array.from({ length: 300 }, (_, i) =>
      toFlat(indianBuffet(foldIn(key(10), i), 3, 10)).reduce((a, b) => a + b, 0),
    )
    expect(Math.abs(meanOf(totals) - 30)).toBeLessThan(4 * Math.sqrt(varianceOf(totals) / 300))
  })

  it('beta process weights decrease geometrically in expectation, E pi_k = (alpha / (1 + alpha))^k', () => {
    const w = Array.from(
      { length: 1500 },
      (_, i) => toFlat(betaProcess(foldIn(key(11), i), { mass: 2 }, uniform(0, 1), 4).weights)[1] ?? 0,
    )
    expect(Math.abs(meanOf(w) - (2 / 3) ** 2)).toBeLessThan(0.02)
    expect(betaProcess(key(12), { mass: 2, concentration: 3 }, uniform(0, 1), 20).weights.shape).toEqual([20])
    expect(() => betaProcess(key(0), { mass: 30, concentration: 2 }, uniform(0, 1), 20)).toThrow()
  })

  it('gamma processes have Gamma(alpha) total mass and gamma increments in time', () => {
    const totals = Array.from({ length: 600 }, (_, i) =>
      item(sum(gammaProcess(foldIn(key(13), i), { alpha: 3 }, normal(0, 1), 50).weights)),
    )
    expect(Math.abs(meanOf(totals) - 3)).toBeLessThan(4 * Math.sqrt(3 / 600))
    const g = gammaSubordinator(key(14), timeGrid(2, 4), { rate: 1.5, scale: 2, paths: 3000 })
    const end = toFlat(valuesAt(g, -1))
    expect(Math.abs(meanOf(end) - 6)).toBeLessThan(4 * Math.sqrt(12 / 3000))
  })
})
