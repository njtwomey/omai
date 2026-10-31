/**
 * Tests for variance reduction. Each estimator's reported standard error is checked by coverage of the truth over many
 * independent runs, and each reported reduction factor against the ratio of empirical variances; the special cases
 * (symmetric integrands for antithetic pairs, odd ones, differing stratum spreads for Neyman, the law of total variance for
 * conditioning) have exact or theoretical answers.
 */

import { describe, expect, it } from 'vitest'
import { key, uniform } from '../random/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import {
  antitheticEstimate,
  controlVariateEstimate,
  monteCarloEstimate,
  quasiMonteCarloEstimate,
  raoBlackwellEstimate,
  stratifiedEstimate,
  uniformStrata,
  varianceReductionAlgorithm,
  varianceReductionEstimates,
} from './index.ts'

const TRUTH = Math.E - 1
const reps = Array.from({ length: 300 }, (_, i) => i)

function coverage(run: (seed: number) => { value: number; standardError?: number }, truth: number): number {
  let covered = 0
  for (const seed of reps) {
    const e = run(seed)
    if (Math.abs(e.value - truth) < 1.96 * (e.standardError ?? 0)) covered += 1
  }
  return covered / reps.length
}

function variance(xs: readonly number[]): number {
  const m = xs.reduce((a, b) => a + b, 0) / xs.length
  return xs.reduce((a, b) => a + (b - m) ** 2, 0) / (xs.length - 1)
}

describe('monteCarloEstimate', () => {
  it('has a calibrated standard error', () => {
    const c = coverage((s) => monteCarloEstimate({ fn: Math.exp, key: key(s), n: 200 }), TRUTH)
    expect(c).toBeGreaterThan(0.91)
    expect(c).toBeLessThan(0.99)
  })
})

describe('controlVariateEstimate', () => {
  const run = (s: number, n = 200): ReturnType<typeof controlVariateEstimate> =>
    controlVariateEstimate({ fn: Math.exp, controls: [{ fn: (u) => u, mean: 0.5 }], key: key(s), n })

  it('has a calibrated standard error and a reduction factor equal to the empirical variance ratio', () => {
    const c = coverage((s) => run(s), TRUTH)
    expect(c).toBeGreaterThan(0.91)
    expect(c).toBeLessThan(0.99)
    const plain = reps.map((s) => monteCarloEstimate({ fn: Math.exp, key: key(10_000 + s), n: 200 }).value)
    const controlled = reps.map((s) => run(10_000 + s).value)
    const empirical = variance(plain) / variance(controlled)
    const reported = run(1).varianceReductionFactor
    expect(reported / empirical).toBeGreaterThan(0.6)
    expect(reported / empirical).toBeLessThan(1.6)
    expect(reported).toBeGreaterThan(30)
  })

  it('fits the regression coefficient and the correlation', () => {
    const e = run(2, 5000)
    // slope of exp(U) on U: Cov(e^U, U) / Var(U), with Cov = int u e^u du - (e - 1) / 2 = 1 - (e - 1) / 2 and Var(U) = 1/12
    const cov = 1 - (Math.E - 1) / 2
    expect(e.beta[0]).toBeCloseTo(cov * 12, 1)
    expect(e.correlation).toBeGreaterThan(0.99)
  })

  it('more controls reduce further, and the estimate stays unbiased', () => {
    const one = run(3, 400)
    const two = controlVariateEstimate({
      fn: Math.exp,
      controls: [
        { fn: (u) => u, mean: 0.5 },
        { fn: (u) => u * u, mean: 1 / 3 },
      ],
      key: key(3),
      n: 400,
    })
    expect(two.beta.length).toBe(2)
    expect(two.varianceReductionFactor).toBeGreaterThan(one.varianceReductionFactor)
    const c = coverage(
      (s) =>
        controlVariateEstimate({
          fn: Math.exp,
          controls: [
            { fn: (u) => u, mean: 0.5 },
            { fn: (u) => u * u, mean: 1 / 3 },
          ],
          key: key(s),
          n: 100,
        }),
      TRUTH,
    )
    expect(c).toBeGreaterThan(0.9)
  })

  it('validates its input', () => {
    expect(() => controlVariateEstimate({ fn: Math.exp, controls: [], key: key(0), n: 10 })).toThrow(/control/)
    expect(() =>
      controlVariateEstimate({ fn: Math.exp, controls: [{ fn: (u) => u, mean: 0.5 }], key: key(0), n: 2 }),
    ).toThrow(/at least 3/)
  })
})

describe('antitheticEstimate', () => {
  it('has a calibrated standard error and reduces variance for a monotone integrand', () => {
    const run = (s: number): ReturnType<typeof antitheticEstimate> =>
      antitheticEstimate({ fn: (x) => Math.exp(x[0] ?? 0), dim: 1, key: key(s), pairs: 100 })
    const c = coverage((s) => run(s), TRUTH)
    expect(c).toBeGreaterThan(0.9)
    const e = run(1)
    expect(e.pairCorrelation).toBeLessThan(-0.9)
    const empirical =
      variance(reps.map((s) => monteCarloEstimate({ fn: Math.exp, key: key(20_000 + s), n: 200 }).value)) /
      variance(reps.map((s) => run(20_000 + s).value))
    expect(e.varianceReductionFactor / empirical).toBeGreaterThan(0.6)
    expect(e.varianceReductionFactor / empirical).toBeLessThan(1.6)
    expect(e.evaluations).toBe(200)
  })

  it('cancels an odd integrand under normal reflection exactly, and gains nothing on a symmetric one', () => {
    const odd = antitheticEstimate({ fn: (x) => x[0] ?? 0, dim: 1, distribution: 'normal', key: key(1), pairs: 50 })
    expect(odd.value).toBeCloseTo(0, 14)
    expect(odd.standardError).toBeCloseTo(0, 14)
    const even = antitheticEstimate({ fn: (x) => ((x[0] ?? 0) - 0.5) ** 2, dim: 1, key: key(1), pairs: 2000 })
    expect(even.pairCorrelation).toBeCloseTo(1, 10)
    // the pairs are copies, so the budget buys half the information
    expect(even.varianceReductionFactor).toBeGreaterThan(0.4)
    expect(even.varianceReductionFactor).toBeLessThan(0.6)
  })
})

describe('stratifiedEstimate', () => {
  it('has a calibrated standard error and a large gain on a smooth integrand', () => {
    const run = (s: number): ReturnType<typeof stratifiedEstimate> =>
      stratifiedEstimate({ strata: uniformStrata(10), fn: Math.exp, n: 200, key: key(s) })
    const c = coverage((s) => run(s), TRUTH)
    expect(c).toBeGreaterThan(0.9)
    expect(c).toBeLessThan(0.99)
    const e = run(0)
    expect(e.counts.reduce((a, b) => a + b, 0)).toBe(200)
    expect(e.varianceReductionFactor).toBeGreaterThan(20)
    const empirical =
      variance(reps.map((s) => monteCarloEstimate({ fn: Math.exp, key: key(30_000 + s), n: 200 }).value)) /
      variance(reps.map((s) => run(30_000 + s).value))
    expect(e.varianceReductionFactor / empirical).toBeGreaterThan(0.5)
    expect(e.varianceReductionFactor / empirical).toBeLessThan(2)
  })

  it('Neyman allocation beats proportional when the strata differ in spread', () => {
    // all the noise is in the last stratum
    const fn = (u: number): number => (u > 0.9 ? 5 * Math.sin(300 * u) : 0)
    const make = (allocation: 'proportional' | 'neyman'): number =>
      stratifiedEstimate({ strata: uniformStrata(10), fn, n: 1000, allocation, key: key(4) }).standardError ?? 0
    expect(make('neyman')).toBeLessThan(0.6 * make('proportional'))
    const e = stratifiedEstimate({ strata: uniformStrata(10), fn, n: 1000, allocation: 'neyman', key: key(4) })
    expect(e.counts[9]).toBeGreaterThan(900)
    expect(e.counts.reduce((a, b) => a + b, 0)).toBe(1000)
    // known spreads skip the pilot
    const known = stratifiedEstimate({
      strata: uniformStrata(10).map((s, j) => ({ ...s, std: j === 9 ? 3.5 : 0 })),
      fn,
      n: 1000,
      allocation: 'neyman',
      key: key(4),
    })
    expect(known.counts[9]).toBeGreaterThan(900)
  })

  it('rejects strata that are not a partition', () => {
    const bad = uniformStrata(3).map((s) => ({ ...s, weight: 0.5 }))
    expect(() => stratifiedEstimate({ strata: bad, fn: Math.exp, n: 30, key: key(0) })).toThrow(/sum to 1/)
    expect(() => stratifiedEstimate({ strata: uniformStrata(10), fn: Math.exp, n: 10, key: key(0) })).toThrow(
      /at least 20/,
    )
  })
})

describe('raoBlackwellEstimate', () => {
  const sample = (k: ReturnType<typeof key>, n: number): [number, number][] => {
    const d = uniform(k, [n, 2]).data
    return Array.from({ length: n }, (_, i) => [d[2 * i] ?? 0, d[2 * i + 1] ?? 0] as [number, number])
  }
  const run = (s: number, n = 20_000): ReturnType<typeof raoBlackwellEstimate<[number, number]>> =>
    raoBlackwellEstimate<[number, number]>({
      sample,
      raw: ([x, y]) => x + y,
      conditional: ([x]) => x + 0.5,
      key: key(s),
      n,
    })

  it('halves the variance here and obeys the law of total variance', () => {
    const e = run(1)
    expect(e.varianceReductionFactor).toBeGreaterThan(1.8)
    expect(e.varianceReductionFactor).toBeLessThan(2.2)
    expect(e.expectedConditionalVariance).toBeCloseTo(1 / 12, 2)
    expect(e.varianceOfConditionalMean).toBeCloseTo(1 / 12, 2)
    const rawVariance = e.plainStandardError ** 2 * 20_000
    expect(
      Math.abs(e.varianceOfConditionalMean + e.expectedConditionalVariance - rawVariance) / rawVariance,
    ).toBeLessThan(0.02)
  })

  it('has a calibrated standard error', () => {
    expect(coverage((s) => run(s, 200), 1)).toBeGreaterThan(0.91)
  })
})

describe('quasiMonteCarloEstimate', () => {
  const fn = (u: Float64Array): number => (u[0] ?? 0) * (u[1] ?? 0) * Math.exp(u[2] ?? 0)
  const truth = 0.25 * (Math.E - 1)

  for (const sequence of ['sobol', 'halton'] as const) {
    it(`${sequence}: is far more accurate than plain Monte Carlo on a smooth integrand`, () => {
      const e = quasiMonteCarloEstimate({ fn, dim: 3, n: 1024, key: key(1), sequence })
      expect(Math.abs(e.value - truth)).toBeLessThan(2e-3)
      expect(e.evaluations).toBe(16 * 1024)
      expect(e.varianceReductionFactor).toBeGreaterThan(sequence === 'sobol' ? 50 : 10)
    })
  }

  it('is unbiased over randomisations: the mean over keys is within its standard error', () => {
    const values = Array.from(
      { length: 40 },
      (_, s) => quasiMonteCarloEstimate({ fn, dim: 3, n: 256, replicates: 4, key: key(s) }).value,
    )
    const m = values.reduce((a, b) => a + b, 0) / values.length
    expect(Math.abs(m - truth)).toBeLessThan(4 * Math.sqrt(variance(values) / values.length))
  })
})

describe('varianceReductionAlgorithm', () => {
  it('runs four estimators whose standard errors are ordered and calibrated, and serialises', () => {
    const alg = varianceReductionAlgorithm({ batches: 300, truth: TRUTH })
    const run = runAlgorithm(alg, { seed: 1, maxSteps: 1000 })
    const e = varianceReductionEstimates(run.state)
    expect(run.state.batches).toBe(300)
    for (const t of ['plain', 'antithetic', 'control', 'stratified'] as const) {
      expect(Math.abs(e[t].value - TRUTH)).toBeLessThan(4.5 * e[t].se)
    }
    expect(e.control.se).toBeLessThan(e.plain.se / 5)
    expect(e.antithetic.se).toBeLessThan(e.plain.se / 3)
    expect(e.stratified.se).toBeLessThan(e.plain.se / 5)
    // plain: se is the Monte Carlo rate sigma / sqrt(10 * 300) with sigma^2 = (e^2 - 1) / 2 - (e - 1)^2
    const sigma = Math.sqrt((Math.E ** 2 - 1) / 2 - TRUTH ** 2)
    expect(e.plain.se / (sigma / Math.sqrt(3000))).toBeGreaterThan(0.9)
    expect(e.plain.se / (sigma / Math.sqrt(3000))).toBeLessThan(1.1)
    const again = alg.deserialise?.(alg.serialise?.(run.state) ?? null)
    expect(again?.strata).toEqual(run.state.strata)
  })

  it('validates the batch size', () => {
    expect(() => varianceReductionAlgorithm({ batch: 3 })).toThrow(/even/)
  })
})
