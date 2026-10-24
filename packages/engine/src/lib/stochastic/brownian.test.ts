import { describe, expect, it } from 'vitest'
import { expectConformant } from '../../testing/check.ts'
import { meanOf, varianceOf } from '../base/index.ts'
import { key } from '../random/index.ts'
import { add, item, mean, mul, sliceAxis, sum, tensor, toArray, toFlat, type Value } from '../tensor/index.ts'
import { recorder } from '../trace/index.ts'
import {
  bridgeRefine,
  brownianBridge,
  brownianIncrements,
  brownianMotion,
  covariation,
  firstPassageCdf,
  firstPassageDistribution,
  firstPassageTimes,
  maximumCdf,
  quadraticVariation,
  reflectPath,
  runningMaximum,
} from './brownian.ts'
import { pathIncrements, timeGrid, valuesAt } from './paths.ts'

/** |sample mean - target| within `z` standard errors. */
function expectMean(xs: readonly number[], target: number, z = 4): void {
  const se = Math.sqrt(varianceOf(xs) / xs.length)
  expect(Math.abs(meanOf(xs) - target)).toBeLessThan(z * se + 1e-12)
}

describe('brownian motion', () => {
  it('has the drift and diffusion of its definition at the end time', () => {
    const w = brownianMotion(key(1), timeGrid(2, 20), { paths: 4000, drift: 0.3, sigma: 0.7, x0: 1 })
    const end = toFlat(valuesAt(w, -1))
    expectMean(end, 1 + 0.6)
    // variance of the sample variance: 2 sigma^4 / (n - 1) for normal data
    const v = 0.49 * 2
    expect(Math.abs(varianceOf(end) - v)).toBeLessThan(4 * v * Math.sqrt(2 / 3999))
  })

  it('reuses given increments exactly: the path is x0 + drift t + sigma * cumulative sum', () => {
    const t = timeGrid(1, 4)
    const dw = brownianIncrements(key(2), t, { paths: 2 })
    const w = brownianMotion(key(99), t, { paths: 2, increments: dw, drift: 1, sigma: 2, x0: 0.5 })
    const inc = toArray(pathIncrements(w.values)) as number[][]
    const raw = toArray(dw) as number[][]
    for (let j = 0; j < 2; j++) {
      for (let i = 0; i < 4; i++) expect(inc[j]?.[i]).toBeCloseTo(0.25 + 2 * (raw[j]?.[i] ?? 0), 12)
    }
  })

  it('correlates coordinates through a covariance, so the covariation is rho t', () => {
    const cov = [
      [1, 0.6],
      [0.6, 2],
    ]
    const w = brownianMotion(key(3), timeGrid(1, 400), { paths: 200, covariance: cov })
    expect(w.values.shape).toEqual([200, 401, 2])
    const x = sliceAxis(w.values, 2, 0, 1)
    const y = sliceAxis(w.values, 2, 1, 2)
    // realised covariation is rho t up to O(sqrt(dt)) noise per path; average over paths
    expect(item(mean(covariation(x, y)))).toBeCloseTo(0.6, 1)
    expect(item(mean(quadraticVariation(y)))).toBeCloseTo(2, 1)
  })

  it('has quadratic variation sigma^2 T with vanishing variance as the grid refines', () => {
    const w = brownianMotion(key(4), timeGrid(3, 3000), { paths: 3, sigma: 0.5 })
    for (const q of toFlat(quadraticVariation(w))) expect(Math.abs(q - 0.75)).toBeLessThan(0.1)
    const running = quadraticVariation(w, { cumulative: true })
    expect(running.shape).toEqual([3, 3001])
  })

  it('is differentiable pathwise in its drift and volatility', () => {
    const t = timeGrid(1, 5)
    const f = (drift: Value, sigma: Value): Value =>
      sum(brownianMotion(key(5), t, { paths: 3, drift, sigma }).values, 1)
    expectConformant(f, [tensor(0.4), tensor(1.3)])
    const g = (x0: Value): Value => valuesAt(brownianMotion(key(5), t, { paths: 2, dim: 2, x0 }), -1)
    expectConformant(g, [tensor([0.1, -0.2])])
  })

  it('records a traced run with the state at each grid time', () => {
    const rec = recorder()
    brownianMotion(key(6), timeGrid(1, 3), { paths: 2, tracer: rec })
    const run = rec.run()
    expect(run.events.filter((e) => e.type === 'metric' && e.name === 't').length).toBe(4)
  })
})

describe('brownian bridges', () => {
  it('starts and ends at the pinned values with the bridge variance in between', () => {
    const t = timeGrid(1, 4)
    const b = brownianBridge(key(7), t, { paths: 3000, x0: 1, x1: -1, sigma: 2 })
    expect(toFlat(valuesAt(b, 0))).toEqual(new Array(3000).fill(1))
    expect(toFlat(valuesAt(b, -1))).toEqual(new Array(3000).fill(-1))
    const mid = toFlat(valuesAt(b, 2))
    expectMean(mid, 0)
    const v = 4 * 0.25
    expect(Math.abs(varianceOf(mid) - v)).toBeLessThan(4 * v * Math.sqrt(2 / 2999))
  })

  it('refines paths by midpoints without moving the old points', () => {
    const w = brownianMotion(key(8), timeGrid(1, 2), { paths: 2000 })
    const r = bridgeRefine(key(9), w)
    expect(r.values.shape).toEqual([2000, 5])
    expect(toFlat(valuesAt(r, 2))).toEqual(toFlat(valuesAt(w, 1)))
    // the refined increments are independent N(0, 1/4)
    const inc = toFlat(sliceAxis(pathIncrements(r.values), 1, 0, 1))
    expectMean(inc, 0)
    expect(Math.abs(varianceOf(inc) - 0.25)).toBeLessThan(4 * 0.25 * Math.sqrt(2 / 1999))
  })
})

describe('the reflection principle and first passage', () => {
  it('computes running maxima and reflected paths', () => {
    expect(toArray(runningMaximum([[0, -1, 2, 1]]))).toEqual([[0, 0, 2, 2]])
    expect(toArray(reflectPath([[2, 1, 0, -1, 0.5]], 0))).toEqual([[2, 1, 0, 1, -0.5]])
  })

  it('matches the simulated maximum and passage times of Brownian motion', () => {
    const w = brownianMotion(key(10), timeGrid(1, 400), { paths: 2000, drift: 0.2 })
    const max = toFlat(valuesAt(runningMaximum(w), -1))
    const p = max.filter((m) => m <= 0.8).length / max.length
    const want = item(maximumCdf(0.8, 1, { drift: 0.2 }))
    // the discrete maximum underestimates the continuous one by O(sqrt(dt)): a small upward bias in p
    expect(Math.abs(p - want)).toBeLessThan(4 * Math.sqrt((want * (1 - want)) / 2000) + 0.04)
    const tau = firstPassageTimes(w, 0.8)
    const hit = Array.from(tau).filter((x) => x <= 1).length / tau.length
    expect(Math.abs(hit - (1 - p))).toBeLessThan(1e-12)
    expect(item(firstPassageCdf(1, 0.8, { drift: 0.2 }))).toBeCloseTo(1 - want, 14)
  })

  it('reflected Brownian motion is Brownian motion: P(W_t >= a) doubles to P(max >= a)', () => {
    const w = brownianMotion(key(11), timeGrid(1, 200), { paths: 3000 })
    const r = toFlat(valuesAt(reflectPath(w, 0.5), -1))
    expectMean(r, 0)
    expect(item(maximumCdf(0.5, 1))).toBeCloseTo(2 * 0.691462461274013 - 1, 12)
  })

  it('has the defective passage law for a drift away from the level', () => {
    const atInfinity = item(firstPassageCdf(1e8, 1, { drift: -0.5 }))
    expect(atInfinity).toBeCloseTo(Math.exp(-1), 6)
    expect(() => firstPassageDistribution(1, { drift: -0.5 })).toThrow()
    expect(firstPassageDistribution(-2, { drift: -1 }).name).toBe('inverseGaussian')
    expect(item(firstPassageCdf(2, -1, { drift: -0.4 }))).toBeCloseTo(item(firstPassageCdf(2, 1, { drift: 0.4 })), 14)
  })

  it('differentiates the reflection-principle laws in their parameters', () => {
    expectConformant(
      (m: Value, mu: Value, s: Value) => maximumCdf(m, 1.5, { drift: mu, sigma: s }),
      [tensor([0.3, 1.2]), tensor(0.4), tensor(0.9)],
    )
    expectConformant((t: Value) => add(firstPassageCdf(t, 1, { drift: 0.3 }), mul(t, 0)), [tensor([0.5, 2])])
  })
})
