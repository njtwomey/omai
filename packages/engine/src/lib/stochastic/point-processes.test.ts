import { describe, expect, it } from 'vitest'
import { expectConformant } from '../../testing/check.ts'
import { meanOf, varianceOf } from '../base/index.ts'
import { exponential, gamma } from '../dists/index.ts'
import { foldIn, key } from '../random/index.ts'
import { tensor, toFlat, type Value } from '../tensor/index.ts'
import { recorder } from '../trace/index.ts'
import { ouCovariance } from './gaussian-processes.ts'
import {
  countingProcess,
  coxProcess,
  hawkesIntensity,
  hawkesLogLikelihood,
  hawkesProcess,
  inhomogeneousPoisson,
  logGaussianCoxProcess,
  markedPoissonProcess,
  maternClusterProcess,
  poissonProcess,
  renewalFunction,
  renewalProcess,
  ripleysK,
  ripleysL,
  spatialPoisson,
  thomasK,
  thomasProcess,
} from './point-processes.ts'

const counts = (f: (k: number) => Float64Array, runs: number): number[] =>
  Array.from({ length: runs }, (_, i) => f(i).length)

describe('Poisson and renewal processes', () => {
  it('homogeneous counts are Poisson(lambda T) with exponential gaps', () => {
    const n = counts((i) => poissonProcess(foldIn(key(1), i), 3, 2), 800)
    expect(Math.abs(meanOf(n) - 6)).toBeLessThan(4 * Math.sqrt(6 / 800))
    expect(Math.abs(varianceOf(n) - 6)).toBeLessThan(1.2)
    const rec = recorder()
    const one = poissonProcess(key(2), 50, 10, { tracer: rec })
    const gaps = Array.from(one, (t, i) => t - (one[i - 1] ?? 0))
    expect(Math.abs(meanOf(gaps) - 1 / 50)).toBeLessThan(4 * Math.sqrt(1 / 2500 / gaps.length))
    expect(rec.run().events.some((e) => e.type === 'metric' && e.name === 'count')).toBe(true)
    expect(Array.from(countingProcess(one, [0, 10]))).toEqual([0, one.length])
  })

  it('thinning produces the integrated intensity and refuses a violated bound', () => {
    const lam = (t: number): number => 2 + 2 * Math.sin(t)
    const n = counts((i) => inhomogeneousPoisson(foldIn(key(3), i), lam, Math.PI, 4), 800)
    // int_0^pi (2 + 2 sin t) dt = 2 pi + 4
    expect(Math.abs(meanOf(n) - (2 * Math.PI + 4))).toBeLessThan(4 * Math.sqrt((2 * Math.PI + 4) / 800))
    expect(() => inhomogeneousPoisson(key(0), () => 5, 10, 4)).toThrow()
  })

  it('renewal counts satisfy the elementary renewal theorem and the renewal function', () => {
    const gap = gamma(2, 2)
    const n = counts((i) => renewalProcess(foldIn(key(4), i), gap, 20), 300)
    const m = renewalFunction(gap, 20, 2000).values.at(-1) ?? 0
    expect(Math.abs(meanOf(n) - m)).toBeLessThan(4 * Math.sqrt(varianceOf(n) / 300))
    expect(m / 20).toBeCloseTo(1, 1)
    expect(renewalFunction(exponential(1), 4, 4000).values.at(-1)).toBeCloseTo(4, 4)
  })

  it('marked and Cox processes', () => {
    const m = markedPoissonProcess(key(5), 4, 5, exponential(1))
    expect(m.marks.every((v) => v > 0)).toBe(true)
    const c = coxProcess(key(6), () => ({ intensity: (t) => (t < 1 ? 10 : 0), max: 10 }), 3)
    expect(c.events.every((t) => t < 1)).toBe(true)
    const lgcp = logGaussianCoxProcess(key(7), [0.5, 1.5], 1, {
      covariance: ouCovariance({ theta: 1, sigma: 0.1 }),
      mean: 2,
    })
    expect(lgcp.intensity.every((v) => v > 5 && v < 10)).toBe(true)
  })
})

describe('Hawkes processes', () => {
  it('have the stationary mean intensity mu / (1 - alpha / beta)', () => {
    const p = { mu: 0.5, alpha: 0.8, beta: 2 }
    const events = hawkesProcess(key(8), p, 2000)
    const want = 0.5 / (1 - 0.4)
    // clustering inflates the variance of the count: a loose band
    expect(Math.abs(events.length / 2000 - want)).toBeLessThan(0.1)
  })

  it('the log-likelihood differentiates in the parameters and the intensity jumps at events', () => {
    const events = [0.3, 0.7, 1.1, 2.5]
    expectConformant(
      (mu: Value, a: Value, b: Value) => hawkesLogLikelihood({ mu, alpha: a, beta: b }, events, 3),
      [tensor(0.8), tensor(0.4), tensor(1.3)],
    )
    const at = toFlat(hawkesIntensity({ mu: 1, alpha: 1, beta: 1 }, [1], [1, 1 + 1e-9]))
    expect(at[0]).toBe(1)
    expect(at[1]).toBeCloseTo(2, 6)
  })
})

describe('spatial point processes', () => {
  it('a homogeneous Poisson pattern has K(r) close to pi r^2', () => {
    const pts = spatialPoisson(key(9), 400)
    const r = [0.05, 0.1]
    const k = ripleysK(pts, r, { correction: 'translation' })
    for (let i = 0; i < 2; i++) expect(Math.abs((k[i] ?? 0) - Math.PI * (r[i] ?? 0) ** 2)).toBeLessThan(0.01)
    const l = ripleysL(pts, r, { correction: 'ripley' })
    expect(Math.abs((l[1] ?? 0) - 0.1)).toBeLessThan(0.02)
    const thin = spatialPoisson(key(10), (x) => (x < 0.5 ? 100 : 0), undefined, 100)
    expect(
      toFlat(thin)
        .filter((_, i) => i % 2 === 0)
        .every((x) => x < 0.5),
    ).toBe(true)
    expect(() => spatialPoisson(key(0), (x) => x)).toThrow()
  })

  it('Thomas patterns cluster: their K matches the closed form', () => {
    const params = { parentRate: 25, meanChildren: 8, sigma: 0.03 }
    const runs = Array.from(
      { length: 8 },
      (_, i) => ripleysK(thomasProcess(foldIn(key(11), i), params), [0.05], { correction: 'translation' })[0] ?? 0,
    )
    const want = toFlat(thomasK([0.05], params))[0] ?? 0
    expect(Math.abs(meanOf(runs) - want)).toBeLessThan(0.25 * want)
    expect(want).toBeGreaterThan(Math.PI * 0.05 ** 2)
    const m = maternClusterProcess(key(12), { parentRate: 20, meanChildren: 5, radius: 0.05 })
    expect(m.shape[1]).toBe(2)
    expect(() => thomasProcess(key(0), { ...params, sigma: 0 })).toThrow()
  })
})
