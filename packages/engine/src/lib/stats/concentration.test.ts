import { describe, expect, it } from 'vitest'
import {
  bennettBound,
  bernsteinBound,
  chebyshevBound,
  chernoffBound,
  chernoffGaussianBound,
  chernoffKlBound,
  dkwBand,
  empiricalBernsteinInterval,
  hoeffdingBound,
  hoeffdingSampleSize,
  markovBound,
  mcDiarmidBound,
} from './concentration.ts'

describe('concentration bounds', () => {
  it('computes Markov and Chebyshev bounds accurately and clamps to [0, 1]', () => {
    expect(markovBound(2, 10)).toBeCloseTo(0.2, 12)
    expect(markovBound(15, 10)).toBe(1)
    expect(markovBound(0, 5)).toBe(0)
    expect(() => markovBound(-1, 5)).toThrow(/mean/)
    expect(() => markovBound(1, 0)).toThrow(/threshold/)

    expect(chebyshevBound(1, 2)).toBeCloseTo(0.25, 12)
    expect(chebyshevBound(1, 2, { twoSided: false })).toBeCloseTo(0.2, 12)
    expect(chebyshevBound(4, 1)).toBe(1)
    expect(() => chebyshevBound(-1, 1)).toThrow(/variance/)
    expect(() => chebyshevBound(1, 0)).toThrow(/deviation/)
  })

  it('computes Chernoff sub-Gaussian bounds and obeys ordering with sample size n', () => {
    const b1 = chernoffGaussianBound(1, 2, { twoSided: false })
    expect(b1).toBeCloseTo(Math.exp(-2), 12)
    expect(chernoffBound(1, 2)).toBeCloseTo(2 * Math.exp(-2), 12)
    const b10 = chernoffGaussianBound(1, 0.5, { n: 10 })
    const b50 = chernoffGaussianBound(1, 0.5, { n: 50 })
    expect(b50).toBeLessThan(b10)
    expect(() => chernoffGaussianBound(0, 1)).toThrow(/sigma/)
  })

  it('computes Hoeffding bounds and sample-size inversion consistently', () => {
    const p = hoeffdingBound(100, 0.1)
    expect(p).toBeCloseTo(2 * Math.exp(-2), 12)
    expect(hoeffdingBound(100, 0.1, { range: [-0.5, 0.5] })).toBeCloseTo(p, 12)
    expect(
      hoeffdingBound(2, 0.2, {
        ranges: [
          [0, 1],
          [0, 2],
        ],
        twoSided: false,
      }),
    ).toBeCloseTo(Math.exp((-2 * 4 * 0.04) / (1 + 4)), 12)

    const nReq = hoeffdingSampleSize(0.05, 0.05)
    expect(nReq).toBe(738)
    expect(hoeffdingBound(nReq, 0.05)).toBeLessThanOrEqual(0.05)
    expect(hoeffdingBound(nReq - 1, 0.05)).toBeGreaterThan(0.05)
  })

  it('satisfies Bennett < Bernstein < Hoeffding when variance is small relative to range', () => {
    const n = 200
    const t = 0.08
    // Variables in [-1, 1] (width 2, M = 1) with small variance sigma^2 = 0.04
    const hBound = hoeffdingBound(n, t, { range: [-1, 1] })
    const bern = bernsteinBound(n, 0.04, 1, t)
    const benn = bennettBound(n, 0.04, 1, t)
    expect(benn).toBeLessThan(bern)
    expect(bern).toBeLessThan(hBound)
  })

  it('matches Hoeffding via McDiarmid when c_i = (b_i - a_i) / n', () => {
    const n = 20
    const t = 0.15
    const c = new Array<number>(n).fill(1 / n)
    expect(mcDiarmidBound(c, t)).toBeCloseTo(hoeffdingBound(n, t, { range: 1 }), 12)
  })

  it('constructs a valid DKW simultaneous confidence band', () => {
    const band = dkwBand([1, 2, 2, 4, 5], { level: 0.95 })
    expect(band.n).toBe(5)
    expect(band.alpha).toBeCloseTo(0.05, 12)
    expect(band.epsilon).toBeCloseTo(Math.sqrt(Math.log(40) / 10), 12)
    expect(Array.from(band.values.data)).toEqual([1, 2, 4, 5])
    expect(Array.from(band.ecdf.data)).toEqual([0.2, 0.6, 0.8, 1])
    expect(band.evaluate(0).ecdf).toBe(0)
    expect(band.evaluate(2).ecdf).toBeCloseTo(0.6, 12)
    expect(band.evaluate(3).ecdf).toBeCloseTo(0.6, 12)
    expect(band.evaluate(10).ecdf).toBe(1)
    for (let i = 0; i < band.values.data.length; i++) {
      expect(band.lower.data[i] ?? 0).toBeGreaterThanOrEqual(0)
      expect(band.upper.data[i] ?? 1).toBeLessThanOrEqual(1)
      expect(band.lower.data[i] ?? 0).toBeLessThanOrEqual(band.ecdf.data[i] ?? 0)
      expect(band.upper.data[i] ?? 1).toBeGreaterThanOrEqual(band.ecdf.data[i] ?? 0)
    }
  })

  it('bounds a binomial tail by exp(-n KL), at least as tight as Hoeffding and valid against the exact tail', () => {
    const n = 100
    const kl = chernoffKlBound(n, 0.5, 0.6)
    const q = 0.6
    const klValue = q * Math.log(q / 0.5) + (1 - q) * Math.log((1 - q) / 0.5)
    expect(kl).toBeCloseTo(Math.exp(-n * klValue), 12)
    expect(kl).toBeLessThan(hoeffdingBound(n, 0.1, { twoSided: false }))
    // Exact upper tail P(Binomial(100, 0.5) >= 60) = 0.02844 sits below the bound.
    let tail = 0
    let logC = 0
    for (let k = 0; k <= n; k++) {
      if (k > 0) logC += Math.log((n - k + 1) / k)
      if (k >= 60) tail += Math.exp(logC + n * Math.log(0.5))
    }
    expect(tail).toBeLessThan(kl)
    expect(chernoffKlBound(n, 0.5, 0.5)).toBe(1)
    expect(chernoffKlBound(n, 0.5, 0.4)).toBeCloseTo(kl, 12)
  })

  it('has an empirical Bernstein interval tighter than Hoeffding for low-variance data and covering the mean', () => {
    const n = 4000
    const x = Array.from({ length: n }, (_, i) => (i % 40 === 0 ? 1 : 0)) // mean 0.025, tiny variance
    const e = empiricalBernsteinInterval(x, { delta: 0.05 })
    expect(e.value).toBeCloseTo(0.025, 12)
    const half = (e.interval?.[1] ?? 0) - (e.value ?? 0)
    // Hoeffding's two-sided half-width at the same delta: sqrt(ln(2/delta) / (2 n)).
    expect(half).toBeLessThan(Math.sqrt(Math.log(2 / 0.05) / (2 * n)))
    const lo = e.interval?.[0] ?? 1
    const hi = e.interval?.[1] ?? 0
    expect(lo).toBeLessThan(0.025)
    expect(hi).toBeGreaterThan(0.025)
    expect(() => empiricalBernsteinInterval([0.5], {})).toThrow()
  })
})
