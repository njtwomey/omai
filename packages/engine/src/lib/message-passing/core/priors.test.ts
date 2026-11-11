import { describe, expect, it } from 'vitest'
import { gaussian } from './families.ts'
import { densityMoments, projectedPrior } from './priors.ts'

const logHalfNormal = (s: number) => (x: Float64Array) =>
  x.map((v) => (v < 0 ? Number.NEGATIVE_INFINITY : -(v * v) / (2 * s * s)))
const logHalfCauchy = (s: number) => (x: Float64Array) =>
  x.map((v) => (v < 0 ? Number.NEGATIVE_INFINITY : -Math.log1p((v / s) ** 2)))
/** The density of log tau: tau p(tau) at tau = exp(l). */
const onLog = (logp: (x: Float64Array) => Float64Array) => (l: Float64Array) => {
  const t = l.map((v) => Math.exp(v))
  const lp = logp(t)
  return lp.map((v, i) => v + (l[i] as number))
}

describe('projecting a density onto a message family', () => {
  it('a half-normal on the log scale has the exact moments of log |Z|: mean -(gamma + log 2) / 2, variance pi^2 / 8', () => {
    const m = densityMoments(onLog(logHalfNormal(2)), -Infinity, Infinity, Math.log(2))
    expect(m.mean).toBeCloseTo(Math.log(2) - (0.5772156649015329 + Math.log(2)) / 2, 5)
    expect(m.variance).toBeCloseTo(Math.PI ** 2 / 8, 5)
  })

  it('a half-Cauchy on the log scale has the exact moments of log |C|: mean 0 (plus log scale), variance pi^2 / 4', () => {
    const m = densityMoments(onLog(logHalfCauchy(5)), -Infinity, Infinity, Math.log(5))
    expect(m.mean).toBeCloseTo(Math.log(5), 5)
    expect(m.variance).toBeCloseTo(Math.PI ** 2 / 4, 5)
  })

  it('a uniform on (0, 10) on the log scale is log 10 minus a unit exponential: mean log 10 - 1, variance 1', () => {
    const m = densityMoments(
      onLog((x) => x.map((v) => (v > 10 + 1e-9 ? Number.NEGATIVE_INFINITY : 0))),
      -Infinity,
      Math.log(10),
      0,
    )
    expect(m.mean).toBeCloseTo(Math.log(10) - 1, 3)
    expect(m.variance).toBeCloseTo(1, 3)
  })

  it('a density far from the starting point, and a narrow one, are found', () => {
    const far = densityMoments((x) => x.map((v) => -((v - 400) ** 2) / 2), -Infinity, Infinity, 0)
    expect(far.mean).toBeCloseTo(400, 4)
    expect(far.variance).toBeCloseTo(1, 4)
    const narrow = densityMoments((x) => x.map((v) => -((v - 3) ** 2) / (2 * 1e-8)), -Infinity, Infinity, 0)
    expect(narrow.mean).toBeCloseTo(3, 6)
    expect(narrow.variance / 1e-8).toBeCloseTo(1, 3)
  })

  it('the Gaussian projection is the prior factor with those moments; the gamma projection matches E[x] and E[log x]', () => {
    const f = projectedPrior('t', onLog(logHalfNormal(1)), { start: 0 })
    expect(f.operator).toBe('prior')
    const eta = f.params?.['eta'] as number[]
    const q = gaussian.moments(eta)
    expect(q[0]).toBeCloseTo(-(0.5772156649015329 + Math.log(2)) / 2, 5)
    // A gamma density projected onto a gamma is itself: shape 3, rate 2.
    const logGamma = (x: Float64Array): Float64Array =>
      x.map((v) => (v <= 0 ? Number.NEGATIVE_INFINITY : 2 * Math.log(v) - 2 * v))
    const g = projectedPrior('g', logGamma, { family: 'gamma' })
    const e = g.params?.['eta'] as number[]
    expect(e[0]).toBeCloseTo(2, 4)
    expect(e[1]).toBeCloseTo(-2, 4)
  })
})
