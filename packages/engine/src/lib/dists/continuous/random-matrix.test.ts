import { describe, expect, it } from 'vitest'
import { quad } from '../../numerics/quadrature/index.ts'
import { key } from '../../random/index.ts'
import { item, mean, mul, sub, tensor, toFlat, type Value } from '../../tensor/index.ts'
import { marchenkoPastur, marchenkoPasturEdges } from './random-matrix.ts'

const num = (v: Value): number => item(v)

describe('marchenkoPasturEdges', () => {
  it('is variance (1 -+ sqrt ratio)^2', () => {
    expect(marchenkoPasturEdges(0.25)).toEqual({ lower: 0.25, upper: 2.25 })
    expect(marchenkoPasturEdges(1, 3)).toEqual({ lower: 0, upper: 12 })
  })

  it('rejects a non-positive ratio or variance', () => {
    expect(() => marchenkoPasturEdges(0)).toThrow(/ratio/)
    expect(() => marchenkoPasturEdges(1, -1)).toThrow(/variance/)
  })
})

describe('marchenkoPastur density', () => {
  for (const [ratio, variance] of [
    [0.1, 1],
    [0.5, 2],
    [0.9, 0.5],
  ] as const) {
    it(`integrates to 1 (ratio ${ratio}, variance ${variance})`, () => {
      const d = marchenkoPastur(ratio, variance)
      const { lower, upper } = marchenkoPasturEdges(ratio, variance)
      const total = quad((x) => d.prob(x) as Value, lower, upper, { epsabs: 1e-12, epsrel: 1e-12, limit: 200 })
      expect(num(total.value)).toBeCloseTo(1, 8)
    })

    it(`has the closed-form cdf as the integral of the density (ratio ${ratio}, variance ${variance})`, () => {
      const d = marchenkoPastur(ratio, variance)
      const { lower, upper } = marchenkoPasturEdges(ratio, variance)
      for (const f of [0.1, 0.4, 0.7, 0.95]) {
        const x = lower + f * (upper - lower)
        const integral = quad((t) => d.prob(t) as Value, lower, x, { epsabs: 1e-12, epsrel: 1e-12, limit: 200 })
        expect(num(d.cdf(x))).toBeCloseTo(num(integral.value), 8)
      }
    })

    it(`has mean variance and variance variance^2 ratio (ratio ${ratio}, variance ${variance})`, () => {
      const d = marchenkoPastur(ratio, variance)
      const { lower, upper } = marchenkoPasturEdges(ratio, variance)
      const m = num(quad((x) => mul(x, d.prob(x)) as Value, lower, upper, { epsabs: 1e-12, limit: 200 }).value)
      const second = num(
        quad((x) => mul(mul(x, x), d.prob(x)) as Value, lower, upper, { epsabs: 1e-12, limit: 200 }).value,
      )
      expect(m).toBeCloseTo(variance, 8)
      expect(num(d.mean())).toBe(variance)
      expect(second - m * m).toBeCloseTo(variance * variance * ratio, 8)
      expect(num(d.variance())).toBeCloseTo(variance * variance * ratio, 14)
    })
  }

  it('is zero outside the support and the cdf is 0 below and 1 above it', () => {
    const d = marchenkoPastur(0.25)
    expect(num(d.prob(0.2))).toBe(0)
    expect(num(d.prob(2.3))).toBe(0)
    expect(num(d.cdf(0.25))).toBe(0)
    expect(num(d.cdf(2.25))).toBe(1)
    expect(num(d.cdf(-1))).toBe(0)
    expect(num(d.cdf(10))).toBe(1)
  })

  it('peaks at variance (1 - ratio)^2 / (1 + ratio)', () => {
    const d = marchenkoPastur(0.5, 2)
    const mode = num(d.mode())
    expect(mode).toBeCloseTo((2 * 0.25) / 1.5, 14)
    expect(num(d.prob(mode))).toBeGreaterThan(num(d.prob(mode * 0.95)))
    expect(num(d.prob(mode))).toBeGreaterThan(num(d.prob(mode * 1.05)))
  })

  it('has an entropy equal to minus the integral of p log p', () => {
    const d = marchenkoPastur(0.4)
    const { lower, upper } = marchenkoPasturEdges(0.4)
    const direct = -num(
      quad((x) => mul(d.prob(x), d.logProb(x)) as Value, lower, upper, { epsabs: 1e-12, limit: 200 }).value,
    )
    expect(num(d.entropy())).toBeCloseTo(direct, 7)
  })
})

describe('marchenkoPastur with a ratio above 1', () => {
  const d = marchenkoPastur(2, 1)
  const atom = 0.5

  it('puts an atom of 1 - 1 / ratio at zero', () => {
    expect(num(d.cdf(0))).toBe(atom)
    expect(num(d.cdf(0.01))).toBe(atom)
    const { lower, upper } = marchenkoPasturEdges(2)
    expect(num(d.cdf(lower))).toBe(atom)
    expect(num(d.cdf(upper))).toBe(1)
    expect(num(d.cdf(-0.5))).toBe(0)
  })

  it('has a density that integrates to 1 / ratio', () => {
    const { lower, upper } = marchenkoPasturEdges(2)
    const total = quad((x) => d.prob(x) as Value, lower, upper, { epsabs: 1e-12, limit: 200 })
    expect(num(total.value)).toBeCloseTo(1 - atom, 8)
  })

  it('has the mean variance and the variance variance^2 ratio of the whole law', () => {
    expect(num(d.mean())).toBe(1)
    expect(num(d.variance())).toBe(2)
  })

  it('inverts the cdf and sends the atom to zero', () => {
    expect(num(d.ppf(0.3))).toBe(0)
    expect(num(d.ppf(0.5))).toBe(0)
    for (const x of [0.2, 1, 2.5, 4]) expect(num(d.ppf(d.cdf(x)))).toBeCloseTo(x, 8)
  })

  it('has no entropy', () => {
    expect(Number.isNaN(num(d.entropy()))).toBe(true)
  })
})

describe('marchenkoPastur quantiles and samples', () => {
  it('inverts the cdf', () => {
    const d = marchenkoPastur(0.3, 1.5)
    const q = tensor([0.01, 0.25, 0.5, 0.75, 0.99])
    expect(toFlat(d.cdf(d.ppf(q) as Value)).map((v) => Math.round(v * 1e9) / 1e9)).toEqual([
      0.01, 0.25, 0.5, 0.75, 0.99,
    ])
    expect(num(d.ppf(0))).toBeCloseTo(marchenkoPasturEdges(0.3, 1.5).lower, 12)
    expect(num(d.ppf(1))).toBeCloseTo(marchenkoPasturEdges(0.3, 1.5).upper, 12)
  })

  it('samples inside the support with the law moments and is reproducible from the key', () => {
    const d = marchenkoPastur(0.5, 2)
    const x = d.sample(key(7), [4000])
    const v = toFlat(x)
    const { lower, upper } = marchenkoPasturEdges(0.5, 2)
    expect(v.every((s) => s >= lower - 1e-12 && s <= upper + 1e-12)).toBe(true)
    expect(num(mean(x))).toBeCloseTo(2, 1)
    const second = num(mean(mul(sub(x, 2), sub(x, 2))))
    expect(second).toBeCloseTo(4 * 0.5, 0)
    expect(toFlat(d.sample(key(7), [4000]))).toEqual(v)
  })

  it('broadcasts over a batch of ratios', () => {
    const d = marchenkoPastur([0.1, 0.5], 1)
    expect(d.batchShape).toEqual([2])
    expect(toFlat(d.mean())).toEqual([1, 1])
    expect(toFlat(d.cdf(1))).toHaveLength(2)
  })

  it('rejects a non-positive parameter', () => {
    expect(() => marchenkoPastur(0)).toThrow(/ratio/)
    expect(() => marchenkoPastur(1, -2)).toThrow(/variance/)
  })
})
