import { describe, expect, it } from 'vitest'
import { messageArg } from '../core/args.ts'
import { ImproperMessageError } from '../core/errors.ts'
import { gaussian, gaussianEta } from '../core/families.ts'
import { apply } from '../operators/likelihoods/functions.ts'
import { centredMoments, resolvedVariance, shiftedSum } from './_moments.ts'

describe('shiftedSum', () => {
  it('shifts by the largest log weight, adds in order and gives the log of the sum', () => {
    const s = shiftedSum([1000, 1000, 998])
    expect(s.top).toBe(1000)
    expect(Array.from(s.weights)).toEqual([1, 1, Math.exp(-2)])
    expect(s.total).toBe(1 + 1 + Math.exp(-2))
    expect(s.logTotal).toBe(1000 + Math.log(s.total))
  })

  it('has no mass when every log weight is minus infinity', () => {
    const s = shiftedSum([Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY])
    expect(s.top).toBe(Number.NEGATIVE_INFINITY)
    expect(Number.isFinite(s.logTotal)).toBe(false)
  })
})

describe('centredMoments', () => {
  it('keeps a spread of 1e-2 at a mean of 1e6, which the difference of second moments cannot hold', () => {
    const xs = [1e6 - 1e-2, 1e6, 1e6 + 1e-2]
    const w = [1, 2, 1]
    const m = centredMoments(w, 4, xs)
    expect(m.mean).toBeCloseTo(1e6, 6)
    expect(m.variance / 5e-5).toBeCloseTo(1, 5)
    const naive = w.reduce((s, v, i) => s + v * (xs[i] as number) ** 2, 0) / 4 - m.mean ** 2
    expect(Math.abs(naive / 5e-5 - 1)).toBeGreaterThan(0.1)
  })

  it('is exactly zero when all the weight is on one value', () => {
    expect(centredMoments([0, 3, 0], 3, [1e6, 1e6 + 0.5, 1e6 - 1]).variance).toBe(0)
  })
})

describe('resolvedVariance', () => {
  it('returns a positive variance and reports a collapse as an improper message, not a floored number', () => {
    expect(resolvedVariance(1e-30, 'site')).toBe(1e-30)
    for (const v of [0, -1e-20, Number.NaN, Number.POSITIVE_INFINITY])
      expect(() => resolvedVariance(v, 'site')).toThrow(ImproperMessageError)
    expect(() => resolvedVariance(0, 'site')).toThrow(/site: the tilted distribution collapsed onto a point/)
  })

  it('is what a numeric site says when its tilted distribution is a point: a rectifier of a cavity far below zero, raised to a power', () => {
    // y = relu(x) with x ~ N(-40, 1) is zero with all the mass: the tilted y is the point mass at 0, and the grid's variance is exactly 0.
    const args = [messageArg(gaussian, gaussianEta(0, 1)), messageArg(gaussian, gaussianEta(-40, 1))]
    const send = (): unknown => apply.ep?.({ mode: 'ep', to: 0, args, params: { fn: 'relu' }, power: 2 })
    expect(send).toThrow(ImproperMessageError)
  })
})
