import { describe, expect, it } from 'vitest'
import { ImproperMessageError } from '../core/errors.ts'
import { tiltedByQuadrature } from './_tilted.ts'
import { findWindow } from './_windows.ts'

const spike = (at: number, sd: number) => (x: number) => -0.5 * ((x - at) / sd) ** 2

describe('findWindow', () => {
  it('narrows onto a spike far narrower than the first window and away from the start', () => {
    const [a, b] = findWindow((xs) => Float64Array.from(xs, spike(2.5, 1e-3)), -100, 100, 0, 201)
    expect(a).toBeLessThan(2.5)
    expect(b).toBeGreaterThan(2.5)
    expect(b - a).toBeLessThan(0.5)
  })

  it('widens from the start while the edges still carry mass, within the support', () => {
    const [a, b] = findWindow((xs) => Float64Array.from(xs, (x) => -0.5 * (x / 30) ** 2), -1000, 1000, 0, 401)
    expect(a).toBeLessThan(-250)
    expect(b).toBeGreaterThan(250)
    expect(a).toBeGreaterThanOrEqual(-1000)
  })

  it('throws when nothing finite is found on the whole bounded support, at once', () => {
    let calls = 0
    const none = (xs: Float64Array): Float64Array => {
      calls++
      return Float64Array.from(xs, () => Number.NEGATIVE_INFINITY)
    }
    expect(() => findWindow(none, -3, 3, 0)).toThrow(ImproperMessageError)
    expect(() => findWindow(none, -3, 3, 0)).toThrow('not finite anywhere on [-3, 3]')
    // the first window already reaches the ends of the support after one widening: not 40 identical scans
    expect(calls).toBeLessThan(2 * 4)
  })

  it('throws, with the window it ended on, when an unbounded line has no finite density in 40 widenings', () => {
    const none = (xs: Float64Array): Float64Array => Float64Array.from(xs, () => Number.NEGATIVE_INFINITY)
    expect(() => findWindow(none, Number.NEGATIVE_INFINITY, Number.POSITIVE_INFINITY, 0)).toThrow(
      /not finite on any window searched \(last window \[/,
    )
  })

  it('throws when the mass does not settle within the rounds (a needle the scan keeps missing the width of)', () => {
    // One node carries the mass in every scan, wherever the scan is: each round narrows onto it and the window never fills
    const needle = (xs: Float64Array): Float64Array => {
      let best = 0
      for (let i = 1; i < xs.length; i++)
        if (Math.abs((xs[i] as number) - 0.1) < Math.abs((xs[best] as number) - 0.1)) best = i
      return Float64Array.from(xs, (_, i) => (i === best ? 0 : -100))
    }
    expect(() => findWindow(needle, -1, 1, 0, 51)).toThrow(/did not settle within 40 rounds/)
  })
})

describe('tiltedByQuadrature with a hint', () => {
  it('finds a factor of spread 1e-3 inside a cavity of spread 100, as the hint-less integral does not', () => {
    const cavity = { mean: 0, variance: 1e4 }
    const logf = spike(37, 1e-3)
    const t = tiltedByQuadrature(cavity, logf, {})
    expect(t.mean).toBeCloseTo(37, 6)
    expect(Math.sqrt(t.variance) / 1e-3).toBeCloseTo(1, 3)
  })

  it('keeps a window given in the scale of a heavy-tailed factor, and integrates its tails', () => {
    const logt = (x: number): number => -2 * Math.log1p((x / 0.1) ** 2 / 3)
    const t = tiltedByQuadrature({ mean: 0, variance: 100 }, logt, { window: { centre: 0, scale: 0.1 }, zoom: false })
    expect(t.mean).toBeCloseTo(0, 10)
    // A Student-t of 3 degrees of freedom and scale 0.1 has variance 0.03; the Gaussian cavity of variance 100 does not touch it.
    expect(t.variance / 0.03).toBeCloseTo(1, 1)
  })

  it('reports an integrand that peaks at the edge of the cavity as improper without a hint, and integrates it with one', () => {
    // exp(2 x^2) against N(0, 1): the tilted density is not normalisable, and grows to the edge of the cavity's window.
    expect(() => tiltedByQuadrature({ mean: 0, variance: 1 }, (x) => 2 * x * x)).toThrow(ImproperMessageError)
    expect(() => tiltedByQuadrature({ mean: 0, variance: 1 }, (x) => 2 * x * x, {})).not.toThrow()
  })
})
