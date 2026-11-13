// The panelled grid: one algorithm for the two-coordinate and the up-to-three-coordinate integrals, its results what the two separate
// implementations it replaced gave (the hash below is that of their outputs on the battery, each number rounded to eight decimals: the
// bits of a float depend on the platform, see the battery), except the variance of `integratePair`, which is centred now (the second
// pass of the weighted squares of the deviations) and so no longer the number the difference of two second moments gave: it is left
// out of the hash and tested on its own.
import { describe, expect, it } from 'vitest'
import { ImproperMessageError, ParameterError } from '../core/errors.ts'
import { betaAxis, gammaAxis, gaussianAxis, integrateAxes, integratePair, pointAxis } from './_grid.ts'

// region battery
/**
 * Every number of a result folded into one hash (FNV-1a over the two words of each double), each rounded to eight decimals first: Math.exp,
 * Math.sin and Math.pow of V8 differ in the last bit between arm64 (macOS) and x64 (the CI runners) for about one input in ten, so the
 * bits of a result are not the same on both, and a hash of the bits was the number of one machine. Rounded to 1e-8 the hash is the same
 * under a last-bit change of every call of those functions (checked), and still changes when those results move by a part in a million.
 */
function fold(h: number, v: unknown, view: DataView): number {
  if (typeof v === 'number') {
    const rounded = Math.round(v * 1e8)
    view.setFloat64(0, Object.is(rounded, -0) ? 0 : rounded)
    for (const o of [0, 4]) h = Math.imul(h ^ view.getUint32(o), 16777619)
    return h
  }
  if (v instanceof Float64Array || Array.isArray(v)) {
    for (const x of Array.from(v as ArrayLike<unknown>)) h = fold(h, x, view)
    return fold(h, (v as ArrayLike<unknown>).length, view)
  }
  if (v !== null && typeof v === 'object')
    for (const k of Object.keys(v)) h = fold(h, (v as Record<string, unknown>)[k], view)
  return h
}

/** A result of `integratePair` without its variance, which is no longer what it was (see the header). */
const withoutVariance = (r: unknown): unknown => {
  const { variance: _variance, ...rest } = r as { variance: unknown }
  return rest
}

/** A battery of integrals over gamma, Gaussian, beta and observed axes (with and without kinks), hashed (see {@link fold}). */
function battery(G: {
  readonly gammaAxis: (s: number, r: number, k?: readonly number[]) => never
  readonly gaussianAxis: (m: number, v: number, k?: readonly number[]) => never
  readonly betaAxis: (a: number, b: number) => never
  readonly pointAxis: (x: number) => never
  readonly integratePair: (axes: never, column: (x1: number) => (x0: number) => number) => unknown
  readonly integrateAxes: (axes: never, f: (t: Float64Array) => number) => unknown
}): number {
  const view = new DataView(new ArrayBuffer(8))
  let h = 2166136261
  const column = (x1: number) => (x0: number) => -0.5 * (x0 - 0.3 * x1) ** 2 - Math.abs(x1 - 0.2) + Math.sin(x0)
  const diff = (x1: number) => (x0: number) => -((x0 - x1) ** 2)
  for (const [shape, rate, m, v] of [
    [2, 1, 0.3, 1.5],
    [0.5, 3, -1, 4],
    [30, 20, 2, 0.01],
    [1, 1, 0, 100],
  ] as const) {
    for (const kinks of [[], [0.5], [-0.5, 1]]) {
      h = fold(
        h,
        withoutVariance(
          G.integratePair([G.gammaAxis(shape, rate, kinks), G.gaussianAxis(m, v, kinks)] as never, column),
        ),
        view,
      )
    }
    h = fold(h, withoutVariance(G.integratePair([G.pointAxis(0.5), G.gammaAxis(shape, rate)] as never, diff)), view)
    h = fold(h, withoutVariance(G.integratePair([G.gaussianAxis(m, v), G.pointAxis(0.2)] as never, diff)), view)
    for (const [a, b] of [
      [2, 3],
      [0.5, 0.5],
      [20, 2],
    ] as const) {
      const f = (t: Float64Array): number => {
        let s = 0
        for (let i = 0; i < t.length; i++) s += Math.sin(1 + i * (t[i] as number)) - 0.1 * (t[i] as number) ** 2
        return s
      }
      const axes = [G.gammaAxis(shape, rate), G.gammaAxis(shape + 1, rate * 2), G.betaAxis(a, b)]
      for (const k of [1, 2, 3]) h = fold(h, G.integrateAxes(axes.slice(0, k) as never, f), view)
      h = fold(h, G.integrateAxes([G.pointAxis(0.3), G.gammaAxis(shape, rate), G.betaAxis(a, b)] as never, f), view)
    }
  }
  return h >>> 0
}
// endregion

describe('the panelled grid', () => {
  it('gives, to eight decimals, the numbers the two grids it replaced gave on a battery of gamma, Gaussian, beta and observed axes', () => {
    expect(battery({ gammaAxis, gaussianAxis, betaAxis, pointAxis, integratePair, integrateAxes } as never)).toBe(
      1809488587,
    )
  })

  it('integrates a Gaussian against a Gaussian density to the known mass and moments', () => {
    // N(0, 1) cavity times exp(-x^2 / 2): the mass is 1 / sqrt(2), the tilted variance 1 / 2; the second axis is an observed point.
    const r = integratePair([gaussianAxis(0, 1), pointAxis(0)], () => (x) => -0.5 * x * x)
    expect(r.logMass).toBeCloseTo(-0.5 * Math.log(2), 8)
    expect(r.variance[0]).toBeCloseTo(0.5, 8)
  })

  it('centres the variance: a mass of spread 0.01 at a mean of 1e8 keeps it', () => {
    // N(1e8, 1e-4) times exp(-(x - 1e8)^2 / 2e-4): the tilted variance is 5e-5, which the difference of second moments (1e16) cannot hold.
    const centre = 1e8
    const r = integratePair([gaussianAxis(centre, 1e-4), pointAxis(0)], () => (x) => (-0.5 * (x - centre) ** 2) / 1e-4)
    expect(r.variance[0] / 5e-5).toBeCloseTo(1, 6)
    expect(r.mean[0] / centre).toBeCloseTo(1, 12)
  })

  it('refuses no axes and more than three, and reports a factor with no mass as improper', () => {
    expect(() => integrateAxes([], () => 0)).toThrow(ParameterError)
    const four = [gammaAxis(2, 1), gammaAxis(2, 1), gammaAxis(2, 1), gammaAxis(2, 1)]
    expect(() => integrateAxes(four, () => 0)).toThrow(ParameterError)
    expect(() => integrateAxes([gammaAxis(2, 1)], () => Number.NEGATIVE_INFINITY)).toThrow(ImproperMessageError)
  })
})
