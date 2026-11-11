// The panelled grid: one algorithm for the two-coordinate and the up-to-three-coordinate integrals, its results bit for bit what
// the two separate implementations it replaced gave (the hash below is that of their outputs on the battery).
import { describe, expect, it } from 'vitest'
import { ImproperMessageError, ParameterError } from '../core/errors.ts'
import { betaAxis, gammaAxis, gaussianAxis, integrateAxes, integratePair, pointAxis } from './_grid.ts'

// region battery
/** The bits of every number of a result folded into one hash (FNV-1a over the two words of each double). */
function fold(h: number, v: unknown, view: DataView): number {
  if (typeof v === 'number') {
    view.setFloat64(0, v)
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

/** A battery of integrals over gamma, Gaussian, beta and observed axes (with and without kinks), hashed bit for bit. */
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
        G.integratePair([G.gammaAxis(shape, rate, kinks), G.gaussianAxis(m, v, kinks)] as never, column),
        view,
      )
    }
    h = fold(h, G.integratePair([G.pointAxis(0.5), G.gammaAxis(shape, rate)] as never, diff), view)
    h = fold(h, G.integratePair([G.gaussianAxis(m, v), G.pointAxis(0.2)] as never, diff), view)
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
  it('gives, bit for bit, the numbers the two grids it replaced gave on a battery of gamma, Gaussian, beta and observed axes', () => {
    expect(battery({ gammaAxis, gaussianAxis, betaAxis, pointAxis, integratePair, integrateAxes } as never)).toBe(
      1986832850,
    )
  })

  it('integrates a Gaussian against a Gaussian density to the known mass and moments', () => {
    // N(0, 1) cavity times exp(-x^2 / 2): the mass is 1 / sqrt(2), the tilted variance 1 / 2; the second axis is an observed point.
    const r = integratePair([gaussianAxis(0, 1), pointAxis(0)], () => (x) => -0.5 * x * x)
    expect(r.logMass).toBeCloseTo(-0.5 * Math.log(2), 8)
    expect(r.variance[0]).toBeCloseTo(0.5, 8)
  })

  it('refuses no axes and more than three, and reports a factor with no mass as improper', () => {
    expect(() => integrateAxes([], () => 0)).toThrow(ParameterError)
    const four = [gammaAxis(2, 1), gammaAxis(2, 1), gammaAxis(2, 1), gammaAxis(2, 1)]
    expect(() => integrateAxes(four, () => 0)).toThrow(ParameterError)
    expect(() => integrateAxes([gammaAxis(2, 1)], () => Number.NEGATIVE_INFINITY)).toThrow(ImproperMessageError)
  })
})
