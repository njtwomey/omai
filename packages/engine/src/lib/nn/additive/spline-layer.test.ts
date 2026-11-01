import { describe, expect, it } from 'vitest'
import { grad, numericalGradient } from '../../autodiff/index.ts'
import { key } from '../../random/index.ts'
import { differencePenalty } from '../../stats/smoothing/index.ts'
import { asTensor, fromStorage, sum, toFlat, type Value } from '../../tensor/index.ts'
import type { Params } from '../module/module.ts'
import { grevilleAbscissae } from './bspline.ts'
import { differenceRoughness, SplineLayer } from './spline-layer.ts'

const grid = (n: number, lo = 0, hi = 1): number[] =>
  Array.from({ length: n }, (_, i) => lo + ((hi - lo) * i) / (n - 1))
const maxAbs = (a: ArrayLike<number>, b: ArrayLike<number>): number =>
  Array.from(a).reduce((m, v, i) => Math.max(m, Math.abs(v - (b[i] ?? 0))), 0)
const withCoef = (layer: SplineLayer, coef: Value): Params => ({ ...layer.params, coef })

describe('SplineLayer', () => {
  it('sums one spline per input and adds the bias: shapes, terms and the parameter count', () => {
    const layer = new SplineLayer({ inFeatures: 3, outFeatures: 2, interiorKnots: 4, key: key(1) })
    const x = asTensor([
      [0.1, 0.5, 0.9],
      [0.3, 0.2, 0.8],
    ])
    expect(layer.forward(x).shape).toEqual([2, 2])
    expect(layer.terms(layer.params, x).shape).toEqual([2, 3, 2])
    expect(layer.basisSize).toBe(8)
    expect(layer.parameterCount()).toBe(2 * 3 * 8 + 2)
    const y = toFlat(layer.forward(x))
    const t = toFlat(layer.terms(layer.params, x))
    for (let n = 0; n < 2; n++)
      for (let o = 0; o < 2; o++)
        expect(y[n * 2 + o]).toBeCloseTo(
          [0, 1, 2].reduce((a, i) => a + (t[(n * 3 + i) * 2 + o] ?? 0), 0),
          12,
        )
  })

  it('reproduces constants and the identity: coefficients 1 give 1, coefficients at the Greville points give x', () => {
    const layer = new SplineLayer({ inFeatures: 1, outFeatures: 1, interiorKnots: 5, bias: false, key: key(2) })
    const xs = grid(25, 0, 1)
    const ones = withCoef(layer, fromStorage(new Float64Array(9).fill(1), [1, 1, 9], 'float64'))
    for (const v of toFlat(layer.apply(ones, asTensor(xs)))) expect(v).toBeCloseTo(1, 12)
    const g = toFlat(grevilleAbscissae(layer.knotVectors(layer.params), 3))
    const linear = withCoef(layer, fromStorage(Float64Array.from(g), [1, 1, 9], 'float64'))
    expect(maxAbs(toFlat(layer.apply(linear, asTensor(xs))), xs)).toBeLessThan(1e-12)
  })

  it('takes its range and quantile knots from a sample, and keeps them in buffers that survive a state dict', () => {
    const sample = asTensor(Array.from({ length: 50 }, (_, i) => [(i / 49) ** 2 * 10]))
    const layer = new SplineLayer({
      inFeatures: 1,
      outFeatures: 1,
      interiorKnots: 4,
      knots: { kind: 'quantile' },
      sample,
      key: key(3),
    })
    const interior = toFlat(layer.interior(layer.params))
    // Quantiles of squares cluster near zero: the first knot sits well below the midrange.
    expect(interior[0]).toBeLessThan(1)
    expect(interior[3]).toBeLessThan(8)
    expect(toFlat(layer.buffers['range'] as Value)).toEqual([0, 10])
    const copy = new SplineLayer({
      inFeatures: 1,
      outFeatures: 1,
      interiorKnots: 4,
      knots: { kind: 'quantile' },
      key: key(9),
    })
    copy.loadStateDict(layer.stateDict())
    const x = asTensor([[0.3], [2.5], [9.9]])
    expect(maxAbs(toFlat(copy.forward(x)), toFlat(layer.forward(x)))).toBeLessThan(1e-14)
  })

  describe('free knots', () => {
    const layer = new SplineLayer({
      inFeatures: 2,
      outFeatures: 1,
      interiorKnots: 4,
      knots: { kind: 'learnable' },
      range: [0, 1],
      initScale: 0.5,
      key: key(4),
    })
    const x = asTensor(Array.from({ length: 12 }, (_, i) => [((i * 7) % 12) / 12 + 0.02, ((i * 5) % 12) / 12 + 0.03]))
    it('starts from evenly spaced ordered knots and trains the logits through the basis', () => {
      expect(
        toFlat(layer.interior(layer.params))
          .slice(0, 4)
          .map((v) => Math.round(v * 1e9) / 1e9),
      ).toEqual([0.2, 0.4, 0.6, 0.8])
      expect(Object.keys(layer.params).sort()).toEqual(['bias', 'coef', 'knotLogits'])
      // Make the output depend on the knots: a non-constant coefficient pattern.
      const coef = fromStorage(
        Float64Array.from({ length: 16 }, (_, i) => Math.sin(i)),
        [1, 2, 8],
        'float64',
      )
      const shaped = (logits: Value): Value =>
        sum(layer.apply({ ...layer.params, coef, knotLogits: logits }, x) as Value)
      const at = layer.params['knotLogits'] as Value
      const analytic = toFlat(grad(shaped)(at) as Value)
      const numeric = toFlat(numericalGradient(shaped, at, 1e-6))
      expect(analytic.some((v) => Math.abs(v) > 1e-4)).toBe(true)
      expect(maxAbs(analytic, numeric)).toBeLessThan(1e-6)
    })
  })

  describe('shape constraints hold for any raw coefficients', () => {
    const raw = (layer: SplineLayer, seed: number): Value =>
      fromStorage(
        Float64Array.from({ length: layer.basisSize }, (_, i) => 3 * Math.sin(seed + 2.3 * i)),
        [1, 1, layer.basisSize],
        'float64',
      )
    const curveOf = (shape: { kind: string; lower?: number; upper?: number }, seed: number): number[] => {
      const layer = new SplineLayer({
        inFeatures: 1,
        outFeatures: 1,
        interiorKnots: 6,
        shape,
        key: key(5),
        knots: { kind: 'learnable' },
      })
      // Uneven knots too: the constraints are in terms of the knots in use.
      const params = {
        ...layer.params,
        coef: raw(layer, seed),
        knotLogits: asTensor([[0.9, -0.4, 0.3, 1.1, -1, 0.2, 0.5]]),
      }
      return toFlat(layer.curve(params, grid(201), 0, 0))
    }
    const second = (v: number[]): number[] =>
      v.slice(2).map((_, i) => (v[i + 2] as number) - 2 * (v[i + 1] as number) + (v[i] as number))

    it('increasing and decreasing splines are monotone', () => {
      for (const seed of [0.3, 1.7, 5]) {
        const up = curveOf({ kind: 'increasing' }, seed)
        const down = curveOf({ kind: 'decreasing' }, seed)
        expect(up.every((v, i) => i === 0 || v >= (up[i - 1] as number) - 1e-12)).toBe(true)
        expect(down.every((v, i) => i === 0 || v <= (down[i - 1] as number) + 1e-12)).toBe(true)
        expect((up.at(-1) as number) - (up[0] as number)).toBeGreaterThan(0)
      }
    })

    it('convex and concave splines have second differences of one sign', () => {
      for (const seed of [0.3, 1.7, 5]) {
        expect(Math.min(...second(curveOf({ kind: 'convex' }, seed)))).toBeGreaterThanOrEqual(-1e-9)
        expect(Math.max(...second(curveOf({ kind: 'concave' }, seed)))).toBeLessThanOrEqual(1e-9)
      }
    })

    it('bounded splines stay inside their bounds', () => {
      for (const seed of [0.3, 1.7, 5]) {
        const v = curveOf({ kind: 'bounded', lower: -2, upper: 3 }, seed)
        expect(Math.min(...v)).toBeGreaterThanOrEqual(-2)
        expect(Math.max(...v)).toBeLessThanOrEqual(3)
      }
    })
  })

  it('has the P-spline penalty of lib/stats/smoothing: the sum of beta S beta over the splines', () => {
    const layer = new SplineLayer({ inFeatures: 2, outFeatures: 2, interiorKnots: 5, key: key(6), initScale: 1 })
    const S = toFlat(differencePenalty(layer.basisSize, 2))
    const c = toFlat(layer.params['coef'] as Value)
    const k = layer.basisSize
    let expected = 0
    for (let s = 0; s < 4; s++) {
      const beta = c.slice(s * k, (s + 1) * k)
      for (let a = 0; a < k; a++)
        for (let b = 0; b < k; b++) expected += (beta[a] as number) * (S[a * k + b] as number) * (beta[b] as number)
    }
    expect(toFlat(layer.penalty(layer.params))[0]).toBeCloseTo(expected, 10)
    expect(toFlat(differenceRoughness([[1, 2, 4, 8]], 2))[0]).toBe(5)
    expect(toFlat(differenceRoughness([[1, 2, 4, 8]], 0))[0]).toBe(85)
  })

  it('has slopes equal to the exact derivative of the curve', () => {
    const layer = new SplineLayer({ inFeatures: 1, outFeatures: 1, interiorKnots: 5, key: key(7), initScale: 1 })
    const xs = grid(9, 0.05, 0.95)
    const h = 1e-6
    const slope = toFlat(layer.termDerivatives(layer.params, asTensor(xs.map((v) => [v]))))
    const up = toFlat(
      layer.curve(
        layer.params,
        xs.map((v) => v + h),
      ),
    )
    const down = toFlat(
      layer.curve(
        layer.params,
        xs.map((v) => v - h),
      ),
    )
    expect(
      maxAbs(
        slope,
        up.map((v, i) => (v - (down[i] as number)) / (2 * h)),
      ),
    ).toBeLessThan(1e-6)
  })
})
