import { describe, expect, it } from 'vitest'
import { grad } from '../../autodiff/index.ts'
import { MdmlError } from '../../base/index.ts'
import { key } from '../../random/index.ts'
import { asTensor, fromStorage, square, sum, toFlat, type Value } from '../../tensor/index.ts'
import type { Params } from '../module/module.ts'
import { AdditiveModel } from './additive-model.ts'
import { grevilleAbscissae } from './bspline.ts'
import { ExULayer, interactionTerm, linearTerm, netTerm, splineTerm, subnetTerm, TensorSpline2d } from './terms.ts'

const rows = (n: number, d: number): Value =>
  asTensor(
    Array.from({ length: n }, (_, i) => Array.from({ length: d }, (_, j) => ((i * (3 + 2 * j) + j) % n) / n + 0.01)),
  )
const maxAbs = (a: ArrayLike<number>, b: ArrayLike<number>): number =>
  Array.from(a).reduce((m, v, i) => Math.max(m, Math.abs(v - (b[i] ?? 0))), 0)

const x = rows(40, 3)

describe('AdditiveModel', () => {
  const terms = [
    splineTerm(0, { interiorKnots: 4, knots: { kind: 'learnable' } }),
    netTerm(1, { width: 6, depth: 2, unit: 'exu' }),
    linearTerm(2),
    interactionTerm([0, 1], { interiorKnots: 3 }),
    subnetTerm([0, 1, 2], { width: 4, depth: 1 }),
  ]
  const model = new AdditiveModel({ inFeatures: 3, terms, sample: x, key: key(1), intercept: 0.5 })

  it('builds every kind of term from the table and labels them', () => {
    expect(model.labels).toEqual(['s(x0)', 'nn(x1)', 'x2', 'te(x0,x1)', 'nn(x0,x1,x2)'])
    expect(model.built.map((t) => t.kind)).toEqual(['spline', 'net', 'linear', 'tensorSpline', 'net'])
    expect(model.forward(x).shape).toEqual([40, 1])
    expect(Object.keys(model.params).sort()).toEqual(['intercept', 't0', 't1', 't2', 't3', 't4'])
  })

  it('is the intercept plus the sum of its terms, and the mean is the inverse link of that', () => {
    const c = toFlat(model.contributions(model.params, x))
    const y = toFlat(model.apply(model.params, x))
    for (let i = 0; i < 40; i++)
      expect(y[i]).toBeCloseTo(0.5 + [0, 1, 2, 3, 4].reduce((a, t) => a + (c[i * 5 + t] ?? 0), 0), 12)
    const logit = new AdditiveModel({ inFeatures: 3, terms: [linearTerm(0)], link: 'logit', key: key(2) })
    const eta = toFlat(logit.apply(logit.params, x))
    const p = toFlat(logit.mean(logit.params, x))
    expect(
      maxAbs(
        p,
        eta.map((e) => 1 / (1 + Math.exp(-e))),
      ),
    ).toBeLessThan(1e-12)
  })

  it('centres each term over the reference rows, and can leave them uncentred', () => {
    const c = toFlat(model.contributions(model.params, x))
    for (let t = 0; t < 5; t++) {
      const col = Array.from({ length: 40 }, (_, i) => c[i * 5 + t] ?? 0)
      expect(col.reduce((a, b) => a + b, 0) / 40).toBeCloseTo(0, 12)
    }
    const raw = new AdditiveModel({ inFeatures: 3, terms: [linearTerm(0)], sample: x, centre: false, key: key(2) })
    expect(toFlat(raw.centres(raw.params))).toEqual([0])
    const centred = new AdditiveModel({ inFeatures: 3, terms: [splineTerm(0)], sample: x, key: key(2), centreRows: 10 })
    expect((centred.buffers['reference'] as Value).shape).toEqual([10, 3])
  })

  it('enforces heredity: an interaction needs its main effects', () => {
    const inter = interactionTerm([0, 1])
    expect(() => new AdditiveModel({ inFeatures: 3, terms: [splineTerm(0), inter], heredity: 'strong' })).toThrow(
      MdmlError,
    )
    expect(() => new AdditiveModel({ inFeatures: 3, terms: [splineTerm(0), inter], heredity: 'weak' })).not.toThrow()
    expect(
      () => new AdditiveModel({ inFeatures: 3, terms: [splineTerm(0), splineTerm(1), inter], heredity: 'strong' }),
    ).not.toThrow()
    expect(() => new AdditiveModel({ inFeatures: 3, terms: [inter], heredity: 'weak' })).toThrow(/heredity/)
  })

  it('drops whole feature nets in training only, and keeps expectations by rescaling', () => {
    const dropping = new AdditiveModel({
      inFeatures: 1,
      terms: [netTerm(0, { width: 4, featureDropout: 0.5 })],
      centre: false,
      intercept: false,
      key: key(3),
    })
    const big = rows(400, 1)
    const eval_ = toFlat(dropping.contributions(dropping.params, big, { train: false, key: key(1) }))
    const trained = toFlat(dropping.contributions(dropping.params, big, { train: true, key: key(1) }))
    const zeroed = trained.filter((v) => v === 0).length
    expect(zeroed).toBeGreaterThan(120)
    expect(zeroed).toBeLessThan(280)
    trained.forEach((v, i) => {
      if (v !== 0) expect(v).toBeCloseTo(2 * (eval_[i] as number), 12)
    })
  })

  it('has the smoothness and output penalties of its terms, and gradients to weights, coefficients and knots', () => {
    const m = new AdditiveModel({
      inFeatures: 2,
      terms: [
        splineTerm(0, { interiorKnots: 5, lambda: 3, knots: { kind: 'learnable' } }),
        netTerm(1, { width: 4, outputPenalty: 2 }),
      ],
      sample: x.shape[1] === 3 ? rows(40, 2) : x,
      key: key(4),
    })
    const { smoothness, regularisation } = m.penalties(m.params)
    const spline = m.built[0]?.smoothness((m.params['t0'] ?? {}) as Params, (m.buffers['t0'] ?? {}) as Params)
    expect(toFlat(smoothness)[0]).toBeCloseTo(toFlat(spline as Value)[0] as number, 12)
    const ref = m.buffers['reference'] as Value
    const f = toFlat(m.built[1]?.module.apply((m.params['t1'] ?? {}) as Params, take1(ref), { train: false }) as Value)
    expect(toFlat(regularisation)[0]).toBeCloseTo((2 * f.reduce((a, v) => a + v * v, 0)) / f.length, 12)
    expect(toFlat(m.penalty(m.params, 40))[0]).toBeCloseTo(
      (toFlat(smoothness)[0] as number) / 40 + (toFlat(regularisation)[0] as number),
      12,
    )
    const g = grad((p: Params) => sum(square(m.apply(p, rows(40, 2)))) as Value)(m.params) as Params
    const t0 = g['t0'] as Params
    expect(Object.keys(t0).sort()).toEqual(['coef', 'knotLogits'])
    expect(toFlat(t0['knotLogits'] as Value).some((v) => Math.abs(v) > 1e-8)).toBe(true)
  })

  it('reports shape functions on grids, and effective degrees of freedom that fall as lambda grows', () => {
    const grid = Array.from({ length: 11 }, (_, i) => i / 10)
    expect(model.termCurve(model.params, 0, grid).shape).toEqual([11])
    expect(model.termSurface(model.params, 3, [0, 0.5, 1], [0, 1]).shape).toEqual([3, 2])
    expect(() => model.termCurve(model.params, 3, grid)).toThrow(MdmlError)
    const dense = rows(120, 1)
    const edfFor = (lambda: number): number =>
      new AdditiveModel({
        inFeatures: 1,
        terms: [splineTerm(0, { interiorKnots: 8, lambda })],
        sample: dense,
        key: key(5),
      }).termEdf(
        new AdditiveModel({
          inFeatures: 1,
          terms: [splineTerm(0, { interiorKnots: 8, lambda })],
          sample: dense,
          key: key(5),
        }).params,
        dense,
      )[0] as number
    expect(edfFor(0)).toBeCloseTo(12, 3)
    expect(edfFor(1e8)).toBeCloseTo(2, 1)
    expect(edfFor(1)).toBeGreaterThan(edfFor(100))
  })
})

const take1 = (v: Value): Value =>
  asTensor(
    toFlat(v)
      .filter((_, i) => i % 2 === 1)
      .map((u) => [u]),
  )

describe('terms', () => {
  it('a tensor-product spline reproduces a bilinear function from the Greville coefficients', () => {
    const t = new TensorSpline2d({ interiorKnots: 4, range: [0, 1], key: key(6) })
    const g = toFlat(grevilleAbscissae(t.margins.knotVectors({}), 3)).slice(0, t.margins.basisSize)
    const k = g.length
    const coef = fromStorage(
      Float64Array.from({ length: k * k }, (_, i) => (g[Math.floor(i / k)] as number) * (g[i % k] as number)),
      [k, k],
      'float64',
    )
    const grid = [0, 0.13, 0.5, 0.77, 1]
    const surface = toFlat(t.surface({ coef }, grid, grid))
    const want = grid.flatMap((a) => grid.map((b) => a * b))
    expect(maxAbs(surface, want)).toBeLessThan(1e-12)
    // Greville points are not equally spaced near the ends, so the index-based difference penalty is not zero here.
    expect(toFlat(t.penalty({ coef }, 2))[0]).toBeGreaterThan(0)
  })

  it('ExU units are the clipped ReLU of the offset input times the exponentiated weights', () => {
    const layer = new ExULayer(1, 3, { key: key(7) })
    const xs = [[0.1], [0.6], [-0.4]]
    const w = toFlat(layer.params['weight'] as Value)
    const b = toFlat(layer.params['bias'] as Value)[0] as number
    const out = toFlat(layer.forward(xs))
    xs.forEach(([v], i) => {
      for (let j = 0; j < 3; j++)
        expect(out[i * 3 + j]).toBeCloseTo(Math.min(1, Math.max(0, ((v as number) - b) * Math.exp(w[j] as number))), 12)
    })
  })
})
