import { describe, expect, it } from 'vitest'
import { checkPrimitive, grad } from '../autodiff/index.ts'
import { isWhiteboxError } from '../base/index.ts'
import { key, normal } from '../random/index.ts'
import {
  abs,
  add,
  div,
  fromStorage,
  mul,
  square,
  sub,
  sum,
  type Tensor,
  tensor,
  toFlat,
  type Value,
} from '../tensor/index.ts'
import { metricSeries, recorder, tracePolicy } from '../trace/index.ts'
import * as O from './index.ts'

const round = (xs: readonly number[], d = 8): number[] => xs.map((v) => Math.round(v * 10 ** d) / 10 ** d)

describe('proximal operators are differentiable', () => {
  it('soft thresholding, block thresholding and elastic net', () => {
    const x = tensor([1.3, -0.4, 2.2, -1.9])
    expect(checkPrimitive((v) => O.proxL1(v, 0.5), [x]).passed).toBe(true)
    expect(checkPrimitive((v) => O.proxL2(v, 0.5), [x]).passed).toBe(true)
    expect(checkPrimitive((v) => O.proxElasticNet(v, 0.5, { l1: 0.3, l2: 2 }), [x]).passed).toBe(true)
    expect(checkPrimitive((v) => O.proxGroupLasso(v, 0.7, [0, 0, 1, 1]), [x]).passed).toBe(true)
    expect(checkPrimitive((v) => O.proxHuber(v, 0.5, 1), [x]).passed).toBe(true)
    expect(checkPrimitive((v) => O.proxLinf(v, 0.5), [x]).passed).toBe(true)
  })

  it('projections onto the simplex, balls and affine sets', () => {
    const x = tensor([0.9, 0.2, -0.3, 0.6])
    expect(checkPrimitive((v) => O.projectSimplex(v), [x]).passed).toBe(true)
    expect(checkPrimitive((v) => O.projectL1Ball(v, 1), [x]).passed).toBe(true)
    expect(checkPrimitive((v) => O.projectL2Ball(v, 0.5), [x]).passed).toBe(true)
    expect(checkPrimitive((v) => O.projectAffine(v, [[1, 1, 0, 2]], [1]), [x]).passed).toBe(true)
    expect(checkPrimitive((v) => O.projectHalfspace(v, [1, -1, 1, 1], 0.2), [x]).passed).toBe(true)
  })

  it('singular value thresholding and PSD projection', () => {
    const m = tensor([
      [3, 1, 0.5],
      [0.2, 2, 0.1],
    ])
    expect(checkPrimitive((v) => O.proxNuclear(v, 0.5), [m], { rtol: 1e-4 }).passed).toBe(true)
    expect(checkPrimitive((v) => O.projectNuclearBall(v, 2), [m], { rtol: 1e-4 }).passed).toBe(true)
  })

  it('one-dimensional total variation in y and lambda', () => {
    const y = tensor([0.1, 0.3, 2.2, 2.0, 2.4, -1, -0.8, 0.5])
    expect(checkPrimitive((v) => O.proxTv1d(v, 0.3), [y]).passed).toBe(true)
    expect(checkPrimitive((v, l) => O.proxTv1d(v, l), [y, tensor(0.3)]).passed).toBe(true)
    // lambda = 0 is the identity; a huge lambda gives the mean.
    expect(toFlat(O.proxTv1d(y, 0))).toEqual(toFlat(y))
    const mean = toFlat(y).reduce((a, b) => a + b, 0) / 8
    expect(round(toFlat(O.proxTv1d(y, 100)))).toEqual(round(new Array(8).fill(mean)))
  })

  it('the Moreau decomposition x = prox_g(x) + prox_{g*}(x)', () => {
    const x = tensor([2, -0.3, 0.7])
    const conj = O.proxConjugate((v, t) => O.proxL1(v, t))
    const both = add(O.proxL1(x, 1), conj(x, 1))
    expect(round(toFlat(both))).toEqual(toFlat(x))
    // The conjugate of |.|_1 is the indicator of the infinity ball: its prox clips to [-1, 1].
    expect(round(toFlat(conj(x, 1)))).toEqual([1, -0.3, 0.7])
  })

  it('projections land in their sets and fix their members', () => {
    const x = normal(key(3), [9])
    const p = toFlat(O.projectSimplex(x, 2))
    expect(Math.abs(p.reduce((a, b) => a + b, 0) - 2)).toBeLessThan(1e-12)
    expect(p.every((v) => v >= 0)).toBe(true)
    expect(round(toFlat(O.projectSimplex(p, 2)), 12)).toEqual(round(p, 12))
    const b = toFlat(O.projectL1Ball(x, 0.5))
    expect(b.reduce((a, v) => a + Math.abs(v), 0)).toBeCloseTo(0.5, 12)
    expect(toFlat(O.projectL1Ball([0.1, -0.1], 1))).toEqual([0.1, -0.1])
    expect(() => O.projectSimplex([1], 0)).toThrow()
  })

  it('TV-2D rejects traced images and vectors', () => {
    expect(() => O.proxTv2d([1, 2], 1)).toThrow()
    try {
      grad((v: Value) => sum(O.proxTv2d(v, 0.1)))(
        tensor([
          [1, 2],
          [3, 4],
        ]),
      )
    } catch (e) {
      expect(isWhiteboxError(e, 'not-differentiable')).toBe(true)
    }
  })
})

describe('proximal and splitting methods', () => {
  it('ISTA traces the objective, the gradient mapping and the backtracking', () => {
    const rec = recorder()
    const f = (x: unknown): Value => sum(square(sub(x as Value, tensor([3, -2, 0.2]))))
    const r = O.ista(f, (x, t) => O.proxL1(x, t), [0, 0, 0], {
      g: (x) => sum(abs(x)),
      tracer: rec,
      policy: tracePolicy({}),
    })
    expect(round(toFlat(r.x as Value))).toEqual([2.5, -1.5, 0])
    const loss = metricSeries(rec.run(), 'loss').values
    expect(loss.every((v, i) => i === 0 || v <= (loss[i - 1] ?? Infinity) + 1e-12)).toBe(true)
    expect(metricSeries(rec.run(), 'trials').values.length).toBeGreaterThan(0)
  })

  it('FISTA needs fewer iterations than ISTA on an ill-conditioned problem', () => {
    const d = tensor([1, 10, 100])
    const f = (x: unknown): Value => mul(sum(mul(d, square(sub(x as Value, 1)))), 0.5)
    const prox = (x: Value, t: number): Value => O.proxL1(x, 0.5 * t)
    const a = O.ista(f, prox, [0, 0, 0], { tol: 1e-12, stepSize: 0.01 })
    const b = O.fista(f, prox, [0, 0, 0], { tol: 1e-12, stepSize: 0.01, restart: 'function' })
    expect(round(toFlat(b.x as Value), 6)).toEqual(round(toFlat(a.x as Value), 6))
    expect(b.steps).toBeLessThan(a.steps)
  })

  it('ADMM by proximal operators and consensus agree with the closed form', () => {
    const r = O.admmProx(
      (v, t) => add(div(sub(v, 3), 1 + t), 3),
      (v, t) => O.proxL1(v, t),
      [0, 0],
      { adaptiveRho: true, relaxation: 1.6 },
    )
    expect(r.converged).toBe(true)
    expect(round(toFlat(r.z))).toEqual([2, 2])
    const terms = [1, 2, 6].map(
      (a): O.Prox =>
        (v, t) =>
          div(add(v, a * t), 1 + t),
    )
    const c = O.consensusAdmm(terms, [0], { proxG: (v, t) => O.proxL1(v, t) })
    // argmin sum (x - a_i)^2 / 2 + |x| = mean - 1/3.
    expect(round(toFlat(c.z), 6)).toEqual([round([3 - 1 / 3], 6)[0]])
  })

  it('Chambolle-Pock denoises an image like the TV prox', () => {
    const m = 4
    const n = 4
    const img = fromStorage(
      Float64Array.from({ length: 16 }, (_, i) => (i % 4 < 2 ? 0 : 2) + 0.1 * Math.sin(i)),
      [m, n],
      'float64',
    )
    const lambda = 0.3
    // K: the forward-difference gradient with zero beyond the boundary, stacked [rows; cols].
    const grad2 = (x: Value): Value => {
      const a = toFlat(x)
      const out = new Float64Array(2 * m * n)
      for (let i = 0; i < m; i++)
        for (let j = 0; j < n; j++) {
          out[i * n + j] = i < m - 1 ? (a[(i + 1) * n + j] ?? 0) - (a[i * n + j] ?? 0) : 0
          out[m * n + i * n + j] = j < n - 1 ? (a[i * n + j + 1] ?? 0) - (a[i * n + j] ?? 0) : 0
        }
      return fromStorage(out, [2 * m * n], 'float64')
    }
    const gradT = (y: Value): Value => {
      const b = toFlat(y)
      const out = new Float64Array(m * n)
      for (let i = 0; i < m; i++)
        for (let j = 0; j < n; j++) {
          const r = b[i * n + j] ?? 0
          const c = b[m * n + i * n + j] ?? 0
          if (i < m - 1) {
            out[(i + 1) * n + j] = (out[(i + 1) * n + j] ?? 0) + r
            out[i * n + j] = (out[i * n + j] ?? 0) - r
          }
          if (j < n - 1) {
            out[i * n + j + 1] = (out[i * n + j + 1] ?? 0) + c
            out[i * n + j] = (out[i * n + j] ?? 0) - c
          }
        }
      return fromStorage(out, [m * n], 'float64')
    }
    const b = Float64Array.from((img as Tensor).data)
    const r = O.chambollePock(
      {
        K: { shape: [2 * m * n, m * n], matvec: grad2, rmatvec: gradT },
        proxG: (x, t) => div(add(x, mul(fromStorage(b, [16], 'float64'), t)), 1 + t),
        // F* is the indicator of the pointwise ball of radius lambda: project each (row, col) pair.
        proxFConjugate: (y) => {
          const v = toFlat(y)
          const out = Float64Array.from(v)
          for (let k = 0; k < m * n; k++) {
            const s = Math.max(1, Math.hypot(v[k] ?? 0, v[m * n + k] ?? 0) / lambda)
            out[k] = (v[k] ?? 0) / s
            out[m * n + k] = (v[m * n + k] ?? 0) / s
          }
          return fromStorage(out, [2 * m * n], 'float64')
        },
      },
      new Float64Array(16),
      { tol: 1e-10, maxIterations: 50000 },
    )
    const tv = toFlat(O.proxTv2d(img, lambda, { maxIterations: 20000, tol: 1e-14 }))
    expect(round(toFlat(r.x as Value), 5)).toEqual(round(tv, 5))
  })
})
