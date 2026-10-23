import { describe, expect, it } from 'vitest'
import { checkPrimitive } from '../../autodiff/index.ts'
import { maxOf } from '../../base/index.ts'
import { key, normal, uniform } from '../../random/index.ts'
import { tensor, toArray, toFlat, type Value } from '../../tensor/index.ts'
import {
  causalConvolution,
  deltaRule,
  discretise,
  discretiseDiagonal,
  linearRecurrence,
  lru,
  lruEigenvalues,
  lruInit,
  selectiveScan,
  ssmKernel,
  ssmRecurrence,
} from './state-space.ts'

const n = (seed: number, shape: number[]): Value => normal(key(seed), shape)
const close = (a: Value, b: Value | number[], digits = 10): void => {
  const x = toFlat(a)
  const y = Array.isArray(b) ? b : toFlat(b)
  expect(x.length).toBe(y.length)
  for (const [i, v] of x.entries()) expect(v).toBeCloseTo(y[i] ?? Number.NaN, digits)
}
const ok = (f: (...xs: Value[]) => Value, xs: Value[]): void => {
  expect(checkPrimitive(f, xs, { rtol: 1e-5, atol: 1e-7 }).failures).toEqual([])
}

// A stable 3-state system.
const A = tensor([
  [-1, 0.5, 0],
  [-0.5, -1, 0.2],
  [0, -0.3, -2],
])
const B = tensor([1, 0, 0.5])
const C = tensor([0.3, -1, 2])

describe('discretisation', () => {
  it('zero-order hold matches the closed form for a diagonal system', () => {
    const lambda = [-1, -0.5, -2]
    const diag = tensor(lambda.map((l, i) => lambda.map((_, j) => (i === j ? l : 0))))
    const full = discretise(diag, [1, 2, 3], 0.3)
    const d = discretiseDiagonal(lambda, [1, 2, 3], 0.3)
    close(full.b, d.b, 12)
    close(full.a, toFlat(tensor(lambda.map((l, i) => lambda.map((_, j) => (i === j ? Math.exp(0.3 * l) : 0))))), 12)
  })

  it('bilinear and zero-order hold agree to second order as dt shrinks', () => {
    for (const dt of [1e-2, 1e-3]) {
      const z = discretise(A, B, dt)
      const b = discretise(A, B, dt, 'bilinear')
      const diff = toFlat(z.a).map((v, i) => Math.abs(v - (toFlat(b.a)[i] ?? 0)))
      expect(maxOf(diff)).toBeLessThan(10 * dt ** 3)
    }
    close(discretiseDiagonal([-1], [1], 0.1, 'bilinear').a, [(1 - 0.05) / (1 + 0.05)], 12)
    close(discretiseDiagonal([0], [2], 0.25).b, [0.5], 14)
  })

  it('discretisation is differentiable', () => {
    ok((a, b) => discretise(a, b, 0.2).b, [A, B])
    ok((a) => discretise(a, B, 0.2, 'bilinear').a, [A])
    ok((l, b) => discretiseDiagonal(l, b, 0.3).b, [tensor([-1, -0.4]), tensor([1, 2])])
  })
})

describe('kernels and recurrences', () => {
  it('the recurrence is a causal convolution with the S4 kernel', () => {
    const ssm = discretise(A, B, 0.1)
    const u = n(0, [7])
    close(ssmRecurrence(ssm, C, u, 0.4), causalConvolution(u, ssmKernel(ssm, C, 7), 0.4), 12)
    const diag = discretiseDiagonal([-1, -0.3], [1, 2], 0.5)
    close(ssmRecurrence(diag, [0.5, -1], u), causalConvolution(u, ssmKernel(diag, [0.5, -1], 7)), 12)
    // A batch of sequences convolves row by row; a short kernel leaves missing taps at zero.
    expect(causalConvolution(n(1, [3, 7]), [1, 0.5]).shape).toEqual([3, 7])
  })

  it('kernels, convolution and the linear recurrence are differentiable', () => {
    ok((a) => ssmKernel({ a, b: B }, C, 4), [discretise(A, B, 0.1).a])
    ok((u, k) => causalConvolution(u, k, 0.5), [n(2, [2, 5]), n(3, [5])])
    ok((a, b) => linearRecurrence(a, b), [uniform(key(4), [4, 2]), n(5, [4, 2])])
    expect(toArray(linearRecurrence([[0.5]], [[1], [1]]))).toEqual([[1], [1.5]])
  })
})

describe('selective scan', () => {
  it('equals a per-channel diagonal recurrence with input-dependent steps', () => {
    const x = n(6, [1, 4, 2])
    const dt = uniform(key(7), [1, 4, 2], { low: 0.1, high: 1 })
    const a = tensor([
      [-1, -0.5],
      [-2, -0.2],
    ])
    const b = n(8, [1, 4, 2])
    const c = n(9, [1, 4, 2])
    const { y } = selectiveScan({ x, dt, a, b, c, d: [0.3, -0.7] })
    const X = toArray(x) as number[][][]
    const DT = toArray(dt) as number[][][]
    const Bv = toArray(b) as number[][][]
    const Cv = toArray(c) as number[][][]
    const Av = [
      [-1, -0.5],
      [-2, -0.2],
    ]
    const want: number[] = []
    const h = [
      [0, 0],
      [0, 0],
    ]
    for (let t = 0; t < 4; t++) {
      for (let d = 0; d < 2; d++) {
        let yd = 0
        for (let s = 0; s < 2; s++) {
          const dtv = DT[0]?.[t]?.[d] ?? 0
          const row = h[d] as number[]
          row[s] =
            Math.exp(dtv * (Av[d]?.[s] ?? 0)) * (row[s] ?? 0) + dtv * (Bv[0]?.[t]?.[s] ?? 0) * (X[0]?.[t]?.[d] ?? 0)
          yd += (Cv[0]?.[t]?.[s] ?? 0) * (row[s] ?? 0)
        }
        want.push(yd + ([0.3, -0.7][d] ?? 0) * (X[0]?.[t]?.[d] ?? 0))
      }
    }
    close(y, want, 12)
  })

  it('is differentiable in every input and gates with silu', () => {
    const a = tensor([[-1, -0.5]])
    ok(
      (x, dt, b, c) => selectiveScan({ x, dt, a, b, c }).y,
      [n(10, [1, 3, 1]), uniform(key(11), [1, 3, 1], { low: 0.2, high: 0.8 }), n(12, [1, 3, 2]), n(13, [1, 3, 2])],
    )
    ok(
      (z) =>
        selectiveScan({ x: n(14, [1, 2, 1]), dt: [[[0.5], [0.5]]], a, b: n(15, [1, 2, 2]), c: n(16, [1, 2, 2]), z }).y,
      [n(17, [1, 2, 1])],
    )
  })
})

describe('delta rule and LRU', () => {
  it('the delta rule matches a naive fast-weight loop', () => {
    const q = n(18, [3, 2])
    const k = n(19, [3, 2])
    const v = n(20, [3, 2])
    const beta = uniform(key(21), [3])
    const S = [
      [0, 0],
      [0, 0],
    ]
    const Q = toArray(q) as number[][]
    const K = toArray(k) as number[][]
    const V = toArray(v) as number[][]
    const bb = toFlat(beta)
    const want: number[] = []
    for (let t = 0; t < 3; t++) {
      const kt = K[t] as number[]
      const sk = S.map((row) => (row[0] ?? 0) * (kt[0] ?? 0) + (row[1] ?? 0) * (kt[1] ?? 0))
      for (let i = 0; i < 2; i++)
        for (let j = 0; j < 2; j++)
          (S[i] as number[])[j] = (S[i]?.[j] ?? 0) + (bb[t] ?? 0) * ((V[t]?.[i] ?? 0) - (sk[i] ?? 0)) * (kt[j] ?? 0)
      const qt = Q[t] as number[]
      for (let i = 0; i < 2; i++) want.push((S[i]?.[0] ?? 0) * (qt[0] ?? 0) + (S[i]?.[1] ?? 0) * (qt[1] ?? 0))
    }
    close(deltaRule(q, k, v, beta).output, want, 12)
    ok((qq, kk, vv) => deltaRule(qq, kk, vv, beta).output, [q, k, v])
    expect(
      deltaRule(n(22, [2, 3, 2]), n(23, [2, 3, 2]), n(24, [2, 3, 4]), uniform(key(25), [2, 3])).state.shape,
    ).toEqual([2, 4, 2])
  })

  it('LRU eigenvalues lie inside the unit circle and the unit is differentiable', () => {
    const p = lruInit(key(26), 6, 3, { rMin: 0.4, rMax: 0.99 })
    const { re, im } = lruEigenvalues(p)
    const mod = toFlat(re).map((r, i) => Math.hypot(r, toFlat(im)[i] ?? 0))
    expect(mod.every((m) => m >= 0.4 - 1e-9 && m < 0.99 + 1e-9)).toBe(true)
    const u = n(27, [4, 3])
    expect(lru(p, u).shape).toEqual([4, 3])
    expect(lru(p, n(28, [2, 4, 3])).shape).toEqual([2, 4, 3])
    ok((uu, b, c, nu) => lru({ ...p, bRe: b, cRe: c, nuLog: nu }, uu), [u, p.bRe, p.cRe, p.nuLog])
  })

  it('an LRU with zero phase is a real diagonal SSM', () => {
    const p = lruInit(key(29), 1, 1)
    const real = { ...p, thetaLog: tensor([-80]), bIm: tensor([[0]]), cIm: tensor([[0]]), d: tensor([0]) }
    const u = n(30, [5, 1])
    const lam = Math.exp(-Math.exp(toFlat(p.nuLog)[0] ?? 0))
    const g = Math.exp(toFlat(p.gammaLog)[0] ?? 0)
    const b = (toFlat(p.bRe)[0] ?? 0) * g
    const c = toFlat(p.cRe)[0] ?? 0
    close(lru(real, u), ssmRecurrence({ a: [lam], b: [b] }, [c], toFlat(u)), 12)
  })
})
