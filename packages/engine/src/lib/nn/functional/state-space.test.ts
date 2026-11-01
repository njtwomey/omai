import { describe, expect, it } from 'vitest'
import { checkPrimitive } from '../../autodiff/index.ts'
import { maxOf } from '../../base/index.ts'
import { key, normal, uniform } from '../../random/index.ts'
import { concat, mul, ones, sliceAxis, tensor, toArray, toFlat, type Value } from '../../tensor/index.ts'
import {
  causalConvolution,
  deltaRule,
  discretise,
  discretiseDiagonal,
  linearRecurrence,
  lru,
  lruEigenvalues,
  lruInit,
  lruScan,
  parallelScan,
  parallelScanComplex,
  selectiveScan,
  ssd,
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

describe('parallel scans and resumed scans', () => {
  it('the parallel scan equals the sequential recurrence, for any length and with an initial state', () => {
    for (const len of [1, 2, 5, 8, 13]) {
      const a = uniform(key(len), [len, 3], { low: -0.9, high: 0.9 })
      const b = n(len + 100, [len, 3])
      close(parallelScan(a, b), linearRecurrence(a, b), 12)
      const h0 = n(7, [3])
      close(parallelScan(a, b, 0, h0), linearRecurrence(a, b, h0), 12)
    }
  })

  it('the complex parallel scan equals a sequential complex recurrence on the time axis 1', () => {
    const [len, d] = [9, 2]
    const ar = uniform(key(1), [2, len, d], { low: -0.6, high: 0.6 })
    const ai = uniform(key(2), [2, len, d], { low: -0.6, high: 0.6 })
    const br = n(3, [2, len, d])
    const bi = n(4, [2, len, d])
    const got = parallelScanComplex({ re: ar, im: ai }, { re: br, im: bi }, 1)
    const A = [toFlat(ar), toFlat(ai)] as const
    const Bv = [toFlat(br), toFlat(bi)] as const
    const hr = new Float64Array(2 * d)
    const hi = new Float64Array(2 * d)
    for (let t = 0; t < len; t++)
      for (let bt = 0; bt < 2; bt++)
        for (let j = 0; j < d; j++) {
          const i = (bt * len + t) * d + j
          const pr = hr[bt * d + j] ?? 0
          const pi = hi[bt * d + j] ?? 0
          const nr = (A[0][i] ?? 0) * pr - (A[1][i] ?? 0) * pi + (Bv[0][i] ?? 0)
          const ni = (A[0][i] ?? 0) * pi + (A[1][i] ?? 0) * pr + (Bv[1][i] ?? 0)
          hr[bt * d + j] = nr
          hi[bt * d + j] = ni
          expect(toFlat(got.re)[i]).toBeCloseTo(nr, 12)
          expect(toFlat(got.im)[i]).toBeCloseTo(ni, 12)
        }
    const h0 = { re: n(5, [2, d]), im: n(6, [2, d]) }
    const resumed = parallelScanComplex({ re: ar, im: ai }, { re: br, im: bi }, 1, h0)
    expect(toFlat(resumed.re)[0]).toBeCloseTo(
      (toFlat(ar)[0] ?? 0) * (toFlat(h0.re)[0] ?? 0) -
        (toFlat(ai)[0] ?? 0) * (toFlat(h0.im)[0] ?? 0) +
        (toFlat(br)[0] ?? 0),
      12,
    )
  })

  it('a selective scan or an LRU resumed from its state equals one pass over the whole sequence', () => {
    const x = n(10, [2, 6, 3])
    const dt = uniform(key(11), [2, 6, 3], { low: 0.1, high: 0.5 })
    const a = uniform(key(12), [3, 2], { low: -2, high: -0.5 })
    const b = n(13, [2, 6, 2])
    const c = n(14, [2, 6, 2])
    const whole = selectiveScan({ x, dt, a, b, c })
    const cut = (v: Value, lo: number, hi: number): Value => sliceAxis(v, 1, lo, hi)
    const first = selectiveScan({ x: cut(x, 0, 4), dt: cut(dt, 0, 4), a, b: cut(b, 0, 4), c: cut(c, 0, 4) })
    const rest = selectiveScan({
      x: cut(x, 4, 6),
      dt: cut(dt, 4, 6),
      a,
      b: cut(b, 4, 6),
      c: cut(c, 4, 6),
      h0: first.state,
    })
    close(concat([first.y, rest.y], 1), whole.y, 12)
    close(rest.state, whole.state, 12)

    const params = lruInit(key(15), 4, 3)
    const u = n(16, [2, 7, 3])
    const all = lruScan(params, u)
    const head = lruScan(params, sliceAxis(u, 1, 0, 3))
    const tail = lruScan(params, sliceAxis(u, 1, 3, 7), head.state)
    close(concat([head.y, tail.y], 1), all.y, 12)
    close(tail.state.re, all.state.re, 12)
  })
})

describe('structured state-space duality', () => {
  const [bs, len, heads, width, dim] = [2, 7, 3, 2, 4]
  const input = {
    x: n(41, [bs, len, heads, width]),
    dt: uniform(key(42), [bs, len, heads], { low: 0.05, high: 0.6 }),
    a: tensor([-0.5, -1.5, -0.1]),
    b: n(43, [bs, len, dim]),
    c: n(44, [bs, len, dim]),
  }

  it('the recurrent, quadratic (attention-like) and chunked forms agree to 1e-10', () => {
    const rec = ssd(input, 'recurrent')
    const quad = ssd(input, 'quadratic')
    close(quad.y, rec.y, 10)
    close(quad.state, rec.state, 10)
    for (const chunk of [1, 3, 7, 16]) {
      const ch = ssd(input, 'chunked', chunk)
      close(ch.y, rec.y, 10)
      close(ch.state, rec.state, 10)
    }
  })

  it('agree from a non-zero initial state, and a split sequence resumes exactly', () => {
    const h0 = n(45, [bs, heads, width, dim])
    const rec = ssd({ ...input, h0 }, 'recurrent')
    close(ssd({ ...input, h0 }, 'quadratic').y, rec.y, 10)
    close(ssd({ ...input, h0 }, 'chunked', 3).state, rec.state, 10)
    const cut = (v: Value, lo: number, hi: number): Value => sliceAxis(v, 1, lo, hi)
    const part = (lo: number, hi: number, init?: Value): ReturnType<typeof ssd> =>
      ssd(
        {
          x: cut(input.x, lo, hi),
          dt: cut(input.dt, lo, hi),
          a: input.a,
          b: cut(input.b, lo, hi),
          c: cut(input.c, lo, hi),
          ...(init === undefined ? {} : { h0: init }),
        },
        'quadratic',
      )
    const head = part(0, 4)
    const tail = part(4, 7, head.state)
    close(concat([head.y, tail.y], 1), ssd(input, 'recurrent').y, 10)
  })

  it('limits: no decay (a = 0) is a causal linear-attention sum, large steps are memoryless', () => {
    const flat = ssd({ ...input, a: tensor([0, 0, 0]) }, 'quadratic').y
    const xs = toFlat(input.x)
    const dts = toFlat(input.dt)
    const bb = toFlat(input.b)
    const cc = toFlat(input.c)
    // y[b, t, h, p] = sum_{s <= t} (C_t . B_s) dt_s x_s.
    const [b0, t0, h0i, p0] = [1, 5, 2, 1]
    let want = 0
    for (let s = 0; s <= t0; s++) {
      let dot = 0
      for (let j = 0; j < dim; j++) dot += (cc[(b0 * len + t0) * dim + j] ?? 0) * (bb[(b0 * len + s) * dim + j] ?? 0)
      want += dot * (dts[(b0 * len + s) * heads + h0i] ?? 0) * (xs[((b0 * len + s) * heads + h0i) * width + p0] ?? 0)
    }
    expect(toFlat(flat)[((b0 * len + t0) * heads + h0i) * width + p0]).toBeCloseTo(want, 10)
    // A very negative a forgets at once: y_t = dt_t (C_t . B_t) x_t.
    const memoryless = toFlat(ssd({ ...input, a: tensor([-1e4, -1e4, -1e4]) }, 'quadratic').y)
    let dot = 0
    for (let j = 0; j < dim; j++) dot += (cc[(b0 * len + t0) * dim + j] ?? 0) * (bb[(b0 * len + t0) * dim + j] ?? 0)
    expect(memoryless[((b0 * len + t0) * heads + h0i) * width + p0]).toBeCloseTo(
      dot * (dts[(b0 * len + t0) * heads + h0i] ?? 0) * (xs[((b0 * len + t0) * heads + h0i) * width + p0] ?? 0),
      10,
    )
  })

  it('passes the gradient check in the quadratic and recurrent forms', () => {
    const small = {
      x: n(46, [1, 4, 2, 2]),
      dt: uniform(key(47), [1, 4, 2], { low: 0.1, high: 0.5 }),
      b: n(48, [1, 4, 3]),
      c: n(49, [1, 4, 3]),
    }
    for (const form of ['quadratic', 'recurrent', 'chunked'] as const)
      ok(
        (x, dt, a, b, c) => ssd({ x, dt, a, b, c }, form, 2).y,
        [small.x, small.dt, tensor([-0.4, -1]), small.b, small.c],
      )
  })
})

describe('limiting cases of the selective scan', () => {
  const [bs, len, dim, st] = [1, 6, 2, 3]
  const x = n(61, [bs, len, dim])
  const dt = uniform(key(62), [bs, len, dim], { low: 0.1, high: 0.5 })
  const b = n(63, [bs, len, st])
  const c = n(64, [bs, len, st])

  it('A = 0 is a causal linear map: y_t = sum_{s <= t} dt_s (C_t . B_s) x_s', () => {
    const y = toFlat(selectiveScan({ x, dt, a: zerosA(dim, st), b, c }).y)
    const [xs, ds, bb, cc] = [toFlat(x), toFlat(dt), toFlat(b), toFlat(c)]
    for (let t = 0; t < len; t++)
      for (let d = 0; d < dim; d++) {
        let want = 0
        for (let s = 0; s <= t; s++) {
          let dot = 0
          for (let j = 0; j < st; j++) dot += (cc[t * st + j] ?? 0) * (bb[s * st + j] ?? 0)
          want += (ds[s * dim + d] ?? 0) * dot * (xs[s * dim + d] ?? 0)
        }
        expect(y[t * dim + d]).toBeCloseTo(want, 12)
      }
  })

  it('a huge step is memoryless: y_t = dt_t (C_t . B_t) x_t', () => {
    const big = mul(dt, 1e4)
    const y = toFlat(selectiveScan({ x, dt: big, a: mul(ones([dim, st]), -1), b, c }).y)
    const [xs, ds, bb, cc] = [toFlat(x), toFlat(big), toFlat(b), toFlat(c)]
    for (let t = 0; t < len; t++)
      for (let d = 0; d < dim; d++) {
        let dot = 0
        for (let j = 0; j < st; j++) dot += (cc[t * st + j] ?? 0) * (bb[t * st + j] ?? 0)
        expect(y[t * dim + d]).toBeCloseTo((ds[t * dim + d] ?? 0) * dot * (xs[t * dim + d] ?? 0), 8)
      }
  })
})

function zerosA(d: number, s: number): Value {
  return mul(ones([d, s]), 0)
}
