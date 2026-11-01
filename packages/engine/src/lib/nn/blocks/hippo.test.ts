import { describe, expect, it } from 'vitest'
import { expm } from '../../numerics/linalg/index.ts'
import { gaussLegendre } from '../../numerics/quadrature/index.ts'
import { toArray, toFlat } from '../../tensor/index.ts'
import { hippoLagT, hippoLegS, hippoLegT, nplrHippo, nplrReconstruct, s4dEigenvalues } from './hippo.ts'

/** Legendre polynomials P_0 .. P_{n-1} at x, by the three-term recurrence. */
function legendre(n: number, x: number): number[] {
  const p = [1, x]
  for (let k = 1; k < n; k++) p.push(((2 * k + 1) * x * (p[k] ?? 0) - k * (p[k - 1] ?? 0)) / (k + 1))
  return p.slice(0, n)
}

/** Laguerre polynomials L_0 .. L_{n-1} at x. */
function laguerre(n: number, x: number): number[] {
  const l = [1, 1 - x]
  for (let k = 1; k < n; k++) l.push(((2 * k + 1 - x) * (l[k] ?? 0) - k * (l[k - 1] ?? 0)) / (k + 1))
  return l.slice(0, n)
}

const rule = gaussLegendre(24)
const nodes = Array.from(rule.nodes.data)
const weights = Array.from(rule.weights.data)

/** The projection onto the orthonormal Legendre basis of the window [lo, hi] (measure 1 / (hi - lo)). */
function project(f: (x: number) => number, lo: number, hi: number, n: number): number[] {
  const out = new Array<number>(n).fill(0)
  nodes.forEach((s, i) => {
    const x = lo + ((s + 1) / 2) * (hi - lo)
    const p = legendre(n, s)
    for (let k = 0; k < n; k++)
      out[k] = (out[k] ?? 0) + ((weights[i] ?? 0) / 2) * f(x) * Math.sqrt(2 * k + 1) * (p[k] ?? 0)
  })
  return out
}

/** Classical RK4 for y' = g(t, y). */
function rk4(g: (t: number, y: number[]) => number[], y0: number[], t0: number, t1: number, steps: number): number[] {
  let y = y0
  const h = (t1 - t0) / steps
  const axpy = (a: number[], s: number, b: number[]): number[] => a.map((v, i) => v + s * (b[i] ?? 0))
  for (let i = 0; i < steps; i++) {
    const t = t0 + i * h
    const k1 = g(t, y)
    const k2 = g(t + h / 2, axpy(y, h / 2, k1))
    const k3 = g(t + h / 2, axpy(y, h / 2, k2))
    const k4 = g(t + h, axpy(y, h, k3))
    y = y.map((v, j) => v + (h / 6) * ((k1[j] ?? 0) + 2 * (k2[j] ?? 0) + 2 * (k3[j] ?? 0) + (k4[j] ?? 0)))
  }
  return y
}

const scaled = (m: number[][], s: number): number[][] => m.map((row) => row.map((v) => v * s))
const poly = (x: number): number => 1 + 0.5 * x - 0.3 * x * x + 0.1 * x ** 3
const mv = (a: number[][], x: number[]): number[] => a.map((row) => row.reduce((s, v, j) => s + v * (x[j] ?? 0), 0))

describe('the HiPPO matrices', () => {
  it('LegS has the closed form of the paper', () => {
    const { a, b } = hippoLegS(4)
    const A = toArray(a) as number[][]
    for (let i = 0; i < 4; i++)
      for (let k = 0; k < 4; k++) {
        const want = i > k ? -Math.sqrt(2 * i + 1) * Math.sqrt(2 * k + 1) : i === k ? -(i + 1) : 0
        expect(A[i]?.[k]).toBeCloseTo(want, 14)
      }
    expect(toFlat(b)).toEqual([1, Math.sqrt(3), Math.sqrt(5), Math.sqrt(7)])
  })

  it('LegS reconstructs a polynomial history: the time-varying ODE tracks the Legendre projection', () => {
    const n = 6
    const { a, b } = hippoLegS(n)
    const A = toArray(a) as number[][]
    const B = toFlat(b)
    const c1 = rk4(
      (t, c) => mv(A, c).map((v, i) => (v + (B[i] ?? 0) * poly(t)) / t),
      project(poly, 0, 1, n),
      1,
      2.5,
      2000,
    )
    const want = project(poly, 0, 2.5, n)
    for (let k = 0; k < n; k++) expect(c1[k]).toBeCloseTo(want[k] ?? Number.NaN, 8)
    // The reconstruction sum_n c_n g_n(x) is the polynomial itself on [0, t].
    const x = 1.3
    const g = legendre(n, (2 * x) / 2.5 - 1)
    const rec = c1.reduce((s, ck, k) => s + ck * Math.sqrt(2 * k + 1) * (g[k] ?? 0), 0)
    expect(rec).toBeCloseTo(poly(x), 8)
  })

  it('LegT reconstructs the last theta time units of a polynomial', () => {
    const n = 6
    const theta = 2
    const { a, b } = hippoLegT(n, { theta })
    const A = toArray(a) as number[][]
    const B = toFlat(b)
    const c1 = rk4(
      (t, c) => mv(A, c).map((v, i) => v + (B[i] ?? 0) * poly(t)),
      project(poly, -theta, 0, n),
      0,
      1.7,
      4000,
    )
    const want = project(poly, 1.7 - theta, 1.7, n)
    for (let k = 0; k < n; k++) expect(c1[k]).toBeCloseTo(want[k] ?? Number.NaN, 8)
    // A signal older than the window is forgotten by the basis: the state is the window's projection only.
    const g = legendre(n, (2 * (0.2 - 1.7)) / theta + 1)
    const rec = c1.reduce((s, ck, k) => s + ck * Math.sqrt(2 * k + 1) * (g[k] ?? 0), 0)
    expect(rec).toBeCloseTo(poly(0.2), 8)
  })

  it('LagT has the Laguerre functions as its impulse response', () => {
    const n = 5
    const { a, b } = hippoLagT(n)
    for (const s of [0, 0.4, 1.7, 3]) {
      const e = toArray(expm(scaled(toArray(a) as number[][], s))) as number[][]
      const got = mv(e, toFlat(b))
      const lag = laguerre(n, s)
      for (let k = 0; k < n; k++) expect(got[k]).toBeCloseTo((lag[k] ?? 0) * Math.exp(-s / 2), 10)
    }
  })
})

describe('the normal-plus-low-rank forms', () => {
  for (const [measure, rank, real] of [
    ['legs', 1, -0.5],
    ['legt', 2, 0],
    ['lagt', 1, 0],
  ] as const) {
    it(`${measure}: A = V Lambda V^H - P P^T with V unitary`, () => {
      const n = 8
      const f = nplrHippo(measure, n, measure === 'legt' ? { theta: 1.5 } : {})
      const dense = measure === 'legs' ? hippoLegS(n) : measure === 'legt' ? hippoLegT(n, { theta: 1.5 }) : hippoLagT(n)
      const back = toFlat(nplrReconstruct(f))
      const want = toFlat(dense.a)
      for (const [i, v] of back.entries()) expect(v).toBeCloseTo(want[i] ?? Number.NaN, 9)
      expect(f.p.shape).toEqual([n, rank])
      for (const v of toFlat(f.lambda.re)) expect(v).toBeCloseTo(real, 12)
      const im = toFlat(f.lambda.im)
      expect([...im].sort((x, y) => x - y)).toEqual(im)
      // V^H V = I.
      const vr = toFlat(f.v.re)
      const vi = toFlat(f.v.im)
      for (let i = 0; i < n; i++)
        for (let j = 0; j < n; j++) {
          let re = 0
          let imag = 0
          for (let k = 0; k < n; k++) {
            re += (vr[k * n + i] ?? 0) * (vr[k * n + j] ?? 0) + (vi[k * n + i] ?? 0) * (vi[k * n + j] ?? 0)
            imag += (vr[k * n + i] ?? 0) * (vi[k * n + j] ?? 0) - (vi[k * n + i] ?? 0) * (vr[k * n + j] ?? 0)
          }
          expect(re).toBeCloseTo(i === j ? 1 : 0, 9)
          expect(imag).toBeCloseTo(0, 9)
        }
      // V (V^H b) = b.
      const bt = { re: toFlat(f.bTilde.re), im: toFlat(f.bTilde.im) }
      const b = toFlat(dense.b)
      for (let i = 0; i < n; i++) {
        let s = 0
        for (let k = 0; k < n; k++) s += (vr[i * n + k] ?? 0) * (bt.re[k] ?? 0) - (vi[i * n + k] ?? 0) * (bt.im[k] ?? 0)
        expect(s).toBeCloseTo(b[i] ?? Number.NaN, 9)
      }
    })
  }
})

describe('the diagonal forms', () => {
  it('S4D-Lin, -Inv and -Real are the closed forms of the paper', () => {
    const lin = s4dEigenvalues('lin', 4)
    expect(toFlat(lin.re)).toEqual([-0.5, -0.5, -0.5, -0.5])
    expect(toFlat(lin.im)).toEqual([0, Math.PI, 2 * Math.PI, 3 * Math.PI])
    const inv = toFlat(s4dEigenvalues('inv', 4).im)
    expect(inv[0]).toBeCloseTo((8 / Math.PI) * 7, 12)
    expect(inv[3]).toBeCloseTo((8 / Math.PI) * (8 / 7 - 1), 12)
    expect(toFlat(s4dEigenvalues('real', 3).re)).toEqual([-1, -2, -3])
    expect(toFlat(s4dEigenvalues('real', 3).im)).toEqual([0, 0, 0])
  })
})
