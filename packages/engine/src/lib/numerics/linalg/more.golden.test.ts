import { describe, expect, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../../testing/golden.ts'
import { checkPrimitive } from '../../autodiff/index.ts'
import { key } from '../../random/index.ts'
import {
  item,
  matmul,
  reshape,
  sliceAxis,
  swapaxes,
  type Tensor,
  tensor,
  toFlat,
  type Value,
} from '../../tensor/index.ts'
import { complexOps as C } from '../differentiate/index.ts'
import * as L from './index.ts'

type Params = Readonly<Record<string, unknown>>
type Fn = (xs: Tensor[], p: Params) => Value | number

const projector = (b: Value): Value => matmul(b, swapaxes(b, 0, 1))

function sortedEigenvalues(a: Tensor): Tensor {
  const { re, im } = L.eigvals(a)
  const pairs = Array.from(re.data, (r, i) => [r, im.data[i] ?? 0] as const).sort((x, y) => x[0] - y[0] || x[1] - y[1])
  return tensor([pairs.map((p) => p[0]), pairs.map((p) => p[1])])
}

const FNS: Readonly<Record<string, Fn>> = {
  toeplitz: ([c, r]) => L.toeplitz(c as Tensor, r),
  circulant: ([c]) => L.circulant(c as Tensor),
  hankel: ([c, r]) => L.hankel(c as Tensor, r),
  blockDiag: (xs) => L.blockDiag(...xs),
  solveTridiagonal: ([l, d, u, b]) => L.solveTridiagonal(l as Tensor, d as Tensor, u as Tensor, b as Tensor),
  solveBanded: ([ab, b], p) => L.solveBanded([p['l'] as number, p['u'] as number], ab as Tensor, b as Tensor),
  solveToeplitz: ([c, b]) => L.solveToeplitz(c as Tensor, undefined, b as Tensor),
  kroneckerSolve: ([a, b, c]) => L.kroneckerSolve(a as Tensor, b as Tensor, c as Tensor),
  eigvalsSorted: ([a]) => sortedEigenvalues(a as Tensor),
  luFactor: ([a], p) =>
    p['part'] === 'lu' ? L.luFactor(a as Tensor).lu : tensor(Array.from(L.luFactor(a as Tensor).piv.data)),
  luSolve: ([a, b], p) => L.luSolve(L.luFactor(a as Tensor), b as Tensor, { transpose: p['transpose'] === true }),
  choleskyUpdate: ([l, x]) => L.choleskyUpdate(l as Tensor, x as Tensor),
  choleskyDowndate: ([l, x]) => L.choleskyDowndate(l as Tensor, x as Tensor),
  polar: ([a], p) => L.polar(a as Tensor, (p['side'] as 'left' | undefined) ?? 'right')[p['part'] as 'u' | 'p'],
  nullSpaceProjector: ([a]) => projector(L.nullSpace(a as Tensor)),
  orthProjector: ([a]) => projector(L.orth(a as Tensor)),
  expm: ([a]) => L.expm(a as Tensor),
  expmFrechet: ([a, e]) => L.expmFrechet(a as Tensor, e as Tensor).frechet,
  sqrtm: ([a]) => L.sqrtm(a as Tensor),
  logm: ([a]) => L.logm(a as Tensor),
  solveSylvester: ([a, b, q]) => L.solveSylvester(a as Tensor, b as Tensor, q as Tensor),
  solveContinuousLyapunov: ([a, q]) => L.solveContinuousLyapunov(a as Tensor, q as Tensor),
  funm: ([a], p) =>
    L.funm(a as Tensor, { exp: C.exp, sin: C.sin, cos: C.cos, sqrt: C.sqrt }[p['fn_name'] as 'exp']).value as Tensor,
  solveDiscreteLyapunov: ([a, q]) => L.solveDiscreteLyapunov(a as Tensor, q as Tensor),
  solveCare: ([a, b, q, r]) => L.solveCare(a as Tensor, b as Tensor, q as Tensor, r as Tensor),
  solveDare: ([a, b, q, r]) => L.solveDare(a as Tensor, b as Tensor, q as Tensor, r as Tensor),
  signm: ([a]) => L.signm(a as Tensor),
  johnsonLindenstraussDim: (_xs, p) => L.johnsonLindenstraussDim(p['n'] as number, p['eps'] as number),
}

for (const file of ['structured', 'decompositions', 'functions']) {
  describe(`linalg/${file} against numpy and scipy`, () => {
    for (const c of loadFixture(`numerics/linalg/${file}`).cases) {
      it(c.name, () => {
        const f = FNS[c.fn]
        if (f === undefined) throw new Error(`no function ${c.fn}`)
        expectMatches(
          f(
            c.inputs.map((x) => decode(x) as Tensor),
            c.params,
          ),
          c.output,
          c.rtol,
          c.atol,
          c.name,
        )
      })
    }
  })
}

describe('eigen solvers', () => {
  it('eigenpairs satisfy A v = lambda v, including complex pairs', () => {
    const a = tensor([
      [0, -2, 1],
      [2, 0, 0.5],
      [0, 0, 3],
    ])
    const { values, vectors } = L.eig(a)
    const av = (re: boolean) => matmul(a, re ? vectors.re : vectors.im)
    const avRe = toFlat(av(true))
    const avIm = toFlat(av(false))
    for (let k = 0; k < 3; k++) {
      const lr = values.re.data[k] ?? 0
      const li = values.im.data[k] ?? 0
      for (let i = 0; i < 3; i++) {
        const vr = vectors.re.data[i * 3 + k] ?? 0
        const vi = vectors.im.data[i * 3 + k] ?? 0
        expect(avRe[i * 3 + k] ?? 0).toBeCloseTo(lr * vr - li * vi, 12)
        expect(avIm[i * 3 + k] ?? 0).toBeCloseTo(lr * vi + li * vr, 12)
      }
    }
  })

  it('Schur and Hessenberg forms reconstruct the matrix with orthogonal factors', () => {
    const a = tensor([
      [4, 1, -2, 2],
      [1, 2, 0, 1],
      [-2, 0, 3, -2],
      [2, 1, -2, -1],
    ])
    const { t, z } = L.schur(a)
    expect(
      toFlat(matmul(matmul(z, t), swapaxes(z, 0, 1))).every((v, i) => Math.abs(v - (a.data[i] ?? 0)) < 1e-12),
    ).toBe(true)
    const { h, q } = L.hessenberg(a)
    expect(
      toFlat(matmul(matmul(q, h), swapaxes(q, 0, 1))).every((v, i) => Math.abs(v - (a.data[i] ?? 0)) < 1e-12),
    ).toBe(true)
  })

  it('power, inverse and Rayleigh quotient iterations find their eigenvalues', () => {
    const a = tensor([
      [2, 1, 0],
      [1, 3, 1],
      [0, 1, 4],
    ])
    const exact = toFlat(L.eigvalsh(a))
    expect(L.powerIteration(a).value).toBeCloseTo(exact[2] ?? 0, 9)
    expect(L.inverseIteration(a, 1.2).value).toBeCloseTo(exact[0] ?? 0, 10)
    const rqi = L.rayleighQuotientIteration(a, { x0: [0, 1, 0.1] })
    expect(exact.some((w) => Math.abs(w - rqi.value) < 1e-12)).toBe(true)
    expect(rqi.iterations).toBeLessThan(8)
  })
})

describe('randomised and stochastic methods', () => {
  it('a randomised SVD recovers the leading singular values', () => {
    const a = matmul(
      L.qr(tensor(Array.from({ length: 20 }, (_, i) => Array.from({ length: 8 }, (_, j) => Math.sin(i * 7 + j * 3)))))
        .q,
      tensor([
        [5, 0, 0, 0, 0, 0, 0, 0],
        [0, 3, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0.1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0.01, 0, 0, 0],
        [0, 0, 0, 0, 0, 0.01, 0, 0],
        [0, 0, 0, 0, 0, 0, 0.01, 0],
        [0, 0, 0, 0, 0, 0, 0, 0.01],
      ]),
    )
    const exact = toFlat(L.svdvals(a)).slice(0, 3)
    toFlat(L.randomizedSvd(a, 3, { key: key(1) }).s).forEach((v, i) => {
      expect(v).toBeCloseTo(exact[i] ?? 0, 8)
    })
  })

  it('Hutchinson and stochastic Lanczos estimates are close to the exact trace and log-determinant', () => {
    const m = tensor([
      [5, 1, 0.5],
      [1, 4, 0.2],
      [0.5, 0.2, 3],
    ])
    const tr = L.hutchinsonTrace(m, { samples: 4000, key: key(2) })
    expect(Math.abs(tr.estimate - 12)).toBeLessThan(4 * tr.standardError + 1e-12)
    expect(L.stochasticLogdet(m, { samples: 50, key: key(3) }).estimate).toBeCloseTo(item(L.slogdet(m).logabsdet), 0)
  })

  it('random projections have the expected scale', () => {
    const p = L.randomProjectionMatrix(key(4), 200, 50)
    const meanSquare = toFlat(p).reduce((s, v) => s + v * v, 0) / (200 * 50)
    expect(meanSquare).toBeCloseTo(1 / 200, 3)
    const sparse = toFlat(L.randomProjectionMatrix(key(5), 200, 100, { kind: 'sparse' }))
    expect(sparse.filter((v) => v !== 0).length / sparse.length).toBeCloseTo(0.1, 1)
  })
})

describe('gradients of matrix functions and equations', () => {
  const a = tensor([
    [0.3, 0.2, -0.1],
    [0.1, -0.4, 0.2],
    [0.05, 0.3, 0.1],
  ])
  const spd = tensor([
    [2, 0.3, 0.1],
    [0.3, 1.5, 0.2],
    [0.1, 0.2, 1.2],
  ])
  for (const [name, f, x] of [
    ['expm', (v: Value) => L.expm(v), a],
    [
      'expm (scaled and squared)',
      (v: Value) => L.expm(L.matrixPower(v, 1)),
      tensor([
        [3, 1, 0],
        [0, -2, 1],
        [1, 0, 2],
      ]),
    ],
    ['sqrtm', (v: Value) => L.sqrtm(v), spd],
    ['logm', (v: Value) => L.logm(v), spd],
    [
      'solveContinuousLyapunov',
      (v: Value) => L.solveContinuousLyapunov(v, spd),
      tensor([
        [-2, 0.3, 0],
        [0.1, -1.5, 0.2],
        [0, 0.1, -1],
      ]),
    ],
    ['solveDare', (v: Value) => L.solveDare(v, [[1], [0.5], [0]], spd, [[1]]), a],
    ['toeplitz', (v: Value) => L.toeplitz(reshape(sliceAxis(v, 0, 0, 1), [3])), a],
    ['gramSchmidt', (v: Value) => L.gramSchmidt(v).q, spd],
    ['polar', (v: Value) => L.polar(v).p, spd],
  ] as const) {
    it(`${name} passes the conformance check`, () => {
      expect(checkPrimitive(f, [x], { rtol: 1e-5, atol: 1e-7 }).failures).toEqual([])
    })
  }
})
