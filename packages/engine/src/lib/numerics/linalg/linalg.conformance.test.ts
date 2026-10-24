import { describe, expect, it } from 'vitest'
import { checkPrimitive } from '../../autodiff/index.ts'
import { add, matmul, mul, square, swapaxes, type Tensor, tensor, tril, type Value } from '../../tensor/index.ts'
import * as L from './index.ts'

// Inputs: a symmetric positive-definite matrix, a general well-conditioned one, rectangular ones.
const spd: Tensor = tensor([
  [4, 1, 0.5],
  [1, 3, 0.2],
  [0.5, 0.2, 2],
])
const general: Tensor = tensor([
  [3, 0.4, -1],
  [0.2, 2.5, 0.3],
  [-0.6, 0.8, 4],
])
const tall: Tensor = tensor([
  [1, 2],
  [0.3, -1],
  [2, 0.5],
  [-0.7, 1.4],
])
const wide: Tensor = swapaxes(tall, 0, 1) as Tensor
const rhs: Tensor = tensor([
  [1, -0.5],
  [0.3, 2],
  [-1, 0.7],
])
const lowerT: Tensor = tril(general) as Tensor

const cases: readonly (readonly [string, (...xs: Value[]) => Value, readonly Tensor[]])[] = [
  ['solveTriangular (lower)', (a, b) => L.solveTriangular(a as Value, b as Value, { lower: true }), [lowerT, rhs]],
  [
    'solveTriangular (upper, transposed, unit)',
    (a, b) => L.solveTriangular(a as Value, b as Value, { transpose: true, unitDiagonal: true }),
    [general, rhs],
  ],
  ['cholesky', (a) => L.cholesky(a as Value), [spd]],
  ['solve', (a, b) => L.solve(a as Value, b as Value), [general, rhs]],
  ['inv', (a) => L.inv(a as Value), [general]],
  [
    'lu (L and U)',
    (a) => {
      const r = L.lu(a as Value)
      return add(r.l, mul(r.u, 2))
    },
    [general],
  ],
  ['slogdet', (a) => L.slogdet(a as Value).logabsdet, [general]],
  ['det', (a) => L.det(a as Value), [general]],
  [
    'qr (tall)',
    (a) => {
      const r = L.qr(a as Value)
      return add(matmul(r.q, r.r), mul(r.q, 2))
    },
    [tall],
  ],
  ['qr R', (a) => L.qr(a as Value).r, [general]],
  ['lstsq (tall)', (a, b) => L.lstsq(a as Value, b as Value), [tall, tensor([1, 0.5, -1, 2])]],
  ['lstsq (wide)', (a, b) => L.lstsq(a as Value, b as Value), [wide, tensor([1, -2])]],
  ['eigh eigenvalues', (a) => L.eigh(a as Value).eigenvalues, [spd]],
  // Eigenvector signs are a convention, so check a sign-invariant function of them.
  ['eigh eigenvectors (squared)', (a) => square(L.eigh(a as Value).eigenvectors), [spd]],
  ['svd singular values', (a) => L.svd(a as Value).s, [tall]],
  ['svd U (squared, tall)', (a) => square(L.svd(a as Value).u), [tall]],
  ['svd Vh (squared, wide)', (a) => square(L.svd(a as Value).vh), [wide]],
  [
    'svd U Vh product (square)',
    (a) => {
      const r = L.svd(a as Value)
      return matmul(r.u, r.vh)
    },
    [general],
  ],
  ['pinv', (a) => L.pinv(a as Value), [tall]],
  ['norm (nuclear)', (a) => L.norm(a as Value, 'nuc'), [tall]],
  ['matrixPower', (a) => L.matrixPower(a as Value, 3), [general]],
]

describe('linear algebra primitives are conformant', () => {
  for (const [name, f, inputs] of cases) {
    it(name, () => {
      const report = checkPrimitive(f, inputs, { rtol: 1e-5, atol: 1e-7 })
      expect(report.failures).toEqual([])
    })
  }

  it('signm (a matrix with eigenvalues on both sides of the imaginary axis)', () => {
    const mixed = tensor([
      [1, 2],
      [0.5, -1.5],
    ])
    const report = checkPrimitive((a) => L.signm(a as Value), [mixed], { rtol: 1e-5, atol: 1e-7, vmap: false })
    expect(report.failures).toEqual([])
  })

  it('reconstruct their inputs', () => {
    const { u, s, vh } = L.svd(tall)
    expect(
      Array.from((add(matmul(mul(u, s), vh), mul(tall, -1)) as Tensor).data).every((x) => Math.abs(x) < 1e-12),
    ).toBe(true)
    const { q, r } = L.qr(tall)
    expect(Array.from((add(matmul(q, r), mul(tall, -1)) as Tensor).data).every((x) => Math.abs(x) < 1e-12)).toBe(true)
  })
})
