import { describe, expect, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../../testing/golden.ts'
import { checkPrimitive, grad } from '../../autodiff/index.ts'
import { item, sum, type Tensor, type TensorLike, tensor, toFlat, type Value } from '../../tensor/index.ts'
import { runAlgorithm } from '../../trace/index.ts'
import { solve } from '../linalg/index.ts'
import * as S from './index.ts'

const SOLVERS = { cg: S.cg, bicgstab: S.bicgstab, gmres: S.gmres, minres: S.minres } as const

describe('sparse matrices and solvers against scipy.sparse', () => {
  for (const c of loadFixture('numerics/sparse/sparse').cases) {
    it(c.name, () => {
      const t = c.inputs.map((v) => decode(v) as Tensor)
      const coo = (): S.CsrMatrix => S.fromCoo(toFlat(t[0] as Tensor), toFlat(t[1] as Tensor), t[2] as Tensor, [3, 4])
      const p = c.params
      const actual: Value =
        c.fn === 'fromCooDense'
          ? S.toDense(coo())
          : c.fn === 'fromCooData'
            ? coo().values
            : c.fn === 'spmv'
              ? S.spmv(coo(), t[3] as Tensor)
              : c.fn === 'spmm'
                ? S.spmm(coo(), t[3] as Tensor)
                : c.fn === 'transpose'
                  ? S.toDense(S.transposeCsr(coo()))
                  : c.fn === 'solve'
                    ? (() => {
                        const r = SOLVERS[p['method'] as keyof typeof SOLVERS](S.csr(t[0] as Tensor), t[1] as Tensor, {
                          rtol: 1e-12,
                          maxIterations: 2000,
                        })
                        expect(r.converged).toBe(true)
                        return r.x
                      })()
                    : c.fn === 'eigsh'
                      ? S.eigsh(S.csr(t[0] as Tensor), 3, { which: p['which'] as S.Which }).values
                      : S.svds(S.csr(t[0] as Tensor), 3).s
      expectMatches(actual, c.output, c.rtol, c.atol, c.name)
    })
  }
})

describe('sparse properties', () => {
  const tri = (n: number): S.CsrMatrix =>
    S.csrAdd(
      S.csrAdd(S.diagCsr(new Array<number>(n).fill(4)), S.diagCsr(new Array<number>(n - 1).fill(-1), 1, n)),
      S.diagCsr(new Array<number>(n - 1).fill(-1), -1, n),
    )

  it('products are differentiable in the values and the vector', () => {
    const a = tri(5)
    expect(
      checkPrimitive((v, x) => S.spmv({ ...a, values: v }, x), [toFlat(a.values), [1, -2, 0.5, 3, 1]]).failures,
    ).toEqual([])
    expect(checkPrimitive((v) => S.toDense(S.transposeCsr({ ...a, values: v })), [toFlat(a.values)]).failures).toEqual(
      [],
    )
  })

  it('preconditioning cuts conjugate-gradient iterations, and IC(0) is exact for tridiagonal matrices', () => {
    const n = 60
    const scaled = S.csr({
      indptr: tri(n).indptr,
      indices: tri(n).indices,
      values: toFlat(tri(n).values).map((v, k) => v * (1 + (tri(n).rows[k] ?? 0) / 4)),
      shape: [n, n],
    })
    const b = Array.from({ length: n }, (_, i) => Math.sin(i))
    const plain = S.cg(scaled, b, { rtol: 1e-10 })
    const ic = S.cg(scaled, b, { rtol: 1e-10, preconditioner: S.incompleteCholesky(tri(n)) })
    expect(plain.converged && ic.converged).toBe(true)
    const exact = S.cg(tri(n), b, { rtol: 1e-12, preconditioner: S.incompleteCholesky(tri(n)) })
    expect(exact.iterations).toBeLessThanOrEqual(2)
    const jac = S.cg(scaled, b, { rtol: 1e-10, preconditioner: S.jacobiPreconditioner(scaled) })
    expect(jac.iterations).toBeLessThan(plain.iterations)
    // MINRES residuals never increase.
    const mr = S.minres(tri(n), b, { rtol: 1e-10 })
    for (let i = 1; i < mr.history.length; i++)
      expect(mr.history[i] ?? 0).toBeLessThanOrEqual((mr.history[i - 1] ?? 0) * (1 + 1e-12))
  })

  it('sparseSolve differentiates by adjoint solves', () => {
    const a = tri(6)
    const b = [1, 2, 0, -1, 3, 0.5]
    const loss = (v: Value, rhs: Value): Value => sum(S.sparseSolve({ ...a, values: v }, rhs))
    const report = checkPrimitive((v, rhs) => S.sparseSolve({ ...a, values: v }, rhs), [toFlat(a.values), b], {
      rtol: 1e-4,
      atol: 1e-6,
    })
    expect(report.failures).toEqual([])
    // d/db sum(A^{-1} b) = A^{-T} 1.
    const gb = toFlat(grad((rhs: Value) => loss(a.values, rhs))(tensor(b)) as Value)
    const expected = toFlat(solve(S.toDense(a), new Array<number>(6).fill(1)))
    for (const [i, v] of gb.entries()) expect(v).toBeCloseTo(expected[i] ?? 0, 9)
    expect(item(loss(a.values, tensor(b)))).toBeCloseTo(
      toFlat(solve(S.toDense(a), b)).reduce((s, v) => s + v, 0),
      9,
    )
  })

  it('Lanczos and Arnoldi produce orthonormal bases', () => {
    const a = tri(12)
    const l = S.lanczos(a, 8)
    const [rows = 0, cols = 0] = l.basis.shape
    const q = toFlat(l.basis)
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < rows; j++) {
        let d = 0
        for (let k = 0; k < cols; k++) d += (q[i * cols + k] ?? 0) * (q[j * cols + k] ?? 0)
        expect(d).toBeCloseTo(i === j ? 1 : 0, 12)
      }
    }
    expect(S.arnoldi(a, 5).basis.shape[0]).toBe(6)
  })

  it('the start vectors, the deflation matrix and the multigrid iterate take tensor-likes (migration, D-009)', () => {
    const a = tri(12)
    const v0 = Array.from({ length: 12 }, (_, i) => 1 + (i % 3))
    const calls: [string, (v: TensorLike) => number[]][] = [
      ['lanczos', (v) => toFlat(S.lanczos(a, 5, { v0: v }).alpha)],
      ['arnoldi', (v) => toFlat(S.arnoldi(a, 4, { v0: v }).hessenberg)],
      [
        'arnoldiAlgorithm',
        (v) => [...(runAlgorithm(S.arnoldiAlgorithm(a, 3, { v0: v }), { maxSteps: 3 }).state.basis[0] ?? [])],
      ],
      ['multigridCycle', (v) => toFlat(S.multigridCycle(S.poisson1d(7), Array(7).fill(1), v, { shape: [7] }).post)],
    ]
    for (const [name, f] of calls) {
      const v = name === 'multigridCycle' ? v0.slice(0, 7) : v0
      const expected = f(v)
      expect(f(Float64Array.from(v)), name).toEqual(expected)
      expect(f(tensor(v)), name).toEqual(expected)
      expect(() => f(tensor([v, v])), name).toThrow(expect.objectContaining({ code: 'shape' }))
    }
    // The deflation vectors: the rows of a matrix, or a list of Float64Arrays.
    const e0 = Float64Array.from({ length: 12 }, (_, i) => (i === 0 ? 1 : 0))
    expect(toFlat(S.lanczos(a, 4, { v0, deflate: [e0] }).alpha)).toEqual(
      toFlat(S.lanczos(a, 4, { v0, deflate: [Array.from(e0)] }).alpha),
    )
    expect(() => S.lanczos(a, 4, { v0, deflate: tensor([1, 2]) })).toThrow(expect.objectContaining({ code: 'shape' }))
  })
})
