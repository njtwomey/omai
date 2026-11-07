import { describe, expect, it } from 'vitest'
import { type Tensor, toFlat } from '../../tensor/index.ts'
import * as S from './index.ts'

// Small systems with known solutions for cg, gmres, minres and bicgstab: the status, the iteration count where theory fixes
// it (a matrix with k distinct eigenvalues takes at most k steps), the reported residual against the true one.
const SOLVERS = { cg: S.cg, minres: S.minres, gmres: S.gmres, bicgstab: S.bicgstab } as const
type Name = keyof typeof SOLVERS
const names = Object.keys(SOLVERS) as Name[]

const tri = (n: number, diagonal: number, lower: number, upper: number): S.CsrMatrix =>
  S.csrAdd(
    S.csrAdd(S.diagCsr(new Array<number>(n).fill(diagonal)), S.diagCsr(new Array<number>(n - 1).fill(upper), 1, n)),
    S.diagCsr(new Array<number>(n - 1).fill(lower), -1, n),
  )
const dense = (a: S.CsrMatrix): number[][] => {
  const flat = toFlat(S.toDense(a))
  const n = a.shape[0]
  return Array.from({ length: n }, (_, i) => flat.slice(i * n, (i + 1) * n))
}
const norm = (v: readonly number[]): number => Math.sqrt(v.reduce((s, x) => s + x * x, 0))
const residual = (a: S.CsrMatrix, x: readonly number[], b: readonly number[]): number =>
  norm(dense(a).map((row, i) => (b[i] ?? 0) - row.reduce((s, v, j) => s + v * (x[j] ?? 0), 0)))

describe('Krylov solvers on small systems with known solutions', () => {
  it('cg, minres, gmres and bicgstab solve the known systems in the iterations theory allows and handle the documented edge cases', () => {
    {
      const failures: string[] = []
      const n = 50
      // The 1-D Poisson matrix tridiag(-1, 2, -1) with b = 1 has the solution u_i = i (n + 1 - i) / 2.
      const poisson = tri(n, 2, -1, -1)
      const ones = new Array<number>(n).fill(1)
      const parabola = Array.from({ length: n }, (_, i) => ((i + 1) * (n - i)) / 2)
      // diag(1, 1, 4, 4, 9) has 3 distinct eigenvalues: every Krylov method stops after at most 3 steps.
      const clustered = S.diagCsr([1, 1, 4, 4, 9])
      // diag(1, -2, 3): indefinite, solved by minres, gmres and bicgstab (cg only because the matrix is diagonal).
      const indefinite = S.diagCsr([1, -2, 3])
      // A non-symmetric diagonally dominant tridiagonal matrix (convection): gmres and bicgstab only.
      const convection = tri(n, 3, -1.5, -0.5)
      const rows: readonly (readonly [
        string,
        S.CsrMatrix,
        readonly number[],
        readonly Name[],
        number | undefined,
        readonly number[] | undefined,
      ])[] = [
        ['Poisson 1-D, n = 50', poisson, ones, names, undefined, parabola],
        ['3 distinct eigenvalues', clustered, [1, 2, 3, 4, 5], names, 3, [1, 2, 0.75, 1, 5 / 9]],
        ['indefinite diagonal', indefinite, [1, 1, 1], names, 4, [1, -0.5, 1 / 3]],
        ['convection', convection, ones, ['gmres', 'bicgstab'], undefined, undefined],
      ]
      for (const [label, a, b, solvers, maxIterations, exact] of rows) {
        for (const name of solvers) {
          const r = SOLVERS[name](a, b, {
            rtol: 1e-10,
            maxIterations: 600,
            ...(name === 'gmres' ? { restart: b.length } : {}),
          })
          const x = toFlat(r.x as Tensor)
          const where = `${name} on ${label}`
          if (r.status !== 'converged' || !r.converged)
            failures.push(`${where}: status ${r.status} after ${r.iterations}`)
          if (maxIterations !== undefined && r.iterations > maxIterations)
            failures.push(`${where}: ${r.iterations} iterations, at most ${maxIterations}`)
          const truth = residual(a, x, b)
          if (!(truth <= 1e-9 * norm(b))) failures.push(`${where}: true residual ${truth}`)
          // The recurrence residual the solver reports agrees with the true one to the tolerance.
          if (!(Math.abs(r.residualNorm - truth) <= 1e-9 * norm(b)))
            failures.push(`${where}: reports ${r.residualNorm}, true ${truth}`)
          if (exact !== undefined) {
            const worst = x.reduce(
              (m, v, i) => Math.max(m, Math.abs(v - (exact[i] ?? 0)) / Math.max(1, Math.abs(exact[i] ?? 0))),
              0,
            )
            if (!(worst <= 1e-8)) failures.push(`${where}: error ${worst}`)
          }
          if (r.history.length !== r.iterations && r.history.length !== r.iterations + 1)
            failures.push(`${where}: history length ${r.history.length}`)
          // GMRES and MINRES minimise the residual over a growing space, so its norm never increases.
          if (
            (name === 'gmres' || name === 'minres') &&
            !r.history.every((h, i) => i === 0 || h <= (r.history[i - 1] ?? 0) * (1 + 1e-10))
          )
            failures.push(`${where}: residual history not monotone`)
        }
      }
      expect(failures).toEqual([])
    }
    {
      const failures: string[] = []
      const n = 20
      const a = tri(n, 2, -1, -1)
      const zero = new Array<number>(n).fill(0)
      const ones = new Array<number>(n).fill(1)
      const parabola = Array.from({ length: n }, (_, i) => ((i + 1) * (n - i)) / 2)
      for (const name of names) {
        const f = SOLVERS[name]
        const at = (what: string, ok: boolean): void => {
          if (!ok) failures.push(`${name}: ${what}`)
        }
        const zeroRhs = f(a, zero, {})
        at(
          'zero right-hand side: converged, 0 iterations, x = 0',
          zeroRhs.status === 'converged' &&
            zeroRhs.iterations === 0 &&
            toFlat(zeroRhs.x as Tensor).every((v) => v === 0),
        )
        const exactStart = f(a, ones, { x0: parabola, rtol: 1e-10 })
        at('an exact start: converged, 0 iterations', exactStart.status === 'converged' && exactStart.iterations === 0)
        const capped = f(a, ones, { rtol: 1e-14, maxIterations: 2 })
        at(
          'maxIterations 2: status completed, not converged, 2 iterations',
          capped.status === 'completed' && !capped.converged && capped.iterations === 2,
        )
        at(
          'the capped residual is the true one',
          Math.abs(capped.residualNorm - residual(a, toFlat(capped.x as Tensor), ones)) <= 1e-8 * norm(ones),
        )
        // The NaN rule (ND-11): a solver rejects non-finite data at its boundary, naming itself; it does not answer.
        for (const [what, call] of [
          ['a NaN right-hand side', () => f(a, new Array<number>(n).fill(Number.NaN), {})],
          ['an infinite start', () => f(a, ones, { x0: new Array<number>(n).fill(Number.POSITIVE_INFINITY) })],
          [
            'a NaN matrix entry',
            () =>
              f(
                [
                  [Number.NaN, 0],
                  [0, 1],
                ],
                [1, 1],
                {},
              ),
          ],
        ] as const) {
          try {
            call()
            failures.push(`${name}: ${what} did not throw`)
          } catch (error) {
            const e = error as { code?: string; message?: string }
            at(
              `${what} throws value naming it (${e.code}: ${e.message})`,
              e.code === 'value' && e.message?.startsWith(`${name}:`) === true,
            )
          }
        }
        const jacobi = f(S.diagCsr([1, 2, 3, 4, 5, 6]), [1, 1, 1, 1, 1, 1], {
          preconditioner: S.jacobiPreconditioner(S.diagCsr([1, 2, 3, 4, 5, 6])),
          rtol: 1e-12,
        })
        at(
          `Jacobi-preconditioned diagonal system: 1 iteration (took ${jacobi.iterations})`,
          jacobi.status === 'converged' && jacobi.iterations <= 1,
        )
      }
      // The eigensolvers take the same rule: a NaN matrix is rejected naming the routine, not answered with `convergence`.
      const nanMatrix = S.csr([
        [2, Number.NaN],
        [Number.NaN, 2],
      ])
      for (const [name, call] of [
        ['eigsh', () => S.eigsh(nanMatrix, 1)],
        ['lanczos', () => S.lanczos(nanMatrix, 2)],
        ['svds', () => S.svds(nanMatrix, 1)],
      ] as const)
        expect(call, name).toThrow(
          expect.objectContaining({ code: 'value', message: expect.stringMatching(`^${name}: `) }),
        )
      // Matrix sizes that do not match, and a rectangular operator, are shape errors.
      expect(() => S.cg(tri(4, 2, -1, -1), [1, 2, 3])).toThrow(expect.objectContaining({ code: 'shape' }))
      expect(failures).toEqual([])
    }
    {
      // ND-30: MINRES takes the Jacobi preconditioner (the Lanczos process in the M^-1 inner product, z = M^-1 r). On an
      // ill-scaled SPD matrix D T D it takes fewer iterations than without, to the same solution.
      const n = 40
      const scale = Array.from({ length: n }, (_, i) => 10 ** ((i % 4) / 2))
      const t = tri(n, 2, -1, -1)
      const scaled = S.csr(dense(t).map((row, i) => row.map((v, j) => v * (scale[i] ?? 1) * (scale[j] ?? 1))))
      const truth = Array.from({ length: n }, (_, i) => Math.sin(i + 1))
      const b = dense(scaled).map((row) => row.reduce((acc, v, j) => acc + v * (truth[j] ?? 0), 0))
      const plain = S.minres(scaled, b, { rtol: 1e-12, maxIterations: 800 })
      const jacobi = S.minres(scaled, b, {
        rtol: 1e-12,
        maxIterations: 800,
        preconditioner: S.jacobiPreconditioner(scaled),
      })
      const worst = (r: S.KrylovResult): number =>
        toFlat(r.x as Tensor).reduce((m, v, i) => Math.max(m, Math.abs(v - (truth[i] ?? 0))), 0)
      expect([plain.status, jacobi.status]).toEqual(['converged', 'converged'])
      expect(jacobi.iterations).toBeLessThan(plain.iterations)
      expect(worst(jacobi)).toBeLessThan(1e-6)
      expect(worst(plain)).toBeLessThan(1e-6)
      // The residual it reports is the M^-1 norm of the true residual.
      const x = toFlat(jacobi.x as Tensor)
      const res = dense(scaled).map((row, i) => (b[i] ?? 0) - row.reduce((acc, v, j) => acc + v * (x[j] ?? 0), 0))
      const diag = dense(scaled).map((row, i) => row[i] ?? 1)
      const mNorm = Math.sqrt(res.reduce((acc, v, i) => acc + (v * v) / (diag[i] ?? 1), 0))
      expect(Math.abs(jacobi.residualNorm - mNorm)).toBeLessThan(1e-9 * Math.sqrt(norm(b)))
      // A preconditioner that is not positive definite is rejected, not followed.
      expect(() => S.minres(scaled, b, { preconditioner: (r) => r.map((v) => -v) })).toThrow(
        expect.objectContaining({ code: 'value' }),
      )
    }
    {
      // ND-31: a singular system with a right-hand side outside the range of A. [[1, 1], [1, 1]] x = [1, 2] has the least
      // residual 0.7071 (at x = [0.75, 0.75]); gmres reported 5e13 and minres 0.106, both `completed`. Now `failed`, with
      // the honest residual of the returned x. A consistent singular system still converges.
      const singular = S.csr([
        [1, 1],
        [1, 1],
      ])
      for (const name of ['gmres', 'minres'] as const) {
        const r = SOLVERS[name](singular, [1, 2], { rtol: 1e-10 })
        const x = toFlat(r.x as Tensor)
        expect([name, r.status, r.converged]).toEqual([name, 'failed', false])
        expect(Math.abs(r.residualNorm - residual(singular, x, [1, 2]))).toBeLessThan(1e-9)
        expect(Math.abs(r.residualNorm - Math.SQRT1_2)).toBeLessThan(1e-9)
        const consistent = SOLVERS[name](singular, [2, 2], { rtol: 1e-10 })
        expect([name, consistent.status]).toEqual([name, 'converged'])
      }
    }
    {
      // The diagonal and the stored count of a CSR matrix: duplicates are summed, explicit zeros are not stored.
      const m = S.csr([
        [1, 0, 2],
        [0, 0, 0],
        [4, 5, 6],
      ])
      expect([toFlat(S.csrDiagonal(m) as Tensor), S.nnz(m)]).toEqual([[1, 0, 6], 5])
      expect(toFlat(S.csrDiagonal(tri(4, 2, -1, -1)) as Tensor)).toEqual([2, 2, 2, 2])
    }
  })
})
