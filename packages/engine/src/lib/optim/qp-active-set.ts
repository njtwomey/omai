/**
 * The primal active-set method for convex quadratic programs (Nocedal and Wright, algorithm 16.3) as a traced algorithm
 * on number arrays: each step solves the equality-constrained problem on the working set, then adds a blocking
 * constraint or drops one with a negative multiplier. It is the exact solver behind `qpActiveSet`.
 */

import { ddot, dnrm2 } from '../numerics/linalg/index.ts'
import { toVector } from '../tensor/index.ts'
import { type Algorithm, channels, runStatusOf } from '../trace/index.ts'
import { matVec, quadraticValue, solveDense } from './_programming.ts'
import { linprogSimplex } from './linear-programming.ts'
import type { QpArrays, QpOptions, QpSolution } from './quadratic.ts'

/** The state of {@link qpActiveSetAlgorithm}. */
export interface QpActiveSetState {
  readonly x: Float64Array
  /** The working set: rows of G treated as equalities. */
  readonly working: readonly number[]
  readonly iterations: number
  /** The last change: a row added (blocking) or removed (negative multiplier), and the step length. */
  readonly last?: { readonly added: number; readonly removed: number; readonly step: number }
  readonly solution: QpSolution
  readonly finished: boolean
}

const row = (m: { data: Float64Array; cols: number }, i: number): Float64Array =>
  m.data.subarray(i * m.cols, (i + 1) * m.cols)

/** Indices of rows independent of `basis` (Gram-Schmidt), appended in order. */
function independent(rows: readonly Float64Array[], basis: Float64Array[]): boolean[] {
  return rows.map((r) => {
    const v = Float64Array.from(r)
    const scale = dnrm2(r)
    if (scale === 0) return false
    for (const q of basis) {
      const c = ddot(v, q)
      for (let j = 0; j < v.length; j++) v[j] = (v[j] ?? 0) - c * (q[j] ?? 0)
    }
    const norm = dnrm2(v)
    if (norm <= 1e-10 * scale) return false
    basis.push(Float64Array.from(v, (x) => x / norm))
    return true
  })
}

/**
 * The primal active-set method for convex quadratic programs (Nocedal and Wright, algorithm 16.3), as a traced
 * algorithm on number arrays. From a feasible point (given, or found by a phase-1 linear program) it keeps a working
 * set of inequality rows held as equalities; each step solves the equality-constrained subproblem through its KKT
 * system, then either moves along the step until a blocking row joins the working set, or (at a stationary point)
 * drops the row with the most negative multiplier, stopping when all multipliers are non-negative. Records the
 * objective, the working-set size and step, and each change of the working set.
 *
 * @param qp - The program (from `qpArrays`).
 * @param options - Tolerance and feasible start.
 * @returns The algorithm.
 * @concept quadratic-programming
 * @example
 * ```ts
 * import { qpActiveSetAlgorithm, qpArrays } from 'mdml-engine/optim'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const qp = qpArrays({ P: [[2, 0], [0, 2]], q: [-2, -5], G: [[1, 2], [-1, 2], [-1, -2]], h: [6, 2, 2], lb: [0, 0] })
 * runAlgorithm(qpActiveSetAlgorithm(qp, { x0: [2, 0] }), { maxSteps: 50 }).state.working // => [1]
 * ```
 */
export function qpActiveSetAlgorithm(qp: QpArrays, options: QpOptions = {}): Algorithm<QpActiveSetState> {
  const tol = options.tol ?? 1e-10
  const start = options.x0 === undefined ? undefined : toVector(options.x0, 'qpActiveSet')
  const { n, G, h, A, b } = qp
  const mi = G.rows
  const scaleP = Math.max(1, Math.sqrt(ddot(qp.P, qp.P)))
  // Independent equality rows (the program is assumed consistent).
  const eqBasis: Float64Array[] = []
  const eqKeep = independent(
    Array.from({ length: A.rows }, (_, i) => row(A, i)),
    eqBasis,
  )
  const eqRows = eqKeep.flatMap((k, i) => (k ? [i] : []))
  const feasible = (x: ArrayLike<number>): boolean => {
    const gx = matVec(G, x)
    for (let i = 0; i < mi; i++) if ((gx[i] ?? 0) > (h[i] ?? 0) + 1e-9 * (1 + Math.abs(h[i] ?? 0))) return false
    const ax = matVec(A, x)
    for (let i = 0; i < A.rows; i++)
      if (Math.abs((ax[i] ?? 0) - (b[i] ?? 0)) > 1e-9 * (1 + Math.abs(b[i] ?? 0))) return false
    return true
  }
  const solutionOf = (
    x: Float64Array,
    working: readonly number[],
    lambda: Float64Array | undefined,
    status: QpSolution['status'],
    iterations: number,
  ): QpSolution => {
    const y = new Float64Array(A.rows)
    const z = new Float64Array(mi)
    if (lambda !== undefined) {
      for (const [k, i] of eqRows.entries()) y[i] = lambda[k] ?? 0
      for (const [k, i] of working.entries()) z[i] = Math.max(0, lambda[eqRows.length + k] ?? 0)
    }
    return {
      x,
      fun: quadraticValue(qp.P, qp.q, x),
      y,
      z,
      status,
      iterations,
      active: [...working].sort((p, q) => p - q),
    }
  }
  // Solve the KKT system [P C^T; C 0] [p; lambda] = [-g; 0] for the working constraints C.
  const eqp = (
    x: Float64Array,
    working: readonly number[],
  ): { p: Float64Array; lambda: Float64Array; regularised: boolean } => {
    const rows = [...eqRows.map((i) => row(A, i)), ...working.map((i) => row(G, i))]
    const k = rows.length
    const size = n + k
    const g = matVec({ data: qp.P, rows: n, cols: n }, x)
    for (let j = 0; j < n; j++) g[j] = (g[j] ?? 0) + (qp.q[j] ?? 0)
    const solveWith = (delta: number): Float64Array => {
      const K = new Float64Array(size * size)
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) K[i * size + j] = qp.P[i * n + j] ?? 0
        K[i * size + i] = (K[i * size + i] ?? 0) + delta
      }
      for (const [r, c] of rows.entries()) {
        for (let j = 0; j < n; j++) {
          K[(n + r) * size + j] = c[j] ?? 0
          K[j * size + n + r] = c[j] ?? 0
        }
      }
      const rhs = new Float64Array(size)
      for (let j = 0; j < n; j++) rhs[j] = -(g[j] ?? 0)
      return solveDense(K, rhs, size)
    }
    let sol = solveWith(0)
    let regularised = false
    if (!sol.every(Number.isFinite)) {
      sol = solveWith(1e-10 * scaleP)
      regularised = true
    }
    return { p: sol.slice(0, n), lambda: sol.slice(n), regularised }
  }
  return {
    name: 'qpActiveSet',
    config: { tol },
    schema: channels({
      metrics: {
        objective: { description: 'objective value', direction: 'lower' },
        workingSet: { description: 'size of the working set' },
        step: { description: 'step length along the subproblem direction' },
      },
      tensors: { x: { role: 'param', description: 'current iterate' } },
      messages: { workingSet: { description: 'a row added to (blocking) or removed from the working set' } },
    }),
    init: () => {
      let x: Float64Array
      if (start !== undefined && feasible(start)) x = Float64Array.from(start)
      else {
        const lp = linprogSimplex({
          c: new Float64Array(n),
          ...(mi === 0 ? {} : { aUb: Array.from({ length: mi }, (_, i) => Array.from(row(G, i))), bUb: h }),
          ...(A.rows === 0 ? {} : { aEq: Array.from({ length: A.rows }, (_, i) => Array.from(row(A, i))), bEq: b }),
          bounds: [null, null],
        })
        if (lp.status !== 'optimal') {
          const x0 = new Float64Array(n)
          return {
            x: x0,
            working: [],
            iterations: 0,
            solution: solutionOf(x0, [], undefined, 'infeasible', 0),
            finished: true,
          }
        }
        x = Float64Array.from(lp.x.data)
      }
      const gx = matVec(G, x)
      const candidates = Array.from({ length: mi }, (_, i) => i).filter(
        (i) => Math.abs((gx[i] ?? 0) - (h[i] ?? 0)) <= 1e-9 * (1 + Math.abs(h[i] ?? 0)),
      )
      const basis = eqBasis.map((q) => Float64Array.from(q))
      const keep = independent(
        candidates.map((i) => row(G, i)),
        basis,
      )
      const working = candidates.filter((_, k) => keep[k] === true)
      return {
        x,
        working,
        iterations: 0,
        solution: solutionOf(x, working, undefined, 'iteration limit', 0),
        finished: false,
      }
    },
    step: (s) => {
      if (s.finished) return s
      const { p, lambda, regularised } = eqp(s.x, s.working)
      const iterations = s.iterations + 1
      const pNorm = p.reduce((m, v) => Math.max(m, Math.abs(v)), 0)
      const xNorm = s.x.reduce((m, v) => Math.max(m, Math.abs(v)), 0)
      if (pNorm <= tol * Math.max(1, xNorm)) {
        let worst = -1
        let most = -tol
        for (const [k, i] of s.working.entries()) {
          const l = lambda[eqRows.length + k] ?? 0
          if (l < most) {
            most = l
            worst = i
          }
        }
        if (worst < 0) {
          const { last: _last, ...rest } = s
          return {
            ...rest,
            iterations,
            solution: solutionOf(s.x, s.working, lambda, 'optimal', iterations),
            finished: true,
          }
        }
        const working = s.working.filter((i) => i !== worst)
        return {
          ...s,
          working,
          iterations,
          last: { added: -1, removed: worst, step: 0 },
          solution: solutionOf(s.x, working, undefined, 'iteration limit', iterations),
        }
      }
      let alpha = 1
      let blocking = -1
      const gx = matVec(G, s.x)
      const gp = matVec(G, p)
      const inWorking = new Set(s.working)
      for (let i = 0; i < mi; i++) {
        if (inWorking.has(i)) continue
        const d = gp[i] ?? 0
        if (d <= 1e-14 * Math.max(1, pNorm)) continue
        const a = Math.max(0, ((h[i] ?? 0) - (gx[i] ?? 0)) / d)
        if (a < alpha) {
          alpha = a
          blocking = i
        }
      }
      if (regularised && blocking < 0 && pNorm > 1e8 * Math.max(1, xNorm)) {
        return {
          ...s,
          iterations,
          solution: solutionOf(s.x, s.working, undefined, 'unbounded', iterations),
          finished: true,
        }
      }
      const x = Float64Array.from(s.x, (v, j) => v + alpha * (p[j] ?? 0))
      const working = blocking < 0 ? s.working : [...s.working, blocking]
      return {
        ...s,
        x,
        working,
        iterations,
        last: { added: blocking, removed: -1, step: alpha },
        solution: solutionOf(x, working, undefined, 'iteration limit', iterations),
      }
    },
    done: (s) => (s.finished ? runStatusOf(s.solution.status) : undefined),
    observe: (s) => ({
      metrics: {
        objective: s.solution.fun,
        workingSet: s.working.length,
        ...(s.last === undefined ? {} : { step: s.last.step }),
      },
      tensors: { x: { role: 'param', value: s.x } },
      ...(s.last === undefined ? {} : { messages: [{ kind: 'workingSet', payload: { ...s.last } }] }),
    }),
  }
}
