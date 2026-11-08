/**
 * A primal-dual interior-point method for convex quadratic programs (Mehrotra's predictor-corrector on the reduced KKT
 * system) as a traced algorithm on number arrays, the solver behind `qpInteriorPoint`.
 */

import { ddot } from '../numerics/linalg/index.ts'
import { type Algorithm, channels } from '../trace/index.ts'
import { matVec, quadraticValue, solveDense } from './_programming.ts'
import { infNorm } from './minimize/objective.ts'
import type { QpArrays, QpOptions, QpSolution } from './quadratic.ts'

/** The state of {@link qpInteriorPointAlgorithm}. */
export interface QpInteriorPointState {
  readonly x: Float64Array
  /** Slacks of G x + s = h. */
  readonly s: Float64Array
  /** Multipliers of G x <= h. */
  readonly z: Float64Array
  /** Multipliers of A x = b. */
  readonly y: Float64Array
  /** Infinity norms of the dual residual P x + q + A^T y + G^T z and of the primal residuals, and mu = s^T z / m. */
  readonly dualResidual: number
  readonly primalResidual: number
  readonly mu: number
  readonly iterations: number
  readonly solution: QpSolution
  readonly finished: boolean
}

/**
 * A primal-dual interior-point method for convex quadratic programs, as a traced algorithm on number arrays. Each step
 * eliminates the slacks and inequality multipliers from the Newton system of the perturbed KKT conditions, solves the
 * reduced system [P + G^T (Z/S) G, A^T; A, 0] for an affine-scaling predictor and Mehrotra's centring corrector
 * (sigma = (mu_aff / mu)^3), and steps 99% of the way to the boundary. Records the objective, residuals and mu.
 *
 * @param qp - The program (from `qpArrays`).
 * @param options - Tolerance (default 1e-9).
 * @returns The algorithm.
 * @concept interior-point-methods
 * @example
 * ```ts
 * import { qpArrays, qpInteriorPointAlgorithm } from 'omai-engine/optim'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const qp = qpArrays({ P: [[2]], q: [-4], ub: [1] })
 * Math.round(runAlgorithm(qpInteriorPointAlgorithm(qp), { maxSteps: 50 }).state.x[0] * 1e6) / 1e6 // => 1
 * ```
 */
export function qpInteriorPointAlgorithm(qp: QpArrays, options: QpOptions = {}): Algorithm<QpInteriorPointState> {
  const tol = options.tol ?? 1e-9
  const { n, G, h, A, b } = qp
  const mi = G.rows
  const me = A.rows
  const size = n + me
  const qScale = 1 + infNorm(qp.q)
  const pScale = 1 + Math.max(infNorm(b), infNorm(h))
  // The reduced KKT matrix [P + G^T W G, A^T; A, 0] for a diagonal W.
  const kkt = (w: Float64Array): Float64Array => {
    const K = new Float64Array(size * size)
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) K[i * size + j] = qp.P[i * n + j] ?? 0
    for (let r = 0; r < mi; r++) {
      const wr = w[r] ?? 0
      if (wr === 0) continue
      for (let i = 0; i < n; i++) {
        const gi = G.data[r * n + i] ?? 0
        if (gi === 0) continue
        for (let j = 0; j < n; j++) K[i * size + j] = (K[i * size + j] ?? 0) + wr * gi * (G.data[r * n + j] ?? 0)
      }
    }
    for (let r = 0; r < me; r++)
      for (let j = 0; j < n; j++) {
        K[(n + r) * size + j] = A.data[r * n + j] ?? 0
        K[j * size + n + r] = A.data[r * n + j] ?? 0
      }
    return K
  }
  const residuals = (x: Float64Array, s: Float64Array, z: Float64Array, y: Float64Array) => {
    const px = matVec({ data: qp.P, rows: n, cols: n }, x)
    const aty = matVec(A, y, true)
    const gtz = matVec(G, z, true)
    const rd = Float64Array.from(px, (v, j) => v + (qp.q[j] ?? 0) + (aty[j] ?? 0) + (gtz[j] ?? 0))
    const ax = matVec(A, x)
    const re = Float64Array.from(ax, (v, i) => v - (b[i] ?? 0))
    const gx = matVec(G, x)
    const ri = Float64Array.from(gx, (v, i) => v + (s[i] ?? 0) - (h[i] ?? 0))
    return { rd, re, ri, mu: mi === 0 ? 0 : ddot(s, z) / mi }
  }
  const make = (
    x: Float64Array,
    s: Float64Array,
    z: Float64Array,
    y: Float64Array,
    iterations: number,
  ): QpInteriorPointState => {
    const r = residuals(x, s, z, y)
    const dualResidual = infNorm(r.rd)
    const primalResidual = Math.max(infNorm(r.re), infNorm(r.ri))
    const converged = dualResidual <= tol * qScale && primalResidual <= tol * pScale && r.mu <= tol
    const active = Array.from({ length: mi }, (_, i) => i).filter((i) => (z[i] ?? 0) > Math.sqrt(tol))
    return {
      x,
      s,
      z,
      y,
      dualResidual,
      primalResidual,
      mu: r.mu,
      iterations,
      solution: {
        x,
        fun: quadraticValue(qp.P, qp.q, x),
        y,
        z,
        status: converged ? 'optimal' : 'iteration limit',
        iterations,
        active,
      },
      finished: converged || !Number.isFinite(dualResidual),
    }
  }
  const maxStep = (v: Float64Array, dv: Float64Array): number => {
    let a = Infinity
    for (let i = 0; i < v.length; i++) if ((dv[i] ?? 0) < 0) a = Math.min(a, -(v[i] ?? 0) / (dv[i] ?? 1))
    return a
  }
  return {
    name: 'qpInteriorPoint',
    config: { tol },
    schema: channels({
      metrics: {
        objective: { description: 'objective value', direction: 'lower' },
        dualResidual: { description: 'infinity norm of P x + q + A^T y + G^T z' },
        primalResidual: { description: 'infinity norm of the equality and inequality residuals' },
        mu: { description: 'complementarity s^T z / m' },
      },
      tensors: { x: { role: 'param', description: 'current iterate' } },
    }),
    init: () => {
      // Start from the minimiser of 1/2 x^T P x + q^T x + 1/2 |G x - h|^2 subject to A x = b, then shift s and z inside.
      const K = kkt(new Float64Array(mi).fill(1))
      const gth = matVec(G, h, true)
      const rhs = new Float64Array(size)
      for (let j = 0; j < n; j++) rhs[j] = -(qp.q[j] ?? 0) + (gth[j] ?? 0)
      for (let r = 0; r < me; r++) rhs[n + r] = b[r] ?? 0
      const sol = solveDense(K, rhs, size)
      const x = sol.slice(0, n)
      const y = sol.slice(n)
      const gx = matVec(G, x)
      const s = Float64Array.from(h, (v, i) => v - (gx[i] ?? 0))
      let lowest = Infinity
      for (const v of s) lowest = Math.min(lowest, v)
      const shift = mi > 0 && lowest < 1 ? 1 - lowest : 0
      for (let i = 0; i < mi; i++) s[i] = (s[i] ?? 0) + shift
      return make(x, s, new Float64Array(mi).fill(1), y, 0)
    },
    step: (st) => {
      if (st.finished) return st
      const { x, s, z, y } = st
      const r = residuals(x, s, z, y)
      const w = Float64Array.from(z, (v, i) => v / (s[i] ?? 1))
      const K = kkt(w)
      const direction = (rsz: Float64Array) => {
        // H dx + A^T dy = -rd + G^T S^{-1} (rsz - Z ri);  A dx = -re.
        const t = Float64Array.from(rsz, (v, i) => (v - (z[i] ?? 0) * (r.ri[i] ?? 0)) / (s[i] ?? 1))
        const gt = matVec(G, t, true)
        const rhs = new Float64Array(size)
        for (let j = 0; j < n; j++) rhs[j] = -(r.rd[j] ?? 0) + (gt[j] ?? 0)
        for (let i = 0; i < me; i++) rhs[n + i] = -(r.re[i] ?? 0)
        const sol = solveDense(K, rhs, size)
        const dx = sol.slice(0, n)
        const dy = sol.slice(n)
        const gdx = matVec(G, dx)
        const ds = Float64Array.from(r.ri, (v, i) => -v - (gdx[i] ?? 0))
        const dz = Float64Array.from(rsz, (v, i) => (-v - (z[i] ?? 0) * (ds[i] ?? 0)) / (s[i] ?? 1))
        return { dx, dy, ds, dz }
      }
      const aff = direction(Float64Array.from(s, (v, i) => v * (z[i] ?? 0)))
      let alpha = Math.min(1, maxStep(s, aff.ds), maxStep(z, aff.dz))
      let next = aff
      if (mi > 0) {
        let muAff = 0
        for (let i = 0; i < mi; i++)
          muAff += ((s[i] ?? 0) + alpha * (aff.ds[i] ?? 0)) * ((z[i] ?? 0) + alpha * (aff.dz[i] ?? 0))
        muAff /= mi
        const sigma = (muAff / r.mu) ** 3
        next = direction(
          Float64Array.from(s, (v, i) => v * (z[i] ?? 0) + (aff.ds[i] ?? 0) * (aff.dz[i] ?? 0) - sigma * r.mu),
        )
        alpha = Math.min(1, 0.99 * Math.min(maxStep(s, next.ds), maxStep(z, next.dz)))
      } else alpha = 1
      const step = (v: Float64Array, dv: Float64Array): Float64Array =>
        Float64Array.from(v, (q, i) => q + alpha * (dv[i] ?? 0))
      return make(step(x, next.dx), step(s, next.ds), step(z, next.dz), step(y, next.dy), st.iterations + 1)
    },
    done: (st) => (st.finished ? (st.solution.status === 'optimal' ? 'converged' : 'diverged') : undefined),
    observe: (st) => ({
      metrics: {
        objective: st.solution.fun,
        dualResidual: st.dualResidual,
        primalResidual: st.primalResidual,
        mu: st.mu,
      },
      tensors: { x: { role: 'param', value: st.x } },
    }),
  }
}
