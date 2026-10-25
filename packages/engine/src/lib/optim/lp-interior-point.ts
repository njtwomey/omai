/**
 * The interior-point algorithm behind `linprogInteriorPoint`: Mehrotra's predictor-corrector on the homogeneous
 * self-dual embedding of Andersen and Andersen (the method of scipy's legacy `interior-point`), as a traced algorithm
 * whose iterates move through the inside of the feasible region.
 */

import { ddot, dnrm2 } from '../numerics/linalg/index.ts'
import type { Algorithm } from '../trace/index.ts'
import { LP_SCHEMA, lpConfig } from './_lp.ts'
import { type Dense, matVec, solveSpd } from './_programming.ts'
import {
  fromStandard,
  type LinprogOptions,
  type LpPathPoint,
  type StandardForm,
  type StandardSolution,
} from './lp-standard-form.ts'

/** The state of {@link lpInteriorPointAlgorithm}: the homogeneous self-dual iterate and its residual indicators. */
export interface LpInteriorPointState {
  readonly x: Float64Array
  readonly y: Float64Array
  readonly z: Float64Array
  readonly tau: number
  readonly kappa: number
  readonly iterations: number
  /** Relative primal and dual residuals, relative gap, and the residual of the gap equation and of complementarity. */
  readonly rhoP: number
  readonly rhoD: number
  readonly rhoA: number
  readonly rhoG: number
  readonly rhoMu: number
  readonly path: readonly LpPathPoint[]
  readonly solution: StandardSolution
  readonly finished: boolean
}

/** Independent equality rows of the standard form (rows with a slack are always independent), by elimination. */
function independentRows(form: StandardForm, tol: number): { keep: number[]; inconsistent: boolean } {
  const N = form.a.cols
  const keep: number[] = []
  const basisRows: Float64Array[] = []
  const pivots: number[] = []
  let inconsistent = false
  for (let r = 0; r < form.a.rows; r++) {
    if ((form.slackOf[r] ?? -1) >= 0) {
      keep.push(r)
      continue
    }
    const row = new Float64Array(N + 1)
    for (let j = 0; j < N; j++) row[j] = form.a.data[r * N + j] ?? 0
    row[N] = form.b[r] ?? 0
    for (const [k, other] of basisRows.entries()) {
      const p = pivots[k] ?? 0
      const f = (row[p] ?? 0) / (other[p] ?? 1)
      if (f !== 0) for (let j = 0; j <= N; j++) row[j] = (row[j] ?? 0) - f * (other[j] ?? 0)
    }
    let p = -1
    let best = tol
    for (let j = 0; j < N; j++) {
      const a = Math.abs(row[j] ?? 0)
      if (a > best) {
        best = a
        p = j
      }
    }
    if (p < 0) {
      if (Math.abs(row[N] ?? 0) > tol * (1 + Math.abs(form.b[r] ?? 0))) inconsistent = true
      continue
    }
    basisRows.push(row)
    pivots.push(p)
    keep.push(r)
  }
  return { keep, inconsistent }
}

/**
 * A primal-dual interior-point method for linear programs: Mehrotra's predictor-corrector on the homogeneous
 * self-dual embedding of Andersen and Andersen (the algorithm of scipy's legacy `interior-point` method), as a traced
 * algorithm. Each step solves the normal equations A D A^T dy = r (D = X Z^{-1}) for an affine-scaling predictor and a
 * centring corrector; the embedding variables tau and kappa certify infeasibility or unboundedness when tau goes to 0.
 * Linearly dependent equality rows are removed first. Records the objective, the relative primal and dual residuals and
 * the relative gap.
 *
 * @param form - A standard form (from `standardForm`).
 * @param options - Tolerance (default 1e-8).
 * @returns The algorithm.
 * @concept interior-point-methods
 * @reference E. D. Andersen and K. D. Andersen, The MOSEK interior point optimizer for linear programming (2000).
 * @example
 * ```ts
 * import { lpInteriorPointAlgorithm, standardForm } from 'whitebox-engine/optim'
 * import { runAlgorithm } from 'whitebox-engine/trace'
 * const form = standardForm({ c: [-1, -2], aUb: [[1, 1], [1, 3]], bUb: [4, 6] })
 * runAlgorithm(lpInteriorPointAlgorithm(form), { maxSteps: 100 }).state.solution.status // => 'optimal'
 * ```
 */
export function lpInteriorPointAlgorithm(
  form: StandardForm,
  options: LinprogOptions = {},
): Algorithm<LpInteriorPointState> {
  const tol = options.tol ?? 1e-8
  const { keep, inconsistent } = independentRows(form, 1e-10)
  const N = form.a.cols
  const m = keep.length
  const aData = new Float64Array(m * N)
  for (const [i, r] of keep.entries()) aData.set(form.a.data.subarray(r * N, (r + 1) * N), i * N)
  const A: Dense = { data: aData, rows: m, cols: N }
  const b = Float64Array.from(keep, (r) => form.b[r] ?? 0)
  const c = form.c
  const alpha0 = 0.99995
  const beta = 0.1

  const rp = (x: Float64Array, tau: number): Float64Array => {
    const ax = matVec(A, x)
    return Float64Array.from(b, (v, i) => v * tau - (ax[i] ?? 0))
  }
  const rd = (y: Float64Array, z: Float64Array, tau: number): Float64Array => {
    const aty = matVec(A, y, true)
    return Float64Array.from(c, (v, j) => v * tau - (aty[j] ?? 0) - (z[j] ?? 0))
  }
  const rg = (x: Float64Array, y: Float64Array, kappa: number): number => kappa + ddot(c, x) - ddot(b, y)
  const mu = (x: Float64Array, z: Float64Array, tau: number, kappa: number): number =>
    (ddot(x, z) + tau * kappa) / (N + 1)
  const x0 = new Float64Array(N).fill(1)
  const y0 = new Float64Array(m)
  const z0 = new Float64Array(N).fill(1)
  const rp0 = Math.max(1, dnrm2(rp(x0, 1)))
  const rd0 = Math.max(1, dnrm2(rd(y0, z0, 1)))
  const rg0 = Math.max(1, Math.abs(rg(x0, y0, 1)))
  const mu0 = mu(x0, z0, 1, 1)

  const pathPoint = (x: Float64Array, tau: number): LpPathPoint => {
    const xs = fromStandard(
      form,
      Float64Array.from(x, (v) => v / tau),
    )
    return { phase: 0, x: Array.from(xs), objective: ddot(form.original.c, xs) }
  }
  const solutionOf = (
    s: Pick<LpInteriorPointState, 'x' | 'y' | 'z' | 'tau' | 'iterations' | 'path'>,
    status: StandardSolution['status'],
  ): StandardSolution => {
    const y = new Float64Array(form.a.rows)
    for (const [i, r] of keep.entries()) y[r] = (s.y[i] ?? 0) / s.tau
    return {
      status,
      z: Float64Array.from(s.x, (v) => v / s.tau),
      y,
      reducedCosts: Float64Array.from(s.z, (v) => v / s.tau),
      iterations: s.iterations,
      path: s.path,
    }
  }
  const indicators = (x: Float64Array, y: Float64Array, z: Float64Array, tau: number, kappa: number) => {
    const by = ddot(b, y)
    return {
      rhoP: dnrm2(rp(x, tau)) / rp0,
      rhoD: dnrm2(rd(y, z, tau)) / rd0,
      rhoA: Math.abs(ddot(c, x) - by) / (tau + Math.abs(by)),
      rhoG: Math.abs(rg(x, y, kappa)) / rg0,
      rhoMu: mu(x, z, tau, kappa) / mu0,
    }
  }
  const stepLength = (
    x: Float64Array,
    dx: Float64Array,
    z: Float64Array,
    dz: Float64Array,
    tau: number,
    dtau: number,
    kappa: number,
    dkappa: number,
    a0: number,
  ): number => {
    let ax = 1
    let az = 1
    for (let j = 0; j < N; j++) {
      if ((dx[j] ?? 0) < 0) ax = Math.min(ax, (x[j] ?? 0) / -(dx[j] ?? 0))
      if ((dz[j] ?? 0) < 0) az = Math.min(az, (z[j] ?? 0) / -(dz[j] ?? 0))
    }
    const at = dtau < 0 ? (a0 * tau) / -dtau : 1
    const ak = dkappa < 0 ? (a0 * kappa) / -dkappa : 1
    return Math.min(1, a0 * ax, at, a0 * az, ak)
  }

  return {
    name: 'lpInteriorPoint',
    config: lpConfig(options),
    schema: LP_SCHEMA,
    init: () => {
      const base = { x: x0, y: y0, z: z0, tau: 1, kappa: 1, iterations: 0, path: [pathPoint(x0, 1)] }
      const ind = indicators(x0, y0, z0, 1, 1)
      if (inconsistent) return { ...base, ...ind, solution: solutionOf(base, 'infeasible'), finished: true }
      return { ...base, ...ind, solution: solutionOf(base, 'iteration limit'), finished: false }
    },
    step: (s) => {
      if (s.finished) return s
      const { x, y, z, tau, kappa } = s
      const rP = rp(x, tau)
      const rD = rd(y, z, tau)
      const rG = rg(x, y, kappa)
      const muNow = mu(x, z, tau, kappa)
      const dinv = Float64Array.from(x, (v, j) => v / (z[j] ?? 1))
      const M = new Float64Array(m * m)
      for (let i = 0; i < m; i++) {
        for (let k = i; k < m; k++) {
          let acc = 0
          for (let j = 0; j < N; j++) acc += (aData[i * N + j] ?? 0) * (dinv[j] ?? 0) * (aData[k * N + j] ?? 0)
          M[i * m + k] = acc
          M[k * m + i] = acc
        }
      }
      const symSolve = (r1: Float64Array, r2: Float64Array): { u: Float64Array; v: Float64Array } => {
        const adr = matVec(
          A,
          Float64Array.from(r1, (v, j) => (dinv[j] ?? 0) * v),
        )
        const v = solveSpd(
          M,
          Float64Array.from(r2, (q, i) => q + (adr[i] ?? 0)),
          m,
        )
        const atv = matVec(A, v, true)
        return { u: Float64Array.from(r1, (q, j) => (dinv[j] ?? 0) * ((atv[j] ?? 0) - q)), v }
      }
      const { u: p, v: q } = symSolve(c, b)
      let gamma = 0
      let dx = new Float64Array(N)
      let dy = new Float64Array(m)
      let dz = new Float64Array(N)
      let dtau = 0
      let dkappa = 0
      for (let pass = 0; pass < 2; pass++) {
        const eta = 1 - gamma
        const rxs = Float64Array.from(
          x,
          (v, j) => gamma * muNow - v * (z[j] ?? 0) - (pass === 1 ? (dx[j] ?? 0) * (dz[j] ?? 0) : 0),
        )
        const rtk = gamma * muNow - tau * kappa - (pass === 1 ? dtau * dkappa : 0)
        const { u, v } = symSolve(
          Float64Array.from(rD, (val, j) => eta * val - (rxs[j] ?? 0) / (x[j] ?? 1)),
          Float64Array.from(rP, (val) => eta * val),
        )
        dtau = (eta * rG + rtk / tau - (-ddot(c, u) + ddot(b, v))) / (kappa / tau + (-ddot(c, p) + ddot(b, q)))
        dx = Float64Array.from(u, (val, j) => val + (p[j] ?? 0) * dtau)
        dy = Float64Array.from(v, (val, i) => val + (q[i] ?? 0) * dtau)
        const dxNow = dx
        dz = Float64Array.from(rxs, (val, j) => (val - (z[j] ?? 0) * (dxNow[j] ?? 0)) / (x[j] ?? 1))
        dkappa = (rtk - kappa * dtau) / tau
        const alpha = stepLength(x, dx, z, dz, tau, dtau, kappa, dkappa, 1)
        gamma = (1 - alpha) ** 2 * Math.min(beta, 1 - alpha)
      }
      const alpha = stepLength(x, dx, z, dz, tau, dtau, kappa, dkappa, alpha0)
      const xn = Float64Array.from(x, (v, j) => v + alpha * (dx[j] ?? 0))
      const yn = Float64Array.from(y, (v, i) => v + alpha * (dy[i] ?? 0))
      const zn = Float64Array.from(z, (v, j) => v + alpha * (dz[j] ?? 0))
      const taun = tau + alpha * dtau
      const kappan = kappa + alpha * dkappa
      const ind = indicators(xn, yn, zn, taun, kappan)
      const next = {
        x: xn,
        y: yn,
        z: zn,
        tau: taun,
        kappa: kappan,
        iterations: s.iterations + 1,
        path: [...s.path, pathPoint(xn, taun)],
        ...ind,
      }
      const optimal = ind.rhoP <= tol && ind.rhoD <= tol && ind.rhoA <= tol
      const inf1 = ind.rhoP < tol && ind.rhoD < tol && ind.rhoG < tol && taun < tol * Math.max(1, kappan)
      const inf2 = ind.rhoMu < tol && taun < tol * Math.min(1, kappan)
      if (optimal) return { ...next, solution: solutionOf(next, 'optimal'), finished: true }
      if (inf1 || inf2) {
        const status = ddot(b, yn) > tol ? 'infeasible' : 'unbounded'
        return { ...next, solution: solutionOf(next, status), finished: true }
      }
      return { ...next, solution: solutionOf(next, 'iteration limit'), finished: false }
    },
    done: (s) => (s.finished ? (s.solution.status === 'optimal' ? 'converged' : 'failed') : undefined),
    observe: (s) => {
      const last = s.path[s.path.length - 1]
      return {
        metrics: {
          primalResidual: s.rhoP,
          dualResidual: s.rhoD,
          gap: s.rhoA,
          ...(last === undefined ? {} : { objective: last.objective }),
        },
        ...(last === undefined ? {} : { tensors: { x: { role: 'param' as const, value: last.x } } }),
      }
    },
  }
}
