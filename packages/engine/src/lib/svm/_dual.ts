// The six duals of the support vector machines as one interface over the library's SMO states: the algorithm to step,
// the kernel expansion a state stands for, the dual objective being raised and the primal-dual gap being closed. Each
// dual is built without shrinking, so the gradient (and with it the objective and the gap) is exact at every step.
// Not exported from the barrel.

import { nuRho, nuSmoAlgorithm, nuSvcStart, nuSvrStart, type SmoState, smoAlgorithm, smoRho } from '../optim/index.ts'
import type { Algorithm } from '../trace/index.ts'

/** What a state of a dual stands for: a kernel expansion `sum_i coef_i K(x_i, x) - rho` over the dual's samples. */
export interface Expansion {
  readonly coef: Float64Array
  readonly rho: number
}

/** A kernel expansion as the plain numbers a state stores (`live` in the classifier's and regressor's states). */
export function toLive(e: Expansion): { readonly coef: number[]; readonly rho: number } {
  return { coef: Array.from(e.coef), rho: e.rho }
}

/** A dual problem on the Gram matrix of its samples. */
export interface Dual {
  /** The number of samples (the dual of a regression has twice as many variables). */
  readonly n: number
  readonly algorithm: Algorithm<SmoState>
  expansion(s: SmoState): Expansion
  /** The dual objective, to be maximised. */
  dual(s: SmoState): number
  /** The primal-dual gap: the primal objective at the state's expansion minus the dual (at least zero). */
  gap(s: SmoState): number
  /** The functional margin of each sample, `y_i f(x_i)` (classification) or the residual `t_i - f(x_i)` (regression). */
  margins(s: SmoState): Float64Array
  /** For a regression: the half-width of the tube (given, or found by the nu form). */
  tube?(s: SmoState): number
}

/** Solver settings shared by the duals. */
export interface DualOptions {
  readonly tol: number
  readonly selection: 'second-order' | 'first-order'
}

/**
 * The minimum over e of k e + sum_i max(0, a_i - e): the optimal slack threshold of the nu-forms' primal (the margin
 * rho, the tube width, the squared radius), found at the (floor(k) + 1)-th largest a_i.
 */
export function hingeMinimum(a: ArrayLike<number>, k: number, lower: number = -Infinity): number {
  const e = hingeArgmin(a, k, lower)
  let total = k * e
  for (const v of Array.from(a)) total += Math.max(0, v - e)
  return total
}

/** The minimiser e of k e + sum_i max(0, a_i - e): the (floor(k) + 1)-th largest a_i. */
export function hingeArgmin(a: ArrayLike<number>, k: number, lower: number = -Infinity): number {
  const sorted = Array.from(a).sort((p, q) => q - p)
  const index = Math.min(Math.floor(k + 1e-12), sorted.length - 1)
  return Math.max(sorted[index] ?? 0, lower)
}

const sumOf = (v: ArrayLike<number>): number => {
  let s = 0
  for (let i = 0; i < v.length; i++) s += v[i] ?? 0
  return s
}

/** The rows and columns `rows` of a flat n x n matrix, as a flat matrix. */
export function submatrix(K: Float64Array, n: number, rows: readonly number[]): Float64Array {
  const m = rows.length
  const out = new Float64Array(m * m)
  for (let i = 0; i < m; i++) for (let j = 0; j < m; j++) out[i * m + j] = K[(rows[i] ?? 0) * n + (rows[j] ?? 0)] ?? 0
  return out
}

function square(K: Float64Array, y: Float64Array | undefined, n: number): number[][] {
  return Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (y === undefined ? 1 : (y[i] ?? 1) * (y[j] ?? 1)) * (K[i * n + j] ?? 0)),
  )
}

/** The C-SVC dual on a Gram matrix with labels +1 or -1 and a penalty per sample. */
export function cSvcDual(K: Float64Array, y: Float64Array, C: Float64Array, options: DualOptions): Dual {
  const n = y.length
  const algorithm = smoAlgorithm(
    { Q: square(K, y, n), p: new Float64Array(n).fill(-1), y, C },
    { tol: options.tol, selection: options.selection, shrinking: false },
  )
  const rho = (s: SmoState): number => smoRho(s.alpha, s.grad, y, C)
  return {
    n,
    algorithm,
    expansion: (s) => ({ coef: s.alpha.map((a, i) => a * (y[i] ?? 1)), rho: rho(s) }),
    dual: (s) => {
      let quad = 0
      for (let i = 0; i < n; i++) quad += (s.alpha[i] ?? 0) * ((s.grad[i] ?? 0) + 1)
      return sumOf(s.alpha) - 0.5 * quad
    },
    gap: (s) => {
      const r = rho(s)
      let quad = 0
      let hinge = 0
      for (let i = 0; i < n; i++) {
        quad += (s.alpha[i] ?? 0) * ((s.grad[i] ?? 0) + 1)
        hinge += (C[i] ?? 0) * Math.max(0, -(s.grad[i] ?? 0) + (y[i] ?? 1) * r)
      }
      return quad + hinge - sumOf(s.alpha)
    },
    margins: (s) => {
      const r = rho(s)
      return Float64Array.from({ length: n }, (_, i) => (s.grad[i] ?? 0) + 1 - (y[i] ?? 1) * r)
    },
  }
}

/** The nu-SVC dual: libsvm's Solver_NU, rescaled by the margin so the expansion is a C-SVC with C = 1 / r. */
export function nuSvcDual(K: Float64Array, y: Float64Array, nu: number, options: DualOptions): Dual {
  const n = y.length
  const algorithm = nuSmoAlgorithm(
    { Q: square(K, y, n), p: new Float64Array(n), y, C: 1, alpha0: nuSvcStart(y, nu) },
    { tol: options.tol },
  )
  // Early on the margin can be zero or negative; the scaling is then left alone, so every state has a model.
  const scaled = (s: SmoState): { rho: number; margin: number; r: number } => {
    const { rho, margin } = nuRho(s.alpha, s.grad, y, 1)
    return { rho, margin, r: margin > 1e-12 ? margin : 1 }
  }
  const margins = (s: SmoState): Float64Array => {
    const { rho } = scaled(s)
    return Float64Array.from({ length: n }, (_, i) => (s.grad[i] ?? 0) - (y[i] ?? 1) * rho)
  }
  return {
    n,
    algorithm,
    expansion: (s) => {
      const { rho, r } = scaled(s)
      return { coef: s.alpha.map((a, i) => (a * (y[i] ?? 1)) / r), rho: rho / r }
    },
    dual: (s) => {
      const { r } = scaled(s)
      let quad = 0
      for (let i = 0; i < n; i++) quad += (s.alpha[i] ?? 0) * (s.grad[i] ?? 0)
      return (nu * n) / r - quad / (r * r)
    },
    gap: (s) => {
      let quad = 0
      for (let i = 0; i < n; i++) quad += (s.alpha[i] ?? 0) * (s.grad[i] ?? 0)
      return (
        quad +
        hingeMinimum(
          margins(s).map((m) => -m),
          nu * n,
        )
      )
    },
    margins: (s) => {
      const { r } = scaled(s)
      return margins(s).map((m) => m / r)
    },
  }
}

/** The epsilon-SVR dual over 2n variables (alpha, alpha*), with a penalty per sample. */
export function svrDual(
  K: Float64Array,
  t: Float64Array,
  C: Float64Array,
  epsilon: number,
  options: DualOptions,
): Dual {
  const n = t.length
  const m = 2 * n
  const y = Float64Array.from({ length: m }, (_, i) => (i < n ? 1 : -1))
  const bounds = Float64Array.from({ length: m }, (_, i) => C[i % n] ?? 0)
  const Q = Array.from({ length: m }, (_, i) =>
    Array.from({ length: m }, (_, j) => (y[i] ?? 1) * (y[j] ?? 1) * (K[(i % n) * n + (j % n)] ?? 0)),
  )
  const p = Float64Array.from({ length: m }, (_, i) => (i < n ? epsilon - (t[i] ?? 0) : epsilon + (t[i - n] ?? 0)))
  const algorithm = smoAlgorithm(
    { Q, p, y, C: bounds },
    { tol: options.tol, selection: options.selection, shrinking: false },
  )
  const rho = (s: SmoState): number => smoRho(s.alpha, s.grad, y, bounds)
  // (K beta)_i from the gradient of the first block: G_i = (K beta)_i + epsilon - t_i.
  const residuals = (s: SmoState): Float64Array => {
    const r = rho(s)
    return Float64Array.from({ length: n }, (_, i) => (t[i] ?? 0) - ((s.grad[i] ?? 0) - epsilon + (t[i] ?? 0) - r))
  }
  const quadOf = (s: SmoState): { quad: number; ty: number; total: number } => {
    let quad = 0
    let ty = 0
    for (let i = 0; i < n; i++) {
      const beta = (s.alpha[i] ?? 0) - (s.alpha[i + n] ?? 0)
      quad += beta * ((s.grad[i] ?? 0) - epsilon + (t[i] ?? 0))
      ty += beta * (t[i] ?? 0)
    }
    return { quad, ty, total: sumOf(s.alpha) }
  }
  return {
    n,
    algorithm,
    expansion: (s) => ({
      coef: Float64Array.from({ length: n }, (_, i) => (s.alpha[i] ?? 0) - (s.alpha[i + n] ?? 0)),
      rho: rho(s),
    }),
    dual: (s) => {
      const { quad, ty, total } = quadOf(s)
      return -0.5 * quad - epsilon * total + ty
    },
    gap: (s) => {
      const { quad, ty, total } = quadOf(s)
      const res = residuals(s)
      let loss = 0
      for (let i = 0; i < n; i++) loss += (C[i] ?? 0) * Math.max(0, Math.abs(res[i] ?? 0) - epsilon)
      return quad + loss + epsilon * total - ty
    },
    margins: residuals,
    tube: () => epsilon,
  }
}

/** The nu-SVR dual over 2n variables: the tube width is found by the solver. */
export function nuSvrDual(K: Float64Array, t: Float64Array, C: number, nu: number, options: DualOptions): Dual {
  const n = t.length
  const m = 2 * n
  const y = Float64Array.from({ length: m }, (_, i) => (i < n ? 1 : -1))
  const Q = Array.from({ length: m }, (_, i) =>
    Array.from({ length: m }, (_, j) => (y[i] ?? 1) * (y[j] ?? 1) * (K[(i % n) * n + (j % n)] ?? 0)),
  )
  const p = Float64Array.from({ length: m }, (_, i) => (i < n ? -(t[i] ?? 0) : (t[i - n] ?? 0)))
  const algorithm = nuSmoAlgorithm({ Q, p, y, C, alpha0: nuSvrStart(n, C, nu) }, { tol: options.tol })
  const offset = (s: SmoState): number => nuRho(s.alpha, s.grad, y, C).rho
  const residuals = (s: SmoState): Float64Array => {
    const r = offset(s)
    return Float64Array.from({ length: n }, (_, i) => (t[i] ?? 0) - ((s.grad[i] ?? 0) + (t[i] ?? 0) - r))
  }
  const quadOf = (s: SmoState): { quad: number; ty: number } => {
    let quad = 0
    let ty = 0
    for (let i = 0; i < n; i++) {
      const beta = (s.alpha[i] ?? 0) - (s.alpha[i + n] ?? 0)
      quad += beta * ((s.grad[i] ?? 0) + (t[i] ?? 0))
      ty += beta * (t[i] ?? 0)
    }
    return { quad, ty }
  }
  return {
    n,
    algorithm,
    expansion: (s) => ({
      coef: Float64Array.from({ length: n }, (_, i) => (s.alpha[i] ?? 0) - (s.alpha[i + n] ?? 0)),
      rho: offset(s),
    }),
    dual: (s) => {
      const { quad, ty } = quadOf(s)
      return ty - 0.5 * quad
    },
    gap: (s) => {
      const { quad, ty } = quadOf(s)
      return quad - ty + C * hingeMinimum(residuals(s).map(Math.abs), nu * n, 0)
    },
    margins: residuals,
    tube: (s) => hingeArgmin(residuals(s).map(Math.abs), nu * n, 0),
  }
}

/** The one-class dual: min 1/2 a^T K a with sum a = nu n and 0 <= a <= 1. */
export function oneClassDual(K: Float64Array, nu: number, options: DualOptions): Dual {
  const n = Math.round(Math.sqrt(K.length))
  const alpha0 = new Float64Array(n)
  let remaining = nu * n
  for (let i = 0; i < n && remaining > 0; i++) {
    alpha0[i] = Math.min(1, remaining)
    remaining -= alpha0[i] ?? 0
  }
  const ones = new Float64Array(n).fill(1)
  const algorithm = smoAlgorithm(
    { Q: square(K, undefined, n), p: new Float64Array(n), y: ones, C: 1, alpha0 },
    { tol: options.tol, selection: options.selection, shrinking: false },
  )
  const rho = (s: SmoState): number => smoRho(s.alpha, s.grad, ones, ones)
  const quadOf = (s: SmoState): number => {
    let quad = 0
    for (let i = 0; i < n; i++) quad += (s.alpha[i] ?? 0) * (s.grad[i] ?? 0)
    return quad
  }
  return {
    n,
    algorithm,
    expansion: (s) => ({ coef: Float64Array.from(s.alpha), rho: rho(s) }),
    dual: (s) => -0.5 * quadOf(s),
    gap: (s) =>
      quadOf(s) +
      hingeMinimum(
        s.grad.map((g) => -g),
        nu * n,
      ),
    margins: (s) => {
      const r = rho(s)
      return s.grad.map((g) => g - r)
    },
  }
}

/** The SVDD dual: min a^T K a - sum a_i K_ii with sum a = 1 and 0 <= a <= C. */
export function svddDual(K: Float64Array, C: number, options: DualOptions): Dual {
  const n = Math.round(Math.sqrt(K.length))
  const diag = Float64Array.from({ length: n }, (_, i) => K[i * n + i] ?? 0)
  const alpha0 = new Float64Array(n)
  let remaining = 1
  for (let i = 0; i < n && remaining > 0; i++) {
    alpha0[i] = Math.min(C, remaining)
    remaining -= alpha0[i] ?? 0
  }
  const ones = new Float64Array(n).fill(1)
  const bounds = new Float64Array(n).fill(C)
  const algorithm = smoAlgorithm(
    { Q: square(K, undefined, n).map((row) => row.map((v) => 2 * v)), p: diag.map((d) => -d), y: ones, C, alpha0 },
    { tol: options.tol, selection: options.selection, shrinking: false },
  )
  const rho = (s: SmoState): number => smoRho(s.alpha, s.grad, ones, bounds)
  // alpha^T K alpha = 1/2 sum a_i (G_i + K_ii); the squared distance of x_i to the centre is alpha^T K alpha - G_i.
  const centre = (s: SmoState): number => {
    let q = 0
    for (let i = 0; i < n; i++) q += 0.5 * (s.alpha[i] ?? 0) * ((s.grad[i] ?? 0) + (diag[i] ?? 0))
    return q
  }
  return {
    n,
    algorithm,
    expansion: (s) => ({ coef: Float64Array.from(s.alpha), rho: rho(s) }),
    dual: (s) => {
      let linear = 0
      for (let i = 0; i < n; i++) linear += (s.alpha[i] ?? 0) * (diag[i] ?? 0)
      return linear - centre(s)
    },
    gap: (s) => {
      const q = centre(s)
      let linear = 0
      for (let i = 0; i < n; i++) linear += (s.alpha[i] ?? 0) * (diag[i] ?? 0)
      return (
        C *
          hingeMinimum(
            s.grad.map((g) => q - g),
            1 / C,
          ) -
        (linear - q)
      )
    },
    margins: (s) => {
      const q = centre(s)
      const radius = q - rho(s)
      return s.grad.map((g) => radius - (q - g))
    },
  }
}
