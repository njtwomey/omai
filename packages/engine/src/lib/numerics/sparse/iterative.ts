/**
 * Krylov solvers for sparse linear systems, scipy.sparse.linalg style: conjugate gradients, BiCGSTAB, restarted GMRES
 * and MINRES, each also available as a traced algorithm, with Jacobi and incomplete Cholesky preconditioners, and
 * `sparseSolve`, which differentiates through the solution by solving the adjoint system.
 *
 * @index function Krylov solvers: CG, BiCGSTAB, GMRES and MINRES with preconditioners.
 */

import { customVjp } from '../../autodiff/index.ts'
import { type Json, MdmlError } from '../../base/index.ts'
import {
  fromStorage,
  mul,
  neg,
  primalOf,
  promoteFloat,
  type Tensor,
  type TensorLike,
  take,
  toFlat,
  type Value,
} from '../../tensor/index.ts'
import { type Algorithm, channels, type RunOptions, type RunStatus, runAlgorithm } from '../../trace/index.ts'
import { daxpy, ddot as dot, dnrm2 as norm } from '../linalg/index.ts'
import { asOperator, type CsrMatrix, csr, type LinearOperator, primalValues, spmv, transposeCsr } from './csr.ts'

/** A preconditioner: z = M^{-1} r. */
export type Preconditioner = (r: Float64Array) => Float64Array

function numeric(op: LinearOperator): (x: Float64Array) => Float64Array {
  return (x) => Float64Array.from(toFlat(primalOf(op.matvec(fromStorage(x, [x.length], 'float64')))))
}

/** Options shared by the Krylov solvers (scipy.sparse.linalg semantics: stop when |r| <= max(rtol |b|, atol)). */
export interface KrylovOptions {
  readonly x0?: ArrayLike<number>
  readonly rtol?: number
  readonly atol?: number
  readonly preconditioner?: Preconditioner
}

/** State of a Krylov solver. */
export interface KrylovState {
  readonly x: Float64Array
  readonly residualNorm: number
  readonly history: readonly number[]
  readonly internal: Readonly<Record<string, unknown>>
}

/** Result of a Krylov solve. */
export interface KrylovResult {
  readonly x: Tensor
  readonly iterations: number
  readonly converged: boolean
  /** How the solve ended: `converged`, `completed` (the iteration cap), or `diverged` (a non-finite residual). */
  readonly status: RunStatus
  readonly residualNorm: number
  /** The residual norm after each iteration (the semi-log convergence plot). */
  readonly history: readonly number[]
}

const SCHEMA = channels({
  metrics: { residualNorm: { description: 'norm of the residual b - A x' } },
  tensors: { x: { role: 'state', description: 'current iterate' } },
})

function setup(a: CsrMatrix | TensorLike | LinearOperator, b: TensorLike, options: KrylovOptions) {
  const op = asOperator(a)
  const apply = numeric(op)
  const bv = Float64Array.from(toFlat(primalOf(promoteFloat(b)[0] as Value)))
  if (op.shape[0] !== bv.length || op.shape[1] !== bv.length)
    throw new MdmlError('shape', 'Krylov solvers need a square operator matching b')
  const x0 = options.x0 === undefined ? new Float64Array(bv.length) : Float64Array.from(options.x0)
  const tol = Math.max((options.rtol ?? 1e-5) * norm(bv), options.atol ?? 0)
  const precondition = options.preconditioner ?? ((r: Float64Array) => Float64Array.from(r))
  return { op, apply, b: bv, x0, tol, precondition }
}

function finish(
  algorithm: Algorithm<KrylovState>,
  run: Partial<RunOptions> & { readonly maxIterations?: number },
  n: number,
): KrylovResult {
  const { maxIterations, ...rest } = run
  const result = runAlgorithm(algorithm, { ...rest, maxSteps: maxIterations ?? 10 * n })
  const s = result.state
  return {
    x: fromStorage(Float64Array.from(s.x), [s.x.length], 'float64'),
    iterations: result.steps,
    converged: result.status === 'converged',
    status: result.status,
    residualNorm: s.residualNorm,
    history: s.history,
  }
}

const observe = (s: KrylovState) => ({
  metrics: { residualNorm: s.residualNorm },
  tensors: { x: { role: 'state' as const, value: s.x } },
})

/**
 * The (preconditioned) conjugate gradient method for symmetric positive definite systems, as a traced algorithm.
 *
 * @param a - The matrix (CSR, dense or operator).
 * @param b - Right-hand side.
 * @param options - Start, tolerances and preconditioner.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { cgAlgorithm } from 'mdml-engine/numerics/sparse'
 * cgAlgorithm([[4, 1], [1, 3]], [1, 2]).name // => 'cg'
 * ```
 */
export function cgAlgorithm(
  a: CsrMatrix | TensorLike | LinearOperator,
  b: TensorLike,
  options: KrylovOptions = {},
): Algorithm<KrylovState> {
  const { apply, b: bv, x0, tol, precondition } = setup(a, b, options)
  return {
    name: 'cg',
    config: { rtol: options.rtol ?? 1e-5, atol: options.atol ?? 0 } as Json,
    schema: SCHEMA,
    init: () => {
      const r = daxpy(-1, apply(x0), bv)
      const z = precondition(r)
      const rn = norm(r)
      return { x: x0, residualNorm: rn, history: [rn], internal: { r, p: z, rz: dot(r, z) } }
    },
    step: (s) => {
      const { r, p, rz } = s.internal as { r: Float64Array; p: Float64Array; rz: number }
      const ap = apply(p)
      const alpha = rz / dot(p, ap)
      const x = daxpy(alpha, p, s.x)
      const rNext = daxpy(-alpha, ap, r)
      const z = precondition(rNext)
      const rzNext = dot(rNext, z)
      const pNext = daxpy(rzNext / rz, p, z)
      const rn = norm(rNext)
      return { x, residualNorm: rn, history: [...s.history, rn], internal: { r: rNext, p: pNext, rz: rzNext } }
    },
    done: (s) => (s.residualNorm <= tol ? 'converged' : Number.isFinite(s.residualNorm) ? undefined : 'diverged'),
    observe,
  }
}

/**
 * Solve A x = b by conjugate gradients (scipy.sparse.linalg.cg).
 *
 * @param a - Symmetric positive definite matrix or operator.
 * @param b - Right-hand side.
 * @param options - Tolerances, start, preconditioner, iteration cap (default 10 n) and trace options.
 * @returns The solution with its residual history.
 * @concept krylov-methods
 * @index function Conjugate gradients for symmetric positive-definite systems.
 * @example
 * ```ts
 * import { cg } from 'mdml-engine/numerics/sparse'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(cg([[4, 1], [1, 3]], [1, 2], { rtol: 1e-12 }).x).map((v) => Math.round(v * 1e10) / 1e10) // => [0.0909090909, 0.6363636364]
 * ```
 */
export function cg(
  a: CsrMatrix | TensorLike | LinearOperator,
  b: TensorLike,
  options: KrylovOptions & Partial<Omit<RunOptions, 'maxSteps'>> & { readonly maxIterations?: number } = {},
): KrylovResult {
  const { x0, rtol, atol, preconditioner, ...run } = options
  const n = asOperator(a).shape[0]
  return finish(
    cgAlgorithm(a, b, {
      ...(x0 === undefined ? {} : { x0 }),
      ...(rtol === undefined ? {} : { rtol }),
      ...(atol === undefined ? {} : { atol }),
      ...(preconditioner === undefined ? {} : { preconditioner }),
    }),
    run,
    n,
  )
}

/**
 * BiCGSTAB (van der Vorst) for general non-symmetric systems, as a traced algorithm.
 *
 * @param a - The matrix or operator.
 * @param b - Right-hand side.
 * @param options - Start, tolerances and (right) preconditioner.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { bicgstabAlgorithm } from 'mdml-engine/numerics/sparse'
 * bicgstabAlgorithm([[4, 1], [2, 3]], [1, 2]).name // => 'bicgstab'
 * ```
 */
export function bicgstabAlgorithm(
  a: CsrMatrix | TensorLike | LinearOperator,
  b: TensorLike,
  options: KrylovOptions = {},
): Algorithm<KrylovState> {
  const { apply, b: bv, x0, tol, precondition } = setup(a, b, options)
  return {
    name: 'bicgstab',
    config: { rtol: options.rtol ?? 1e-5, atol: options.atol ?? 0 } as Json,
    schema: SCHEMA,
    init: () => {
      const r = daxpy(-1, apply(x0), bv)
      const rn = norm(r)
      return {
        x: x0,
        residualNorm: rn,
        history: [rn],
        internal: {
          r,
          rhat: Float64Array.from(r),
          p: new Float64Array(r.length),
          v: new Float64Array(r.length),
          rho: 1,
          alpha: 1,
          omega: 1,
        },
      }
    },
    step: (s) => {
      const st = s.internal as {
        r: Float64Array
        rhat: Float64Array
        p: Float64Array
        v: Float64Array
        rho: number
        alpha: number
        omega: number
      }
      const rho = dot(st.rhat, st.r)
      if (rho === 0) return { ...s, residualNorm: Number.NaN }
      const beta = (rho / st.rho) * (st.alpha / st.omega)
      const p = st.r.map((ri, i) => ri + beta * ((st.p[i] ?? 0) - st.omega * (st.v[i] ?? 0)))
      const phat = precondition(p)
      const v = apply(phat)
      const alpha = rho / dot(st.rhat, v)
      const sv = daxpy(-alpha, v, st.r)
      let x = daxpy(alpha, phat, s.x)
      if (norm(sv) <= tol) {
        const rn = norm(sv)
        return { x, residualNorm: rn, history: [...s.history, rn], internal: { ...st, r: sv, p, v, rho, alpha } }
      }
      const shat = precondition(sv)
      const t = apply(shat)
      const omega = dot(t, sv) / dot(t, t)
      x = daxpy(omega, shat, x)
      const r = daxpy(-omega, t, sv)
      const rn = norm(r)
      return {
        x,
        residualNorm: rn,
        history: [...s.history, rn],
        internal: { r, rhat: st.rhat, p, v, rho, alpha, omega },
      }
    },
    done: (s) => (s.residualNorm <= tol ? 'converged' : Number.isFinite(s.residualNorm) ? undefined : 'diverged'),
    observe,
  }
}

/**
 * Solve A x = b by BiCGSTAB (scipy.sparse.linalg.bicgstab).
 *
 * @param a - The matrix or operator.
 * @param b - Right-hand side.
 * @param options - Tolerances, start, preconditioner, iteration cap and trace options.
 * @returns The solution with its residual history.
 * @example
 * ```ts
 * import { bicgstab } from 'mdml-engine/numerics/sparse'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(bicgstab([[4, 1], [2, 3]], [1, 2], { rtol: 1e-12 }).x).map((v) => Math.round(v * 1e10) / 1e10) // => [0.1, 0.6]
 * ```
 */
export function bicgstab(
  a: CsrMatrix | TensorLike | LinearOperator,
  b: TensorLike,
  options: KrylovOptions & Partial<Omit<RunOptions, 'maxSteps'>> & { readonly maxIterations?: number } = {},
): KrylovResult {
  const { x0, rtol, atol, preconditioner, ...run } = options
  return finish(
    bicgstabAlgorithm(a, b, {
      ...(x0 === undefined ? {} : { x0 }),
      ...(rtol === undefined ? {} : { rtol }),
      ...(atol === undefined ? {} : { atol }),
      ...(preconditioner === undefined ? {} : { preconditioner }),
    }),
    run,
    asOperator(a).shape[0],
  )
}

/**
 * Restarted GMRES (Saad and Schultz) for general systems: each step extends the Arnoldi basis by one vector and updates
 * the least-squares residual by Givens rotations; every `restart` steps (default 20, as scipy) the iterate is formed
 * and the basis restarted. Left preconditioning.
 *
 * @param a - The matrix or operator.
 * @param b - Right-hand side.
 * @param options - Start, tolerances, preconditioner and restart length.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { gmresAlgorithm } from 'mdml-engine/numerics/sparse'
 * gmresAlgorithm([[4, 1], [2, 3]], [1, 2]).name // => 'gmres'
 * ```
 */
export function gmresAlgorithm(
  a: CsrMatrix | TensorLike | LinearOperator,
  b: TensorLike,
  options: KrylovOptions & { readonly restart?: number } = {},
): Algorithm<KrylovState> {
  const { apply: rawApply, b: rawB, x0, tol: rawTol, precondition } = setup(a, b, options)
  const apply = (x: Float64Array): Float64Array => precondition(rawApply(x))
  const bv = precondition(rawB)
  const tol =
    options.preconditioner === undefined ? rawTol : Math.max((options.rtol ?? 1e-5) * norm(bv), options.atol ?? 0)
  const m = Math.min(options.restart ?? 20, rawB.length)
  type Cycle = { V: Float64Array[]; H: number[][]; cs: number[]; sn: number[]; g: number[]; base: Float64Array }
  const begin = (x: Float64Array): { cycle: Cycle; rn: number } => {
    const r = daxpy(-1, apply(x), bv)
    const beta = norm(r)
    return { cycle: { V: [r.map((v) => v / (beta || 1))], H: [], cs: [], sn: [], g: [beta], base: x }, rn: beta }
  }
  // The iterate from the first j columns: x = base + V_j y with H_j y = g_j (upper triangular after the rotations).
  const solution = (c: Cycle): Float64Array => {
    const j = c.H.length
    const y = new Array<number>(j).fill(0)
    for (let i = j - 1; i >= 0; i--) {
      let s = c.g[i] ?? 0
      for (let k = i + 1; k < j; k++) s -= (c.H[k]?.[i] ?? 0) * (y[k] ?? 0)
      y[i] = s / (c.H[i]?.[i] ?? 1)
    }
    let x = c.base
    y.forEach((yi, i) => {
      x = daxpy(yi, c.V[i] as Float64Array, x)
    })
    return x
  }
  return {
    name: 'gmres',
    config: { rtol: options.rtol ?? 1e-5, atol: options.atol ?? 0, restart: m } as Json,
    schema: SCHEMA,
    init: () => {
      const { cycle, rn } = begin(x0)
      return { x: x0, residualNorm: rn, history: [rn], internal: { cycle } }
    },
    step: (s) => {
      const c = (s.internal as { cycle: Cycle }).cycle
      const j = c.H.length
      // Arnoldi with modified Gram-Schmidt: column j of H.
      let w = apply(c.V[j] as Float64Array)
      const h: number[] = []
      for (let i = 0; i <= j; i++) {
        h[i] = dot(w, c.V[i] as Float64Array)
        w = daxpy(-(h[i] ?? 0), c.V[i] as Float64Array, w)
      }
      const hn = norm(w)
      h[j + 1] = hn
      // Apply the previous rotations, then a new one to zero h[j + 1].
      for (let i = 0; i < j; i++) {
        const t = (c.cs[i] ?? 1) * (h[i] ?? 0) + (c.sn[i] ?? 0) * (h[i + 1] ?? 0)
        h[i + 1] = -(c.sn[i] ?? 0) * (h[i] ?? 0) + (c.cs[i] ?? 1) * (h[i + 1] ?? 0)
        h[i] = t
      }
      const r = Math.hypot(h[j] ?? 0, h[j + 1] ?? 0)
      const cs = r === 0 ? 1 : (h[j] ?? 0) / r
      const sn = r === 0 ? 0 : (h[j + 1] ?? 0) / r
      h[j] = r
      h[j + 1] = 0
      const g = [...c.g]
      g[j + 1] = -sn * (g[j] ?? 0)
      g[j] = cs * (g[j] ?? 0)
      const cycle: Cycle = {
        V: [...c.V, w.map((v) => v / (hn || 1))],
        H: [...c.H, h],
        cs: [...c.cs, cs],
        sn: [...c.sn, sn],
        g,
        base: c.base,
      }
      const rn = Math.abs(g[j + 1] ?? 0)
      const x = solution(cycle)
      if (rn <= tol || hn === 0 || j + 1 >= m) {
        // Restart from the current iterate (its true residual replaces the estimate).
        const next = begin(x)
        return { x, residualNorm: next.rn, history: [...s.history, next.rn], internal: { cycle: next.cycle } }
      }
      return { x, residualNorm: rn, history: [...s.history, rn], internal: { cycle } }
    },
    done: (s) => (s.residualNorm <= tol ? 'converged' : Number.isFinite(s.residualNorm) ? undefined : 'diverged'),
    observe,
  }
}

/**
 * Solve A x = b by restarted GMRES (scipy.sparse.linalg.gmres).
 *
 * @param a - The matrix or operator.
 * @param b - Right-hand side.
 * @param options - Tolerances, start, preconditioner, restart, iteration cap and trace options.
 * @returns The solution with its residual history.
 * @example
 * ```ts
 * import { gmres } from 'mdml-engine/numerics/sparse'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(gmres([[4, 1], [2, 3]], [1, 2], { rtol: 1e-12 }).x).map((v) => Math.round(v * 1e10) / 1e10) // => [0.1, 0.6]
 * ```
 */
export function gmres(
  a: CsrMatrix | TensorLike | LinearOperator,
  b: TensorLike,
  options: KrylovOptions & { readonly restart?: number } & Partial<Omit<RunOptions, 'maxSteps'>> & {
      readonly maxIterations?: number
    } = {},
): KrylovResult {
  const { x0, rtol, atol, preconditioner, restart, ...run } = options
  return finish(
    gmresAlgorithm(a, b, {
      ...(x0 === undefined ? {} : { x0 }),
      ...(rtol === undefined ? {} : { rtol }),
      ...(atol === undefined ? {} : { atol }),
      ...(preconditioner === undefined ? {} : { preconditioner }),
      ...(restart === undefined ? {} : { restart }),
    }),
    run,
    asOperator(a).shape[0],
  )
}

/**
 * MINRES (Paige and Saunders) for symmetric, possibly indefinite systems: minimises the residual over the Krylov space
 * through the Lanczos recurrence and Givens rotations. The residual norm decreases monotonically.
 *
 * @param a - Symmetric matrix or operator.
 * @param b - Right-hand side.
 * @param options - Start and tolerances.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { minresAlgorithm } from 'mdml-engine/numerics/sparse'
 * minresAlgorithm([[1, 2], [2, -1]], [1, 0]).name // => 'minres'
 * ```
 */
export function minresAlgorithm(
  a: CsrMatrix | TensorLike | LinearOperator,
  b: TensorLike,
  options: KrylovOptions = {},
): Algorithm<KrylovState> {
  const { apply, b: bv, x0, tol } = setup(a, b, options)
  type M = {
    vOld: Float64Array
    v: Float64Array
    beta: number
    wOld: Float64Array
    wOlder: Float64Array
    gamma: number
    gammaOld: number
    sigma: number
    sigmaOld: number
    eta: number
  }
  return {
    name: 'minres',
    config: { rtol: options.rtol ?? 1e-5, atol: options.atol ?? 0 } as Json,
    schema: SCHEMA,
    init: () => {
      const r = daxpy(-1, apply(x0), bv)
      const beta = norm(r)
      const n = r.length
      const zero = new Float64Array(n)
      return {
        x: x0,
        residualNorm: beta,
        history: [beta],
        internal: {
          vOld: zero,
          v: r.map((v) => v / (beta || 1)),
          beta,
          wOld: zero,
          wOlder: zero,
          gamma: 1,
          gammaOld: 1,
          sigma: 0,
          sigmaOld: 0,
          eta: beta,
        } satisfies M,
      }
    },
    step: (s) => {
      const st = s.internal as M
      // Lanczos step: A v_j = beta_j v_{j-1} + alpha_j v_j + beta_{j+1} v_{j+1}.
      const av = apply(st.v)
      const alpha = dot(st.v, av)
      let next = daxpy(-st.beta, st.vOld, daxpy(-alpha, st.v, av))
      const betaNext = norm(next)
      next = next.map((v) => v / (betaNext || 1))
      // Apply the previous two rotations to the new column (beta_j, alpha_j, beta_{j+1}).
      const delta = st.gamma * alpha - st.gammaOld * st.sigma * st.beta
      const rho1 = Math.hypot(delta, betaNext)
      const rho2 = st.sigma * alpha + st.gammaOld * st.gamma * st.beta
      const rho3 = st.sigmaOld * st.beta
      const gammaNext = delta / rho1
      const sigmaNext = betaNext / rho1
      const w = st.v.map((v, i) => (v - rho3 * (st.wOlder[i] ?? 0) - rho2 * (st.wOld[i] ?? 0)) / rho1)
      const x = daxpy(gammaNext * st.eta, w, s.x)
      const rn = Math.abs(sigmaNext) * s.residualNorm
      return {
        x,
        residualNorm: rn,
        history: [...s.history, rn],
        internal: {
          vOld: st.v,
          v: next,
          beta: betaNext,
          wOld: w,
          wOlder: st.wOld,
          gamma: gammaNext,
          gammaOld: st.gamma,
          sigma: sigmaNext,
          sigmaOld: st.sigma,
          eta: -sigmaNext * st.eta,
        } satisfies M,
      }
    },
    done: (s) => (s.residualNorm <= tol ? 'converged' : Number.isFinite(s.residualNorm) ? undefined : 'diverged'),
    observe,
  }
}

/**
 * Solve a symmetric system by MINRES (scipy.sparse.linalg.minres).
 *
 * @param a - Symmetric matrix or operator.
 * @param b - Right-hand side.
 * @param options - Tolerances, start, iteration cap and trace options.
 * @returns The solution with its residual history.
 * @example
 * ```ts
 * import { minres } from 'mdml-engine/numerics/sparse'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(minres([[1, 2], [2, -1]], [1, 0], { rtol: 1e-12 }).x).map((v) => Math.round(v * 1e10) / 1e10) // => [0.2, 0.4]
 * ```
 */
export function minres(
  a: CsrMatrix | TensorLike | LinearOperator,
  b: TensorLike,
  options: KrylovOptions & Partial<Omit<RunOptions, 'maxSteps'>> & { readonly maxIterations?: number } = {},
): KrylovResult {
  const { x0, rtol, atol, preconditioner: _p, ...run } = options
  return finish(
    minresAlgorithm(a, b, {
      ...(x0 === undefined ? {} : { x0 }),
      ...(rtol === undefined ? {} : { rtol }),
      ...(atol === undefined ? {} : { atol }),
    }),
    run,
    asOperator(a).shape[0],
  )
}

/**
 * The Jacobi (diagonal) preconditioner z = r / diag(A).
 *
 * @param a - The matrix.
 * @returns The preconditioner.
 * @example
 * ```ts
 * import { csr, jacobiPreconditioner } from 'mdml-engine/numerics/sparse'
 * Array.from(jacobiPreconditioner(csr([[2, 0], [0, 4]]))(Float64Array.from([2, 2]))) // => [1, 0.5]
 * ```
 */
export function jacobiPreconditioner(a: CsrMatrix | TensorLike): Preconditioner {
  const m = typeof a === 'object' && a !== null && 'indptr' in a ? (a as CsrMatrix) : csr(a as TensorLike)
  const values = primalValues(m)
  const diag = new Float64Array(m.shape[0]).fill(1)
  m.rows.forEach((r, k) => {
    if (m.indices[k] === r) diag[r] = values[k] ?? 1
  })
  return (r) => r.map((v, i) => v / (diag[i] || 1))
}

/**
 * The incomplete Cholesky preconditioner IC(0): L with the sparsity of the lower triangle of A and L L^T close to A
 * on that pattern; z = L^-T L^-1 r. For symmetric positive definite A (it may break down otherwise).
 *
 * @param a - Symmetric positive definite matrix.
 * @returns The preconditioner.
 * @example
 * ```ts
 * import { csr, incompleteCholesky } from 'mdml-engine/numerics/sparse'
 * // For a tridiagonal matrix IC(0) is the exact Cholesky factor, so the preconditioner solves A z = r.
 * Array.from(incompleteCholesky(csr([[4, 1], [1, 3]]))(Float64Array.from([5, 4]))).map((v) => Math.round(v * 1e12) / 1e12) // => [1, 1]
 * ```
 */
export function incompleteCholesky(a: CsrMatrix | TensorLike): Preconditioner {
  const m = typeof a === 'object' && a !== null && 'indptr' in a ? (a as CsrMatrix) : csr(a as TensorLike)
  const n = m.shape[0]
  const values = primalValues(m)
  // Row-wise lower-triangular pattern as maps column -> value.
  const lower: Map<number, number>[] = Array.from({ length: n }, () => new Map())
  m.rows.forEach((r, k) => {
    const c = m.indices[k] ?? 0
    if (c <= r) lower[r]?.set(c, (lower[r]?.get(c) ?? 0) + (values[k] ?? 0))
  })
  for (let i = 0; i < n; i++) {
    const row = lower[i] as Map<number, number>
    const cols = [...row.keys()].sort((p, q) => p - q)
    for (const j of cols) {
      let s = row.get(j) ?? 0
      const rowJ = lower[j] as Map<number, number>
      for (const [k, v] of row) if (k < j && rowJ.has(k)) s -= v * (rowJ.get(k) ?? 0)
      if (j < i) row.set(j, s / (rowJ.get(j) ?? 1))
      else {
        if (!(s > 0))
          throw new MdmlError('value', 'incompleteCholesky: breakdown (the matrix is not positive definite enough)')
        row.set(i, Math.sqrt(s))
      }
    }
  }
  return (r) => {
    const y = new Float64Array(n)
    for (let i = 0; i < n; i++) {
      let s = r[i] ?? 0
      for (const [k, v] of lower[i] as Map<number, number>) if (k < i) s -= v * (y[k] ?? 0)
      y[i] = s / ((lower[i] as Map<number, number>).get(i) ?? 1)
    }
    const z = new Float64Array(n)
    for (let i = n - 1; i >= 0; i--) {
      let s = y[i] ?? 0
      // L^T: column i of L is row i of L^T.
      for (let j = i + 1; j < n; j++) {
        const v = (lower[j] as Map<number, number>).get(i)
        if (v !== undefined) s -= v * (z[j] ?? 0)
      }
      z[i] = s / ((lower[i] as Map<number, number>).get(i) ?? 1)
    }
    return z
  }
}

/** The Krylov method used by {@link sparseSolve}. */
export type SolveMethod = 'cg' | 'bicgstab' | 'gmres' | 'minres'

/**
 * Solve A x = b for a sparse A, differentiably in A's values and b: the forward solve is iterative; the vjp solves the
 * adjoint system A^T lambda = x_bar (same method), giving b_bar = lambda and A_bar_ij = -lambda_i x_j on the pattern;
 * the jvp solves A dx = db - dA x.
 *
 * @param a - The CSR matrix.
 * @param b - Right-hand side.
 * @param options - Method (default gmres, or cg for symmetric positive definite A) and tolerances (default rtol 1e-12).
 * @returns The solution.
 * @throws MdmlError with code `convergence` when the iteration does not converge: the return is a bare value, with no
 *   place for a status (the Krylov solvers `cg`, `gmres`, `minres` and `bicgstab` return one).
 * @concept implicit-differentiation
 * @example
 * ```ts
 * import { csr, sparseSolve } from 'mdml-engine/numerics/sparse'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(sparseSolve(csr([[2, 0], [0, 4]]), [2, 2])) // => [1, 0.5]
 * ```
 */
export function sparseSolve(
  a: CsrMatrix,
  b: TensorLike,
  options: {
    readonly method?: SolveMethod
    readonly rtol?: number
    readonly atol?: number
    readonly maxIterations?: number
  } = {},
): Value {
  const method = options.method ?? 'gmres'
  const solveWith = (m: CsrMatrix, rhs: Float64Array): Float64Array => {
    const opts = {
      rtol: options.rtol ?? 1e-12,
      atol: options.atol ?? 0,
      maxIterations: options.maxIterations ?? 20 * m.shape[0],
    }
    const f = method === 'cg' ? cg : method === 'bicgstab' ? bicgstab : method === 'minres' ? minres : gmres
    const r = f(m, rhs, opts)
    if (!r.converged)
      throw new MdmlError('convergence', `sparseSolve: ${method} did not converge (residual ${r.residualNorm})`)
    return Float64Array.from(r.x.data)
  }
  const [bv] = promoteFloat(b) as [Value]
  const structure = (values: Value): CsrMatrix => ({ ...a, values })
  const op = customVjp({
    name: 'sparseSolve',
    forward: (values: Value, rhs: Value) => {
      const x = solveWith(structure(primalOf(values)), Float64Array.from(toFlat(primalOf(rhs))))
      return fromStorage(x, [x.length], 'float64')
    },
    vjp: (g, [values], out) => {
      const at = transposeCsr(structure(primalOf(values as Value)))
      const lambda = fromStorage(solveWith(at, Float64Array.from(toFlat(primalOf(g)))), [a.shape[0]], 'float64')
      const rowsIdx = fromStorage(a.rows, [a.rows.length], 'int32')
      const colsIdx = fromStorage(a.indices, [a.indices.length], 'int32')
      const valuesBar = neg(mul(take(lambda, rowsIdx, 0), take(out, colsIdx, 0)))
      return [valuesBar, lambda]
    },
    jvp: ([values, rhs], [dvalues, drhs], out) => {
      if ((dvalues === null || dvalues === undefined) && (drhs === null || drhs === undefined)) return null
      let rhsTangent = Float64Array.from(toFlat(primalOf(drhs ?? mul(rhs as Value, 0))))
      if (dvalues !== null && dvalues !== undefined) {
        const dax = toFlat(primalOf(spmv(structure(primalOf(dvalues)), primalOf(out))))
        rhsTangent = rhsTangent.map((v, i) => v - (dax[i] ?? 0))
      }
      const dx = solveWith(structure(primalOf(values as Value)), rhsTangent)
      return fromStorage(dx, [dx.length], 'float64')
    },
  })
  return op(a.values, bv)
}
