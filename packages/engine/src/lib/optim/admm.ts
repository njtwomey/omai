/**
 * The alternating direction method of multipliers (ADMM, Boyd et al.): splitting min f(x) + g(z) subject to A x + B z =
 * c into alternating partial minimisations and a dual update, as a traced algorithm. Also the proximal
 * (Douglas-Rachford) form, a lasso solver with a cached Cholesky factor, and global consensus ADMM for sums of terms.
 */

import { type Json, MdmlError } from '../base/index.ts'
import { cholesky, choSolve, daxpy, dgemv, dnrm2, dscal } from '../numerics/linalg/index.ts'
import {
  add,
  asTensor,
  eye,
  fromStorage,
  matmul,
  mul,
  primalOf,
  swapaxes,
  type Tensor,
  type TensorLike,
  toFlat,
  toVector,
  type Value,
} from '../tensor/index.ts'
import { type Algorithm, channels, type RunOptions, type RunStatus, runAlgorithm } from '../trace/index.ts'
import { flatOf } from './_vector.ts'
import { type Prox, proxL1 } from './proximal.ts'

/**
 * The problem min f(x) + g(z) subject to A x + B z = c, given by its two partial minimisations (scaled form, u the
 * scaled dual variable). A defaults to the identity, B to minus the identity and c to zero (the splitting x = z). The
 * two updates are callbacks on plain arrays, run once per iteration (D-009 keeps them native; `admmProx` and `admmLasso`
 * hand them tensors); `c` takes a tensor-like (migration, D-009: was `ArrayLike<number>`).
 */
export interface AdmmProblem {
  /** argmin_x f(x) + (rho / 2) |A x + B z - c + u|^2. */
  readonly xUpdate: (z: Float64Array, u: Float64Array, rho: number) => ArrayLike<number>
  /** argmin_z g(z) + (rho / 2) |A x + B z - c + u|^2, where `ax` is A x (over-relaxed when requested). */
  readonly zUpdate: (ax: Float64Array, u: Float64Array, rho: number, x: Float64Array) => ArrayLike<number>
  /** The constraint matrices `[p, n]` and `[p, m]`, and c `[p]`. */
  readonly A?: TensorLike
  readonly B?: TensorLike
  readonly c?: TensorLike
  /** f(x) + g(z), for the trace. */
  readonly objective?: (x: Float64Array, z: Float64Array) => number
}

/**
 * Options of {@link admmAlgorithm}. The starting points are tensor-likes (migration, D-009: they were
 * `ArrayLike<number>`).
 */
export interface AdmmOptions {
  readonly x0?: TensorLike
  readonly z0?: TensorLike
  readonly u0?: TensorLike
  /** The penalty (default 1). */
  readonly rho?: number
  /** Residual balancing (He et al. 2000): scale rho by `tauIncr` when one residual exceeds `mu` times the other. */
  readonly adaptiveRho?: boolean
  readonly mu?: number
  readonly tauIncr?: number
  /** Over-relaxation alpha in (0, 2) (default 1, none). */
  readonly relaxation?: number
  /** Boyd et al.'s stopping tolerances (default 1e-9 each). */
  readonly absTol?: number
  readonly relTol?: number
}

/** State of {@link admmAlgorithm}. */
export interface AdmmState {
  readonly x: Float64Array
  readonly z: Float64Array
  readonly u: Float64Array
  readonly rho: number
  readonly primalResidual: number
  readonly dualResidual: number
  readonly epsPrimal: number
  readonly epsDual: number
  readonly objective: number
}

/** Result of the ADMM solvers. */
export interface AdmmResult {
  readonly x: Tensor
  readonly z: Tensor
  /** The scaled dual variable (the unscaled multiplier is rho u). */
  readonly u: Tensor
  readonly rho: number
  readonly objective: number
  readonly primalResidual: number
  readonly dualResidual: number
  readonly iterations: number
  readonly status: RunStatus
  readonly converged: boolean
}

interface Dense {
  readonly data: Float64Array | null
  readonly rows: number
  readonly cols: number
  readonly scale: number
}

function dense(a: TensorLike | undefined, rows: number, cols: number, scale: number): Dense {
  if (a === undefined) return { data: null, rows, cols, scale }
  const t = primalOf(asTensor(a))
  if (t.shape.length !== 2) throw new MdmlError('shape', 'admm: constraint matrices must be 2-D')
  return { data: Float64Array.from(t.data), rows: t.shape[0] ?? 0, cols: t.shape[1] ?? 0, scale: 1 }
}

/** A starting vector as a fresh array: the tensor-like when given, else the fallback. */
const startOf = (v: TensorLike | undefined, fallback: ArrayLike<number>): Float64Array =>
  Float64Array.from(v === undefined ? fallback : toVector(v, 'admm'))

const apply = (m: Dense, x: Float64Array, transpose = false): Float64Array =>
  m.data === null ? dscal(m.scale, x) : dgemv(m.data, m.rows, m.cols, x, transpose)

const SCHEMA = channels({
  metrics: {
    objective: { description: 'f(x) + g(z)', direction: 'lower' },
    primalResidual: { description: '|A x + B z - c|' },
    dualResidual: { description: '|rho A^T B (z - z_prev)|' },
    rho: { description: 'penalty parameter' },
  },
  tensors: {
    x: { role: 'param', description: 'x iterate' },
    z: { role: 'state', description: 'z iterate' },
    u: { role: 'state', description: 'scaled dual variable' },
  },
})

/**
 * The alternating direction method of multipliers (Boyd et al. 2011) in scaled form, as a traced algorithm:
 * x+ = argmin_x L(x, z, u), z+ = argmin_z L(x+, z, u), u+ = u + A x+ + B z+ - c, with optional over-relaxation and
 * residual-balancing penalty updates, stopping on Boyd's primal and dual residual tests.
 *
 * @param problem - The partial minimisations and the constraint.
 * @param options - Starting points (vectors; sizes from x0 and z0), penalty, adaptation, relaxation and tolerances.
 * @returns The algorithm.
 * @throws MdmlError with code `shape` unless a starting point or `c` has rank 1, and `not-differentiable` for a traced one.
 * @concept admm
 * @example
 * ```ts
 * import { admmAlgorithm } from 'mdml-engine/optim'
 * const alg = admmAlgorithm({ xUpdate: (z, u) => z.map((v, i) => v - (u[i] ?? 0)), zUpdate: (ax, u) => ax.map((v, i) => v + (u[i] ?? 0)) }, { x0: [0] })
 * alg.name // => 'admm'
 * ```
 */
export function admmAlgorithm(problem: AdmmProblem, options: AdmmOptions = {}): Algorithm<AdmmState> {
  const x0 = startOf(options.x0 ?? options.z0, [])
  const z0 = startOf(options.z0, x0)
  const A = dense(problem.A, z0.length, x0.length, 1)
  const B = dense(problem.B, A.rows, z0.length, -1)
  const p = A.data === null ? (B.data === null ? x0.length : B.rows) : A.rows
  const c = startOf(problem.c, new Float64Array(p))
  const u0 = startOf(options.u0, new Float64Array(p))
  const absTol = options.absTol ?? 1e-9
  const relTol = options.relTol ?? 1e-9
  const alpha = options.relaxation ?? 1
  const mu = options.mu ?? 10
  const tauIncr = options.tauIncr ?? 2
  const objective = (x: Float64Array, z: Float64Array): number => problem.objective?.(x, z) ?? Number.NaN
  return {
    name: 'admm',
    config: {
      rho: options.rho ?? 1,
      adaptiveRho: options.adaptiveRho ?? false,
      relaxation: alpha,
      absTol,
      relTol,
    } as Json,
    schema: SCHEMA,
    init: () => ({
      x: x0,
      z: z0,
      u: u0,
      rho: options.rho ?? 1,
      primalResidual: Infinity,
      dualResidual: Infinity,
      epsPrimal: 0,
      epsDual: 0,
      objective: objective(x0, z0),
    }),
    step: (s) => {
      const rho = s.rho
      const x = Float64Array.from(problem.xUpdate(s.z, s.u, rho))
      const ax = apply(A, x)
      const bzOld = apply(B, s.z)
      // Over-relaxation: replace A x by alpha A x - (1 - alpha) (B z - c).
      const axHat =
        alpha === 1 ? ax : Float64Array.from(ax, (v, i) => alpha * v - (1 - alpha) * ((bzOld[i] ?? 0) - (c[i] ?? 0)))
      const z = Float64Array.from(problem.zUpdate(axHat, s.u, rho, x))
      const bz = apply(B, z)
      let u: Float64Array = Float64Array.from(s.u, (v, i) => v + (axHat[i] ?? 0) + (bz[i] ?? 0) - (c[i] ?? 0))
      const r = Float64Array.from(ax, (v, i) => v + (bz[i] ?? 0) - (c[i] ?? 0))
      const sDual = dscal(rho, apply(A, daxpy(-1, bzOld, bz), true))
      const primalResidual = dnrm2(r)
      const dualResidual = dnrm2(sDual)
      const epsPrimal = Math.sqrt(p) * absTol + relTol * Math.max(dnrm2(ax), dnrm2(bz), dnrm2(c))
      const epsDual = Math.sqrt(x.length) * absTol + relTol * dnrm2(dscal(rho, apply(A, u, true)))
      let next = rho
      if (options.adaptiveRho === true) {
        if (primalResidual > mu * dualResidual) next = rho * tauIncr
        else if (dualResidual > mu * primalResidual) next = rho / tauIncr
        if (next !== rho) u = dscal(rho / next, u)
      }
      return { x, z, u, rho: next, primalResidual, dualResidual, epsPrimal, epsDual, objective: objective(x, z) }
    },
    done: (s) =>
      s.primalResidual <= s.epsPrimal && s.dualResidual <= s.epsDual
        ? 'converged'
        : Number.isNaN(s.primalResidual)
          ? 'diverged'
          : undefined,
    observe: (s) => ({
      metrics: {
        objective: s.objective,
        primalResidual: s.primalResidual,
        dualResidual: s.dualResidual,
        rho: s.rho,
      },
      tensors: {
        x: { role: 'param', value: s.x },
        z: { role: 'state', value: s.z },
        u: { role: 'state', value: s.u },
      },
    }),
  }
}

function finish(
  algorithm: Algorithm<AdmmState>,
  run: Partial<Omit<RunOptions, 'maxSteps'>> & { readonly maxIterations?: number },
  shapeX: readonly number[],
  shapeZ: readonly number[],
): AdmmResult {
  const { maxIterations, ...rest } = run
  const result = runAlgorithm(algorithm, { ...rest, maxSteps: maxIterations ?? 10000 })
  const s = result.state
  return {
    x: fromStorage(s.x, shapeX, 'float64'),
    z: fromStorage(s.z, shapeZ, 'float64'),
    u: fromStorage(s.u, [s.u.length], 'float64'),
    rho: s.rho,
    objective: s.objective,
    primalResidual: s.primalResidual,
    dualResidual: s.dualResidual,
    iterations: result.steps,
    status: result.status,
    converged: result.status === 'converged',
  }
}

/** Run options of the ADMM convenience functions. */
export type AdmmRunOptions = AdmmOptions &
  Partial<Omit<RunOptions, 'maxSteps' | 'start'>> & { readonly maxIterations?: number }

function splitRun(options: AdmmRunOptions): {
  admm: AdmmOptions
  run: Partial<Omit<RunOptions, 'maxSteps'>> & { readonly maxIterations?: number }
} {
  const { x0, z0, u0, rho, adaptiveRho, mu, tauIncr, relaxation, absTol, relTol, ...run } = options
  const admm: AdmmOptions = Object.fromEntries(
    Object.entries({ x0, z0, u0, rho, adaptiveRho, mu, tauIncr, relaxation, absTol, relTol }).filter(
      ([, v]) => v !== undefined,
    ),
  )
  return { admm, run }
}

/**
 * Solve min f(x) + g(z) subject to A x + B z = c by ADMM.
 *
 * @param problem - The partial minimisations and the constraint.
 * @param options - Starting points, penalty, adaptation, relaxation, tolerances, iteration cap (default 10000).
 * @returns x, z, the scaled dual u and the residuals.
 * @concept admm
 * @index function Alternating direction method of multipliers.
 * @example
 * ```ts
 * import { admm } from 'mdml-engine/optim'
 * import { toArray } from 'mdml-engine/tensor'
 * // min (x - 1)^2 / 2 + (z + 1)^2 / 2 subject to x = z: both meet at 0.
 * const r = admm({
 *   xUpdate: (z, u, rho) => z.map((v, i) => (1 + rho * (v - (u[i] ?? 0))) / (1 + rho)),
 *   zUpdate: (ax, u, rho) => ax.map((v, i) => (-1 + rho * (v + (u[i] ?? 0))) / (1 + rho)),
 * }, { x0: [0] })
 * toArray(r.x).map((v) => Math.round(v * 1e8) / 1e8) // => [0]
 * ```
 */
export function admm(problem: AdmmProblem, options: AdmmRunOptions = {}): AdmmResult {
  const { admm: o, run } = splitRun(options)
  const n = startOf(options.x0 ?? options.z0, []).length
  const m = startOf(options.z0 ?? options.x0, []).length
  return finish(admmAlgorithm(problem, o), run, [n], [m])
}

/**
 * min f(x) + g(x) by ADMM on the splitting x = z, from the two proximal operators:
 * x+ = prox_{f / rho}(z - u), z+ = prox_{g / rho}(x+ + u) (Douglas-Rachford splitting).
 *
 * @param proxF - The proximal operator of f.
 * @param proxG - The proximal operator of g.
 * @param x0 - The starting point (any shape; the operators see that shape).
 * @param options - Penalty, adaptation, relaxation, tolerances, iteration cap and run options.
 * @returns The result (x and z agree at convergence).
 * @concept admm
 * @example
 * ```ts
 * import { admmProx, proxL1, proxSquaredL2 } from 'mdml-engine/optim'
 * import { add, sub, toArray } from 'mdml-engine/tensor'
 * // min |x - 3|^2 / 2 + |x|_1: the soft threshold of 3 at 1.
 * const r = admmProx((v, t) => add(proxSquaredL2(sub(v, 3), t), 3), proxL1, [0])
 * toArray(r.z).map((v) => Math.round(v * 1e8) / 1e8) // => [2]
 * ```
 */
export function admmProx(proxF: Prox, proxG: Prox, x0: TensorLike, options: AdmmRunOptions = {}): AdmmResult {
  const start = primalOf(asTensor(x0))
  const shape = start.shape
  const shaped = (v: ArrayLike<number>): Value => fromStorage(Float64Array.from(v), shape, 'float64')
  const { admm: o, run } = splitRun({ ...options, x0: options.x0 ?? Float64Array.from(start.data) })
  const problem: AdmmProblem = {
    xUpdate: (z, u, rho) => flatOf(proxF(shaped(daxpy(-1, u, z)), 1 / rho)),
    zUpdate: (ax, u, rho) => flatOf(proxG(shaped(daxpy(1, u, ax)), 1 / rho)),
  }
  return finish(admmAlgorithm(problem, o), run, shape, shape)
}

/** Options of {@link admmLasso}. */
export type AdmmLassoOptions = AdmmRunOptions

/**
 * The lasso min |A x - b|^2 / 2 + lambda |x|_1 by ADMM: the x-update solves (A^T A + rho I) x = A^T b + rho (z - u)
 * with one cached Cholesky factorisation, the z-update soft-thresholds.
 *
 * @param a - The design `[m, n]`.
 * @param b - The response `[m]`.
 * @param lambda - The L1 weight.
 * @param options - Penalty, adaptation, relaxation, tolerances, iteration cap and run options.
 * @returns The result; `z` is the sparse solution.
 * @throws MdmlError with code `value` when `lambda` is negative or not finite.
 * @concept admm
 * @example
 * ```ts
 * import { admmLasso } from 'mdml-engine/optim'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(admmLasso([[1, 0], [0, 1]], [3, 0.5], 1).z).map((v) => Math.round(v * 1e6) / 1e6) // => [2, 0]
 * ```
 */
export function admmLasso(a: TensorLike, b: TensorLike, lambda: number, options: AdmmLassoOptions = {}): AdmmResult {
  if (!(lambda >= 0) || !Number.isFinite(lambda))
    throw new MdmlError('value', `admmLasso: lambda must be finite and non-negative, got ${lambda}`)
  const av = primalOf(asTensor(a))
  const n = av.shape[1] ?? 0
  const at = Float64Array.from(av.data)
  const rows = av.shape[0] ?? 0
  const atb = dgemv(at, rows, n, toFlat(primalOf(asTensor(b))), true)
  const gram = matmul(swapaxes(av, 0, 1), av)
  const factors = new Map<number, Value>()
  const factorOf = (rho: number): Value => {
    let l = factors.get(rho)
    if (l === undefined) {
      l = cholesky(add(gram, mul(eye(n), rho)))
      factors.set(rho, l)
    }
    return l
  }
  const vec = (v: ArrayLike<number>): Value => fromStorage(Float64Array.from(v), [v.length], 'float64')
  const { admm: o, run } = splitRun({ ...options, x0: options.x0 ?? new Float64Array(n) })
  const problem: AdmmProblem = {
    xUpdate: (z, u, rho) =>
      flatOf(choSolve(factorOf(rho), vec(Float64Array.from(atb, (v, i) => v + rho * ((z[i] ?? 0) - (u[i] ?? 0)))))),
    zUpdate: (ax, u, rho) => flatOf(proxL1(vec(daxpy(1, u, ax)), lambda / rho)),
    objective: (_x, z) => {
      const r = dgemv(at, rows, n, z)
      const bb = toFlat(primalOf(asTensor(b)))
      let sq = 0
      for (let i = 0; i < rows; i++) sq += ((r[i] ?? 0) - (bb[i] ?? 0)) ** 2
      let l1 = 0
      for (const v of z) l1 += Math.abs(v)
      return sq / 2 + lambda * l1
    },
  }
  return finish(admmAlgorithm(problem, o), run, [n], [n])
}

/** State of {@link consensusAdmmAlgorithm}. */
export interface ConsensusState {
  readonly xs: readonly Float64Array[]
  readonly us: readonly Float64Array[]
  readonly z: Float64Array
  readonly primalResidual: number
  readonly dualResidual: number
}

/**
 * Global consensus ADMM (Boyd et al. 2011, section 7.1) for min sum_i f_i(x) + g(x): each term keeps its own copy,
 * x_i+ = prox_{f_i / rho}(z - u_i), z+ = prox_{g / (N rho)}(mean(x_i+ + u_i)), u_i+ = u_i + x_i+ - z+. The terms'
 * updates are independent (the distributed form).
 *
 * @param proxes - The proximal operators of the f_i.
 * @param x0 - The starting point (any shape).
 * @param options - The proximal operator of g, penalty and tolerances.
 * @returns The algorithm.
 * @concept admm
 * @example
 * ```ts
 * import { consensusAdmmAlgorithm, proxSquaredL2 } from 'mdml-engine/optim'
 * consensusAdmmAlgorithm([proxSquaredL2], [0]).name // => 'consensusAdmm'
 * ```
 */
export function consensusAdmmAlgorithm(
  proxes: readonly Prox[],
  x0: TensorLike,
  options: { readonly proxG?: Prox; readonly rho?: number; readonly absTol?: number; readonly relTol?: number } = {},
): Algorithm<ConsensusState> {
  const start = primalOf(asTensor(x0))
  const shape = start.shape
  const n = start.data.length
  const N = proxes.length
  if (N === 0) throw new MdmlError('value', 'consensusAdmm: give at least one term')
  const rho = options.rho ?? 1
  const absTol = options.absTol ?? 1e-9
  const relTol = options.relTol ?? 1e-9
  const shaped = (v: ArrayLike<number>): Value => fromStorage(Float64Array.from(v), shape, 'float64')
  return {
    name: 'consensusAdmm',
    config: { rho, terms: N, absTol, relTol } as Json,
    schema: channels({
      metrics: {
        primalResidual: { description: 'sqrt(sum_i |x_i - z|^2)' },
        dualResidual: { description: 'sqrt(N) rho |z - z_prev|' },
      },
      tensors: { z: { role: 'param', description: 'consensus variable' } },
    }),
    init: () => {
      const z = Float64Array.from(start.data)
      return {
        xs: proxes.map(() => Float64Array.from(z)),
        us: proxes.map(() => new Float64Array(n)),
        z,
        primalResidual: Infinity,
        dualResidual: Infinity,
      }
    },
    step: (s) => {
      const xs = proxes.map((prox, i) => flatOf(prox(shaped(daxpy(-1, s.us[i] as Float64Array, s.z)), 1 / rho)))
      const mean = new Float64Array(n)
      for (const [i, x] of xs.entries())
        for (let j = 0; j < n; j++) mean[j] = (mean[j] ?? 0) + ((x[j] ?? 0) + ((s.us[i] as Float64Array)[j] ?? 0)) / N
      const z = options.proxG === undefined ? mean : flatOf(options.proxG(shaped(mean), 1 / (N * rho)))
      const us = xs.map((x, i) => Float64Array.from(s.us[i] as Float64Array, (v, j) => v + (x[j] ?? 0) - (z[j] ?? 0)))
      let pr = 0
      for (const x of xs) pr += dnrm2(daxpy(-1, z, x)) ** 2
      return { xs, us, z, primalResidual: Math.sqrt(pr), dualResidual: Math.sqrt(N) * rho * dnrm2(daxpy(-1, s.z, z)) }
    },
    done: (s) => {
      let xn = 0
      for (const x of s.xs) xn += dnrm2(x) ** 2
      let un = 0
      for (const u of s.us) un += dnrm2(u) ** 2
      const epsP = Math.sqrt(n * N) * absTol + relTol * Math.max(Math.sqrt(xn), Math.sqrt(N) * dnrm2(s.z))
      const epsD = Math.sqrt(n * N) * absTol + relTol * rho * Math.sqrt(un)
      return s.primalResidual <= epsP && s.dualResidual <= epsD ? 'converged' : undefined
    },
    observe: (s) => ({
      metrics: { primalResidual: s.primalResidual, dualResidual: s.dualResidual },
      tensors: { z: { role: 'param', value: s.z, shape } },
    }),
  }
}

/**
 * Solve min sum_i f_i(x) + g(x) by global consensus ADMM.
 *
 * @param proxes - The proximal operators of the f_i.
 * @param x0 - The starting point.
 * @param options - The proximal operator of g, penalty, tolerances, iteration cap (default 10000) and run options.
 * @returns The consensus solution z (also as x), with the residuals.
 * @concept admm
 * @example
 * ```ts
 * import { consensusAdmm, type Prox } from 'mdml-engine/optim'
 * import { add, div, toArray } from 'mdml-engine/tensor'
 * // Terms (x - a_i)^2 / 2 for a = 1, 2, 6: the consensus is their mean.
 * const proxAt = (a: number): Prox => (v, t) => div(add(v, a * t), 1 + t)
 * toArray(consensusAdmm([proxAt(1), proxAt(2), proxAt(6)], [0]).z).map((v) => Math.round(v * 1e8) / 1e8) // => [3]
 * ```
 */
export function consensusAdmm(
  proxes: readonly Prox[],
  x0: TensorLike,
  options: {
    readonly proxG?: Prox
    readonly rho?: number
    readonly absTol?: number
    readonly relTol?: number
  } & Partial<Omit<RunOptions, 'maxSteps' | 'start'>> & { readonly maxIterations?: number } = {},
): AdmmResult {
  const { proxG, rho, absTol, relTol, maxIterations, ...run } = options
  const alg = consensusAdmmAlgorithm(proxes, x0, {
    ...(proxG === undefined ? {} : { proxG }),
    ...(rho === undefined ? {} : { rho }),
    ...(absTol === undefined ? {} : { absTol }),
    ...(relTol === undefined ? {} : { relTol }),
  })
  const result = runAlgorithm(alg, { ...run, maxSteps: maxIterations ?? 10000 })
  const s = result.state
  const shape = primalOf(asTensor(x0)).shape
  return {
    x: fromStorage(Float64Array.from(s.z), shape, 'float64'),
    z: fromStorage(Float64Array.from(s.z), shape, 'float64'),
    u: fromStorage(Float64Array.from(s.us.flatMap((u) => Array.from(u))), [s.us.length, s.z.length], 'float64'),
    rho: rho ?? 1,
    objective: Number.NaN,
    primalResidual: s.primalResidual,
    dualResidual: s.dualResidual,
    iterations: result.steps,
    status: result.status,
    converged: result.status === 'converged',
  }
}
