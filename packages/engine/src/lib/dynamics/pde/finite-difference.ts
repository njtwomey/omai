/**
 * Time-dependent and steady PDEs solved by finite differences on grids: the heat equation by the theta method
 * (explicit, implicit, Crank-Nicolson), the 1-D wave equation by leapfrog, linear advection by upwind, Lax-Friedrichs
 * or Lax-Wendroff, and the Poisson problem by Jacobi, Gauss-Seidel, SOR, conjugate gradients or a direct solve, all as
 * traced algorithms.
 */

import { OmaiError } from '../../base/index.ts'
import { dnrm2 } from '../../numerics/linalg/index.ts'
import { type CsrMatrix, cg, fromCoo, gmres } from '../../numerics/sparse/index.ts'
import { asTensor, fromStorage, type Tensor, type TensorLike, toFlat } from '../../tensor/index.ts'
import {
  type Algorithm,
  type Contract,
  channels,
  defineContract,
  type RunOptions,
  runAlgorithm,
  type StepContext,
  scalar,
  stateFromJson,
  stateToJson,
  tensor,
} from '../../trace/index.ts'
import { csrApply, rowOf, shiftedLu, sparseLuSolve } from './_linear.ts'
import { advectionSchemes, heatSchemes } from './finite-difference-options.ts'
import {
  applyDirichlet,
  type BoundaryCondition,
  boundaryValue,
  type DiscreteOperator,
  type Grid1d,
  type Grid2d,
  laplacian1d,
  laplacian2d,
  type RectangleConditions,
} from './grids.ts'

/** Tracing options shared by the PDE solvers. */
export type PdeTraceOptions = Partial<Omit<RunOptions, 'maxSteps' | 'start'>>

/** A time-dependent solution on a grid: snapshot times and nodal values `[nt, nodes]`. */
export interface PdeSolution {
  readonly t: readonly number[]
  readonly u: Tensor
  readonly status: 'success' | 'failed'
  readonly message: string
}

/** The state of the time-stepping PDE algorithms. */
export interface TimeStepState {
  /** Steps taken. */
  readonly step: number
  /** The time reached. */
  readonly t: number
  /** The field `u(t)` at the nodes. */
  readonly u: Float64Array
  /** The field one step back (the leapfrog scheme's second level), `undefined` before the first step. */
  readonly prev: Float64Array | undefined
  /** The snapshot times and fields kept (every n-th step, the last, and a failing one). */
  readonly ts: readonly number[]
  readonly us: readonly Float64Array[]
  /** Whether the field became non-finite. */
  readonly failed: boolean
}

/** The paths the time-stepping PDE algorithms log, as a type. */
export type PdePath = 'pde/t' | 'pde/u' | 'pde/norm' | 'pde/max' | 'pde/failed'

/**
 * What the time-stepping PDE algorithms log, by path: the time reached, the field at the nodes, its l2 norm and its
 * largest absolute value, and whether it has become non-finite.
 */
export const pdeContract: Contract<TimeStepState, PdePath> = defineContract({
  'pde/t': scalar({ description: 'the time reached' }),
  'pde/u': tensor({ role: 'state', shape: ['node'], description: 'the field at the nodes' }),
  'pde/norm': scalar({ description: 'the l2 norm of the field' }),
  'pde/max': scalar({ description: 'the largest |u|' }),
  'pde/failed': scalar({ description: '1 when the field has become non-finite, else 0' }),
})

const STEP_SCHEMA = channels({
  metrics: {
    t: { description: 'time' },
    norm: { description: 'the l2 norm of the solution' },
    max: { description: 'the largest |u|' },
  },
  tensors: { u: { role: 'state', description: 'the nodal values' } },
})

function maxAbs(u: Float64Array): number {
  let m = 0
  for (const v of u) m = Math.max(m, Math.abs(v))
  return m
}

function observeState(s: TimeStepState): {
  metrics: Record<string, number>
  tensors: Record<string, { role: 'state'; value: Float64Array }>
} {
  return { metrics: { t: s.t, norm: dnrm2(s.u), max: maxAbs(s.u) }, tensors: { u: { role: 'state', value: s.u } } }
}

function finite(u: Float64Array): boolean {
  for (const v of u) if (!Number.isFinite(v)) return false
  return true
}

/** The number of steps of size dt that reach tEnd (the last one shortened to land on it). */
function stepCount(name: string, dt: number, tEnd: number): number {
  if (!(dt > 0)) throw new OmaiError('value', `${name}: dt must be positive`)
  if (!(tEnd >= 0)) throw new OmaiError('value', `${name}: tEnd must be non-negative`)
  return Math.ceil(tEnd / dt - 1e-9)
}

function timeStepAlgorithm(
  name: string,
  config: Record<string, number | string>,
  u0: Float64Array,
  dt: number,
  tEnd: number,
  every: number,
  advance: (s: TimeStepState, t: number, h: number) => { u: Float64Array; prev: Float64Array | undefined },
): Algorithm<TimeStepState, PdePath> {
  const steps = stepCount(name, dt, tEnd)
  return {
    name,
    config: { ...config, steps },
    schema: STEP_SCHEMA,
    init: () => ({ step: 0, t: 0, u: u0, prev: undefined, ts: [0], us: [u0], failed: false }),
    step: (s, ctx) => {
      const h = Math.min(dt, tEnd - s.t)
      const { u, prev } = advance(s, s.t, h)
      const t = ctx.t + 1 === steps ? tEnd : s.t + h
      const failed = !finite(u)
      const keep = (ctx.t + 1) % every === 0 || ctx.t + 1 === steps || failed
      if (ctx.wants('pde/t')) ctx.log('pde/t', t)
      if (ctx.wants('pde/u')) ctx.log('pde/u', u)
      if (ctx.wants('pde/norm')) ctx.log('pde/norm', dnrm2(u))
      if (ctx.wants('pde/max')) ctx.log('pde/max', maxAbs(u))
      if (ctx.wants('pde/failed')) ctx.log('pde/failed', failed ? 1 : 0)
      return {
        step: s.step + 1,
        t,
        u,
        prev,
        ts: keep ? [...s.ts, t] : s.ts,
        us: keep ? [...s.us, u] : s.us,
        failed,
      }
    },
    done: (s) => (s.failed ? 'diverged' : s.step >= steps ? 'converged' : undefined),
    observe: observeState,
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as TimeStepState,
  }
}

/** A time-stepping algorithm run to its end and its snapshots as a {@link PdeSolution}. */
function solveSteps(
  algorithm: Algorithm<TimeStepState, PdePath>,
  dt: number,
  tEnd: number,
  options: PdeTraceOptions,
): PdeSolution {
  const name = algorithm.name
  const st = runAlgorithm(algorithm, { maxSteps: stepCount(name, dt, tEnd), ...options }).state
  const n = st.u.length
  const flat = new Float64Array(st.us.length * n)
  st.us.forEach((u, i) => {
    flat.set(u, i * n)
  })
  return {
    t: [...st.ts],
    u: fromStorage(flat, [st.us.length, n], 'float64'),
    status: st.failed ? 'failed' : 'success',
    message: st.failed ? `${name}: the solution became non-finite at t = ${st.t} (unstable step?)` : '',
  }
}

function initial(u0: TensorLike | ((x: number) => number), grid: Grid1d): Float64Array {
  if (typeof u0 === 'function') return Float64Array.from(grid.x, (x) => u0(x))
  const v = Float64Array.from(toFlat(asTensor(u0)))
  if (v.length !== grid.n) throw new OmaiError('shape', `the initial condition needs ${grid.n} values`)
  return v
}

/** The time scheme of a parabolic solve. */
export type HeatScheme = 'explicit' | 'implicit' | 'crank-nicolson' | 'theta'

/** Options of {@link heat}. */
export interface HeatOptions extends PdeTraceOptions {
  readonly dt: number
  readonly tEnd: number
  /** FTCS, BTCS, Crank-Nicolson (default) or a weighted `theta` (see {@link heatSchemes}). */
  readonly scheme?: HeatScheme
  /** The weight of the new level for the `theta` scheme (default 0.75). */
  readonly theta?: number
  /** A source term s(node index, t) added to the right-hand side. */
  readonly source?: (i: number, t: number) => number
  /** Keep every n-th step (default 1). */
  readonly every?: number
}

/**
 * The parabolic problem u_t = L u + s on any discretised operator as a steppable algorithm: each step advances the
 * field by the theta method (explicit FTCS, implicit BTCS, or Crank-Nicolson; see {@link heatSchemes}). The state is the
 * field at the nodes, the time and the snapshots, so it serialises and a run resumes; every step logs the time, the
 * field, its norm and its largest value, and an unstable step ends the run `diverged` (see {@link pdeContract}).
 *
 * @param op - The operator with its boundary conditions.
 * @param u0 - Initial nodal values.
 * @param options - Step, final time, scheme, source and thinning.
 * @returns The algorithm; it is done after the last step.
 * @throws OmaiError with code `value` for a non-positive step, or `shape` for an initial condition of the wrong size.
 * @concept heat-equation
 * @example
 * ```ts
 * import { dirichlet, grid1d, heatAlgorithm, laplacian1d } from 'omai-engine/dynamics/pde'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const g = grid1d(0, 1, 11)
 * const alg = heatAlgorithm(laplacian1d(g, dirichlet(), dirichlet()), Array.from(g.x, (x) => Math.sin(Math.PI * x)), { dt: 0.01, tEnd: 0.05 })
 * runAlgorithm(alg, { maxSteps: 5 }).state.t // => 0.05
 * ```
 */
export function heatAlgorithm(
  op: DiscreteOperator,
  u0: TensorLike,
  options: HeatOptions,
): Algorithm<TimeStepState, PdePath> {
  const scheme = options.scheme ?? 'crank-nicolson'
  const theta = heatSchemes.make({ kind: scheme, ...(options.theta === undefined ? {} : { theta: options.theta }) })
  const n = op.size
  const start = applyDirichlet(Float64Array.from(toFlat(asTensor(u0))), op, 0)
  if (start.length !== n) throw new OmaiError('shape', `heatAlgorithm: the initial condition needs ${n} values`)
  const fixedRows = new Set(op.fixed.map((f) => f.index))
  const factorCache = new Map<number, ReturnType<typeof shiftedLu>>()
  const rhsOf = (t: number): Float64Array => {
    const b = op.forcing(t)
    if (options.source !== undefined)
      for (let i = 0; i < n; i++) if (!fixedRows.has(i)) b[i] = (b[i] ?? 0) + options.source(i, t)
    return b
  }
  return timeStepAlgorithm(
    `heat-${scheme}`,
    { scheme, theta, dt: options.dt, tEnd: options.tEnd },
    start,
    options.dt,
    options.tEnd,
    Math.max(1, options.every ?? 1),
    (s, t, h) => {
      const au = csrApply(op.matrix, s.u)
      const b0 = rhsOf(t)
      if (theta === 0) {
        const next = Float64Array.from(s.u, (v, i) => v + h * ((au[i] ?? 0) + (b0[i] ?? 0)))
        return { u: applyDirichlet(next, op, t + h), prev: s.u }
      }
      let lu = factorCache.get(h)
      if (lu === undefined) {
        lu = shiftedLu(op.matrix, 1, -theta * h, fixedRows)
        factorCache.set(h, lu)
      }
      const b1 = rhsOf(t + h)
      const rhs = Float64Array.from(
        s.u,
        (v, i) => v + (1 - theta) * h * ((au[i] ?? 0) + (b0[i] ?? 0)) + theta * h * (b1[i] ?? 0),
      )
      for (const f of op.fixed) rhs[f.index] = boundaryValue(f.bc, t + h, ...f.at)
      return { u: sparseLuSolve(lu, rhs), prev: s.u }
    },
  )
}

/**
 * The parabolic problem u_t = L u + s on any discretised operator (1-D or 2-D, from {@link laplacian1d} or
 * {@link laplacian2d}) by the theta method: explicit (FTCS, theta = 0), implicit (BTCS, theta = 1) or Crank-Nicolson
 * (theta = 1/2). The implicit matrices are factorised once. A traced algorithm with the solution norm per step; an
 * unstable explicit step ends the run with a failed status.
 *
 * @param op - The operator with its boundary conditions.
 * @param u0 - Initial nodal values.
 * @param options - Step, final time, scheme, source and thinning.
 * @returns Snapshots of the solution.
 * @concept heat-equation
 * @index function Heat equation on a 1-D or 2-D grid with boundary conditions.
 * @example
 * ```ts
 * import { dirichlet, grid1d, heat, laplacian1d } from 'omai-engine/dynamics/pde'
 * const g = grid1d(0, 1, 21)
 * const u0 = Array.from(g.x, (x) => Math.sin(Math.PI * x))
 * const s = heat(laplacian1d(g, dirichlet(), dirichlet()), u0, { dt: 0.001, tEnd: 0.1 })
 * Math.abs((s.u.data[s.u.data.length - 11] ?? 0) - Math.exp(-(Math.PI ** 2) * 0.1)) < 1e-3 // => true
 * ```
 */
export function heat(op: DiscreteOperator, u0: TensorLike, options: HeatOptions): PdeSolution {
  return solveSteps(heatAlgorithm(op, u0, options), options.dt, options.tEnd, options)
}

/** Options of {@link heat1d}. */
export type Heat1dOptions = HeatOptions & {
  readonly alpha?: number
  readonly left: BoundaryCondition
  readonly right: BoundaryCondition
  readonly sourceFn?: (x: number, t: number) => number
}

/**
 * The 1-D heat equation u_t = alpha u_xx (plus a source) on a grid with boundary conditions, as a steppable algorithm
 * (see {@link heatAlgorithm}).
 *
 * @param u0 - Initial values or a function of x.
 * @param grid - The grid.
 * @param options - Diffusivity, boundary conditions and the options of {@link heat}.
 * @returns The algorithm; it is done after the last step.
 * @concept heat-equation
 * @example
 * ```ts
 * import { dirichlet, grid1d, heat1dAlgorithm } from 'omai-engine/dynamics/pde'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const alg = heat1dAlgorithm((x) => x * (1 - x), grid1d(0, 1, 11), { dt: 0.01, tEnd: 0.05, left: dirichlet(), right: dirichlet() })
 * runAlgorithm(alg, { maxSteps: 5 }).state.us.length // => 6
 * ```
 */
export function heat1dAlgorithm(
  u0: TensorLike | ((x: number) => number),
  grid: Grid1d,
  options: Heat1dOptions,
): Algorithm<TimeStepState, PdePath> {
  const op = laplacian1d(grid, options.left, options.right, options.alpha ?? 1)
  const source = options.sourceFn
  return heatAlgorithm(op, initial(u0, grid), {
    ...options,
    ...(source === undefined ? {} : { source: (i: number, t: number) => source(grid.x[i] ?? 0, t) }),
  })
}

/**
 * The 1-D heat equation u_t = alpha u_xx (plus a source) on a grid with boundary conditions (see {@link heat}).
 *
 * @param u0 - Initial values or a function of x.
 * @param grid - The grid.
 * @param options - Diffusivity, boundary conditions and the options of {@link heat}.
 * @returns Snapshots of the solution.
 * @concept heat-equation
 * @example
 * ```ts
 * import { dirichlet, grid1d, heat1d } from 'omai-engine/dynamics/pde'
 * heat1d((x) => x * (1 - x), grid1d(0, 1, 11), { dt: 0.01, tEnd: 0.05, left: dirichlet(), right: dirichlet() }).t.length // => 6
 * ```
 */
export function heat1d(u0: TensorLike | ((x: number) => number), grid: Grid1d, options: Heat1dOptions): PdeSolution {
  return solveSteps(heat1dAlgorithm(u0, grid, options), options.dt, options.tEnd, options)
}

/** Options of {@link wave1d}. */
export interface WaveOptions extends PdeTraceOptions {
  readonly c?: number
  readonly dt: number
  readonly tEnd: number
  readonly left: BoundaryCondition
  readonly right: BoundaryCondition
  readonly every?: number
}

/**
 * The 1-D wave equation u_tt = c^2 u_xx by the leapfrog scheme as a steppable algorithm: the first step is a Taylor
 * step from the initial velocity, the rest u^{n+1} = 2 u^n - u^{n-1} + dt^2 c^2 D2 u^n, so the state carries the field
 * and the one before it. Stable for c dt / dx <= 1 (exact on the grid at c dt = dx). Logs the paths of {@link pdeContract}.
 *
 * @param u0 - Initial displacement (values or a function of x).
 * @param v0 - Initial velocity (values or a function of x).
 * @param grid - The grid.
 * @param options - Speed, step, final time, boundary conditions and thinning.
 * @returns The algorithm; it is done after the last step.
 * @concept wave-equation
 * @example
 * ```ts
 * import { dirichlet, grid1d, wave1dAlgorithm } from 'omai-engine/dynamics/pde'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const alg = wave1dAlgorithm((x) => Math.sin(Math.PI * x), () => 0, grid1d(0, 1, 21), { dt: 0.05, tEnd: 0.5, left: dirichlet(), right: dirichlet() })
 * runAlgorithm(alg, { maxSteps: 10 }).state.prev?.length // => 21
 * ```
 */
export function wave1dAlgorithm(
  u0: TensorLike | ((x: number) => number),
  v0: TensorLike | ((x: number) => number),
  grid: Grid1d,
  options: WaveOptions,
): Algorithm<TimeStepState, PdePath> {
  const c2 = (options.c ?? 1) ** 2
  const op = laplacian1d(grid, options.left, options.right, c2)
  const v = initial(v0, grid)
  return timeStepAlgorithm(
    'wave-leapfrog',
    { dt: options.dt, tEnd: options.tEnd, c: options.c ?? 1 },
    applyDirichlet(initial(u0, grid), op, 0),
    options.dt,
    options.tEnd,
    Math.max(1, options.every ?? 1),
    (s, t, h) => {
      const au = csrApply(op.matrix, s.u)
      const b = op.forcing(t)
      const next =
        s.prev === undefined
          ? Float64Array.from(s.u, (u, i) => u + h * (v[i] ?? 0) + 0.5 * h * h * ((au[i] ?? 0) + (b[i] ?? 0)))
          : Float64Array.from(s.u, (u, i) => 2 * u - (s.prev?.[i] ?? 0) + h * h * ((au[i] ?? 0) + (b[i] ?? 0)))
      return { u: applyDirichlet(next, op, t + h), prev: s.u }
    },
  )
}

/**
 * The 1-D wave equation u_tt = c^2 u_xx by the leapfrog scheme u^{n+1} = 2 u^n - u^{n-1} + dt^2 c^2 D2 u^n, started
 * by a Taylor step. Stable for c dt / dx <= 1 (exact on the grid at c dt = dx).
 *
 * @param u0 - Initial displacement (values or a function of x).
 * @param v0 - Initial velocity (values or a function of x).
 * @param grid - The grid.
 * @param options - Speed, step, final time, boundary conditions and thinning.
 * @returns Snapshots of the displacement.
 * @concept wave-equation
 * @example
 * ```ts
 * import { dirichlet, grid1d, wave1d } from 'omai-engine/dynamics/pde'
 * const g = grid1d(0, 1, 41)
 * const s = wave1d((x) => Math.sin(Math.PI * x), () => 0, g, { dt: 0.025, tEnd: 1, left: dirichlet(), right: dirichlet() })
 * Math.abs((s.u.data[s.u.data.length - 21] ?? 0) + 1) < 1e-2 // => true
 * ```
 */
export function wave1d(
  u0: TensorLike | ((x: number) => number),
  v0: TensorLike | ((x: number) => number),
  grid: Grid1d,
  options: WaveOptions,
): PdeSolution {
  return solveSteps(wave1dAlgorithm(u0, v0, grid, options), options.dt, options.tEnd, options)
}

/** The advection schemes. */
export type AdvectionScheme = 'upwind' | 'lax-friedrichs' | 'lax-wendroff'

/** Options of {@link advection1d}. */
export interface AdvectionOptions extends PdeTraceOptions {
  /** The constant velocity a of u_t + a u_x = 0. */
  readonly velocity: number
  readonly dt: number
  readonly tEnd: number
  readonly scheme?: AdvectionScheme
  /** `periodic` (default) or an inflow value at the upwind boundary. */
  readonly boundary?: 'periodic' | { readonly inflow: number }
  readonly every?: number
}

/**
 * Linear advection u_t + a u_x = 0 as a steppable algorithm: each step applies a stencil (upwind, Lax-Friedrichs or
 * Lax-Wendroff; see {@link advectionSchemes}) to the field, with periodic or inflow and outflow boundaries. Each is
 * stable for |a| dt / dx <= 1. Logs the paths of {@link pdeContract}.
 *
 * @param u0 - Initial values or a function of x.
 * @param grid - The grid.
 * @param options - Velocity, step, final time, scheme and boundary.
 * @returns The algorithm; it is done after the last step.
 * @concept advection-equation
 * @example
 * ```ts
 * import { advection1dAlgorithm, grid1d } from 'omai-engine/dynamics/pde'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const alg = advection1dAlgorithm((x) => Math.sin(2 * Math.PI * x), grid1d(0, 1, 101), { velocity: 1, dt: 0.01, tEnd: 0.5 })
 * runAlgorithm(alg, { maxSteps: 50 }).state.step // => 50
 * ```
 */
export function advection1dAlgorithm(
  u0: TensorLike | ((x: number) => number),
  grid: Grid1d,
  options: AdvectionOptions,
): Algorithm<TimeStepState, PdePath> {
  const a = options.velocity
  const scheme = options.scheme ?? 'upwind'
  const stencil = advectionSchemes.make({ kind: scheme })
  const bc = options.boundary ?? 'periodic'
  const n = grid.n
  const isPeriodic = bc === 'periodic'
  const m = isPeriodic ? n - 1 : n
  return timeStepAlgorithm(
    `advection-${scheme}`,
    { scheme, velocity: a, dt: options.dt, tEnd: options.tEnd },
    initial(u0, grid),
    options.dt,
    options.tEnd,
    Math.max(1, options.every ?? 1),
    (s, _t, h) => {
      const nu = (a * h) / grid.dx
      const u = s.u
      const at = (i: number): number => {
        if (isPeriodic) return u[((i % m) + m) % m] ?? 0
        if (i < 0) return a > 0 ? (bc as { inflow: number }).inflow : (u[0] ?? 0)
        if (i >= n) return a < 0 ? (bc as { inflow: number }).inflow : (u[n - 1] ?? 0)
        return u[i] ?? 0
      }
      const next = new Float64Array(n)
      for (let i = 0; i < m; i++) next[i] = stencil(at(i - 1), at(i), at(i + 1), nu, a)
      if (isPeriodic) next[n - 1] = next[0] ?? 0
      else if (a > 0) next[0] = (bc as { inflow: number }).inflow
      else if (a < 0) next[n - 1] = (bc as { inflow: number }).inflow
      return { u: next, prev: u }
    },
  )
}

/**
 * Linear advection u_t + a u_x = 0 by first-order upwind, Lax-Friedrichs or Lax-Wendroff, with periodic or inflow and
 * outflow boundaries. Each is stable for |a| dt / dx <= 1.
 *
 * @param u0 - Initial values or a function of x.
 * @param grid - The grid.
 * @param options - Velocity, step, final time, scheme and boundary.
 * @returns Snapshots of the solution.
 * @concept advection-equation
 * @example
 * ```ts
 * import { advection1d, grid1d } from 'omai-engine/dynamics/pde'
 * const g = grid1d(0, 1, 101)
 * const s = advection1d((x) => Math.sin(2 * Math.PI * x), g, { velocity: 1, dt: 0.01, tEnd: 1 })
 * Math.abs((s.u.data[s.u.data.length - 76] ?? 0) - 1) < 1e-12 // => true
 * ```
 */
export function advection1d(
  u0: TensorLike | ((x: number) => number),
  grid: Grid1d,
  options: AdvectionOptions,
): PdeSolution {
  return solveSteps(advection1dAlgorithm(u0, grid, options), options.dt, options.tEnd, options)
}

/** The Poisson iterations. */
export type PoissonMethod = 'jacobi' | 'gauss-seidel' | 'sor' | 'cg' | 'multigrid'

/** Options of {@link poisson}. */
export interface PoissonOptions extends PdeTraceOptions {
  readonly method?: PoissonMethod
  /** Relaxation factor of SOR (default: the optimal 2 / (1 + sin(pi h)) for the unit square). */
  readonly omega?: number
  /** Relative residual tolerance (default 1e-10). */
  readonly tol?: number
  readonly maxIterations?: number
  /** Multigrid: smoothing sweeps before and after the coarse-grid correction (default 2). */
  readonly sweeps?: number
  /** Multigrid: sweeps that solve the coarsest grid (default 50). */
  readonly coarseSweeps?: number
}

/** The result of {@link poisson}. */
export interface PoissonResult {
  readonly u: Tensor
  readonly iterations: number
  readonly converged: boolean
  readonly residualHistory: readonly number[]
}

/** The state of {@link poissonAlgorithm}: the iterate, its relative residual and the residuals so far. */
export interface PoissonState {
  /** The iterate at the nodes (fixed nodes hold their boundary values). */
  readonly u: Float64Array
  /** ||f - A u|| / ||f|| over the free nodes. */
  readonly residual: number
  /** The residual at the start and after every sweep or cycle. */
  readonly history: readonly number[]
}

/** The paths {@link poissonAlgorithm} logs, as a type. */
export type PoissonPath = 'poisson/residual' | 'poisson/iterate' | 'poisson/change'

/** What {@link poissonAlgorithm} logs, by path: the relative residual, the iterate and the size of the last update. */
export const poissonContract: Contract<PoissonState, PoissonPath> = defineContract({
  'poisson/residual': scalar({ description: 'the relative residual after this sweep or cycle', direction: 'lower' }),
  'poisson/iterate': tensor({ role: 'state', shape: ['node'], description: 'the iterate at the nodes' }),
  'poisson/change': scalar({ description: 'the l2 norm of the update this sweep or cycle made', direction: 'lower' }),
})

interface PoissonProblem {
  readonly op: DiscreteOperator
  readonly n: number
  readonly fv: Float64Array
  readonly rhs: Float64Array
  readonly u: Float64Array
  readonly fixed: ReadonlySet<number>
  readonly method: PoissonMethod
  readonly tol: number
  readonly maxIterations: number
  readonly residualOf: (x: Float64Array) => number
}

function poissonProblem(
  f: TensorLike | ((...x: number[]) => number),
  grid: Grid1d | Grid2d,
  bcs: readonly [BoundaryCondition, BoundaryCondition] | RectangleConditions,
  options: PoissonOptions,
): PoissonProblem {
  const op =
    'a' in grid
      ? laplacian1d(
          grid,
          (bcs as readonly BoundaryCondition[])[0] as BoundaryCondition,
          (bcs as readonly BoundaryCondition[])[1] as BoundaryCondition,
        )
      : laplacian2d(grid, bcs as RectangleConditions)
  const n = op.size
  const fv = new Float64Array(n)
  if (typeof f === 'function') {
    if ('a' in grid) for (let i = 0; i < n; i++) fv[i] = f(grid.x[i] ?? 0)
    else
      for (let i = 0; i < grid.nx; i++)
        for (let j = 0; j < grid.ny; j++) fv[i * grid.ny + j] = f(grid.x.x[i] ?? 0, grid.y.x[j] ?? 0)
  } else fv.set(toFlat(asTensor(f)))
  const forcing = op.forcing(0)
  const fixed = new Set(op.fixed.map((q) => q.index))
  // A u = f - forcing on the free nodes; u = g on the fixed ones.
  const rhs = Float64Array.from(fv, (v, i) => (fixed.has(i) ? 0 : v - (forcing[i] ?? 0)))
  const u = applyDirichlet(new Float64Array(n), op, 0)
  const method = options.method ?? 'cg'
  const tol = options.tol ?? 1e-10
  const maxIterations = options.maxIterations ?? (method === 'multigrid' ? 50 : method === 'cg' ? 10 * n : 100 * n)
  const bnorm = Math.max(dnrm2(rhs), 1e-300)
  const residualOf = (x: Float64Array): number => {
    const ax = csrApply(op.matrix, x)
    let s = 0
    for (let i = 0; i < n; i++) if (!fixed.has(i)) s += ((rhs[i] ?? 0) - (ax[i] ?? 0)) ** 2
    return Math.sqrt(s) / bnorm
  }
  return { op, n, fv, rhs, u, fixed, method, tol, maxIterations, residualOf }
}

/**
 * The Poisson problem Delta u = f with boundary conditions on a 1-D or 2-D grid as a steppable algorithm: each step is
 * one Jacobi, Gauss-Seidel or SOR sweep, or one geometric multigrid V-cycle (Dirichlet conditions, 2^k + 1 nodes per
 * side); see {@link poissonMethods}. The state is the iterate and the residual history, so it serialises and a run
 * resumes; it logs the paths of {@link poissonContract} and is done when the relative residual reaches the tolerance
 * (`diverged` when it stops being finite).
 *
 * @param f - The right-hand side at the nodes (values, or a function of the node's coordinates).
 * @param grid - A 1-D or 2-D grid.
 * @param bcs - The conditions: [left, right] in 1-D, the four sides in 2-D.
 * @param options - Method, relaxation, tolerance and, for multigrid, the sweeps.
 * @returns The algorithm.
 * @throws OmaiError with code `value` for conjugate gradients (a Krylov solver, see `cgAlgorithm`) or when multigrid does not fit the grid.
 * @concept poisson-equation
 * @example
 * ```ts
 * import { dirichlet, grid1d, poissonAlgorithm } from 'omai-engine/dynamics/pde'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const alg = poissonAlgorithm(() => 2, grid1d(0, 1, 9), [dirichlet(), dirichlet()], { method: 'gauss-seidel', tol: 1e-8 })
 * runAlgorithm(alg, { maxSteps: 1000 }).state.residual < 1e-8 // => true
 * ```
 */
export function poissonAlgorithm(
  f: TensorLike | ((...x: number[]) => number),
  grid: Grid1d | Grid2d,
  bcs: readonly [BoundaryCondition, BoundaryCondition] | RectangleConditions,
  options: PoissonOptions = {},
): Algorithm<PoissonState, PoissonPath> {
  const problem = poissonProblem(f, grid, bcs, options)
  if (problem.method === 'cg')
    throw new OmaiError(
      'value',
      'poissonAlgorithm: conjugate gradients is a Krylov solver (see cgAlgorithm in numerics/sparse)',
    )
  return sweepAlgorithm(problem, grid, options)
}

function sweepAlgorithm(
  problem: PoissonProblem,
  grid: Grid1d | Grid2d,
  options: PoissonOptions,
): Algorithm<PoissonState, PoissonPath> {
  const { op, n, fv, rhs, u, fixed, method, tol, residualOf } = problem
  if (method === 'multigrid') return multigridPoisson(grid, fv, u, fixed, tol, residualOf, options)
  const rows = Array.from({ length: n }, (_, r) => rowOf(op.matrix, r))
  const h = 'a' in grid ? grid.dx : Math.max(grid.x.dx, grid.y.dx)
  const omega = method === 'sor' ? (options.omega ?? 2 / (1 + Math.sin(Math.PI * h))) : 1
  return {
    name: `poisson-${method}`,
    config: { method, omega, tol },
    schema: channels({ metrics: { residual: { description: 'relative residual', direction: 'lower' } } }),
    init: () => ({ u, residual: residualOf(u), history: [residualOf(u)] }),
    step: (s, ctx) => {
      const next = Float64Array.from(s.u)
      const src = s.u
      for (let r = 0; r < n; r++) {
        if (fixed.has(r)) continue
        const { diag, off } = rows[r] as ReturnType<typeof rowOf>
        let acc = rhs[r] ?? 0
        for (const { c, v } of off) acc -= v * ((method === 'jacobi' ? src : next)[c] ?? 0)
        const gs = acc / diag
        next[r] = (1 - omega) * (src[r] ?? 0) + omega * gs
      }
      return advancePoisson(s, next, residualOf, ctx)
    },
    done: (s) => (s.residual <= tol ? 'converged' : Number.isFinite(s.residual) ? undefined : 'diverged'),
    observe: (s) => ({ metrics: { residual: s.residual } }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as PoissonState,
  }
}

function advancePoisson(
  s: PoissonState,
  next: Float64Array,
  residualOf: (x: Float64Array) => number,
  ctx: StepContext<PoissonPath>,
): PoissonState {
  const residual = residualOf(next)
  if (ctx.wants('poisson/residual')) ctx.log('poisson/residual', residual)
  if (ctx.wants('poisson/iterate')) ctx.log('poisson/iterate', next)
  if (ctx.wants('poisson/change'))
    ctx.log('poisson/change', dnrm2(Float64Array.from(next, (v, i) => v - (s.u[i] ?? 0))))
  return { u: next, residual, history: [...s.history, residual] }
}

/**
 * Solve the Poisson problem Delta u = f with boundary conditions on a 1-D or 2-D grid, by Jacobi, Gauss-Seidel or SOR
 * sweeps, conjugate gradients (GMRES when Neumann or Robin rows make the system unsymmetric) or geometric multigrid
 * V-cycles (Dirichlet conditions, 2^k + 1 nodes per side). Each sweep or cycle is one traced step with the relative
 * residual as a metric ({@link poissonAlgorithm} run to the end).
 *
 * @param f - The right-hand side at the nodes (values, or a function of the node's coordinates).
 * @param grid - A 1-D or 2-D grid.
 * @param bcs - The conditions: [left, right] in 1-D, the four sides in 2-D.
 * @param options - Method, relaxation, tolerance and iteration cap.
 * @returns The solution, iterations and residual history.
 * @throws OmaiError with code `value` when multigrid does not fit the grid.
 * @concept poisson-equation
 * @example
 * ```ts
 * import { dirichlet, grid1d, poisson } from 'omai-engine/dynamics/pde'
 * // u'' = 2, u(0) = u(1) = 0: u = x^2 - x.
 * const r = poisson(() => 2, grid1d(0, 1, 9), [dirichlet(), dirichlet()], { method: 'cg' })
 * Math.abs((r.u.data[4] ?? 0) + 0.25) < 1e-10 // => true
 * ```
 */
export function poisson(
  f: TensorLike | ((...x: number[]) => number),
  grid: Grid1d | Grid2d,
  bcs: readonly [BoundaryCondition, BoundaryCondition] | RectangleConditions,
  options: PoissonOptions = {},
): PoissonResult {
  const problem = poissonProblem(f, grid, bcs, options)
  if (problem.method === 'cg') {
    const r = krylovPoisson(
      problem.op,
      problem.rhs,
      problem.u,
      problem.fixed,
      problem.tol,
      problem.maxIterations,
      options,
    )
    return { ...r, u: shaped(Float64Array.from(r.u.data), grid) }
  }
  const run = runAlgorithm(sweepAlgorithm(problem, grid, options), { maxSteps: problem.maxIterations, ...options })
  return {
    u: shaped(run.state.u, grid),
    iterations: run.steps,
    converged: run.state.residual <= problem.tol,
    residualHistory: [...run.state.history],
  }
}

function shaped(u: Float64Array, grid: Grid1d | Grid2d): Tensor {
  return 'a' in grid ? fromStorage(u, [grid.n], 'float64') : fromStorage(u, [grid.nx, grid.ny], 'float64')
}

function krylovPoisson(
  op: DiscreteOperator,
  rhs: Float64Array,
  u: Float64Array,
  fixed: ReadonlySet<number>,
  tol: number,
  maxIterations: number,
  options: PdeTraceOptions,
): PoissonResult {
  // The free-node system -A_ff u_f = -(rhs_f - A_fd g): symmetric positive definite for Dirichlet problems.
  const n = op.size
  const free = Array.from({ length: n }, (_, i) => i).filter((i) => !fixed.has(i))
  const pos = new Map(free.map((g, k) => [g, k]))
  const rows: number[] = []
  const cols: number[] = []
  const vals: number[] = []
  const b = new Float64Array(free.length)
  let symmetric = true
  const entries = new Map<string, number>()
  free.forEach((g, k) => {
    const { diag, off } = rowOf(op.matrix, g)
    rows.push(k)
    cols.push(k)
    vals.push(-diag)
    let acc = rhs[g] ?? 0
    for (const { c, v } of off) {
      const kc = pos.get(c)
      if (kc === undefined) acc -= v * (u[c] ?? 0)
      else {
        rows.push(k)
        cols.push(kc)
        vals.push(-v)
        entries.set(`${k},${kc}`, -v)
      }
    }
    b[k] = -acc
  })
  for (const [key, v] of entries) {
    const [i, j] = key.split(',')
    if (Math.abs((entries.get(`${j},${i}`) ?? 0) - v) > 1e-12 * Math.abs(v)) {
      symmetric = false
      break
    }
  }
  const a: CsrMatrix = fromCoo(rows, cols, vals, [free.length, free.length])
  const solver = symmetric ? cg : gmres
  const r = solver(a, b, { rtol: tol, maxIterations, ...options })
  const out = Float64Array.from(u)
  free.forEach((g, k) => {
    out[g] = r.x.data[k] ?? 0
  })
  return {
    u: fromStorage(out, [n], 'float64'),
    iterations: r.iterations,
    converged: r.converged,
    residualHistory: r.history.map((h) => h / Math.max(dnrm2(b), 1e-300)),
  }
}

function isPowerOfTwoPlusOne(n: number): boolean {
  const m = n - 1
  return m >= 2 && (m & (m - 1)) === 0
}

// The geometric multigrid behind `method: 'multigrid'`: matrix-free, on the five-point stencil itself (red-black
// Gauss-Seidel smoothing, full-weighting restriction, bilinear prolongation, grids of 2^k + 1 nodes). It is not
// numerics/sparse's `multigridAlgorithm` on purpose: that one builds Galerkin coarse operators from any CSR matrix with
// Jacobi or SOR smoothers; here the structured grid gives the stencil's own transfer operators and a smoother that
// needs no matrix, which is what the Poisson solvers of this module compare against Jacobi, Gauss-Seidel and SOR.
function multigridPoisson(
  grid: Grid1d | Grid2d,
  f: Float64Array,
  u0: Float64Array,
  fixed: ReadonlySet<number>,
  tol: number,
  residualOf: (x: Float64Array) => number,
  options: PoissonOptions,
): Algorithm<PoissonState, PoissonPath> {
  const sweeps = options.sweeps ?? 2
  const coarseSweeps = options.coarseSweeps ?? 50
  const twoD = !('a' in grid)
  const nx = twoD ? grid.nx : grid.n
  const ny = twoD ? grid.ny : 1
  const hx = twoD ? grid.x.dx : grid.dx
  const hy = twoD ? grid.y.dx : 1
  const boundaryCount = twoD ? 2 * nx + 2 * ny - 4 : 2
  if (
    !isPowerOfTwoPlusOne(nx) ||
    (twoD && (!isPowerOfTwoPlusOne(ny) || nx !== ny || hx !== hy)) ||
    fixed.size !== boundaryCount
  ) {
    throw new OmaiError(
      'value',
      'poisson: multigrid needs Dirichlet conditions and 2^k + 1 nodes per side (square cells)',
    )
  }
  const h = hx
  // Residual equation in the interior with homogeneous boundaries at the coarse levels.
  const index = (i: number, j: number, m: number): number => (twoD ? i * m + j : i)
  const smooth = (u: Float64Array, rhs: Float64Array, m: number, hh: number, sweeps: number): void => {
    for (let s = 0; s < sweeps; s++) {
      for (let colour = 0; colour < 2; colour++) {
        for (let i = 1; i < m - 1; i++) {
          if (!twoD) {
            if ((i & 1) !== colour) continue
            u[i] = 0.5 * ((u[i - 1] ?? 0) + (u[i + 1] ?? 0) - hh * hh * (rhs[i] ?? 0))
            continue
          }
          for (let j = 1; j < m - 1; j++) {
            if (((i + j) & 1) !== colour) continue
            const k = index(i, j, m)
            u[k] =
              0.25 *
              ((u[index(i - 1, j, m)] ?? 0) +
                (u[index(i + 1, j, m)] ?? 0) +
                (u[index(i, j - 1, m)] ?? 0) +
                (u[index(i, j + 1, m)] ?? 0) -
                hh * hh * (rhs[k] ?? 0))
          }
        }
      }
    }
  }
  const residual = (u: Float64Array, rhs: Float64Array, m: number, hh: number): Float64Array => {
    const r = new Float64Array(u.length)
    for (let i = 1; i < m - 1; i++) {
      if (!twoD) {
        r[i] = (rhs[i] ?? 0) - ((u[i - 1] ?? 0) - 2 * (u[i] ?? 0) + (u[i + 1] ?? 0)) / (hh * hh)
        continue
      }
      for (let j = 1; j < m - 1; j++) {
        const k = index(i, j, m)
        const lap =
          ((u[index(i - 1, j, m)] ?? 0) +
            (u[index(i + 1, j, m)] ?? 0) +
            (u[index(i, j - 1, m)] ?? 0) +
            (u[index(i, j + 1, m)] ?? 0) -
            4 * (u[k] ?? 0)) /
          (hh * hh)
        r[k] = (rhs[k] ?? 0) - lap
      }
    }
    return r
  }
  const restrict = (r: Float64Array, m: number): Float64Array => {
    const mc = (m - 1) / 2 + 1
    const out = new Float64Array(twoD ? mc * mc : mc)
    for (let I = 1; I < mc - 1; I++) {
      if (!twoD) {
        out[I] = 0.25 * (r[2 * I - 1] ?? 0) + 0.5 * (r[2 * I] ?? 0) + 0.25 * (r[2 * I + 1] ?? 0)
        continue
      }
      for (let J = 1; J < mc - 1; J++) {
        const [i, j] = [2 * I, 2 * J]
        let s = 4 * (r[index(i, j, m)] ?? 0)
        s +=
          2 *
          ((r[index(i - 1, j, m)] ?? 0) +
            (r[index(i + 1, j, m)] ?? 0) +
            (r[index(i, j - 1, m)] ?? 0) +
            (r[index(i, j + 1, m)] ?? 0))
        s +=
          (r[index(i - 1, j - 1, m)] ?? 0) +
          (r[index(i + 1, j - 1, m)] ?? 0) +
          (r[index(i - 1, j + 1, m)] ?? 0) +
          (r[index(i + 1, j + 1, m)] ?? 0)
        out[index(I, J, mc)] = s / 16
      }
    }
    return out
  }
  const prolong = (e: Float64Array, mc: number): Float64Array => {
    const m = 2 * (mc - 1) + 1
    const out = new Float64Array(twoD ? m * m : m)
    for (let i = 0; i < m; i++) {
      const I0 = Math.floor(i / 2)
      const I1 = Math.ceil(i / 2)
      if (!twoD) {
        out[i] = 0.5 * ((e[I0] ?? 0) + (e[I1] ?? 0))
        continue
      }
      for (let j = 0; j < m; j++) {
        const J0 = Math.floor(j / 2)
        const J1 = Math.ceil(j / 2)
        out[index(i, j, m)] =
          0.25 *
          ((e[index(I0, J0, mc)] ?? 0) +
            (e[index(I1, J0, mc)] ?? 0) +
            (e[index(I0, J1, mc)] ?? 0) +
            (e[index(I1, J1, mc)] ?? 0))
      }
    }
    return out
  }
  const vcycle = (u: Float64Array, rhs: Float64Array, m: number, hh: number): void => {
    if (m <= 3) {
      smooth(u, rhs, m, hh, coarseSweeps)
      return
    }
    smooth(u, rhs, m, hh, sweeps)
    const rc = restrict(residual(u, rhs, m, hh), m)
    const mc = (m - 1) / 2 + 1
    const ec = new Float64Array(rc.length)
    vcycle(ec, rc, mc, 2 * hh)
    const e = prolong(ec, mc)
    for (let k = 0; k < u.length; k++) u[k] = (u[k] ?? 0) + (e[k] ?? 0)
    smooth(u, rhs, m, hh, sweeps)
  }
  return {
    name: 'poisson-multigrid',
    config: { method: 'multigrid', tol, sweeps, coarseSweeps },
    schema: channels({ metrics: { residual: { description: 'relative residual', direction: 'lower' } } }),
    init: () => ({ u: u0, residual: residualOf(u0), history: [residualOf(u0)] }),
    step: (s, ctx) => {
      const u = Float64Array.from(s.u)
      vcycle(u, f, nx, h)
      return advancePoisson(s, u, residualOf, ctx)
    },
    done: (s) => (s.residual <= tol ? 'converged' : Number.isFinite(s.residual) ? undefined : 'diverged'),
    observe: (s) => ({ metrics: { residual: s.residual } }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as PoissonState,
  }
}
