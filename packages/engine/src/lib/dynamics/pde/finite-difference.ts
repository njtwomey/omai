/**
 * Time-dependent and steady PDEs solved by finite differences on grids: the heat equation by the theta method
 * (explicit, implicit, Crank-Nicolson), the 1-D wave equation by leapfrog, linear advection by upwind, Lax-Friedrichs
 * or Lax-Wendroff, and the Poisson problem by Jacobi, Gauss-Seidel, SOR, conjugate gradients or a direct solve, all as
 * traced algorithms.
 */

import { WhiteboxError } from '../../base/index.ts'
import { dnrm2 } from '../../numerics/linalg/index.ts'
import { type CsrMatrix, cg, fromCoo, gmres } from '../../numerics/sparse/index.ts'
import { asTensor, fromStorage, type Tensor, type TensorLike, toFlat } from '../../tensor/index.ts'
import { type Algorithm, channels, type RunOptions, runAlgorithm } from '../../trace/index.ts'
import { csrApply, rowOf, shiftedLu, sparseLuSolve } from './_linear.ts'
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
  readonly t: number
  readonly u: Float64Array
  readonly prev: Float64Array | undefined
  readonly ts: number[]
  readonly us: Float64Array[]
  readonly failed: boolean
}

const STEP_SCHEMA = channels({
  metrics: {
    t: { description: 'time' },
    norm: { description: 'the l2 norm of the solution' },
    max: { description: 'the largest |u|' },
  },
  tensors: { u: { role: 'state', description: 'the nodal values' } },
})

function observeState(s: TimeStepState): {
  metrics: Record<string, number>
  tensors: Record<string, { role: 'state'; value: Float64Array }>
} {
  let m = 0
  for (const v of s.u) m = Math.max(m, Math.abs(v))
  return { metrics: { t: s.t, norm: dnrm2(s.u), max: m }, tensors: { u: { role: 'state', value: s.u } } }
}

function finite(u: Float64Array): boolean {
  for (const v of u) if (!Number.isFinite(v)) return false
  return true
}

function runSteps(
  name: string,
  config: Record<string, number | string>,
  u0: Float64Array,
  dt: number,
  tEnd: number,
  every: number,
  advance: (s: TimeStepState, t: number, h: number) => { u: Float64Array; prev: Float64Array | undefined },
  options: PdeTraceOptions,
): PdeSolution {
  if (!(dt > 0)) throw new WhiteboxError('value', `${name}: dt must be positive`)
  if (!(tEnd >= 0)) throw new WhiteboxError('value', `${name}: tEnd must be non-negative`)
  const steps = Math.ceil(tEnd / dt - 1e-9)
  const algorithm: Algorithm<TimeStepState> = {
    name,
    config,
    schema: STEP_SCHEMA,
    init: () => ({ t: 0, u: u0, prev: undefined, ts: [0], us: [u0], failed: false }),
    step: (s, ctx) => {
      const h = Math.min(dt, tEnd - s.t)
      const { u, prev } = advance(s, s.t, h)
      const t = ctx.t + 1 === steps ? tEnd : s.t + h
      const failed = !finite(u)
      if ((ctx.t + 1) % every === 0 || ctx.t + 1 === steps || failed) {
        s.ts.push(t)
        s.us.push(u)
      }
      return { ...s, t, u, prev, failed }
    },
    done: (s) => (s.failed ? 'diverged' : undefined),
    observe: observeState,
  }
  const run = runAlgorithm(algorithm, { maxSteps: steps, ...options })
  const st = run.state
  const n = u0.length
  const flat = new Float64Array(st.us.length * n)
  st.us.forEach((u, i) => {
    flat.set(u, i * n)
  })
  return {
    t: st.ts,
    u: fromStorage(flat, [st.us.length, n], 'float64'),
    status: st.failed ? 'failed' : 'success',
    message: st.failed ? `${name}: the solution became non-finite at t = ${st.t} (unstable step?)` : '',
  }
}

function initial(u0: TensorLike | ((x: number) => number), grid: Grid1d): Float64Array {
  if (typeof u0 === 'function') return Float64Array.from(grid.x, (x) => u0(x))
  const v = Float64Array.from(toFlat(asTensor(u0)))
  if (v.length !== grid.n) throw new WhiteboxError('shape', `the initial condition needs ${grid.n} values`)
  return v
}

/** The time scheme of a parabolic solve. */
export type HeatScheme = 'explicit' | 'implicit' | 'crank-nicolson'

/** Options of {@link heat}. */
export interface HeatOptions extends PdeTraceOptions {
  readonly dt: number
  readonly tEnd: number
  /** FTCS, BTCS or Crank-Nicolson (default). */
  readonly scheme?: HeatScheme
  /** A source term s(node index, t) added to the right-hand side. */
  readonly source?: (i: number, t: number) => number
  /** Keep every n-th step (default 1). */
  readonly every?: number
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
 * @example
 * ```ts
 * import { dirichlet, grid1d, heat, laplacian1d } from 'whitebox-engine/dynamics/pde'
 * const g = grid1d(0, 1, 21)
 * const u0 = Array.from(g.x, (x) => Math.sin(Math.PI * x))
 * const s = heat(laplacian1d(g, dirichlet(), dirichlet()), u0, { dt: 0.001, tEnd: 0.1 })
 * Math.abs((s.u.data[s.u.data.length - 11] ?? 0) - Math.exp(-(Math.PI ** 2) * 0.1)) < 1e-3 // => true
 * ```
 */
export function heat(op: DiscreteOperator, u0: TensorLike, options: HeatOptions): PdeSolution {
  const scheme = options.scheme ?? 'crank-nicolson'
  const theta = scheme === 'explicit' ? 0 : scheme === 'implicit' ? 1 : 0.5
  const n = op.size
  const start = applyDirichlet(Float64Array.from(toFlat(asTensor(u0))), op, 0)
  if (start.length !== n) throw new WhiteboxError('shape', `heat: the initial condition needs ${n} values`)
  const fixedRows = new Set(op.fixed.map((f) => f.index))
  const factorCache = new Map<number, ReturnType<typeof shiftedLu>>()
  const rhsOf = (t: number): Float64Array => {
    const b = op.forcing(t)
    if (options.source !== undefined)
      for (let i = 0; i < n; i++) if (!fixedRows.has(i)) b[i] = (b[i] ?? 0) + options.source(i, t)
    return b
  }
  return runSteps(
    `heat-${scheme}`,
    { scheme, dt: options.dt, tEnd: options.tEnd },
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
    options,
  )
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
 * import { dirichlet, grid1d, heat1d } from 'whitebox-engine/dynamics/pde'
 * heat1d((x) => x * (1 - x), grid1d(0, 1, 11), { dt: 0.01, tEnd: 0.05, left: dirichlet(), right: dirichlet() }).t.length // => 6
 * ```
 */
export function heat1d(
  u0: TensorLike | ((x: number) => number),
  grid: Grid1d,
  options: HeatOptions & {
    readonly alpha?: number
    readonly left: BoundaryCondition
    readonly right: BoundaryCondition
    readonly sourceFn?: (x: number, t: number) => number
  },
): PdeSolution {
  const op = laplacian1d(grid, options.left, options.right, options.alpha ?? 1)
  const source = options.sourceFn
  return heat(op, initial(u0, grid), {
    ...options,
    ...(source === undefined ? {} : { source: (i: number, t: number) => source(grid.x[i] ?? 0, t) }),
  })
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
 * import { dirichlet, grid1d, wave1d } from 'whitebox-engine/dynamics/pde'
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
  const c2 = (options.c ?? 1) ** 2
  const op = laplacian1d(grid, options.left, options.right, c2)
  const v = initial(v0, grid)
  return runSteps(
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
    options,
  )
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
 * import { advection1d, grid1d } from 'whitebox-engine/dynamics/pde'
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
  const a = options.velocity
  const scheme = options.scheme ?? 'upwind'
  const bc = options.boundary ?? 'periodic'
  const n = grid.n
  const isPeriodic = bc === 'periodic'
  const m = isPeriodic ? n - 1 : n
  return runSteps(
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
      for (let i = 0; i < m; i++) {
        const [l, c, r] = [at(i - 1), at(i), at(i + 1)]
        if (scheme === 'upwind') next[i] = a >= 0 ? c - nu * (c - l) : c - nu * (r - c)
        else if (scheme === 'lax-friedrichs') next[i] = 0.5 * (l + r) - 0.5 * nu * (r - l)
        else next[i] = c - 0.5 * nu * (r - l) + 0.5 * nu * nu * (r - 2 * c + l)
      }
      if (isPeriodic) next[n - 1] = next[0] ?? 0
      else if (a > 0) next[0] = (bc as { inflow: number }).inflow
      else if (a < 0) next[n - 1] = (bc as { inflow: number }).inflow
      return { u: next, prev: u }
    },
    options,
  )
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
}

/** The result of {@link poisson}. */
export interface PoissonResult {
  readonly u: Tensor
  readonly iterations: number
  readonly converged: boolean
  readonly residualHistory: readonly number[]
}

/**
 * Solve the Poisson problem Delta u = f with boundary conditions on a 1-D or 2-D grid, by Jacobi, Gauss-Seidel or SOR
 * sweeps, conjugate gradients (GMRES when Neumann or Robin rows make the system unsymmetric) or geometric multigrid
 * V-cycles (Dirichlet conditions, 2^k + 1 nodes per side). Each sweep or cycle is one traced step with the relative
 * residual as a metric.
 *
 * @param f - The right-hand side at the nodes (values, or a function of the node's coordinates).
 * @param grid - A 1-D or 2-D grid.
 * @param bcs - The conditions: [left, right] in 1-D, the four sides in 2-D.
 * @param options - Method, relaxation, tolerance and iteration cap.
 * @returns The solution, iterations and residual history.
 * @throws WhiteboxError with code `value` when multigrid does not fit the grid.
 * @concept poisson-equation
 * @example
 * ```ts
 * import { dirichlet, grid1d, poisson } from 'whitebox-engine/dynamics/pde'
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
  if (method === 'cg') {
    const r = krylovPoisson(op, rhs, u, fixed, tol, maxIterations, options)
    return { ...r, u: shaped(Float64Array.from(r.u.data), grid) }
  }
  if (method === 'multigrid') return multigridPoisson(grid, fv, u, fixed, tol, maxIterations, options, residualOf)
  const rows = Array.from({ length: n }, (_, r) => rowOf(op.matrix, r))
  const h = 'a' in grid ? grid.dx : Math.max(grid.x.dx, grid.y.dx)
  const omega = method === 'sor' ? (options.omega ?? 2 / (1 + Math.sin(Math.PI * h))) : 1
  const algorithm: Algorithm<{ u: Float64Array; residual: number; history: number[] }> = {
    name: `poisson-${method}`,
    config: { method, omega, tol },
    schema: channels({ metrics: { residual: { description: 'relative residual', direction: 'lower' } } }),
    init: () => ({ u, residual: residualOf(u), history: [residualOf(u)] }),
    step: (s) => {
      const next = method === 'jacobi' ? Float64Array.from(s.u) : s.u
      const src = s.u
      for (let r = 0; r < n; r++) {
        if (fixed.has(r)) continue
        const { diag, off } = rows[r] as ReturnType<typeof rowOf>
        let acc = rhs[r] ?? 0
        for (const { c, v } of off) acc -= v * ((method === 'jacobi' ? src : next)[c] ?? 0)
        const gs = acc / diag
        next[r] = (1 - omega) * (src[r] ?? 0) + omega * gs
      }
      const residual = residualOf(next)
      s.history.push(residual)
      return { u: method === 'jacobi' ? next : Float64Array.from(next), residual, history: s.history }
    },
    done: (s) => (s.residual <= tol ? 'converged' : Number.isFinite(s.residual) ? undefined : 'diverged'),
    observe: (s) => ({ metrics: { residual: s.residual } }),
  }
  const run = runAlgorithm(algorithm, { maxSteps: maxIterations, ...options })
  return {
    u: shaped(run.state.u, grid),
    iterations: run.steps,
    converged: run.state.residual <= tol,
    residualHistory: run.state.history,
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

function multigridPoisson(
  grid: Grid1d | Grid2d,
  f: Float64Array,
  u0: Float64Array,
  fixed: ReadonlySet<number>,
  tol: number,
  maxIterations: number,
  options: PdeTraceOptions,
  residualOf: (x: Float64Array) => number,
): PoissonResult {
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
    throw new WhiteboxError(
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
      smooth(u, rhs, m, hh, 50)
      return
    }
    smooth(u, rhs, m, hh, 2)
    const rc = restrict(residual(u, rhs, m, hh), m)
    const mc = (m - 1) / 2 + 1
    const ec = new Float64Array(rc.length)
    vcycle(ec, rc, mc, 2 * hh)
    const e = prolong(ec, mc)
    for (let k = 0; k < u.length; k++) u[k] = (u[k] ?? 0) + (e[k] ?? 0)
    smooth(u, rhs, m, hh, 2)
  }
  const algorithm: Algorithm<{ u: Float64Array; residual: number; history: number[] }> = {
    name: 'poisson-multigrid',
    config: { method: 'multigrid', tol },
    schema: channels({ metrics: { residual: { description: 'relative residual', direction: 'lower' } } }),
    init: () => ({ u: u0, residual: residualOf(u0), history: [residualOf(u0)] }),
    step: (s) => {
      const u = Float64Array.from(s.u)
      vcycle(u, f, nx, h)
      const r = residualOf(u)
      s.history.push(r)
      return { u, residual: r, history: s.history }
    },
    done: (s) => (s.residual <= tol ? 'converged' : Number.isFinite(s.residual) ? undefined : 'diverged'),
    observe: (s) => ({ metrics: { residual: s.residual } }),
  }
  const run = runAlgorithm(algorithm, { maxSteps: maxIterations, ...options })
  return {
    u: shaped(run.state.u, grid),
    iterations: run.steps,
    converged: run.state.residual <= tol,
    residualHistory: run.state.history,
  }
}
