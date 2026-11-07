/**
 * Geometric multigrid for model problems: the Poisson matrices on 1-D and 2-D grids, linear and bilinear
 * prolongation between nested grids, and the two-grid, V and W cycles with Galerkin coarse operators and a stationary
 * smoother, each cycle one step of a traced algorithm whose state serialises and whose three stages (pre-smooth,
 * coarse-grid correction, post-smooth) are reported separately.
 */

import { type Json, MdmlError } from '../../base/index.ts'
import { fromStorage, primalOf, promoteFloat, type Tensor, type TensorLike, toFlat } from '../../tensor/index.ts'
import { type Algorithm, channels, type RunOptions, runAlgorithm } from '../../trace/index.ts'
import { solveKernel } from '../linalg/_kernels.ts'
import { dnrm2 as norm } from '../linalg/blas.ts'
import { csrOf, type Rows, residualNormOf, rowsOf, sweep } from './_rows.ts'
import { type CsrMatrix, csr, fromCoo, primalValues, transposeCsr } from './csr.ts'
import type { KrylovResult } from './iterative.ts'
import type { StationaryMethod } from './stationary.ts'

/**
 * The 1-D Poisson matrix tridiag(-1, 2, -1) of `n` interior points (the second difference with the grid spacing
 * absorbed into the right-hand side).
 *
 * @param n - The number of interior grid points.
 * @returns The `[n, n]` CSR matrix.
 * @example
 * ```ts
 * import { poisson1d } from 'mdml-engine/numerics/sparse'
 * Array.from(poisson1d(3).indices) // => [0, 1, 0, 1, 2, 1, 2]
 * ```
 */
export function poisson1d(n: number): CsrMatrix {
  if (!Number.isInteger(n) || n < 1) throw new MdmlError('value', 'poisson1d needs a positive size')
  const rows: number[] = []
  const cols: number[] = []
  const vals: number[] = []
  for (let i = 0; i < n; i++) {
    if (i > 0) {
      rows.push(i)
      cols.push(i - 1)
      vals.push(-1)
    }
    rows.push(i)
    cols.push(i)
    vals.push(2)
    if (i < n - 1) {
      rows.push(i)
      cols.push(i + 1)
      vals.push(-1)
    }
  }
  return fromCoo(rows, cols, vals, [n, n])
}

/**
 * The 2-D Poisson matrix (the five-point Laplacian, 4 on the diagonal and -1 for each neighbour) on an `nx` by `ny`
 * grid of interior points, numbered row by row (x fastest).
 *
 * @param nx - Points in x.
 * @param ny - Points in y (default `nx`).
 * @returns The `[nx ny, nx ny]` CSR matrix.
 * @example
 * ```ts
 * import { poisson2d } from 'mdml-engine/numerics/sparse'
 * poisson2d(3).shape // => [9, 9]
 * ```
 */
export function poisson2d(nx: number, ny: number = nx): CsrMatrix {
  if (!Number.isInteger(nx) || !Number.isInteger(ny) || nx < 1 || ny < 1)
    throw new MdmlError('value', 'poisson2d needs positive sizes')
  const rows: number[] = []
  const cols: number[] = []
  const vals: number[] = []
  const at = (i: number, j: number): number => j * nx + i
  for (let j = 0; j < ny; j++)
    for (let i = 0; i < nx; i++) {
      const r = at(i, j)
      for (const [di, dj] of [
        [0, -1],
        [-1, 0],
      ] as const)
        if (i + di >= 0 && j + dj >= 0) {
          rows.push(r)
          cols.push(at(i + di, j + dj))
          vals.push(-1)
        }
      rows.push(r)
      cols.push(r)
      vals.push(4)
      for (const [di, dj] of [
        [1, 0],
        [0, 1],
      ] as const)
        if (i + di < nx && j + dj < ny) {
          rows.push(r)
          cols.push(at(i + di, j + dj))
          vals.push(-1)
        }
    }
  return fromCoo(rows, cols, vals, [nx * ny, nx * ny])
}

/** Linear interpolation from `nc` coarse interior points to the `2 nc + 1` fine ones, as a dense row-major array. */
function interpolation1d(nc: number): Float64Array {
  const nf = 2 * nc + 1
  const p = new Float64Array(nf * nc)
  for (let j = 0; j < nc; j++) {
    p[(2 * j + 1) * nc + j] = 1
    p[2 * j * nc + j] = 0.5
    p[(2 * j + 2) * nc + j] = 0.5
  }
  return p
}

/**
 * The prolongation from the coarse interior grid that every second fine point makes to the fine grid: linear
 * interpolation in 1-D (`[n]` points, `n = 2 m + 1`), bilinear in 2-D (`[nx, ny]`). Each coarse point is the fine
 * point between two others; the coarse values are copied there and averaged in between.
 *
 * @param fine - The fine grid's interior points per axis (each odd).
 * @returns The `[fineSize, coarseSize]` CSR matrix.
 * @example
 * ```ts
 * import { prolongation } from 'mdml-engine/numerics/sparse'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(prolongation([3]).values) // => [0.5, 1, 0.5]
 * ```
 */
export function prolongation(fine: readonly number[]): CsrMatrix {
  if (fine.length < 1 || fine.length > 2) throw new MdmlError('value', 'prolongation takes one or two axis sizes')
  const coarse = fine.map((n) => {
    if (!Number.isInteger(n) || n < 3 || n % 2 === 0)
      throw new MdmlError('value', 'prolongation needs an odd number (at least 3) of interior points per axis')
    return (n - 1) / 2
  })
  const [nx, ny] = fine as [number, number?]
  const [cx, cy] = coarse as [number, number?]
  const px = interpolation1d(cx)
  if (ny === undefined || cy === undefined) return csr({ ...denseTriplets(px, nx, cx) })
  const py = interpolation1d(cy)
  const rows: number[] = []
  const cols: number[] = []
  const vals: number[] = []
  for (let jf = 0; jf < ny; jf++)
    for (let jc = 0; jc < cy; jc++) {
      const wy = py[jf * cy + jc] ?? 0
      if (wy === 0) continue
      for (let i = 0; i < nx; i++)
        for (let ic = 0; ic < cx; ic++) {
          const wx = px[i * cx + ic] ?? 0
          if (wx === 0) continue
          rows.push(jf * nx + i)
          cols.push(jc * cx + ic)
          vals.push(wx * wy)
        }
    }
  return fromCoo(rows, cols, vals, [nx * ny, cx * cy])
}

function denseTriplets(
  p: Float64Array,
  nf: number,
  nc: number,
): { shape: [number, number]; values: number[]; indptr: number[]; indices: number[] } {
  const indptr = [0]
  const indices: number[] = []
  const values: number[] = []
  for (let i = 0; i < nf; i++) {
    for (let j = 0; j < nc; j++) {
      const v = p[i * nc + j] ?? 0
      if (v !== 0) {
        indices.push(j)
        values.push(v)
      }
    }
    indptr.push(indices.length)
  }
  return { shape: [nf, nc], values, indptr, indices }
}

/** The prolongations of a geometric hierarchy from the finest grid to `levels` grids (fine to coarse). */
function geometricProlongations(fine: readonly number[], levels: number): CsrMatrix[] {
  const out: CsrMatrix[] = []
  let shape = [...fine]
  for (let l = 1; l < levels; l++) {
    out.push(prolongation(shape))
    shape = shape.map((n) => (n - 1) / 2)
  }
  return out
}

/** The smoother and cycle shape of a multigrid cycle. */
export interface MultigridOptions {
  /**
   * The grids: the finest grid's interior points per axis (`[n]` or `[nx, ny]`, each `2^k - 1`-like, odd at every
   * level), with `levels` grids in all. Alternatively give `prolongations`.
   */
  readonly shape?: readonly number[]
  /** The number of grids, the finest included (default 2: the two-grid method). */
  readonly levels?: number
  /** The prolongations from each grid to the one above it, fine to coarse, instead of `shape` (any sparse matrices). */
  readonly prolongations?: readonly (CsrMatrix | TensorLike)[]
  /** The smoother (default damped Jacobi). */
  readonly smoother?: StationaryMethod
  /** The smoother's relaxation (default 2/3 for Jacobi, 1 for Gauss-Seidel, 1.2 for SOR). */
  readonly omega?: number
  /** Smoothing sweeps before restriction and after prolongation (defaults 1 and 1). */
  readonly preSweeps?: number
  readonly postSweeps?: number
  /** Coarse-grid corrections per level: `'v'` (one) or `'w'` (two). Default `'v'`. */
  readonly cycle?: 'v' | 'w'
  readonly x0?: ArrayLike<number>
  /** Stop when the residual norm is at most `max(rtol |b|, atol)` (defaults 1e-8 and 0). */
  readonly rtol?: number
  readonly atol?: number
}

/** One grid of a hierarchy: its matrix and the prolongation from the next coarser grid. */
interface Level {
  readonly rows: Rows
  readonly a: CsrMatrix
  readonly p?: CsrMatrix
  readonly pt?: CsrMatrix
  /** The dense matrix of the coarsest grid, for the direct solve. */
  readonly dense?: Float64Array
}

function denseOf(a: CsrMatrix): Float64Array {
  const [m, n] = a.shape
  const out = new Float64Array(m * n)
  const values = primalValues(a)
  for (let k = 0; k < a.indices.length; k++) {
    const at = (a.rows[k] ?? 0) * n + (a.indices[k] ?? 0)
    out[at] = (out[at] ?? 0) + (values[k] ?? 0)
  }
  return out
}

/** The Galerkin coarse matrix P^T A P, by rows of the sparse operands. */
function galerkin(a: CsrMatrix, p: CsrMatrix): CsrMatrix {
  const nf = a.shape[0]
  const nc = p.shape[1]
  const pv = primalValues(p)
  const av = primalValues(a)
  // A P, row by row, as dense rows of the coarse width.
  const ap = new Float64Array(nf * nc)
  for (let i = 0; i < nf; i++)
    for (let k = a.indptr[i] ?? 0; k < (a.indptr[i + 1] ?? 0); k++) {
      const j = a.indices[k] ?? 0
      for (let q = p.indptr[j] ?? 0; q < (p.indptr[j + 1] ?? 0); q++) {
        const at = i * nc + (p.indices[q] ?? 0)
        ap[at] = (ap[at] ?? 0) + (av[k] ?? 0) * (pv[q] ?? 0)
      }
    }
  const out = new Float64Array(nc * nc)
  for (let i = 0; i < nf; i++)
    for (let q = p.indptr[i] ?? 0; q < (p.indptr[i + 1] ?? 0); q++) {
      const c = p.indices[q] ?? 0
      const w = pv[q] ?? 0
      for (let j = 0; j < nc; j++) out[c * nc + j] = (out[c * nc + j] ?? 0) + w * (ap[i * nc + j] ?? 0)
    }
  return csr(fromStorage(out, [nc, nc], 'float64'))
}

function hierarchy(a: CsrMatrix | TensorLike, options: MultigridOptions): Level[] {
  const fine = csrOf(a)
  let ps: CsrMatrix[]
  if (options.prolongations !== undefined) ps = options.prolongations.map(csrOf)
  else if (options.shape !== undefined) ps = geometricProlongations(options.shape, options.levels ?? 2)
  else throw new MdmlError('value', 'multigrid needs a grid `shape` or explicit `prolongations`')
  const levels: Level[] = []
  let current = fine
  for (const p of ps) {
    if (p.shape[0] !== current.shape[0])
      throw new MdmlError(
        'shape',
        `multigrid: a prolongation has ${p.shape[0]} rows, its grid has ${current.shape[0]} points`,
      )
    levels.push({ rows: rowsOf(current), a: current, p, pt: transposeCsr(p) })
    current = galerkin(current, p)
  }
  levels.push({ rows: rowsOf(current), a: current, dense: denseOf(current) })
  return levels
}

function matvec(a: CsrMatrix, v: ArrayLike<number>): Float64Array {
  const av = primalValues(a)
  const out = new Float64Array(a.shape[0])
  for (let i = 0; i < a.shape[0]; i++) {
    let s = 0
    for (let k = a.indptr[i] ?? 0; k < (a.indptr[i + 1] ?? 0); k++) s += (av[k] ?? 0) * (v[a.indices[k] ?? 0] ?? 0)
    out[i] = s
  }
  return out
}

function residualOf(level: Level, b: ArrayLike<number>, x: ArrayLike<number>): Float64Array {
  const ax = matvec(level.a, x)
  return Float64Array.from(ax, (v, i) => (b[i] ?? 0) - v)
}

interface CycleSettings {
  readonly smoother: StationaryMethod
  readonly omega: number
  readonly pre: number
  readonly post: number
  readonly corrections: number
}

function smooth(level: Level, b: ArrayLike<number>, x0: ArrayLike<number>, sweeps: number, s: CycleSettings): number[] {
  let x = Array.from(x0)
  for (let k = 0; k < sweeps; k++) x = sweep(level.rows, b, x, s.smoother, s.omega)
  return x
}

/** One cycle from grid `l`, returning the iterate after the pre-smoothing, the correction and the post-smoothing. */
function cycleFrom(
  levels: readonly Level[],
  l: number,
  b: ArrayLike<number>,
  x0: ArrayLike<number>,
  s: CycleSettings,
): { pre: number[]; corrected: number[]; post: number[] } {
  const level = levels[l] as Level
  if (level.dense !== undefined) {
    const x = Array.from(solveKernel(level.dense, Float64Array.from(b), level.rows.n, 1))
    return { pre: x, corrected: x, post: x }
  }
  const pre = smooth(level, b, x0, s.pre, s)
  const r = residualOf(level, b, pre)
  const rc = matvec(level.pt as CsrMatrix, r)
  let ec: number[] = new Array<number>(rc.length).fill(0)
  for (let g = 0; g < s.corrections; g++) ec = cycleFrom(levels, l + 1, rc, ec, s).post
  const pe = matvec(level.p as CsrMatrix, ec)
  const corrected = pre.map((v, i) => v + (pe[i] ?? 0))
  return { pre, corrected, post: smooth(level, b, corrected, s.post, s) }
}

/** The state of a multigrid algorithm: the iterate, residual history, and the residual norm after each stage of the last cycle. */
export interface MultigridState {
  readonly x: readonly number[]
  readonly residualNorm: number
  readonly history: readonly number[]
  /** The residual norms after the pre-smoothing, the coarse-grid correction and the post-smoothing of the last cycle. */
  readonly stages: readonly [number, number, number]
}

function settingsOf(options: MultigridOptions): CycleSettings {
  const smoother = options.smoother ?? 'jacobi'
  const omega = options.omega ?? (smoother === 'jacobi' ? 2 / 3 : smoother === 'sor' ? 1.2 : 1)
  const pre = options.preSweeps ?? 1
  const post = options.postSweeps ?? 1
  if (!Number.isInteger(pre) || !Number.isInteger(post) || pre < 0 || post < 0)
    throw new MdmlError('value', 'multigrid sweep counts are non-negative integers')
  return { smoother, omega, pre, post, corrections: options.cycle === 'w' ? 2 : 1 }
}

const SCHEMA = channels({
  metrics: {
    residualNorm: { description: 'norm of the residual b - A x after the cycle', direction: 'lower' },
    preSmooth: { description: 'residual norm after the pre-smoothing', direction: 'lower' },
    coarseCorrection: { description: 'residual norm after the coarse-grid correction' },
    postSmooth: { description: 'residual norm after the post-smoothing', direction: 'lower' },
  },
  tensors: { x: { role: 'state', description: 'current iterate' } },
})

/**
 * A multigrid method as a traced algorithm, one cycle per step: smooth, restrict the residual, correct on the coarser
 * grids (a direct solve at the coarsest), prolong and add, smooth again. The coarse operators are Galerkin products
 * P^T A P and the restriction is P^T, so the coarse correction is the A-orthogonal projection of the error onto the
 * range of P. Two grids is `levels: 2` (the default), a V-cycle more, a W-cycle with `cycle: 'w'`. This is the
 * algebraic form, for any matrix with a grid hierarchy; `poissonAlgorithm` of `dynamics/pde` with `method: 'multigrid'` is
 * the matrix-free geometric V-cycle on the five-point stencil (red-black Gauss-Seidel, full weighting).
 *
 * @param a - The matrix of the finest grid (for example {@link poisson1d}).
 * @param b - The right-hand side.
 * @param options - The grids, smoother, sweeps, cycle shape, start and tolerances.
 * @returns The algorithm, named `two-grid`, `v-cycle` or `w-cycle`.
 * @example
 * ```ts
 * import { multigridAlgorithm, poisson1d } from 'mdml-engine/numerics/sparse'
 * multigridAlgorithm(poisson1d(7), new Array(7).fill(1), { shape: [7], levels: 3 }).name // => 'v-cycle'
 * ```
 */
export function multigridAlgorithm(
  a: CsrMatrix | TensorLike,
  b: TensorLike,
  options: MultigridOptions = {},
): Algorithm<MultigridState> {
  const levels = hierarchy(a, options)
  const top = levels[0] as Level
  const settings = settingsOf(options)
  const bv = Float64Array.from(toFlat(primalOf(promoteFloat(b)[0] as Tensor)))
  if (bv.length !== top.rows.n) throw new MdmlError('shape', 'multigrid needs b with one entry per grid point')
  const rtol = options.rtol ?? 1e-8
  const atol = options.atol ?? 0
  const tol = Math.max(rtol * norm(bv), atol)
  const x0 = options.x0 === undefined ? new Array<number>(top.rows.n).fill(0) : Array.from(options.x0)
  if (x0.length !== top.rows.n) throw new MdmlError('shape', 'multigrid needs x0 with one entry per grid point')
  const name = levels.length === 2 ? 'two-grid' : settings.corrections === 2 ? 'w-cycle' : 'v-cycle'
  return {
    name,
    config: {
      levels: levels.length,
      sizes: levels.map((l) => l.rows.n),
      smoother: settings.smoother,
      omega: settings.omega,
      preSweeps: settings.pre,
      postSweeps: settings.post,
      corrections: settings.corrections,
      rtol,
      atol,
    } as Json,
    schema: SCHEMA,
    init: () => {
      const rn = residualNormOf(top.rows, x0, bv)
      return { x: x0, residualNorm: rn, history: [rn], stages: [rn, rn, rn] }
    },
    step: (s) => {
      const c = cycleFrom(levels, 0, bv, s.x, settings)
      const stage = (x: number[]): number => residualNormOf(top.rows, x, bv)
      const rn = stage(c.post)
      return {
        x: c.post,
        residualNorm: rn,
        history: [...s.history, rn],
        stages: [stage(c.pre), stage(c.corrected), rn],
      }
    },
    done: (s) => (s.residualNorm <= tol ? 'converged' : Number.isFinite(s.residualNorm) ? undefined : 'diverged'),
    observe: (s) => ({
      metrics: {
        residualNorm: s.residualNorm,
        preSmooth: s.stages[0],
        coarseCorrection: s.stages[1],
        postSmooth: s.stages[2],
      },
      tensors: { x: { role: 'state' as const, value: s.x } },
    }),
    serialise: (s) =>
      ({ x: [...s.x], residualNorm: s.residualNorm, history: [...s.history], stages: [...s.stages] }) as Json,
    deserialise: (j) => {
      const o = j as { x: number[]; residualNorm: number; history: number[]; stages: [number, number, number] }
      return { x: o.x, residualNorm: o.residualNorm, history: o.history, stages: o.stages }
    },
  }
}

/**
 * One multigrid cycle from an iterate, with the iterate after each of its three stages (for the error by frequency
 * after the smoothing and after the coarse-grid correction).
 *
 * @param a - The matrix of the finest grid.
 * @param b - The right-hand side.
 * @param x - The iterate to start from.
 * @param options - The grids, smoother, sweeps and cycle shape.
 * @returns The iterate after pre-smoothing, after the coarse-grid correction, and after post-smoothing.
 * @example
 * ```ts
 * import { multigridCycle, poisson1d } from 'mdml-engine/numerics/sparse'
 * multigridCycle(poisson1d(7), new Array(7).fill(1), new Array(7).fill(0), { shape: [7] }).post.length // => 7
 * ```
 */
export function multigridCycle(
  a: CsrMatrix | TensorLike,
  b: TensorLike,
  x: ArrayLike<number>,
  options: MultigridOptions = {},
): { readonly pre: number[]; readonly corrected: number[]; readonly post: number[] } {
  const levels = hierarchy(a, options)
  const bv = Float64Array.from(toFlat(primalOf(promoteFloat(b)[0] as Tensor)))
  return cycleFrom(levels, 0, bv, x, settingsOf(options))
}

/**
 * Solve A x = b by multigrid cycles to a residual tolerance or a cap on cycles (default 100).
 *
 * @param a - The matrix of the finest grid.
 * @param b - The right-hand side.
 * @param options - The grids, smoother, sweeps, cycle shape, start, tolerances, cycle cap and trace options.
 * @returns The solution with its residual history (one entry per cycle).
 * @example
 * ```ts
 * import { multigridSolve, poisson1d } from 'mdml-engine/numerics/sparse'
 * multigridSolve(poisson1d(15), new Array(15).fill(1), { shape: [15], levels: 3 }).converged // => true
 * ```
 */
export function multigridSolve(
  a: CsrMatrix | TensorLike,
  b: TensorLike,
  options: MultigridOptions & Partial<Omit<RunOptions, 'maxSteps'>> & { readonly maxIterations?: number } = {},
): KrylovResult {
  const { maxIterations, seed, key, tracer, policy, runId, parent, ...rest } = options
  const algorithm = multigridAlgorithm(a, b, rest)
  const run: Partial<RunOptions> = {
    ...(seed === undefined ? {} : { seed }),
    ...(key === undefined ? {} : { key }),
    ...(tracer === undefined ? {} : { tracer }),
    ...(policy === undefined ? {} : { policy }),
    ...(runId === undefined ? {} : { runId }),
    ...(parent === undefined ? {} : { parent }),
  }
  const result = runAlgorithm(algorithm, { ...run, maxSteps: maxIterations ?? 100 })
  const n = result.state.x.length
  return {
    x: fromStorage(Float64Array.from(result.state.x), [n], 'float64'),
    iterations: result.steps,
    converged: result.status === 'converged',
    status: result.status,
    residualNorm: result.state.residualNorm,
    history: result.state.history,
  }
}
