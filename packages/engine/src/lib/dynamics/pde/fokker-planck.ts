import { WhiteboxError } from '../../base/index.ts'
import { fromStorage, type Tensor } from '../../tensor/index.ts'
import { type Algorithm, channels, runAlgorithm } from '../../trace/index.ts'
import { sparseLu, sparseLuSolve } from '../ode/_sparse-lu.ts'
import type { PdeTraceOptions } from './finite-difference.ts'

/**
 * A 1-D diffusion process dX = mu(X) dt + sigma(X) dW, described for its Fokker-Planck equation
 * p_t = -(mu p)_x + (D p)_xx with D = sigma^2 / 2.
 */
export interface DiffusionProcess {
  readonly drift: (x: number) => number
  /** sigma(x) (the diffusion coefficient is sigma^2 / 2). */
  readonly sigma: (x: number) => number
}

/** The Bernoulli function B(z) = z / (e^z - 1), with B(0) = 1. */
function bernoulli(z: number): number {
  if (Math.abs(z) < 1e-8) return 1 - z / 2
  return z / Math.expm1(z)
}

/** A finite-volume discretisation on N cells of [a, b]: dp/dt = G p with zero-flux walls. */
export interface FokkerPlanckGrid {
  readonly centres: Float64Array
  readonly dx: number
  /** The generator as rows of (column, value) pairs. */
  readonly rows: readonly (readonly [number, number][])[]
}

/**
 * The Scharfetter-Gummel discretisation of the Fokker-Planck operator: the flux between cells i and i + 1 is
 * (D/dx) [B(-P) p_i - B(P) p_{i+1}] with the cell Peclet number P = mu dx / D at the face and the Bernoulli function
 * B. Exponential fitting keeps densities positive and makes the discrete stationary state exact for constant
 * coefficients; zero flux at the walls conserves mass exactly.
 *
 * @param process - Drift and sigma.
 * @param domain - [a, b].
 * @param cells - Number of cells.
 * @returns Cell centres and the generator G (dp/dt = G p).
 * @throws WhiteboxError with code `value` for fewer than two cells or a vanishing diffusion.
 * @concept fokker-planck
 * @example
 * ```ts
 * import { scharfetterGummel } from 'whitebox-engine/dynamics/pde'
 * scharfetterGummel({ drift: (x) => -x, sigma: () => 1 }, [-3, 3], 10).centres.length // => 10
 * ```
 */
export function scharfetterGummel(
  process: DiffusionProcess,
  domain: readonly [number, number],
  cells: number,
): FokkerPlanckGrid {
  if (!Number.isInteger(cells) || cells < 2) throw new WhiteboxError('value', 'scharfetterGummel: at least two cells')
  const [a, b] = domain
  const dx = (b - a) / cells
  const centres = Float64Array.from({ length: cells }, (_, i) => a + (i + 0.5) * dx)
  const rows: [number, number][][] = Array.from({ length: cells }, () => [])
  for (let i = 0; i < cells - 1; i++) {
    const xf = a + (i + 1) * dx
    const s = process.sigma(xf)
    const D = 0.5 * s * s
    if (!(D > 0)) throw new WhiteboxError('value', `scharfetterGummel: the diffusion vanishes at x = ${xf}`)
    const P = (process.drift(xf) * dx) / D
    // Flux J = c_left p_i - c_right p_{i+1}; cell i loses J / dx, cell i + 1 gains it.
    const cl = (D / dx) * bernoulli(-P)
    const cr = (D / dx) * bernoulli(P)
    rows[i]?.push([i, -cl / dx], [i + 1, cr / dx])
    rows[i + 1]?.push([i, cl / dx], [i + 1, -cr / dx])
  }
  return { centres, dx, rows }
}

/** Options of {@link fokkerPlanck}. */
export interface FokkerPlanckOptions extends PdeTraceOptions {
  readonly dt: number
  readonly tEnd: number
  /** 1 backward Euler (default, positivity preserving), 0.5 Crank-Nicolson. */
  readonly theta?: number
  readonly every?: number
}

/** The result of {@link fokkerPlanck}. */
export interface DensityEvolution {
  readonly t: readonly number[]
  readonly x: Float64Array
  /** Densities `[nt, cells]` (each integrates to the initial mass). */
  readonly p: Tensor
  readonly mean: readonly number[]
  readonly variance: readonly number[]
}

function assemble(grid: FokkerPlanckGrid, alpha: number, beta: number): ReturnType<typeof sparseLu> {
  const n = grid.centres.length
  const r: number[] = []
  const c: number[] = []
  const v: number[] = []
  for (let i = 0; i < n; i++) {
    r.push(i)
    c.push(i)
    v.push(alpha)
    for (const [j, w] of grid.rows[i] ?? []) {
      r.push(i)
      c.push(j)
      v.push(beta * w)
    }
  }
  return sparseLu(n, r, c, v)
}

function applyGenerator(grid: FokkerPlanckGrid, p: Float64Array): Float64Array {
  const out = new Float64Array(p.length)
  grid.rows.forEach((row, i) => {
    let s = 0
    for (const [j, w] of row) s += w * (p[j] ?? 0)
    out[i] = s
  })
  return out
}

function moments(x: Float64Array, p: Float64Array, dx: number): { mean: number; variance: number } {
  let m0 = 0
  let m1 = 0
  let m2 = 0
  for (let i = 0; i < p.length; i++) {
    const w = (p[i] ?? 0) * dx
    m0 += w
    m1 += w * (x[i] ?? 0)
    m2 += w * (x[i] ?? 0) ** 2
  }
  const mean = m1 / m0
  return { mean, variance: m2 / m0 - mean * mean }
}

/**
 * Evolve a probability density under the Fokker-Planck equation of a 1-D diffusion with the Scharfetter-Gummel
 * discretisation and the theta method (backward Euler by default: positive and mass conserving). A traced algorithm
 * with the mean, variance and mass per step.
 *
 * @param process - Drift and sigma.
 * @param p0 - The initial density on the cell centres (or a function of x).
 * @param domain - [a, b] (wide enough that the density vanishes at the walls).
 * @param cells - Number of cells.
 * @param options - Step, final time, theta and thinning.
 * @returns Densities with their moments.
 * @concept fokker-planck
 * @example
 * ```ts
 * import { fokkerPlanck } from 'whitebox-engine/dynamics/pde'
 * // Ornstein-Uhlenbeck dX = -X dt + dW from a point mass near 1: the mean decays as e^{-t}.
 * const ev = fokkerPlanck({ drift: (x) => -x, sigma: () => 1 }, (x) => Math.exp(-((x - 1) ** 2) / 0.02), [-4, 4], 200, { dt: 0.01, tEnd: 1 })
 * Math.abs((ev.mean.at(-1) ?? 0) - Math.exp(-1)) < 1e-2 // => true
 * ```
 */
export function fokkerPlanck(
  process: DiffusionProcess,
  p0: ArrayLike<number> | ((x: number) => number),
  domain: readonly [number, number],
  cells: number,
  options: FokkerPlanckOptions,
): DensityEvolution {
  const grid = scharfetterGummel(process, domain, cells)
  const x = grid.centres
  const start = typeof p0 === 'function' ? Float64Array.from(x, (v) => p0(v)) : Float64Array.from(p0)
  if (start.length !== cells) throw new WhiteboxError('value', `fokkerPlanck: give ${cells} initial densities`)
  // Normalise to a probability density.
  const mass0 = start.reduce((s, v) => s + v * grid.dx, 0)
  if (!(mass0 > 0)) throw new WhiteboxError('value', 'fokkerPlanck: the initial density has no mass')
  for (let i = 0; i < cells; i++) start[i] = (start[i] ?? 0) / mass0
  const theta = options.theta ?? 1
  const dt = options.dt
  if (!(dt > 0)) throw new WhiteboxError('value', 'fokkerPlanck: dt must be positive')
  const lu = assemble(grid, 1, -theta * dt)
  const steps = Math.ceil(options.tEnd / dt - 1e-9)
  const every = Math.max(1, options.every ?? 1)
  const algorithm: Algorithm<{ p: Float64Array; t: number; ts: number[]; ps: Float64Array[] }> = {
    name: 'fokker-planck',
    config: { cells, dt, theta, tEnd: options.tEnd },
    schema: channels({
      metrics: {
        t: { description: 'time' },
        mean: { description: 'mean' },
        variance: { description: 'variance' },
        mass: { description: 'total probability' },
      },
      tensors: { p: { role: 'state', description: 'the density on the cells' } },
    }),
    init: () => ({ p: start, t: 0, ts: [0], ps: [start] }),
    step: (s, ctx) => {
      const gp = applyGenerator(grid, s.p)
      const rhs = Float64Array.from(s.p, (v, i) => v + (1 - theta) * dt * (gp[i] ?? 0))
      const p = sparseLuSolve(lu, rhs)
      const t = ctx.t + 1 === steps ? options.tEnd : s.t + dt
      if ((ctx.t + 1) % every === 0 || ctx.t + 1 === steps) {
        s.ts.push(t)
        s.ps.push(p)
      }
      return { ...s, p, t }
    },
    observe: (s) => {
      const m = moments(x, s.p, grid.dx)
      return {
        metrics: { t: s.t, mean: m.mean, variance: m.variance, mass: s.p.reduce((a, v) => a + v * grid.dx, 0) },
        tensors: { p: { role: 'state', value: s.p } },
      }
    },
  }
  const run = runAlgorithm(algorithm, { maxSteps: steps, ...options })
  const flat = new Float64Array(run.state.ps.length * cells)
  run.state.ps.forEach((p, i) => {
    flat.set(p, i * cells)
  })
  const ms = run.state.ps.map((p) => moments(x, p, grid.dx))
  return {
    t: run.state.ts,
    x,
    p: fromStorage(flat, [run.state.ps.length, cells], 'float64'),
    mean: ms.map((m) => m.mean),
    variance: ms.map((m) => m.variance),
  }
}

/**
 * The stationary density of a 1-D diffusion with reflecting walls: p(x) proportional to exp(int mu / D) / D with
 * D = sigma^2 / 2, by cumulative trapezoidal quadrature on a fine grid and normalisation (the zero-flux solution).
 *
 * @param process - Drift and sigma.
 * @param domain - [a, b].
 * @param n - Grid points (default 401).
 * @returns The grid and the normalised density.
 * @concept fokker-planck
 * @example
 * ```ts
 * import { stationaryDensity } from 'whitebox-engine/dynamics/pde'
 * // OU with theta = 1, sigma = 1: N(0, 1/2), density 1/sqrt(pi) at 0.
 * const s = stationaryDensity({ drift: (x) => -x, sigma: () => 1 }, [-6, 6], 401)
 * Math.abs((s.p[200] ?? 0) - 1 / Math.sqrt(Math.PI)) < 1e-4 // => true
 * ```
 */
export function stationaryDensity(
  process: DiffusionProcess,
  domain: readonly [number, number],
  n = 401,
): { readonly x: Float64Array; readonly p: Float64Array } {
  const [a, b] = domain
  const x = Float64Array.from({ length: n }, (_, i) => a + ((b - a) * i) / (n - 1))
  const ratio = Float64Array.from(x, (v) => process.drift(v) / (0.5 * process.sigma(v) ** 2))
  const phi = new Float64Array(n)
  for (let i = 1; i < n; i++)
    phi[i] = (phi[i - 1] ?? 0) + 0.5 * ((ratio[i] ?? 0) + (ratio[i - 1] ?? 0)) * ((x[i] ?? 0) - (x[i - 1] ?? 0))
  let top = Number.NEGATIVE_INFINITY
  for (const v of phi) top = Math.max(top, v)
  const p = Float64Array.from(x, (v, i) => Math.exp((phi[i] ?? 0) - top) / (0.5 * process.sigma(v) ** 2))
  let mass = 0
  for (let i = 1; i < n; i++) mass += 0.5 * ((p[i] ?? 0) + (p[i - 1] ?? 0)) * ((x[i] ?? 0) - (x[i - 1] ?? 0))
  for (let i = 0; i < n; i++) p[i] = (p[i] ?? 0) / mass
  return { x, p }
}

/**
 * The stationary density of the discrete Scharfetter-Gummel generator: the normalised null vector of G (one equation
 * replaced by the normalisation sum p dx = 1).
 *
 * @param process - Drift and sigma.
 * @param domain - [a, b].
 * @param cells - Number of cells.
 * @returns Cell centres and the density.
 * @concept fokker-planck
 * @example
 * ```ts
 * import { stationaryDensityDiscrete } from 'whitebox-engine/dynamics/pde'
 * const s = stationaryDensityDiscrete({ drift: (x) => -x, sigma: () => 1 }, [-6, 6], 120)
 * Math.abs(s.p.reduce((a, v) => a + v * 0.1, 0) - 1) < 1e-12 // => true
 * ```
 */
export function stationaryDensityDiscrete(
  process: DiffusionProcess,
  domain: readonly [number, number],
  cells: number,
): { readonly x: Float64Array; readonly p: Float64Array } {
  const grid = scharfetterGummel(process, domain, cells)
  const r: number[] = []
  const c: number[] = []
  const v: number[] = []
  for (let i = 0; i < cells - 1; i++) {
    for (const [j, w] of grid.rows[i] ?? []) {
      r.push(i)
      c.push(j)
      v.push(w)
    }
  }
  for (let j = 0; j < cells; j++) {
    r.push(cells - 1)
    c.push(j)
    v.push(grid.dx)
  }
  const rhs = new Float64Array(cells)
  rhs[cells - 1] = 1
  return { x: grid.centres, p: sparseLuSolve(sparseLu(cells, r, c, v), rhs) }
}
