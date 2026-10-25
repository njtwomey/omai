/**
 * Finite-volume solvers for scalar conservation laws u_t + f(u)_x = 0, which update cell averages by fluxes across cell
 * faces and so capture shocks. Burgers' and linear fluxes, the Godunov and Rusanov numerical fluxes, the classical TVD
 * slope limiters, Godunov and MUSCL-Hancock schemes, and the exact Burgers Riemann solution for comparison.
 */

import { MdmlError } from '../../base/index.ts'
import { fromStorage, type Tensor } from '../../tensor/index.ts'
import { type Algorithm, channels, runAlgorithm } from '../../trace/index.ts'
import type { PdeTraceOptions } from './finite-difference.ts'

/** A scalar flux function f(u) with its derivative (the characteristic speed). */
export interface Flux {
  readonly f: (u: number) => number
  readonly df: (u: number) => number
  /** For convex (or concave) fluxes, the point where f' = 0 (the sonic point), used by the exact Godunov flux. */
  readonly critical?: number
}

/** Burgers' flux u^2 / 2 (convex, sonic point 0). */
export const BURGERS: Flux = { f: (u) => 0.5 * u * u, df: (u) => u, critical: 0 }

/**
 * The linear flux a u.
 *
 * @param a - The velocity.
 * @returns The flux.
 * @example
 * ```ts
 * import { linearFlux } from 'mdml-engine/dynamics/pde'
 * linearFlux(2).f(3) // => 6
 * ```
 */
export function linearFlux(a: number): Flux {
  return { f: (u) => a * u, df: () => a }
}

/** A numerical flux F(uL, uR) at a cell interface. */
export type NumericalFlux = (flux: Flux, uL: number, uR: number) => number

/**
 * Godunov's flux: the flux of the exact Riemann solution at the interface, min over [uL, uR] of f when uL <= uR and
 * max over [uR, uL] otherwise. Exact for convex fluxes with a known sonic point; otherwise the extremum is taken over
 * the end points and a dense sample.
 *
 * @param flux - The flux.
 * @param uL - Left state.
 * @param uR - Right state.
 * @returns The interface flux.
 * @concept godunov-method
 * @example
 * ```ts
 * import { BURGERS, godunovFlux } from 'mdml-engine/dynamics/pde'
 * // A transonic rarefaction takes the sonic value f(0) = 0.
 * godunovFlux(BURGERS, -1, 1) // => 0
 * ```
 */
export function godunovFlux(flux: Flux, uL: number, uR: number): number {
  const lo = Math.min(uL, uR)
  const hi = Math.max(uL, uR)
  const candidates = [flux.f(uL), flux.f(uR)]
  if (flux.critical !== undefined) {
    if (flux.critical > lo && flux.critical < hi) candidates.push(flux.f(flux.critical))
  } else {
    for (let k = 1; k < 32; k++) candidates.push(flux.f(lo + ((hi - lo) * k) / 32))
  }
  let out = candidates[0] ?? 0
  for (const c of candidates) out = uL <= uR ? Math.min(out, c) : Math.max(out, c)
  return out
}

/**
 * The Rusanov (local Lax-Friedrichs) flux (f(uL) + f(uR))/2 - max(|f'(uL)|, |f'(uR)|) (uR - uL)/2.
 *
 * @param flux - The flux.
 * @param uL - Left state.
 * @param uR - Right state.
 * @returns The interface flux.
 * @example
 * ```ts
 * import { BURGERS, rusanovFlux } from 'mdml-engine/dynamics/pde'
 * rusanovFlux(BURGERS, 1, 1) // => 0.5
 * ```
 */
export function rusanovFlux(flux: Flux, uL: number, uR: number): number {
  const s = Math.max(Math.abs(flux.df(uL)), Math.abs(flux.df(uR)))
  return 0.5 * (flux.f(uL) + flux.f(uR)) - 0.5 * s * (uR - uL)
}

/** A slope limiter phi(r) of the ratio of consecutive differences. */
export type Limiter = (r: number) => number

/** The classical slope limiters (all TVD). */
export const LIMITERS: Readonly<Record<'minmod' | 'superbee' | 'vanLeer' | 'mc' | 'vanAlbada' | 'none', Limiter>> = {
  minmod: (r) => Math.max(0, Math.min(1, r)),
  superbee: (r) => Math.max(0, Math.min(2 * r, 1), Math.min(r, 2)),
  vanLeer: (r) => (r + Math.abs(r)) / (1 + Math.abs(r)),
  mc: (r) => Math.max(0, Math.min(2 * r, 0.5 * (1 + r), 2)),
  vanAlbada: (r) => (r > 0 ? (r * r + r) / (r * r + 1) : 0),
  none: () => 0,
}

/** Options of {@link conservationLaw}. */
export interface ConservationOptions extends PdeTraceOptions {
  readonly flux: Flux
  readonly tEnd: number
  /** A fixed step, or else the CFL number (default 0.5) sets each step from the largest speed. */
  readonly dt?: number
  readonly cfl?: number
  /** First-order Godunov or second-order MUSCL-Hancock with a limiter (default godunov). */
  readonly scheme?: 'godunov' | 'muscl'
  readonly limiter?: keyof typeof LIMITERS | Limiter
  /** The interface flux (default Godunov's). */
  readonly riemann?: 'godunov' | 'rusanov' | NumericalFlux
  readonly boundary?: 'periodic' | 'outflow'
  /** Snapshots every n steps (default 1). */
  readonly every?: number
  /** Step budget (default 100000). */
  readonly maxSteps?: number
}

/** The result of {@link conservationLaw}. */
export interface ConservationSolution {
  readonly t: readonly number[]
  /** Cell averages `[nt, cells]`. */
  readonly u: Tensor
  /** Cell centres. */
  readonly x: readonly number[]
  /** The total mass sum u dx at each snapshot. */
  readonly mass: readonly number[]
  /** The total variation at each snapshot. */
  readonly totalVariation: readonly number[]
  readonly status: 'success' | 'failed'
}

interface FvState {
  readonly t: number
  readonly u: Float64Array
  readonly ts: number[]
  readonly us: Float64Array[]
  readonly failed: boolean
}

function totalVariation(u: Float64Array, periodic: boolean): number {
  let s = 0
  for (let i = 1; i < u.length; i++) s += Math.abs((u[i] ?? 0) - (u[i - 1] ?? 0))
  if (periodic) s += Math.abs((u[0] ?? 0) - (u[u.length - 1] ?? 0))
  return s
}

/**
 * A scalar conservation law u_t + f(u)_x = 0 on [a, b] with N cells by finite volumes: Godunov's first-order scheme
 * or MUSCL-Hancock (limited linear reconstruction, half-step predictor, Riemann fluxes), periodic or outflow
 * boundaries, steps fixed or set by a CFL number. Conservative to rounding; TVD with a TVD limiter. A traced
 * algorithm with mass and total variation per step.
 *
 * @param u0 - The initial condition, as a function of x (sampled at cell centres) or cell averages.
 * @param domain - [a, b].
 * @param cells - The number of cells.
 * @param options - Flux, final time, step or CFL, scheme, limiter, Riemann solver and boundary.
 * @returns Snapshots with mass and total variation.
 * @throws MdmlError with code `value` for a non-positive cell count or a mismatched initial condition.
 * @concept finite-volume
 * @example
 * ```ts
 * import { BURGERS, conservationLaw } from 'mdml-engine/dynamics/pde'
 * const s = conservationLaw((x) => (x < 0.5 ? 1 : 0), [0, 1], 100, { flux: BURGERS, tEnd: 0.2, boundary: 'outflow' })
 * Math.abs((s.mass.at(-1) ?? 0) - (s.mass[0] ?? 0) - 0.2 * 0.5) < 1e-12 // => true
 * ```
 */
export function conservationLaw(
  u0: ((x: number) => number) | readonly number[],
  domain: readonly [number, number],
  cells: number,
  options: ConservationOptions,
): ConservationSolution {
  if (!Number.isInteger(cells) || cells < 2) throw new MdmlError('value', 'conservationLaw: at least two cells')
  const [a, b] = domain
  const dx = (b - a) / cells
  const x = Array.from({ length: cells }, (_, i) => a + (i + 0.5) * dx)
  const init = typeof u0 === 'function' ? Float64Array.from(x, (xi) => u0(xi)) : Float64Array.from(u0)
  if (init.length !== cells) throw new MdmlError('value', `conservationLaw: give ${cells} cell averages`)
  const flux = options.flux
  const periodic = (options.boundary ?? 'periodic') === 'periodic'
  const limiter = typeof options.limiter === 'function' ? options.limiter : LIMITERS[options.limiter ?? 'minmod']
  const riemann: NumericalFlux =
    typeof options.riemann === 'function' ? options.riemann : options.riemann === 'rusanov' ? rusanovFlux : godunovFlux
  const muscl = options.scheme === 'muscl'
  const every = Math.max(1, options.every ?? 1)
  const at = (u: Float64Array, i: number): number => {
    if (periodic) return u[((i % cells) + cells) % cells] ?? 0
    return u[Math.min(cells - 1, Math.max(0, i))] ?? 0
  }
  const stepOf = (u: Float64Array): number => {
    if (options.dt !== undefined) return options.dt
    let s = 0
    for (const v of u) s = Math.max(s, Math.abs(flux.df(v)))
    return s === 0 ? options.tEnd : ((options.cfl ?? 0.5) * dx) / s
  }
  const advance = (u: Float64Array, h: number): Float64Array => {
    const faces = new Float64Array(cells + 1)
    if (!muscl) {
      for (let i = 0; i <= cells; i++) faces[i] = riemann(flux, at(u, i - 1), at(u, i))
    } else {
      // MUSCL-Hancock: limited slopes, half-step evolution of the face values, then Riemann fluxes.
      const slope = new Float64Array(cells + 2)
      for (let i = -1; i <= cells; i++) {
        const dl = at(u, i) - at(u, i - 1)
        const dr = at(u, i + 1) - at(u, i)
        slope[i + 1] = dr === 0 ? 0 : limiter(dl / dr) * dr
      }
      const left = (i: number): number => {
        // Right face of cell i, evolved by half a step.
        const s = slope[i + 1] ?? 0
        const ul = at(u, i) - 0.5 * s
        const ur = at(u, i) + 0.5 * s
        return ur - (0.5 * h * (flux.f(ur) - flux.f(ul))) / dx
      }
      const right = (i: number): number => {
        // Left face of cell i, evolved by half a step.
        const s = slope[i + 1] ?? 0
        const ul = at(u, i) - 0.5 * s
        const ur = at(u, i) + 0.5 * s
        return ul - (0.5 * h * (flux.f(ur) - flux.f(ul))) / dx
      }
      for (let i = 0; i <= cells; i++) faces[i] = riemann(flux, left(i - 1), right(i))
    }
    return Float64Array.from(u, (v, i) => v - (h / dx) * ((faces[i + 1] ?? 0) - (faces[i] ?? 0)))
  }
  const algorithm: Algorithm<FvState> = {
    name: muscl ? 'muscl-hancock' : 'godunov',
    config: { cells, tEnd: options.tEnd, periodic: periodic ? 1 : 0 },
    schema: channels({
      metrics: {
        t: { description: 'time' },
        mass: { description: 'total mass' },
        totalVariation: { description: 'total variation', direction: 'lower' },
      },
      tensors: { u: { role: 'state', description: 'cell averages' } },
    }),
    init: () => ({ t: 0, u: init, ts: [0], us: [init], failed: false }),
    step: (s, ctx) => {
      const h = Math.min(stepOf(s.u), options.tEnd - s.t)
      const u = advance(s.u, h)
      const t = s.t + h
      const done = t >= options.tEnd - 1e-14
      const failed = !u.every((v) => Number.isFinite(v))
      if ((ctx.t + 1) % every === 0 || done || failed) {
        s.ts.push(done ? options.tEnd : t)
        s.us.push(u)
      }
      return { t: done ? options.tEnd : t, u, ts: s.ts, us: s.us, failed }
    },
    done: (s) => (s.failed ? 'diverged' : s.t >= options.tEnd ? 'completed' : undefined),
    observe: (s) => ({
      metrics: { t: s.t, mass: s.u.reduce((acc, v) => acc + v * dx, 0), totalVariation: totalVariation(s.u, periodic) },
      tensors: { u: { role: 'state', value: s.u } },
    }),
  }
  const run = runAlgorithm(algorithm, { maxSteps: options.maxSteps ?? 100000, ...options })
  const st = run.state
  const flat = new Float64Array(st.us.length * cells)
  st.us.forEach((u, i) => {
    flat.set(u, i * cells)
  })
  return {
    t: st.ts,
    u: fromStorage(flat, [st.us.length, cells], 'float64'),
    x,
    mass: st.us.map((u) => u.reduce((acc, v) => acc + v * dx, 0)),
    totalVariation: st.us.map((u) => totalVariation(u, periodic)),
    status: st.failed || st.t < options.tEnd ? 'failed' : 'success',
  }
}

/**
 * The exact solution of the Riemann problem for Burgers' equation, u(x, 0) = uL for x < x0 and uR for x > x0: a shock
 * moving at (uL + uR)/2 when uL > uR, a rarefaction fan u = (x - x0)/t between uL t and uR t otherwise.
 *
 * @param uL - Left state.
 * @param uR - Right state.
 * @param x - Position.
 * @param t - Time (> 0).
 * @param x0 - Initial discontinuity (default 0).
 * @returns u(x, t).
 * @concept riemann-problem
 * @example
 * ```ts
 * import { burgersRiemann } from 'mdml-engine/dynamics/pde'
 * [burgersRiemann(1, 0, 0.4, 1), burgersRiemann(1, 0, 0.6, 1), burgersRiemann(0, 1, 0.5, 1)] // => [1, 0, 0.5]
 * ```
 */
export function burgersRiemann(uL: number, uR: number, x: number, t: number, x0 = 0): number {
  const xi = (x - x0) / t
  if (uL > uR) return xi < 0.5 * (uL + uR) ? uL : uR
  if (xi <= uL) return uL
  if (xi >= uR) return uR
  return xi
}
