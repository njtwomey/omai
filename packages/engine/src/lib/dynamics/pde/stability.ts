/**
 * Von Neumann stability analysis of finite-difference schemes: the amplification factor of each Fourier mode per time
 * step, a stability check over all phase angles, stencils for the standard heat and advection schemes (FTCS, BTCS,
 * Crank-Nicolson, upwind, Lax-Friedrichs, Lax-Wendroff), the Courant number and stable step sizes, and numerical
 * dispersion and dissipation.
 */

import { MdmlError } from '../../base/index.ts'
import { complexOps as C, type ComplexNumber } from '../../numerics/differentiate/index.ts'

/**
 * A two-level linear scheme on a uniform grid, sum_j b_j u_{i+j}^{n+1} = sum_j a_j u_{i+j}^n, as stencil weights by
 * offset j (the implicit side defaults to the identity).
 */
export interface TwoLevelScheme {
  readonly name: string
  readonly explicit: Readonly<Record<number, number>>
  readonly implicit?: Readonly<Record<number, number>>
}

function symbol(stencil: Readonly<Record<number, number>>, theta: number): ComplexNumber {
  let acc = C.of(0)
  for (const [off, w] of Object.entries(stencil)) {
    const j = Number(off)
    acc = C.add(acc, C.of(w * Math.cos(j * theta), w * Math.sin(j * theta)))
  }
  return acc
}

/**
 * The von Neumann amplification factor G(theta) of a two-level scheme: a Fourier mode e^{i i theta} is multiplied by
 * G = A(theta) / B(theta) per step, with A and B the symbols of the two stencils.
 *
 * @param scheme - The scheme.
 * @param theta - The phase angle k dx in [0, pi].
 * @returns G(theta).
 * @concept von-neumann-stability
 * @example
 * ```ts
 * import { amplificationFactor, ftcs } from 'mdml-engine/dynamics/pde'
 * // FTCS for the heat equation: G = 1 - 4 r sin^2(theta / 2).
 * amplificationFactor(ftcs(0.25), Math.PI).re // => 0
 * ```
 */
export function amplificationFactor(scheme: TwoLevelScheme, theta: number): ComplexNumber {
  const a = symbol(scheme.explicit, theta)
  const b = scheme.implicit === undefined ? C.of(1) : symbol(scheme.implicit, theta)
  return C.div(a, b)
}

/** The outcome of {@link vonNeumann}. */
export interface VonNeumannReport {
  readonly stable: boolean
  /** max over theta of |G(theta)|. */
  readonly maxGrowth: number
  /** The phase angle where the growth is largest. */
  readonly worstTheta: number
}

/**
 * Von Neumann stability: |G(theta)| <= 1 for every phase angle (sampled on [0, pi]).
 *
 * @param scheme - The scheme.
 * @param samples - Number of angles (default 721).
 * @returns Whether it is stable, and the worst growth.
 * @concept von-neumann-stability
 * @example
 * ```ts
 * import { ftcs, vonNeumann } from 'mdml-engine/dynamics/pde'
 * [vonNeumann(ftcs(0.5)).stable, vonNeumann(ftcs(0.6)).stable] // => [true, false]
 * ```
 */
export function vonNeumann(scheme: TwoLevelScheme, samples = 721): VonNeumannReport {
  let maxGrowth = 0
  let worstTheta = 0
  for (let k = 0; k < samples; k++) {
    const theta = (Math.PI * k) / (samples - 1)
    const g = amplificationFactor(scheme, theta)
    const m = Math.hypot(g.re, g.im)
    if (m > maxGrowth) {
      maxGrowth = m
      worstTheta = theta
    }
  }
  return { stable: maxGrowth <= 1 + 1e-12, maxGrowth, worstTheta }
}

/**
 * FTCS for u_t = alpha u_xx with r = alpha dt / dx^2 (stable for r <= 1/2).
 *
 * @param r - The diffusion number.
 * @returns The scheme.
 * @example
 * ```ts
 * import { ftcs } from 'mdml-engine/dynamics/pde'
 * ftcs(0.2).explicit[0] // => 0.6
 * ```
 */
export function ftcs(r: number): TwoLevelScheme {
  return { name: 'ftcs', explicit: { [-1]: r, 0: 1 - 2 * r, 1: r } }
}

/**
 * BTCS (backward Euler) for the heat equation (unconditionally stable).
 *
 * @param r - The diffusion number.
 * @returns The scheme.
 * @example
 * ```ts
 * import { btcs, vonNeumann } from 'mdml-engine/dynamics/pde'
 * vonNeumann(btcs(10)).stable // => true
 * ```
 */
export function btcs(r: number): TwoLevelScheme {
  return { name: 'btcs', explicit: { 0: 1 }, implicit: { [-1]: -r, 0: 1 + 2 * r, 1: -r } }
}

/**
 * Crank-Nicolson for the heat equation (unconditionally stable, |G| -> 1 for high modes at large r).
 *
 * @param r - The diffusion number.
 * @returns The scheme.
 * @example
 * ```ts
 * import { crankNicolsonScheme, vonNeumann } from 'mdml-engine/dynamics/pde'
 * vonNeumann(crankNicolsonScheme(5)).stable // => true
 * ```
 */
export function crankNicolsonScheme(r: number): TwoLevelScheme {
  return {
    name: 'crank-nicolson',
    explicit: { [-1]: r / 2, 0: 1 - r, 1: r / 2 },
    implicit: { [-1]: -r / 2, 0: 1 + r, 1: -r / 2 },
  }
}

/**
 * First-order upwind for u_t + a u_x = 0 with Courant number c = a dt / dx (stable for 0 <= c <= 1).
 *
 * @param c - The Courant number (its sign picks the upwind side).
 * @returns The scheme.
 * @example
 * ```ts
 * import { upwindScheme, vonNeumann } from 'mdml-engine/dynamics/pde'
 * vonNeumann(upwindScheme(1.2)).stable // => false
 * ```
 */
export function upwindScheme(c: number): TwoLevelScheme {
  return c >= 0
    ? { name: 'upwind', explicit: { [-1]: c, 0: 1 - c } }
    : { name: 'upwind', explicit: { 0: 1 + c, 1: -c } }
}

/**
 * Lax-Friedrichs for linear advection (stable for |c| <= 1, strongly dissipative).
 *
 * @param c - The Courant number.
 * @returns The scheme.
 * @example
 * ```ts
 * import { laxFriedrichsScheme, vonNeumann } from 'mdml-engine/dynamics/pde'
 * vonNeumann(laxFriedrichsScheme(0.9)).stable // => true
 * ```
 */
export function laxFriedrichsScheme(c: number): TwoLevelScheme {
  return { name: 'lax-friedrichs', explicit: { [-1]: (1 + c) / 2, 1: (1 - c) / 2 } }
}

/**
 * Lax-Wendroff for linear advection (second order, stable for |c| <= 1, dispersive).
 *
 * @param c - The Courant number.
 * @returns The scheme.
 * @example
 * ```ts
 * import { laxWendroffScheme, vonNeumann } from 'mdml-engine/dynamics/pde'
 * Math.round(vonNeumann(laxWendroffScheme(1)).maxGrowth * 1e12) / 1e12 // => 1
 * ```
 */
export function laxWendroffScheme(c: number): TwoLevelScheme {
  return { name: 'lax-wendroff', explicit: { [-1]: (c * c + c) / 2, 0: 1 - c * c, 1: (c * c - c) / 2 } }
}

/**
 * The Courant (CFL) number |a| dt / dx of an advection step.
 *
 * @param velocity - The wave speed a.
 * @param dt - Time step.
 * @param dx - Grid spacing.
 * @returns The Courant number.
 * @concept cfl-condition
 * @example
 * ```ts
 * import { courantNumber } from 'mdml-engine/dynamics/pde'
 * courantNumber(1, 0.5, 0.25) // => 2
 * ```
 */
export function courantNumber(velocity: number, dt: number, dx: number): number {
  if (!(dx > 0)) throw new MdmlError('value', 'courantNumber: dx must be positive')
  return (Math.abs(velocity) * dt) / dx
}

/**
 * The largest stable explicit step: dx / |a| times the CFL limit for advection, dx^2 / (2 alpha) for diffusion,
 * whichever is smaller (with a safety factor).
 *
 * @param options - Grid spacing, speed, diffusivity, CFL limit (default 1) and safety (default 1).
 * @returns The step bound (Infinity when nothing limits it).
 * @concept cfl-condition
 * @example
 * ```ts
 * import { stableTimeStep } from 'mdml-engine/dynamics/pde'
 * stableTimeStep({ dx: 0.1, diffusivity: 1 }) // => 0.005000000000000001
 * ```
 */
export function stableTimeStep(options: {
  readonly dx: number
  readonly velocity?: number
  readonly diffusivity?: number
  readonly cfl?: number
  readonly safety?: number
}): number {
  const s = options.safety ?? 1
  let dt = Number.POSITIVE_INFINITY
  if (options.velocity !== undefined && options.velocity !== 0)
    dt = Math.min(dt, ((options.cfl ?? 1) * options.dx) / Math.abs(options.velocity))
  if (options.diffusivity !== undefined && options.diffusivity > 0)
    dt = Math.min(dt, (options.dx * options.dx) / (2 * options.diffusivity))
  return s * dt
}

/** The numerical dispersion and dissipation of an advection scheme at one wavenumber. */
export interface DispersionReport {
  /** |G|: 1 is no dissipation. */
  readonly amplitude: number
  /** The numerical over the exact phase speed (1 is no dispersion). */
  readonly relativePhaseSpeed: number
}

/**
 * Numerical dispersion and dissipation of a scheme for u_t + a u_x = 0 at Courant number c: the exact solution moves
 * a mode by phase -c theta per step, the scheme by arg G(theta).
 *
 * @param scheme - The scheme.
 * @param c - Its Courant number.
 * @param theta - The phase angle k dx in (0, pi].
 * @returns Amplitude per step and relative phase speed.
 * @concept numerical-dispersion
 * @example
 * ```ts
 * import { dispersion, upwindScheme } from 'mdml-engine/dynamics/pde'
 * // At c = 1 upwind is exact.
 * const d = dispersion(upwindScheme(1), 1, 0.7)
 * [Math.round(d.amplitude * 1e12) / 1e12, Math.round(d.relativePhaseSpeed * 1e12) / 1e12] // => [1, 1]
 * ```
 */
export function dispersion(scheme: TwoLevelScheme, c: number, theta: number): DispersionReport {
  const g = amplificationFactor(scheme, theta)
  const phase = Math.atan2(g.im, g.re)
  return { amplitude: Math.hypot(g.re, g.im), relativePhaseSpeed: c === 0 ? 1 : -phase / (c * theta) }
}
