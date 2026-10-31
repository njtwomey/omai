/**
 * The lattice Boltzmann method on the D2Q9 lattice with the BGK collision: nine populations per site relaxing toward
 * the local equilibrium at a rate set by the relaxation time tau, which sets the viscosity `nu = (tau - 1/2) / 3`. A
 * body force enters by Guo's scheme, solid sites and walls bounce populations back (the wall lies half-way between a
 * fluid node and the solid), and an inlet and outlet drive a flow past an obstacle. The macroscopic fields, the
 * analytic Poiseuille profile it is checked against, and the solver as a traced algorithm that logs the density, the
 * velocity, the conserved mass and the Reynolds number.
 */

import { MdmlError } from '../../base/index.ts'
import {
  type Algorithm,
  type Contract,
  contractSchema,
  defineContract,
  derived,
  scalar,
  tensor,
} from '../../trace/index.ts'
import { codec, mod, positiveInt } from './_core.ts'
import { curl2d } from './lattice-gas.ts'

const CX = [0, 1, 0, -1, 0, 1, -1, -1, 1] as const
const CY = [0, 0, 1, 0, -1, 1, 1, -1, -1] as const
const W = [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36] as const
const OPP = [0, 3, 4, 1, 2, 7, 8, 5, 6] as const

/**
 * The D2Q9 equilibrium population of direction `k`: `w_k rho (1 + 3 c.u + 4.5 (c.u)^2 - 1.5 u.u)`.
 *
 * @param k - The direction, 0 (rest) to 8.
 * @param rho - The density.
 * @param ux - The x velocity.
 * @param uy - The y velocity.
 * @returns The population.
 * @concept lattice-boltzmann
 * @example
 * ```ts
 * import { automata } from 'mdml-engine/dynamics'
 * [automata.d2q9Equilibrium(0, 1, 0, 0), automata.d2q9Equilibrium(1, 1, 0, 0)] // => [0.4444444444444444, 0.1111111111111111]
 * ```
 */
export function d2q9Equilibrium(k: number, rho: number, ux: number, uy: number): number {
  const cu = (CX[k] ?? 0) * ux + (CY[k] ?? 0) * uy
  return (W[k] ?? 0) * rho * (1 + 3 * cu + 4.5 * cu * cu - 1.5 * (ux * ux + uy * uy))
}

/** Options of {@link latticeBoltzmann}. */
export interface LatticeBoltzmannOptions {
  /** Columns and rows (default 64 by 32). */
  readonly width?: number
  readonly height?: number
  /** The relaxation time (default 0.8; stable above 1/2). The viscosity is `(tau - 1/2) / 3`. */
  readonly tau?: number
  /** A uniform body force per unit mass `[gx, gy]`, applied by Guo's scheme (default none). */
  readonly force?: readonly [number, number]
  /** The top and bottom rows are walls (default false: periodic). */
  readonly walls?: boolean
  /** The speed of a flow in from the left, with an outlet on the right (default none: periodic left and right). */
  readonly inflow?: number
  /** A solid disc `{ x, y, radius }` in lattice units (default none). */
  readonly cylinder?: { readonly x: number; readonly y: number; readonly radius: number }
  /** Solid sites (nonzero) in addition to the cylinder and the walls (row-major). */
  readonly solid?: Uint8Array
  /** The starting fields (default density 1 and velocity `(inflow, 0)`). */
  readonly initial?: (x: number, y: number) => { readonly rho: number; readonly ux: number; readonly uy: number }
}

/** The state of {@link latticeBoltzmann}: nine populations per site, direction-major (`f[k * sites + site]`). */
export interface LatticeBoltzmannState {
  readonly f: Float64Array
  readonly time: number
}

/** The paths of {@link latticeBoltzmann}. */
export type LatticeBoltzmannPath =
  | 'lbm/density'
  | 'lbm/velocity'
  | 'flow/mass'
  | 'flow/maxSpeed'
  | 'flow/kineticEnergy'
  | 'flow/reynolds'
  | 'field/vorticity'

/**
 * The density and velocity of a set of populations (with the half-force correction when a body force acts).
 *
 * @param f - Populations, direction-major.
 * @param sites - Number of sites.
 * @param force - The body force (default none).
 * @returns Density, ux and uy per site.
 * @concept lattice-boltzmann
 * @example
 * ```ts
 * import { automata } from 'mdml-engine/dynamics'
 * const f = Float64Array.from({ length: 9 }, (_, k) => automata.d2q9Equilibrium(k, 1.2, 0.05, 0))
 * const m = automata.d2q9Moments(f, 1)
 * [m.rho[0]?.toFixed(6), m.ux[0]?.toFixed(6)] // => ['1.200000', '0.050000']
 * ```
 */
export function d2q9Moments(
  f: Float64Array,
  sites: number,
  force: readonly [number, number] = [0, 0],
): { rho: Float64Array; ux: Float64Array; uy: Float64Array } {
  const rho = new Float64Array(sites)
  const ux = new Float64Array(sites)
  const uy = new Float64Array(sites)
  for (let i = 0; i < sites; i++) {
    let r = 0
    let mx = 0
    let my = 0
    for (let k = 0; k < 9; k++) {
      const v = f[k * sites + i] ?? 0
      r += v
      mx += v * (CX[k] ?? 0)
      my += v * (CY[k] ?? 0)
    }
    rho[i] = r
    // A solid site holds nothing: no density, no velocity.
    ux[i] = r === 0 ? 0 : (mx + force[0] / 2) / r
    uy[i] = r === 0 ? 0 : (my + force[1] / 2) / r
  }
  return { rho, ux, uy }
}

/**
 * The steady velocity of plane Poiseuille flow between two walls with half-way bounce-back, driven by a body force `g`:
 * `u(j) = g / (2 nu) (j + 1/2) (H - 1/2 - j)` for the fluid nodes `j = 0..H-1` (the walls lie at `-1/2` and `H - 1/2`).
 *
 * @param height - H, the number of fluid nodes across the channel.
 * @param force - The body force g along the channel.
 * @param tau - The relaxation time.
 * @returns The velocity at each node.
 * @concept lattice-boltzmann
 * @example
 * ```ts
 * import { automata } from 'mdml-engine/dynamics'
 * automata.poiseuilleProfile(3, 1e-4, 1).map((u) => u.toExponential(3)) // => ['3.750e-4', '6.750e-4', '3.750e-4']
 * ```
 */
export function poiseuilleProfile(height: number, force: number, tau: number): number[] {
  const nu = (tau - 0.5) / 3
  return Array.from({ length: height }, (_, j) => (force / (2 * nu)) * (j + 0.5) * (height - 0.5 - j))
}

/**
 * The log contract of {@link latticeBoltzmann}. `field/vorticity` is derived from the velocity on demand.
 *
 * @param width - Columns.
 * @param height - Rows.
 * @param force - The body force (the velocity includes its half-step).
 * @returns The contract.
 * @example
 * ```ts
 * import { automata } from 'mdml-engine/dynamics'
 * Object.keys(automata.latticeBoltzmannContract(8, 8, [0, 0])).length // => 7
 * ```
 */
export function latticeBoltzmannContract(
  width: number,
  height: number,
  force: readonly [number, number],
): Contract<LatticeBoltzmannState, LatticeBoltzmannPath> {
  return defineContract({
    'lbm/density': tensor({ role: 'state', shape: ['height', 'width'], description: 'fluid density' }),
    'lbm/velocity': tensor({
      role: 'state',
      shape: ['height', 'width', 2],
      description: 'fluid velocity (ux, uy), y up',
    }),
    'flow/mass': scalar({ description: 'total mass (conserved by periodic and bounce-back boundaries)' }),
    'flow/maxSpeed': scalar({
      description: 'the largest fluid speed (the method needs it far below 0.577, the lattice sound speed)',
    }),
    'flow/kineticEnergy': scalar({ description: 'half the sum of density times speed squared' }),
    'flow/reynolds': scalar({
      description: 'speed times length over viscosity (inflow or max speed, obstacle diameter or channel height)',
    }),
    'field/vorticity': derived<LatticeBoltzmannState>({
      description: 'the curl of the velocity, central differences (periodic)',
      compute: (s) => {
        const m = d2q9Moments(s.f, width * height, force)
        return Array.from(curl2d(m.ux, m.uy, width, height))
      },
    }),
  })
}

/**
 * D2Q9 lattice Boltzmann (BGK) as a traced algorithm: each step relaxes the populations toward equilibrium (adding Guo's
 * forcing term when a body force acts) and streams them, with bounce-back on solid sites and walls, and equilibrium
 * inflow with a zero-gradient outflow when `inflow` is set. The state is the populations and serialises. Units are
 * lattice units (spacing and time step 1).
 *
 * @param options - Size, relaxation time, force, boundaries, obstacle and starting fields.
 * @returns The algorithm.
 * @throws MdmlError with code `value` for a relaxation time of at most 1/2, a bad size or a solid mask of the wrong size.
 * @concept lattice-boltzmann
 * @example
 * ```ts
 * import { automata } from 'mdml-engine/dynamics'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * // A fluid at rest stays at rest, with its mass.
 * const s = runAlgorithm(automata.latticeBoltzmann({ width: 8, height: 8 }), { seed: 0, maxSteps: 20 }).state
 * automata.d2q9Moments(s.f, 64).rho.every((r) => Math.abs(r - 1) < 1e-12) // => true
 * ```
 */
export function latticeBoltzmann(
  options: LatticeBoltzmannOptions = {},
): Algorithm<LatticeBoltzmannState, LatticeBoltzmannPath> {
  const width = options.width ?? 64
  const height = options.height ?? 32
  const tau = options.tau ?? 0.8
  const force = options.force ?? [0, 0]
  const walls = options.walls ?? false
  const inflow = options.inflow
  positiveInt('latticeBoltzmann', 'width', width)
  positiveInt('latticeBoltzmann', 'height', height)
  if (!(tau > 0.5)) throw new MdmlError('value', `latticeBoltzmann: tau must exceed 1/2, got ${tau}`)
  if (options.solid !== undefined && options.solid.length !== width * height)
    throw new MdmlError('value', 'latticeBoltzmann: solid must be width x height')
  const sites = width * height
  const nu = (tau - 0.5) / 3
  const solid = new Uint8Array(sites)
  options.solid?.forEach((v, i) => {
    if (v) solid[i] = 1
  })
  const cyl = options.cylinder
  if (cyl !== undefined)
    for (let r = 0; r < height; r++)
      for (let c = 0; c < width; c++) if (Math.hypot(c - cyl.x, r - cyl.y) <= cyl.radius) solid[r * width + c] = 1
  const length = cyl !== undefined ? 2 * cyl.radius : walls ? height : width
  const contract = latticeBoltzmannContract(width, height, force)
  const guo = 1 - 1 / (2 * tau)
  return {
    name: 'dynamics/automata/lattice-boltzmann',
    config: {
      width,
      height,
      tau,
      force: [...force],
      walls,
      inflow: inflow ?? null,
      cylinder: cyl ?? null,
      solids: solid.reduce((a, b) => a + b, 0),
      initial: options.initial === undefined ? 'uniform' : 'custom',
    },
    schema: contractSchema(contract),
    init: () => {
      const f = new Float64Array(9 * sites)
      for (let r = 0; r < height; r++)
        for (let c = 0; c < width; c++) {
          const i = r * width + c
          if (solid[i]) continue
          const init = options.initial?.(c, r) ?? { rho: 1, ux: inflow ?? 0, uy: 0 }
          for (let k = 0; k < 9; k++) f[k * sites + i] = d2q9Equilibrium(k, init.rho, init.ux, init.uy)
        }
      return { f, time: 0 }
    },
    step: (s) => {
      const m = d2q9Moments(s.f, sites, force)
      // Collide (BGK plus Guo's forcing), on fluid sites only.
      const post = new Float64Array(9 * sites)
      for (let i = 0; i < sites; i++) {
        if (solid[i]) {
          for (let k = 0; k < 9; k++) post[k * sites + i] = s.f[k * sites + i] ?? 0
          continue
        }
        const rho = m.rho[i] ?? 1
        const ux = m.ux[i] ?? 0
        const uy = m.uy[i] ?? 0
        for (let k = 0; k < 9; k++) {
          const fk = s.f[k * sites + i] ?? 0
          const cu = (CX[k] ?? 0) * ux + (CY[k] ?? 0) * uy
          const forcing =
            guo *
            (W[k] ?? 0) *
            (3 * (((CX[k] ?? 0) - ux) * force[0] + ((CY[k] ?? 0) - uy) * force[1]) +
              9 * cu * ((CX[k] ?? 0) * force[0] + (CY[k] ?? 0) * force[1]))
          post[k * sites + i] = fk - (fk - d2q9Equilibrium(k, rho, ux, uy)) / tau + forcing
        }
      }
      // Stream by pulling from the upstream site; a solid or wall upstream bounces the opposite population back.
      const next = new Float64Array(9 * sites)
      for (let r = 0; r < height; r++)
        for (let c = 0; c < width; c++) {
          const i = r * width + c
          if (solid[i]) continue
          for (let k = 0; k < 9; k++) {
            // Rows increase downward but c_y points up, so the upstream row is r + c_y.
            const rr = r + (CY[k] ?? 0)
            const cc = c - (CX[k] ?? 0)
            const wallHit = walls && (rr < 0 || rr >= height)
            const wrappedRow = mod(rr, height)
            const wrappedCol = inflow === undefined ? mod(cc, width) : Math.min(width - 1, Math.max(0, cc))
            const j = wrappedRow * width + wrappedCol
            next[k * sites + i] =
              wallHit || solid[j] ? (post[(OPP[k] ?? 0) * sites + i] ?? 0) : (post[k * sites + j] ?? 0)
          }
        }
      if (inflow !== undefined) {
        for (let r = 0; r < height; r++) {
          const inlet = r * width
          const outlet = r * width + width - 1
          if (!solid[inlet]) for (let k = 0; k < 9; k++) next[k * sites + inlet] = d2q9Equilibrium(k, 1, inflow, 0)
          if (!solid[outlet]) for (let k = 0; k < 9; k++) next[k * sites + outlet] = next[k * sites + outlet - 1] ?? 0
        }
      }
      return { f: next, time: s.time + 1 }
    },
    observe: (s) => {
      const m = d2q9Moments(s.f, sites, force)
      const velocity = new Float64Array(2 * sites)
      let mass = 0
      let top = 0
      let energy = 0
      for (let i = 0; i < sites; i++) {
        if (solid[i]) {
          m.rho[i] = 0
          continue
        }
        velocity[2 * i] = m.ux[i] ?? 0
        velocity[2 * i + 1] = m.uy[i] ?? 0
        const speed = Math.hypot(m.ux[i] ?? 0, m.uy[i] ?? 0)
        mass += m.rho[i] ?? 0
        top = Math.max(top, speed)
        energy += 0.5 * (m.rho[i] ?? 0) * speed * speed
      }
      return {
        metrics: {
          'flow/mass': mass,
          'flow/maxSpeed': top,
          'flow/kineticEnergy': energy,
          'flow/reynolds': ((inflow ?? top) * length) / nu,
        },
        tensors: {
          'lbm/density': { role: 'state', value: m.rho, shape: [height, width] },
          'lbm/velocity': { role: 'state', value: velocity, shape: [height, width, 2] },
        },
      }
    },
    ...codec<LatticeBoltzmannState>(),
  }
}
