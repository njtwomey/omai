/**
 * The n-body problem of Newtonian gravity as a Hamiltonian system: H(q, p) = sum |p_i|^2 / (2 m_i) - G sum_{i<j} m_i
 * m_j / |q_i - q_j| (optionally softened), its force by vectorised tensor operations (so it differentiates), the
 * conserved quantities (energy, momentum, angular momentum, centre of mass), and a stepped integration with the
 * integrator as a choice (the symplectic splittings against Runge-Kutta) that logs the energy and its drift, the
 * angular momentum and the closest approach, and fires an event at a near collision.
 */

import { type Json, OmaiError } from '../../base/index.ts'
import { add, expandDims, mul, pow, sub, sum, type TensorLike, type Value } from '../../tensor/index.ts'
import {
  type Algorithm,
  type Contract,
  defineContract,
  event,
  scalar,
  stateFromJson,
  stateToJson,
  tensor,
} from '../../trace/index.ts'
import { floatState, fromNumbers, numbersOf } from '../ode/_vec.ts'
import type { ButcherTableau } from '../ode/explicit.ts'
import { EULER, HEUN, MIDPOINT, RK4, RK38, rkStep } from '../ode/explicit.ts'
import {
  FOREST_RUTH,
  type SeparableSystem,
  type SplittingMethod,
  STORMER_VERLET,
  SYMPLECTIC_EULER,
  separableField,
  symplecticStep,
  VELOCITY_VERLET,
  YOSHIDA4,
} from '../ode/symplectic.ts'

/** Options of {@link nBodyHamiltonian}. */
export interface NBodyOptions {
  /** The gravitational constant (default 1). */
  readonly G?: number
  /** Plummer softening length: distances become sqrt(r^2 + eps^2), taming close encounters (default 0). */
  readonly softening?: number
}

/** The closest pair of bodies. */
export interface ClosestPair {
  readonly distance: number
  readonly i: number
  readonly j: number
}

/** An n-body gravitational system: a separable Hamiltonian with its diagnostics. Positions and momenta are `[n, dim]`. */
export interface NBodySystem extends SeparableSystem {
  readonly masses: readonly number[]
  /** Spatial dimension (2 or 3). */
  readonly dim: number
  readonly G: number
  readonly softening: number
  /** H(q, p). */
  readonly energy: (q: Value, p: Value) => number
  /** The kinetic energy of momenta given as flat numbers. */
  kinetic(p: ArrayLike<number>): number
  /** The potential energy of positions given as flat numbers. */
  potential(q: ArrayLike<number>): number
  /** The total momentum. */
  momentum(p: ArrayLike<number>): number[]
  /** The angular momentum about the origin: a vector in 3 dimensions, `[Lz]` in 2. */
  angularMomentum(q: ArrayLike<number>, p: ArrayLike<number>): number[]
  /** The centre of mass. */
  centreOfMass(q: ArrayLike<number>): number[]
  /** The closest pair. */
  closest(q: ArrayLike<number>): ClosestPair
}

/**
 * The Hamiltonian of n point masses under Newtonian gravity. The force is computed for all pairs at once with tensor
 * operations, so it is differentiable and the symplectic integrators of `dynamics/ode` run it as they would any
 * separable system.
 *
 * @param masses - The masses.
 * @param dim - The spatial dimension, 2 or 3 (default 3).
 * @param options - G and the softening.
 * @returns The system.
 * @throws OmaiError with code `value` for fewer than two bodies, a non-positive mass or a dimension other than 2 or 3.
 * @concept n-body-problem
 * @example
 * ```ts
 * import { nBodyHamiltonian } from 'omai-engine/dynamics/celestial'
 * import { tensor } from 'omai-engine/tensor'
 * const two = nBodyHamiltonian([1, 1], 2)
 * // Two unit masses a unit apart at rest: the energy is the potential, -1.
 * two.energy(tensor([[0, 0], [1, 0]]), tensor([[0, 0], [0, 0]])) // => -1
 * ```
 */
export function nBodyHamiltonian(masses: readonly number[], dim: 2 | 3 = 3, options: NBodyOptions = {}): NBodySystem {
  const n = masses.length
  if (n < 2) throw new OmaiError('value', 'nBodyHamiltonian: give at least two bodies')
  if (masses.some((m) => !(m > 0) || !Number.isFinite(m)))
    throw new OmaiError('value', 'nBodyHamiltonian: masses must be positive')
  if (dim !== 2 && dim !== 3) throw new OmaiError('value', 'nBodyHamiltonian: dim is 2 or 3')
  const G = options.G ?? 1
  const eps2 = (options.softening ?? 0) ** 2
  // G m_i m_j off the diagonal, and a unit diagonal added to r^2 so that the self-term never divides by zero.
  const weights = new Float64Array(n * n)
  const diagonal = new Float64Array(n * n)
  for (let i = 0; i < n; i++) {
    diagonal[i * n + i] = 1
    for (let j = 0; j < n; j++) if (i !== j) weights[i * n + j] = G * (masses[i] as number) * (masses[j] as number)
  }
  const W = fromNumbers(weights, [n, n])
  const D = fromNumbers(diagonal, [n, n])
  const mcol = fromNumbers(masses, [n, 1])
  const separation = (q: ArrayLike<number>, i: number, j: number): number => {
    let s = eps2
    for (let k = 0; k < dim; k++) s += ((q[j * dim + k] ?? 0) - (q[i * dim + k] ?? 0)) ** 2
    return Math.sqrt(s)
  }
  const kinetic = (p: ArrayLike<number>): number => {
    let t = 0
    for (let i = 0; i < n; i++) {
      for (let k = 0; k < dim; k++) t += (p[i * dim + k] ?? 0) ** 2 / (2 * (masses[i] as number))
    }
    return t
  }
  const potential = (q: ArrayLike<number>): number => {
    let v = 0
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) v -= (G * (masses[i] as number) * (masses[j] as number)) / separation(q, i, j)
    }
    return v
  }
  return {
    masses,
    dim,
    G,
    softening: options.softening ?? 0,
    velocity: (p) => mul(p, pow(mcol, -1)),
    force: (q) => {
      const diff = sub(expandDims(q, 0), expandDims(q, 1))
      const r2 = add(add(sum(mul(diff, diff), 2), eps2), D)
      const w = mul(W, pow(r2, -1.5))
      return sum(mul(expandDims(w, 2), diff), 1)
    },
    energy: (q, p) => kinetic(numbersOf(p)) + potential(numbersOf(q)),
    kinetic,
    potential,
    momentum: (p) => {
      const out = new Array<number>(dim).fill(0)
      for (let i = 0; i < n; i++) for (let k = 0; k < dim; k++) out[k] = (out[k] ?? 0) + (p[i * dim + k] ?? 0)
      return out
    },
    angularMomentum: (q, p) => {
      const L = [0, 0, 0]
      for (let i = 0; i < n; i++) {
        const x = q[i * dim] ?? 0
        const y = q[i * dim + 1] ?? 0
        const z = dim === 3 ? (q[i * dim + 2] ?? 0) : 0
        const px = p[i * dim] ?? 0
        const py = p[i * dim + 1] ?? 0
        const pz = dim === 3 ? (p[i * dim + 2] ?? 0) : 0
        L[0] = (L[0] ?? 0) + (y * pz - z * py)
        L[1] = (L[1] ?? 0) + (z * px - x * pz)
        L[2] = (L[2] ?? 0) + (x * py - y * px)
      }
      return dim === 3 ? L : [L[2] ?? 0]
    },
    centreOfMass: (q) => {
      const total = masses.reduce((a, b) => a + b, 0)
      const c = new Array<number>(dim).fill(0)
      for (let i = 0; i < n; i++) {
        for (let k = 0; k < dim; k++) c[k] = (c[k] ?? 0) + ((masses[i] as number) * (q[i * dim + k] ?? 0)) / total
      }
      return c
    },
    closest: (q) => {
      let best: ClosestPair = { distance: Number.POSITIVE_INFINITY, i: 0, j: 1 }
      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
          const d = separation(q, i, j)
          if (d < best.distance) best = { distance: d, i, j }
        }
      }
      return best
    },
  }
}

/** The integrators an n-body run can use: the symplectic splittings, and Runge-Kutta methods that are not. */
export const nBodyIntegrators: Readonly<Record<string, SplittingMethod | ButcherTableau>> = {
  'symplectic-euler': SYMPLECTIC_EULER,
  'stormer-verlet': STORMER_VERLET,
  'velocity-verlet': VELOCITY_VERLET,
  'forest-ruth': FOREST_RUTH,
  yoshida4: YOSHIDA4,
  euler: EULER,
  heun: HEUN,
  midpoint: MIDPOINT,
  rk4: RK4,
  rk38: RK38,
}

/** The name of an n-body integrator. */
export type NBodyIntegrator =
  | 'symplectic-euler'
  | 'stormer-verlet'
  | 'velocity-verlet'
  | 'forest-ruth'
  | 'yoshida4'
  | 'euler'
  | 'heun'
  | 'midpoint'
  | 'rk4'
  | 'rk38'

/**
 * One step of an integrator on a Hamiltonian system, from flat positions and momenta.
 *
 * @param system - The separable system, whose state is `[n, dim]`.
 * @param q - Positions.
 * @param p - Momenta.
 * @param h - The step.
 * @param integrator - The method (default Stormer-Verlet).
 * @returns The new positions and momenta.
 * @example
 * ```ts
 * import { nBodyHamiltonian, nBodyStep } from 'omai-engine/dynamics/celestial'
 * const two = nBodyHamiltonian([1, 1], 2)
 * Array.from(nBodyStep(two, [0, 0, 1, 0], [0, 0, 0, 0], 0.0).q) // => [0, 0, 1, 0]
 * ```
 */
export function nBodyStep(
  system: NBodySystem,
  q: ArrayLike<number>,
  p: ArrayLike<number>,
  h: number,
  integrator: NBodyIntegrator = 'stormer-verlet',
): { readonly q: Float64Array; readonly p: Float64Array } {
  const n = system.masses.length
  const shape = [n, system.dim]
  const method = nBodyIntegrators[integrator]
  if (method === undefined) throw new OmaiError('value', `nBodyStep: unknown integrator '${integrator}'`)
  if ('drift' in method) {
    const out = symplecticStep(system, fromNumbers(q, shape), fromNumbers(p, shape), h, method)
    return { q: numbersOf(out.q).slice(), p: numbersOf(out.p).slice() }
  }
  const y = new Float64Array(2 * n * system.dim)
  y.set(Array.from(q), 0)
  y.set(Array.from(p), n * system.dim)
  const next = numbersOf(rkStep(separableField(system), 0, fromNumbers(y, [2, n, system.dim]), h, method))
  return { q: next.slice(0, n * system.dim), p: next.slice(n * system.dim) }
}

/** The state of {@link nBodyAlgorithm}. Everything is numbers, so it serialises. */
export interface NBodyState {
  readonly t: number
  readonly step: number
  /** Positions, flat `[n * dim]`. */
  readonly q: Float64Array
  /** Momenta, flat `[n * dim]`. */
  readonly p: Float64Array
  /** The energy and angular momentum at the start, for the drift. */
  readonly energy0: number
  readonly angular0: number
  /** Whether the pair that last came close was already inside the threshold (an event fires on entering). */
  readonly inside: boolean
}

/** The paths of {@link nBodyAlgorithm}. */
export type NBodyPath =
  | 'bodies/t'
  | 'bodies/positions'
  | 'bodies/momenta'
  | 'bodies/energy'
  | 'bodies/energyDrift'
  | 'bodies/angularMomentum'
  | 'bodies/momentum'
  | 'bodies/minSeparation'
  | 'bodies/close'

/** What an n-body run logs. */
export const nBodyContract: Contract<NBodyState, NBodyPath> = defineContract({
  'bodies/t': scalar({ description: 'time' }),
  'bodies/positions': tensor({ role: 'state', shape: ['n', 'dim'], description: 'positions of the bodies' }),
  'bodies/momenta': tensor({ role: 'state', shape: ['n', 'dim'], description: 'momenta of the bodies' }),
  'bodies/energy': scalar({ description: 'the Hamiltonian H(q, p), conserved by the flow' }),
  'bodies/energyDrift': scalar({ description: 'the energy error |H - H0| / |H0|', direction: 'lower' }),
  'bodies/angularMomentum': scalar({ description: 'the length of the angular momentum about the origin (conserved)' }),
  'bodies/momentum': scalar({ description: 'the length of the total momentum (conserved)' }),
  'bodies/minSeparation': scalar({ description: 'the distance between the closest pair' }),
  'bodies/close': event({
    description: 'two bodies came closer than the threshold',
    context: { i: 'body', j: 'body', distance: 'their separation' },
  }),
})

/** Options of {@link nBodyAlgorithm}. */
export interface NBodyRunOptions {
  /** The step. */
  readonly dt: number
  /** The number of steps to take (the algorithm stops after them). */
  readonly steps: number
  /** The integrator (default Stormer-Verlet). */
  readonly integrator?: NBodyIntegrator
  /** Fire `bodies/close` when two bodies come within this distance (default: never). */
  readonly closeBelow?: number
}

const norm = (v: readonly number[]): number => Math.hypot(...v)

/**
 * An n-body integration as a stepped algorithm: one step of the chosen integrator per algorithm step, logging time,
 * positions, momenta, the energy and its drift, the angular momentum, the total momentum and the closest approach,
 * and firing `bodies/close` as a pair first comes within the threshold. Symplectic splittings keep the energy error
 * bounded; Runge-Kutta methods let it drift.
 *
 * @param system - The system.
 * @param q0 - Initial positions `[n, dim]`.
 * @param p0 - Initial momenta `[n, dim]`.
 * @param options - Step, step count, integrator and the close-encounter threshold.
 * @returns The algorithm.
 * @throws OmaiError with code `value` for a non-positive step or an unknown integrator.
 * @concept n-body-problem
 * @example
 * ```ts
 * import { nBodyAlgorithm, nBodyHamiltonian } from 'omai-engine/dynamics/celestial'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const two = nBodyHamiltonian([1, 1], 2)
 * // Two unit masses on a circular orbit about their centre (speed 1/2 each at separation 1).
 * const alg = nBodyAlgorithm(two, [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5], { dt: 0.01, steps: 100 })
 * Math.abs(runAlgorithm(alg, { maxSteps: 1000 }).state.t - 1) < 1e-12 // => true
 * ```
 */
export function nBodyAlgorithm(
  system: NBodySystem,
  q0: TensorLike,
  p0: TensorLike,
  options: NBodyRunOptions,
): Algorithm<NBodyState, NBodyPath> {
  const { dt, steps } = options
  if (!(dt > 0)) throw new OmaiError('value', 'nBodyAlgorithm: dt must be positive')
  const integrator = options.integrator ?? 'stormer-verlet'
  if (nBodyIntegrators[integrator] === undefined)
    throw new OmaiError('value', `nBodyAlgorithm: unknown integrator '${integrator}'`)
  const n = system.masses.length
  const take = (x: TensorLike): Float64Array => numbersOf(floatState(x)).slice()
  const q = take(q0)
  const p = take(p0)
  if (q.length !== n * system.dim || p.length !== n * system.dim) {
    throw new OmaiError('shape', `nBodyAlgorithm: positions and momenta are [${n}, ${system.dim}]`)
  }
  const drift = (e: number, e0: number): number => Math.abs(e - e0) / Math.max(Math.abs(e0), Number.MIN_VALUE)
  return {
    name: 'nBody',
    config: {
      masses: [...system.masses],
      dim: system.dim,
      G: system.G,
      softening: system.softening,
      dt,
      steps,
      integrator,
      closeBelow: options.closeBelow ?? null,
    } as Json,
    init: () => ({
      t: 0,
      step: 0,
      q: q.slice(),
      p: p.slice(),
      energy0: system.kinetic(p) + system.potential(q),
      angular0: norm(system.angularMomentum(q, p)),
      inside: false,
    }),
    step: (s, ctx) => {
      const next = nBodyStep(system, s.q, s.p, dt, integrator)
      const t = (s.step + 1) * dt
      const energy = system.kinetic(next.p) + system.potential(next.q)
      const nearest = system.closest(next.q)
      const inside = options.closeBelow !== undefined && nearest.distance < options.closeBelow
      ctx.log('bodies/t', t)
      ctx.log('bodies/positions', next.q)
      ctx.log('bodies/momenta', next.p)
      ctx.log('bodies/energy', energy)
      ctx.log('bodies/energyDrift', drift(energy, s.energy0))
      ctx.log('bodies/angularMomentum', norm(system.angularMomentum(next.q, next.p)))
      ctx.log('bodies/momentum', norm(system.momentum(next.p)))
      ctx.log('bodies/minSeparation', nearest.distance)
      if (inside && !s.inside) ctx.emit('bodies/close', { i: nearest.i, j: nearest.j, distance: nearest.distance })
      return { ...s, t, step: s.step + 1, q: next.q, p: next.p, inside }
    },
    done: (s) => (s.step >= steps ? 'completed' : undefined),
    serialise: (s) => stateToJson(s),
    deserialise: (j: Json) => stateFromJson(j) as NBodyState,
  }
}
