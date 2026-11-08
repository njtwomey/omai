/**
 * Symplectic integrators for separable Hamiltonian systems, which preserve the geometry of phase space so that energy
 * errors stay bounded over long runs. Splitting methods (symplectic Euler, Stormer-Verlet, velocity Verlet,
 * Forest-Ruth, Yoshida), a traced integrator with the energy as a metric, and energy-drift diagnostics.
 */

import { grad } from '../../autodiff/index.ts'
import { OmaiError } from '../../base/index.ts'
import { item, mul, reshape, sliceAxis, stack, sum, type TensorLike, type Value } from '../../tensor/index.ts'
import { type Algorithm, channels, type RunOptions, runAlgorithm } from '../../trace/index.ts'
import { combine, floatState, numbersOf } from './_vec.ts'
import type { OdeVectorField } from './problem.ts'

/**
 * A separable Hamiltonian system H(q, p) = T(p) + V(q): the velocity dH/dp = dT/dp and the force -dH/dq = -dV/dq,
 * with the energy when it is known.
 */
export interface SeparableSystem {
  /** dT/dp, the velocity. */
  readonly velocity: (p: Value) => Value
  /** -dV/dq, the force. */
  readonly force: (q: Value) => Value
  /** H(q, p), for energy diagnostics. */
  readonly energy?: (q: Value, p: Value) => number
}

/**
 * A separable system from its kinetic and potential energies, the derivatives taken by automatic differentiation.
 *
 * @param kinetic - T(p), a scalar (default p.p / 2, unit mass).
 * @param potential - V(q), a scalar.
 * @returns The system with its energy.
 * @concept hamiltonian-mechanics
 * @example
 * ```ts
 * import { separableHamiltonian } from 'omai-engine/dynamics/ode'
 * import { mul, sum, tensor } from 'omai-engine/tensor'
 * const sho = separableHamiltonian((q) => mul(sum(mul(q, q)), 0.5))
 * sho.energy?.(tensor([1]), tensor([1])) // => 1
 * ```
 */
export function separableHamiltonian(
  potential: (q: Value) => Value,
  kinetic: (p: Value) => Value = (p) => mul(sum(mul(p, p)), 0.5),
): SeparableSystem {
  const dT = grad(kinetic)
  const dV = grad(potential)
  return {
    velocity: (p) => dT(p),
    force: (q) => mul(dV(q), -1),
    energy: (q, p) => item(kinetic(p)) + item(potential(q)),
  }
}

/**
 * The vector field of a separable Hamiltonian system on the stacked phase-space state `[2, ...]` (positions in row 0,
 * momenta in row 1): dq/dt = dT/dp, dp/dt = -dV/dq. It lets any integrator (a Runge-Kutta method, an adaptive pair)
 * run a Hamiltonian system, to set against the symplectic splittings.
 *
 * @param system - The separable system.
 * @returns A field `(t, y) => dy/dt` with y of shape `[2, ...shape]`.
 * @concept hamiltonian-mechanics
 * @example
 * ```ts
 * import { separableField, separableHamiltonian } from 'omai-engine/dynamics/ode'
 * import { mul, sum, tensor, toFlat } from 'omai-engine/tensor'
 * const f = separableField(separableHamiltonian((q) => mul(sum(mul(q, q)), 0.5)))
 * toFlat(f(0, tensor([[1], [2]]))) // => [2, -1]
 * ```
 */
export function separableField(system: SeparableSystem): OdeVectorField {
  return (_t, y) => {
    const q = sliceAxis(y, 0, 0, 1)
    const p = sliceAxis(y, 0, 1, 2)
    const shape = y.shape.slice(1)
    const qs = reshape(q, shape)
    const ps = reshape(p, shape)
    return stack([system.velocity(ps), system.force(qs)])
  }
}

/**
 * A splitting method as alternating drifts q += c_i h dT/dp(p) and kicks p += d_i h F(q), applied in order, with its
 * order.
 */
export interface SplittingMethod {
  readonly name: string
  readonly drift: readonly number[]
  readonly kick: readonly number[]
  readonly order: number
}

const CBRT2 = 2 ** (1 / 3)
const W1 = 1 / (2 - CBRT2)
const W0 = -CBRT2 / (2 - CBRT2)

/** Symplectic Euler (drift then kick), order 1. */
export const SYMPLECTIC_EULER: SplittingMethod = { name: 'symplectic-euler', drift: [1], kick: [1], order: 1 }
/** Stormer-Verlet in position form, the leapfrog (half drift, kick, half drift), order 2. */
export const STORMER_VERLET: SplittingMethod = { name: 'stormer-verlet', drift: [0.5, 0.5], kick: [1, 0], order: 2 }
/** Velocity Verlet (half kick, drift, half kick), order 2. */
export const VELOCITY_VERLET: SplittingMethod = { name: 'velocity-verlet', drift: [0, 1], kick: [0.5, 0.5], order: 2 }
/** Forest and Ruth's fourth-order method (the position-form triple jump of the leapfrog). */
export const FOREST_RUTH: SplittingMethod = {
  name: 'forest-ruth',
  drift: [W1 / 2, (W0 + W1) / 2, (W0 + W1) / 2, W1 / 2],
  kick: [W1, W0, W1, 0],
  order: 4,
}
/** Yoshida's fourth-order composition of velocity Verlet with weights (w1, w0, w1). */
export const YOSHIDA4: SplittingMethod = {
  name: 'yoshida4',
  drift: [0, W1, W0, W1],
  kick: [W1 / 2, (W0 + W1) / 2, (W0 + W1) / 2, W1 / 2],
  order: 4,
}

/**
 * One step of a splitting method on a separable system.
 *
 * @param system - The system.
 * @param q - Positions.
 * @param p - Momenta.
 * @param h - Step.
 * @param method - The splitting (default Stormer-Verlet).
 * @returns The new positions and momenta (differentiable through the step).
 * @concept symplectic-integrator
 * @example
 * ```ts
 * import { separableHamiltonian, symplecticStep } from 'omai-engine/dynamics/ode'
 * import { mul, sum, tensor, toArray } from 'omai-engine/tensor'
 * const free = separableHamiltonian(() => mul(sum(tensor([0])), 0))
 * toArray(symplecticStep(free, tensor([0]), tensor([2]), 0.5).q) // => [1]
 * ```
 */
export function symplecticStep(
  system: SeparableSystem,
  q: Value,
  p: Value,
  h: number,
  method: SplittingMethod = STORMER_VERLET,
): { readonly q: Value; readonly p: Value } {
  let qq = q
  let pp = p
  for (let i = 0; i < method.drift.length; i++) {
    const c = method.drift[i] ?? 0
    if (c !== 0) qq = combine(qq, [[c * h, system.velocity(pp)]])
    const d = method.kick[i] ?? 0
    if (d !== 0) pp = combine(pp, [[d * h, system.force(qq)]])
  }
  return { q: qq, p: pp }
}

/** The state of {@link symplecticAlgorithm}. */
export interface SymplecticState {
  readonly t: number
  readonly q: Value
  readonly p: Value
  readonly ts: number[]
  readonly qs: Value[]
  readonly ps: Value[]
}

/** Options of {@link symplecticIntegrate}. */
export interface SymplecticOptions extends Partial<Omit<RunOptions, 'maxSteps' | 'start'>> {
  readonly dt: number
  readonly method?: SplittingMethod
  /** Keep every n-th step (default 1). */
  readonly every?: number
}

/** A trajectory of a Hamiltonian system. */
export interface SymplecticTrajectory {
  readonly t: readonly number[]
  /** Positions `[nt, ...]`. */
  readonly q: Value
  /** Momenta `[nt, ...]`. */
  readonly p: Value
  /** The energy at each kept step, when the system knows it. */
  readonly energy: readonly number[]
}

const SCHEMA = channels({
  metrics: {
    t: { description: 'time' },
    energy: { description: 'the Hamiltonian H(q, p)' },
  },
  tensors: { q: { role: 'state', description: 'positions' }, p: { role: 'state', description: 'momenta' } },
})

/**
 * The fixed-step integration of a separable Hamiltonian system as a traced algorithm (one step per algorithm step,
 * the energy as a metric).
 *
 * @param system - The system.
 * @param q0 - Initial positions.
 * @param p0 - Initial momenta.
 * @param tspan - [t0, t1].
 * @param options - Step, method and thinning.
 * @returns The algorithm and its step count.
 * @throws OmaiError with code `value` for a non-positive step.
 * @example
 * ```ts
 * import { separableHamiltonian, symplecticAlgorithm } from 'omai-engine/dynamics/ode'
 * import { mul, sum } from 'omai-engine/tensor'
 * symplecticAlgorithm(separableHamiltonian((q) => mul(sum(mul(q, q)), 0.5)), [1], [0], [0, 1], { dt: 0.1 }).steps // => 10
 * ```
 */
export function symplecticAlgorithm(
  system: SeparableSystem,
  q0: TensorLike,
  p0: TensorLike,
  tspan: readonly [number, number],
  options: SymplecticOptions,
): { readonly algorithm: Algorithm<SymplecticState>; readonly steps: number } {
  const dt = options.dt
  if (!(dt > 0)) throw new OmaiError('value', 'symplecticIntegrate: dt must be positive')
  const method = options.method ?? STORMER_VERLET
  const span = tspan[1] - tspan[0]
  const steps = Math.max(0, Math.round(Math.abs(span) / dt))
  const h = steps === 0 ? 0 : span / steps
  const every = Math.max(1, options.every ?? 1)
  const qv = floatState(q0)
  const pv = floatState(p0)
  const algorithm: Algorithm<SymplecticState> = {
    name: 'symplecticIntegrate',
    config: { method: method.name, dt, steps },
    schema: SCHEMA,
    init: () => ({ t: tspan[0], q: qv, p: pv, ts: [tspan[0]], qs: [qv], ps: [pv] }),
    step: (s, ctx) => {
      const { q, p } = symplecticStep(system, s.q, s.p, h, method)
      const k = ctx.t + 1
      const t = tspan[0] + k * h
      if (k % every === 0 || k === steps) {
        s.ts.push(t)
        s.qs.push(q)
        s.ps.push(p)
      }
      return { ...s, t, q, p }
    },
    observe: (s) => ({
      metrics: { t: s.t, ...(system.energy === undefined ? {} : { energy: system.energy(s.q, s.p) }) },
      tensors: {
        q: { role: 'state', value: numbersOf(s.q), shape: s.q.shape },
        p: { role: 'state', value: numbersOf(s.p), shape: s.p.shape },
      },
    }),
  }
  return { algorithm, steps }
}

/**
 * Integrate a separable Hamiltonian system with a symplectic splitting method (symplectic Euler, Stormer-Verlet,
 * velocity Verlet, Forest-Ruth, Yoshida 4): the energy error stays bounded over long times instead of drifting.
 *
 * @param system - The system.
 * @param q0 - Initial positions.
 * @param p0 - Initial momenta.
 * @param tspan - [t0, t1] (the step is adjusted to divide it evenly).
 * @param options - Step, method, thinning and tracing.
 * @returns The trajectory with its energies.
 * @concept symplectic-integrator
 * @example
 * ```ts
 * import { FOREST_RUTH, separableHamiltonian, symplecticIntegrate } from 'omai-engine/dynamics/ode'
 * import { mul, sum } from 'omai-engine/tensor'
 * const sho = separableHamiltonian((q) => mul(sum(mul(q, q)), 0.5))
 * const tr = symplecticIntegrate(sho, [1], [0], [0, 100], { dt: 0.1, method: FOREST_RUTH })
 * Math.abs((tr.energy.at(-1) ?? 0) - 0.5) < 1e-5 // => true
 * ```
 */
export function symplecticIntegrate(
  system: SeparableSystem,
  q0: TensorLike,
  p0: TensorLike,
  tspan: readonly [number, number],
  options: SymplecticOptions,
): SymplecticTrajectory {
  const { algorithm, steps } = symplecticAlgorithm(system, q0, p0, tspan, options)
  const { state } = runAlgorithm(algorithm, {
    maxSteps: steps,
    ...(options.tracer === undefined ? {} : { tracer: options.tracer }),
    ...(options.policy === undefined ? {} : { policy: options.policy }),
    ...(options.parent === undefined ? {} : { parent: options.parent }),
  })
  const energy =
    system.energy === undefined
      ? []
      : state.qs.map((q, i) => (system.energy as NonNullable<SeparableSystem['energy']>)(q, state.ps[i] as Value))
  return { t: state.ts, q: stack(state.qs), p: stack(state.ps), energy }
}

/** Energy-drift statistics of a trajectory. */
export interface EnergyDrift {
  /** max |H - H0| / max(|H0|, tiny). */
  readonly maxRelative: number
  /** The least-squares slope of H against time (secular drift). */
  readonly slope: number
  /** H at the last point minus H0. */
  readonly final: number
}

/**
 * Energy-drift diagnostics: the largest relative energy error and the secular trend. Symplectic methods show a bounded
 * oscillating error with near-zero slope; non-symplectic ones drift.
 *
 * @param t - Times.
 * @param energy - The energies at those times.
 * @returns The drift statistics.
 * @throws OmaiError with code `value` when the lengths differ or there is no data.
 * @concept symplectic-integrator
 * @example
 * ```ts
 * import { energyDrift } from 'omai-engine/dynamics/ode'
 * Math.round(energyDrift([0, 1, 2], [1, 1.1, 1.2]).slope * 10) / 10 // => 0.1
 * ```
 */
export function energyDrift(t: readonly number[], energy: readonly number[]): EnergyDrift {
  if (t.length !== energy.length || t.length === 0) {
    throw new OmaiError('value', 'energyDrift: give one energy per time')
  }
  const h0 = energy[0] ?? 0
  const scale = Math.max(Math.abs(h0), Number.MIN_VALUE)
  let maxRelative = 0
  for (const e of energy) maxRelative = Math.max(maxRelative, Math.abs(e - h0) / scale)
  const mt = t.reduce((a, b) => a + b, 0) / t.length
  const me = energy.reduce((a, b) => a + b, 0) / energy.length
  let sxy = 0
  let sxx = 0
  for (let i = 0; i < t.length; i++) {
    sxy += ((t[i] ?? 0) - mt) * ((energy[i] ?? 0) - me)
    sxx += ((t[i] ?? 0) - mt) ** 2
  }
  return { maxRelative, slope: sxx === 0 ? 0 : sxy / sxx, final: (energy[energy.length - 1] ?? h0) - h0 }
}
