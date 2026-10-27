/**
 * Named dynamical systems as vector fields for the ODE solvers and the phase-portrait tools: predator and prey
 * (Lotka-Volterra), the Lorenz attractor, the Van der Pol oscillator, a damped pendulum and the SIR epidemic. Each is
 * written in tensor operations, so a field can be differentiated (Jacobians, fixed points) as well as integrated, and
 * comes with a start, a time span that shows its behaviour and the names of its components.
 */

import { mul, neg, sin, sliceAxis, stack, sub, sum, type Value } from '../tensor/index.ts'
import type { VectorField } from './ode/problem.ts'

/** Component `i` of a state vector, as a scalar. */
const at = (y: Value, i: number): Value => sum(sliceAxis(y, 0, i, i + 1))

/** The parameters of {@link lotkaVolterraField}. */
export interface LotkaVolterraParameters {
  /** The prey's growth rate (default 1). */
  readonly alpha?: number
  /** How fast predators eat prey (default 0.1). */
  readonly beta?: number
  /** How fast eating turns into new predators (default 0.075). */
  readonly delta?: number
  /** The predators' death rate (default 1.5). */
  readonly gamma?: number
}

/**
 * The Lotka-Volterra predator and prey equations on `(prey, predators)`: `x' = alpha x - beta x y` and
 * `y' = delta x y - gamma y`. Solutions circle the equilibrium `(gamma / delta, alpha / beta)` and conserve
 * `delta x - gamma ln x + beta y - alpha ln y`.
 *
 * @param parameters - The four rates.
 * @returns The vector field.
 * @example
 * ```ts
 * import { lotkaVolterraField } from 'mdml-engine/dynamics'
 * import { tensor, toFlat } from 'mdml-engine/tensor'
 * toFlat(lotkaVolterraField()(0, tensor([20, 10]))) // => [0, 0]
 * ```
 */
export function lotkaVolterraField(parameters: LotkaVolterraParameters = {}): VectorField {
  const { alpha = 1, beta = 0.1, delta = 0.075, gamma = 1.5 } = parameters
  return (_t, y) => {
    const x = at(y, 0)
    const p = at(y, 1)
    const eaten = mul(x, p)
    return stack([sub(mul(x, alpha), mul(eaten, beta)), sub(mul(eaten, delta), mul(p, gamma))])
  }
}

/** The parameters of {@link lorenzField}. */
export interface LorenzParameters {
  /** The Prandtl number (default 10). */
  readonly sigma?: number
  /** The Rayleigh number (default 28, where the flow is chaotic). */
  readonly rho?: number
  /** The aspect ratio (default 8/3). */
  readonly beta?: number
}

/**
 * The Lorenz equations on `(x, y, z)`: `x' = sigma (y - x)`, `y' = x (rho - z) - y`, `z' = x y - beta z`. At the
 * classic parameters the flow settles on a strange attractor; its divergence is the constant `-(sigma + 1 + beta)`.
 *
 * @param parameters - sigma, rho and beta.
 * @returns The vector field.
 * @example
 * ```ts
 * import { lorenzField } from 'mdml-engine/dynamics'
 * import { tensor, toFlat } from 'mdml-engine/tensor'
 * toFlat(lorenzField()(0, tensor([1, 1, 1]))) // => [0, 26, -1.6666666666666665]
 * ```
 */
export function lorenzField(parameters: LorenzParameters = {}): VectorField {
  const { sigma = 10, rho = 28, beta = 8 / 3 } = parameters
  return (_t, s) => {
    const x = at(s, 0)
    const y = at(s, 1)
    const z = at(s, 2)
    return stack([mul(sub(y, x), sigma), sub(mul(x, sub(rho, z)), y), sub(mul(x, y), mul(z, beta))])
  }
}

/**
 * The Van der Pol oscillator on `(position, velocity)`: `x' = v`, `v' = mu (1 - x^2) v - x`. For `mu > 0` every
 * orbit but the origin is drawn to one limit cycle, which turns from a sine into a relaxation oscillation as `mu` grows.
 *
 * @param mu - The nonlinearity (default 2).
 * @returns The vector field.
 * @example
 * ```ts
 * import { vanDerPolField } from 'mdml-engine/dynamics'
 * import { tensor, toFlat } from 'mdml-engine/tensor'
 * toFlat(vanDerPolField(2)(0, tensor([2, 1]))) // => [1, -8]
 * ```
 */
export function vanDerPolField(mu = 2): VectorField {
  return (_t, y) => {
    const x = at(y, 0)
    const v = at(y, 1)
    return stack([v, sub(mul(mul(sub(1, mul(x, x)), v), mu), x)])
  }
}

/** The parameters of {@link dampedPendulumField}. */
export interface DampedPendulumParameters {
  /** Gravity over the length (default 1). */
  readonly gravityOverLength?: number
  /** The drag on the angular velocity (default 0.25). */
  readonly damping?: number
}

/**
 * A damped pendulum on `(angle, angular velocity)`: `theta' = omega`, `omega' = -(g / L) sin theta - c omega`. The
 * downward equilibria `(2 k pi, 0)` are stable spirals and the upward ones `((2 k + 1) pi, 0)` saddles.
 *
 * @param parameters - g over L, and the damping.
 * @returns The vector field.
 * @example
 * ```ts
 * import { dampedPendulumField } from 'mdml-engine/dynamics'
 * import { tensor, toFlat } from 'mdml-engine/tensor'
 * toFlat(dampedPendulumField({ gravityOverLength: 1, damping: 0.5 })(0, tensor([0, 2]))) // => [2, -1]
 * ```
 */
export function dampedPendulumField(parameters: DampedPendulumParameters = {}): VectorField {
  const { gravityOverLength = 1, damping = 0.25 } = parameters
  return (_t, y) => {
    const omega = at(y, 1)
    return stack([omega, sub(neg(mul(sin(at(y, 0)), gravityOverLength)), mul(omega, damping))])
  }
}

/** The parameters of {@link sirField}. */
export interface SirParameters {
  /** Infections per infectious person per day in a fully susceptible population (default 0.4). */
  readonly beta?: number
  /** One over the mean infectious period, per day (default 0.1). */
  readonly gamma?: number
}

/**
 * The SIR epidemic on the shares `(susceptible, infectious, recovered)`: `S' = -beta S I`, `I' = beta S I - gamma I`,
 * `R' = gamma I`. The shares always sum to one, and the infection grows while `S > gamma / beta`.
 *
 * @param parameters - The transmission and recovery rates, per day.
 * @returns The vector field.
 * @concept compartmental-models
 * @example
 * ```ts
 * import { sirField } from 'mdml-engine/dynamics'
 * import { tensor, toFlat } from 'mdml-engine/tensor'
 * toFlat(sirField()(0, tensor([0.5, 0.25, 0.25]))) // => [-0.05, 0.025, 0.025]
 * ```
 */
export function sirField(parameters: SirParameters = {}): VectorField {
  const { beta = 0.4, gamma = 0.1 } = parameters
  return (_t, y) => {
    const s = at(y, 0)
    const i = at(y, 1)
    const infections = mul(mul(s, i), beta)
    const recoveries = mul(i, gamma)
    return stack([neg(infections), sub(infections, recoveries), recoveries])
  }
}

/** The names of the systems in {@link namedSystems}. */
export type NamedSystemName = 'lotka-volterra' | 'lorenz' | 'van-der-pol' | 'damped-pendulum' | 'sir'

/** A named system ready to integrate: its field, a start, a span that shows its behaviour and its components. */
export interface NamedSystem {
  readonly label: string
  readonly description: string
  readonly field: VectorField
  /** The initial state. */
  readonly y0: readonly number[]
  /** A time span long enough to show the behaviour (many cycles, or until the epidemic has run its course). */
  readonly span: readonly [number, number]
  /** The names of the state's components. */
  readonly names: readonly string[]
}

/**
 * The named systems at their default parameters, each with a start and a time span that shows it: the field to hand to
 * `solveIvp` or `solveIvpAlgorithm`.
 *
 * @example
 * ```ts
 * import { namedSystems } from 'mdml-engine/dynamics'
 * Object.keys(namedSystems) // => ['lotka-volterra', 'lorenz', 'van-der-pol', 'damped-pendulum', 'sir']
 * ```
 */
export const namedSystems: Readonly<Record<NamedSystemName, NamedSystem>> = {
  'lotka-volterra': {
    label: 'Lotka-Volterra',
    description: 'Predators and prey: each boom feeds the other and the populations circle for ever.',
    field: lotkaVolterraField(),
    y0: [10, 5],
    span: [0, 300],
    names: ['prey', 'predators'],
  },
  lorenz: {
    label: 'Lorenz',
    description: 'Three equations of convection whose flow is chaotic: nearby starts part ways on the attractor.',
    field: lorenzField(),
    y0: [1, 1, 1],
    span: [0, 100],
    names: ['x', 'y', 'z'],
  },
  'van-der-pol': {
    label: 'Van der Pol',
    description: 'A self-sustaining oscillator: every start is drawn to the same limit cycle.',
    field: vanDerPolField(),
    y0: [0.5, 0],
    span: [0, 150],
    names: ['position', 'velocity'],
  },
  'damped-pendulum': {
    label: 'Damped pendulum',
    description: 'A swinging weight losing energy to drag: the swings shrink to the downward rest.',
    field: dampedPendulumField(),
    y0: [2.5, 0],
    span: [0, 60],
    names: ['angle', 'angular velocity'],
  },
  sir: {
    label: 'SIR epidemic',
    description: 'Susceptible, infectious and recovered shares of a population as an infection runs through it.',
    field: sirField(),
    y0: [0.99, 0.01, 0],
    span: [0, 160],
    names: ['susceptible', 'infectious', 'recovered'],
  },
}
