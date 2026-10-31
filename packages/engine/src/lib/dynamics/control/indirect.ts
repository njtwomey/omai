/**
 * Optimal control by Pontryagin's maximum principle, the indirect method: the Hamiltonian H = L + lambda . f of a
 * problem, the state-costate system that the principle makes of it (the state forward, the costate backward in effect,
 * the control minimising H pointwise), bang-bang control for problems linear in the control, and the two-point boundary
 * value problem of finding the initial costate (and, for minimum time, the horizon) that lands the state where it must,
 * solved by Newton shooting as a stepped algorithm: one shot per step, the miss at the far boundary shrinking.
 */

import { grad } from '../../autodiff/index.ts'
import { type Json, MdmlError } from '../../base/index.ts'
import { solve } from '../../numerics/linalg/index.ts'
import { add, concat, item, mul, neg, sliceAxis, sum, tanh, type Value } from '../../tensor/index.ts'
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
import { RK4, rkStep } from '../ode/explicit.ts'
import type { VectorField } from '../ode/problem.ts'

/** An optimal control problem x' = f(x, u), cost integral of L(x, u) dt, with the control that minimises H pointwise. */
export interface ControlProblem {
  /** The state dimension. */
  readonly stateSize: number
  /** The control dimension. */
  readonly controlSize: number
  /** The dynamics f(x, u). */
  readonly dynamics: (x: Value, u: Value) => Value
  /** The running cost L(x, u), a scalar. */
  readonly runningCost: (x: Value, u: Value) => Value
  /** The minimiser of the Hamiltonian over the admissible controls, u*(x, lambda) (the stationarity condition, or a bang-bang rule). */
  readonly control: (x: Value, lambda: Value) => Value
  /** Optionally the switching function sigma(x, lambda) = dH/du, whose sign changes are the switches of a bang-bang control. */
  readonly switching?: (x: Value, lambda: Value) => number
}

/**
 * The Hamiltonian H(x, lambda, u) = L(x, u) + lambda . f(x, u): the quantity Pontryagin's principle minimises over the
 * control along an optimal trajectory (and which is constant in time when the problem does not depend on it, zero when the
 * horizon is free).
 *
 * @param problem - The problem.
 * @param x - The state.
 * @param lambda - The costate.
 * @param u - The control.
 * @returns H.
 * @concept pontryagin-maximum-principle
 * @example
 * ```ts
 * import { pontryaginHamiltonian, type ControlProblem } from 'mdml-engine/dynamics/control'
 * import { add, item, mul, sum, tensor } from 'mdml-engine/tensor'
 * // x' = u with cost u^2 / 2: H = u^2 / 2 + lambda u.
 * const p: ControlProblem = { stateSize: 1, controlSize: 1, dynamics: (_x, u) => u, runningCost: (_x, u) => mul(sum(mul(u, u)), 0.5), control: (_x, l) => mul(l, -1) }
 * item(pontryaginHamiltonian(p, tensor([0]), tensor([2]), tensor([3]))) // => 10.5
 * ```
 */
export function pontryaginHamiltonian(problem: ControlProblem, x: Value, lambda: Value, u: Value): Value {
  return add(problem.runningCost(x, u), sum(mul(lambda, problem.dynamics(x, u))))
}

const part = (y: Value, from: number, to: number): Value => sliceAxis(y, 0, from, to)

/**
 * The state-costate system of the maximum principle on `y = [x, lambda]`: x' = f(x, u*), lambda' = -dH/dx at the
 * minimising control u*(x, lambda). It is an ordinary differential equation with the state fixed at one end and the costate
 * (up to the transversality conditions) at the other: integrating it forward from a guessed initial costate is shooting;
 * integrating it with a negative step from a terminal costate is the backward sweep.
 *
 * @param problem - The problem.
 * @returns The vector field on `[x, lambda]` (size `2 n`).
 * @concept pontryagin-maximum-principle
 * @example
 * ```ts
 * import { doubleIntegratorEnergy, hamiltonianSystem } from 'mdml-engine/dynamics/control'
 * import { tensor, toFlat } from 'mdml-engine/tensor'
 * // Double integrator, cost u^2 / 2: u* = -lambda2, so x' = (v, -lambda2) and lambda' = (0, -lambda1).
 * toFlat(hamiltonianSystem(doubleIntegratorEnergy())(0, tensor([0, 0, 2, 3]))).map((v) => v + 0) // => [0, -3, 0, -2]
 * ```
 */
export function hamiltonianSystem(problem: ControlProblem): VectorField {
  const n = problem.stateSize
  return (_t, y) => {
    const x = part(y, 0, n)
    const lambda = part(y, n, 2 * n)
    const u = problem.control(x, lambda)
    const dHdx = grad((xx: Value) => pontryaginHamiltonian(problem, xx, lambda, u))(x)
    return concat([problem.dynamics(x, u), neg(dHdx)])
  }
}

/**
 * The control that minimises sigma u over the interval [lo, hi] when the Hamiltonian is linear in the control with
 * switching function sigma = dH/du: the lower bound where sigma is positive, the upper where it is negative (a singular arc,
 * sigma = 0 over an interval, is not handled). A minimum-time problem's control is this, switching where sigma changes sign.
 *
 * @param sigma - The switching function.
 * @param bounds - The control limits.
 * @returns The extreme control.
 * @concept pontryagin-maximum-principle
 * @example
 * ```ts
 * import { bangBang } from 'mdml-engine/dynamics/control'
 * [bangBang(0.3, [-1, 1]), bangBang(-0.3, [-1, 1])] // => [-1, 1]
 * ```
 */
export function bangBang(sigma: number, bounds: readonly [number, number]): number {
  return sigma > 0 ? bounds[0] : bounds[1]
}

/** The state of {@link pontryaginAlgorithm}: the unknowns, the shot they gave and how far it missed. */
export interface ShotState {
  /** The initial costate being solved for. */
  readonly lambda0: Float64Array
  /** The horizon (the guess, or the unknown when it is free). */
  readonly horizon: number
  /** The residual at the far boundary: the state miss, plus the Hamiltonian at the end for a free horizon. */
  readonly miss: Float64Array
  readonly missNorm: number
  /** The shot's states, costates and controls at the nodes, flat row-major `[steps + 1, n]` and `[steps + 1, m]`. */
  readonly states: Float64Array
  readonly costates: Float64Array
  readonly controls: Float64Array
  /** The running cost integrated along the shot (trapezoid rule). */
  readonly cost: number
  /** The times at which the switching function changed sign. */
  readonly switches: readonly number[]
  readonly iteration: number
}

/** The paths of {@link pontryaginAlgorithm}. */
export type ShootingPath =
  | 'shot/initialCostate'
  | 'shot/horizon'
  | 'shot/miss'
  | 'shot/cost'
  | 'trajectory/state'
  | 'trajectory/costate'
  | 'trajectory/control'
  | 'trajectory/switch'

/** What a shooting run logs. */
export const shootingContract: Contract<ShotState, ShootingPath> = defineContract({
  'shot/initialCostate': tensor({ role: 'param', shape: ['n'], description: 'the initial costate being solved for' }),
  'shot/horizon': scalar({ description: 'the horizon T (an unknown when it is free)' }),
  'shot/miss': scalar({ description: 'the norm of the residual at the far boundary', direction: 'lower' }),
  'shot/cost': scalar({ description: 'the running cost along the shot' }),
  'trajectory/state': tensor({ role: 'state', shape: ['steps', 'n'], description: 'the shot’s states' }),
  'trajectory/costate': tensor({ role: 'state', shape: ['steps', 'n'], description: 'the shot’s costates' }),
  'trajectory/control': tensor({ role: 'output', shape: ['steps', 'm'], description: 'the control along the shot' }),
  'trajectory/switch': event({ description: 'the switching function changed sign', context: { time: 'when' } }),
})

/** Options of {@link pontryaginAlgorithm}. */
export interface ShootingOptions {
  /** The initial state. */
  readonly x0: readonly number[]
  /** A fixed final state. */
  readonly target?: readonly number[]
  /** For a free final state, the terminal cost phi(x(T)): the transversality condition is lambda(T) = grad phi. */
  readonly terminalCost?: (x: Value) => Value
  /** The horizon T, or its starting guess when `freeHorizon` is set. */
  readonly horizon: number
  /** Solve for the horizon too, with the extra condition H(T) = 0 (minimum time and the like). */
  readonly freeHorizon?: boolean
  /** The starting guess of the initial costate. */
  readonly lambda0: readonly number[]
  /** Runge-Kutta steps across the horizon (default 100). */
  readonly steps?: number
  /** Stop when the residual's norm is below this (default 1e-8). */
  readonly tol?: number
}

interface Shot {
  readonly miss: Float64Array
  readonly states: Float64Array
  readonly costates: Float64Array
  readonly controls: Float64Array
  readonly cost: number
  readonly switches: number[]
}

/**
 * Pontryagin's maximum principle as a stepped shooting algorithm. The unknowns are the initial costate (and the horizon when
 * it is free); a shot integrates the state-costate system forward from them with fixed-step RK4, taking the control that
 * minimises the Hamiltonian at every node; the residual is how far the state lands from the target (for a free final state,
 * how far the costate lands from the gradient of the terminal cost; for a free horizon, the Hamiltonian at the end must also
 * vanish). Each algorithm step is one damped Newton update of the unknowns, the Jacobian by finite differences of the shot,
 * followed by the shot it gives, which is logged with its states, costates, controls and the time of any switch.
 *
 * @param problem - The control problem.
 * @param options - The boundary conditions, the guess and the resolution.
 * @returns The algorithm; it converges when the residual is below the tolerance.
 * @throws MdmlError with code `value` when neither a target nor a terminal cost is given, or the guess has the wrong size.
 * @concept pontryagin-maximum-principle
 * @example
 * ```ts
 * import { doubleIntegratorEnergy, pontryaginAlgorithm } from 'mdml-engine/dynamics/control'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * // Rest to rest over a unit distance in time 2 at least energy: Newton finds the costate that does it.
 * const alg = pontryaginAlgorithm(doubleIntegratorEnergy(), { x0: [0, 0], target: [1, 0], horizon: 2, lambda0: [0, 0] })
 * runAlgorithm(alg, { maxSteps: 10 }).state.missNorm < 1e-8 // => true
 * ```
 */
export function pontryaginAlgorithm(
  problem: ControlProblem,
  options: ShootingOptions,
): Algorithm<ShotState, ShootingPath> {
  const n = problem.stateSize
  const m = problem.controlSize
  const free = options.freeHorizon === true
  if (options.target === undefined && options.terminalCost === undefined) {
    throw new MdmlError('value', 'pontryaginAlgorithm: give a target state or a terminal cost')
  }
  if (options.lambda0.length !== n || options.x0.length !== n) {
    throw new MdmlError('value', `pontryaginAlgorithm: the initial state and costate guess have ${n} entries`)
  }
  const steps = options.steps ?? 100
  const tol = options.tol ?? 1e-8
  const field = hamiltonianSystem(problem)

  const shoot = (z: Float64Array): Shot => {
    const T = free ? (z[n] as number) : options.horizon
    const h = T / steps
    const states = new Float64Array((steps + 1) * n)
    const costates = new Float64Array((steps + 1) * n)
    const controls = new Float64Array((steps + 1) * m)
    let y: Value = fromNumbers(Float64Array.from([...options.x0, ...z.slice(0, n)]), [2 * n])
    let cost = 0
    let prevCost = 0
    let prevSigma = Number.NaN
    const switches: number[] = []
    for (let k = 0; k <= steps; k++) {
      const yk = numbersOf(y)
      states.set(yk.slice(0, n), k * n)
      costates.set(yk.slice(n), k * n)
      const x = part(y, 0, n)
      const lambda = part(y, n, 2 * n)
      const u = problem.control(x, lambda)
      controls.set(numbersOf(u), k * m)
      const here = item(problem.runningCost(x, u))
      if (k > 0) cost += 0.5 * h * (here + prevCost)
      prevCost = here
      if (problem.switching !== undefined) {
        const sigma = problem.switching(x, lambda)
        if (k > 0 && Math.sign(sigma) !== Math.sign(prevSigma) && !Number.isNaN(prevSigma)) {
          switches.push((k - 1 + prevSigma / (prevSigma - sigma)) * h)
        }
        prevSigma = sigma
      }
      if (k < steps) y = rkStep(field, k * h, y, h, RK4)
    }
    const end = numbersOf(y)
    const xT = end.slice(0, n)
    const lT = end.slice(n)
    const miss: number[] = []
    if (options.target !== undefined) {
      for (let i = 0; i < n; i++) miss.push((xT[i] as number) - (options.target[i] as number))
    } else {
      const phi = options.terminalCost as (x: Value) => Value
      const g = numbersOf(grad(phi)(fromNumbers(xT, [n])))
      for (let i = 0; i < n; i++) miss.push((lT[i] as number) - (g[i] as number))
    }
    if (free) {
      const x = part(y, 0, n)
      const lambda = part(y, n, 2 * n)
      miss.push(item(pontryaginHamiltonian(problem, x, lambda, problem.control(x, lambda))))
    }
    return { miss: Float64Array.from(miss), states, costates, controls, cost, switches }
  }

  const unknowns = n + (free ? 1 : 0)
  const norm = (v: Float64Array): number => Math.hypot(...v)
  const stateOf = (z: Float64Array, shot: Shot, iteration: number): ShotState => ({
    lambda0: z.slice(0, n),
    horizon: free ? (z[n] as number) : options.horizon,
    miss: shot.miss,
    missNorm: norm(shot.miss),
    states: shot.states,
    costates: shot.costates,
    controls: shot.controls,
    cost: shot.cost,
    switches: shot.switches,
    iteration,
  })
  const pack = (s: ShotState): Float64Array => (free ? Float64Array.from([...s.lambda0, s.horizon]) : s.lambda0.slice())

  return {
    name: 'pontryagin',
    config: {
      x0: [...options.x0],
      target: options.target === undefined ? null : [...options.target],
      horizon: options.horizon,
      freeHorizon: free,
      steps,
      tol,
    } as Json,
    init: () => {
      const z = Float64Array.from([...options.lambda0, ...(free ? [options.horizon] : [])])
      return stateOf(z, shoot(z), 0)
    },
    step: (s, ctx) => {
      const z = pack(s)
      // The Jacobian of the residual in the unknowns, by forward differences of the shot.
      const jac = new Float64Array(unknowns * unknowns)
      for (let j = 0; j < unknowns; j++) {
        const dz = 1e-6 * Math.max(1, Math.abs(z[j] as number))
        const zp = z.slice()
        zp[j] = (zp[j] as number) + dz
        const mp = shoot(zp).miss
        for (let i = 0; i < unknowns; i++) jac[i * unknowns + j] = ((mp[i] as number) - (s.miss[i] as number)) / dz
      }
      const direction = numbersOf(solve(fromNumbers(jac, [unknowns, unknowns]), fromNumbers(s.miss, [unknowns])))
      // Halve the step until the residual falls (a damped Newton step).
      let scale = 1
      let next = z
      let shot: Shot = shoot(z)
      for (let tries = 0; tries < 12; tries++) {
        next = z.map((v, i) => v - scale * (direction[i] as number))
        shot = shoot(next)
        const admissible = !free || (next[n] as number) > 0
        if (admissible && norm(shot.miss) < s.missNorm) break
        scale /= 2
      }
      const state = stateOf(next, shot, s.iteration + 1)
      ctx.log('shot/initialCostate', state.lambda0)
      ctx.log('shot/horizon', state.horizon)
      ctx.log('shot/miss', state.missNorm)
      ctx.log('shot/cost', state.cost)
      ctx.log('trajectory/state', fromNumbers(state.states, [steps + 1, n]))
      ctx.log('trajectory/costate', fromNumbers(state.costates, [steps + 1, n]))
      ctx.log('trajectory/control', fromNumbers(state.controls, [steps + 1, m]))
      for (const time of state.switches) ctx.emit('trajectory/switch', { time })
      return state
    },
    done: (s) => (s.missNorm <= tol ? 'converged' : undefined),
    serialise: (s) => stateToJson(s),
    deserialise: (j: Json) => stateFromJson(j) as ShotState,
  }
}

/**
 * The double integrator x' = v, v' = u with the control energy u^2 / 2 as the cost: the standard test of the maximum
 * principle. The principle gives u* = -lambda2, lambda1' = 0, lambda2' = -lambda1, so the control is a straight line in
 * time; for rest to rest (0, 0) to (d, 0) in time T it is u(t) = 6 d (T - 2 t) / T^3, at cost 6 d^2 / T^3 (the truth the
 * tests check the shooting against).
 *
 * @returns The problem.
 * @concept pontryagin-maximum-principle
 * @example
 * ```ts
 * import { doubleIntegratorEnergy } from 'mdml-engine/dynamics/control'
 * doubleIntegratorEnergy().stateSize // => 2
 * ```
 */
export function doubleIntegratorEnergy(): ControlProblem {
  return {
    stateSize: 2,
    controlSize: 1,
    dynamics: (x, u) => concat([part(x, 1, 2), u]),
    runningCost: (_x, u) => mul(sum(mul(u, u)), 0.5),
    control: (_x, lambda) => neg(part(lambda, 1, 2)),
    switching: (_x, lambda) => item(part(lambda, 1, 2)),
  }
}

/**
 * The double integrator to rest in minimum time with |u| <= bound: cost 1 per unit time (so H = 1 + lambda . f and the
 * horizon is free), and the control that minimises H is bang-bang, u* = -bound sign(lambda2), switching where lambda2 changes
 * sign: along the switching curve x = -v |v| / (2 bound), and from rest at distance d the time is 2 sqrt(d / bound) with the
 * switch at half of it. A discontinuous control makes the shot's residual piecewise constant in the costate (Newton has no
 * slope to follow), so `smoothing` replaces the sign by tanh(lambda2 / smoothing): a smooth problem to solve, whose solution
 * approaches the bang-bang one as the smoothing falls. Solve with a large smoothing first and shrink it, each solution the
 * next one's guess ({@link minimumTimeShooting} does that).
 *
 * @param bound - The control limit (default 1).
 * @param smoothing - The width of the tanh that stands for the sign; 0 is the exact bang-bang rule (default 0).
 * @returns The problem, with its switching function lambda2.
 * @concept pontryagin-maximum-principle
 * @example
 * ```ts
 * import { minimumTimeDoubleIntegrator } from 'mdml-engine/dynamics/control'
 * import { tensor, toFlat } from 'mdml-engine/tensor'
 * // lambda2 > 0: brake with the full negative control.
 * toFlat(minimumTimeDoubleIntegrator().control(tensor([1, 0]), tensor([1, 0.5]))) // => [-1]
 * ```
 */
export function minimumTimeDoubleIntegrator(bound = 1, smoothing = 0): ControlProblem {
  return {
    stateSize: 2,
    controlSize: 1,
    dynamics: (x, u) => concat([part(x, 1, 2), u]),
    runningCost: () => floatState(1),
    control: (_x, lambda) =>
      smoothing > 0
        ? mul(tanh(mul(part(lambda, 1, 2), 1 / smoothing)), -bound)
        : floatState([bangBang(item(part(lambda, 1, 2)), [-bound, bound])]),
    switching: (_x, lambda) => item(part(lambda, 1, 2)),
  }
}

/** Options of {@link minimumTimeShooting}. */
export interface MinimumTimeOptions {
  /** The control limit (default 1). */
  readonly bound?: number
  /** The smoothings to solve for in turn, from a large one to a small one (default 1, 0.3, 0.1, 0.03, 0.01). */
  readonly smoothings?: readonly number[]
  /** Runge-Kutta steps across the horizon (default 400). */
  readonly steps?: number
  /** The starting guess of the initial costate (default `[0.8, 0.8]`) and horizon (default 3 sqrt(distance)). */
  readonly lambda0?: readonly number[]
  readonly horizon?: number
}

/**
 * The minimum-time problem of the double integrator from `(x0, 0)` to rest at the origin, solved by shooting with continuation
 * in the control's smoothing: each stage is a {@link pontryaginAlgorithm} run to convergence from the previous stage's
 * costate and horizon. The result is the last stage's shot, a smoothed bang-bang control whose switch is a few smoothing
 * widths wide.
 *
 * @param x0 - The starting position (rest is assumed; a negative position is mirrored).
 * @param options - Bound, the smoothing schedule, resolution and guess.
 * @returns The final shot, with the stages' final states.
 * @throws MdmlError with code `convergence` when a stage fails to converge in 60 iterations.
 * @concept pontryagin-maximum-principle
 * @example
 * ```ts
 * import { minimumTimeShooting } from 'mdml-engine/dynamics/control'
 * Math.abs(minimumTimeShooting(1).final.horizon - 2) < 0.02 // => true
 * ```
 */
export function minimumTimeShooting(
  x0: number,
  options: MinimumTimeOptions = {},
): { readonly final: ShotState; readonly stages: readonly ShotState[] } {
  const bound = options.bound ?? 1
  const stages: ShotState[] = []
  let lambda0 = options.lambda0 ?? [0.8, 0.8]
  let horizon = options.horizon ?? 3 * Math.sqrt(Math.abs(x0) / bound)
  for (const smoothing of options.smoothings ?? [1, 0.3, 0.1, 0.03, 0.01]) {
    const alg = pontryaginAlgorithm(minimumTimeDoubleIntegrator(bound, smoothing), {
      x0: [x0, 0],
      target: [0, 0],
      horizon,
      freeHorizon: true,
      lambda0,
      steps: options.steps ?? 400,
    })
    let state = alg.init(null as never)
    const ctx = { log: () => undefined, emit: () => undefined, wants: () => false } as never
    for (let k = 0; k < 60 && alg.done?.(state, k) === undefined; k++) state = alg.step(state, ctx)
    if (alg.done?.(state, 60) === undefined) {
      throw new MdmlError('convergence', `minimumTimeShooting: no convergence at smoothing ${smoothing}`)
    }
    stages.push(state)
    lambda0 = Array.from(state.lambda0)
    horizon = state.horizon
  }
  return { final: stages[stages.length - 1] as ShotState, stages }
}
