/**
 * Evolutionary game theory: how population shares of strategies change when better-performing strategies grow. The
 * replicator dynamics of symmetric games and its two-population form for bimatrix games, as in nashpy, and the
 * projected replicator dynamics of Lanctot et al. as a traced algorithm.
 */

import { OmaiError } from '../base/index.ts'
import { projectSimplex } from '../optim/index.ts'
import { dict, type OptionTable, optionTable } from '../space/index.ts'
import { asTensor, type TensorLike, toFlat } from '../tensor/index.ts'
import {
  type Algorithm,
  type Contract,
  channels,
  defineContract,
  runAlgorithm,
  scalar,
  stateFromJson,
  stateToJson,
  tensor,
} from '../trace/index.ts'
import { actionValues, type GameRunOptions, type NormalFormGame, nashConv } from './normal-form/index.ts'

/** A vector field on population states. */
type Field = (x: readonly number[]) => number[]

/** One step of size h of an integrator of a vector field. */
export type FieldStepper = (f: (x: readonly number[]) => number[], x: readonly number[], h: number) => number[]

/** What an integrator can do, for validity rules and selectors. */
export interface FieldStepperProperties {
  /** The order of its global error in the step size. */
  readonly order: number
}

/**
 * The integrators of the replicator equation as one slot: `rk4` (the classical Runge-Kutta, as nashpy's odeint agrees
 * with to about 1e-8), `midpoint` (second order) and `euler` (first order, which drifts off the simplex's invariants
 * faster).
 *
 * @concept replicator-dynamics
 * @example
 * ```ts
 * import { replicatorIntegrators } from 'omai-engine/games'
 * replicatorIntegrators.make({ kind: 'euler' })((x) => x.map((v) => -v), [1], 0.5) // => [0.5]
 * ```
 */
export const replicatorIntegrators: OptionTable<FieldStepper, FieldStepperProperties> = optionTable(
  {
    rk4: {
      description: 'the classical fourth-order Runge-Kutta step',
      properties: { order: 4 },
      space: dict({}),
      make: () => (f, x, h) => {
        const k1 = f(x)
        const k2 = f(x.map((v, i) => v + 0.5 * h * (k1[i] ?? 0)))
        const k3 = f(x.map((v, i) => v + 0.5 * h * (k2[i] ?? 0)))
        const k4 = f(x.map((v, i) => v + h * (k3[i] ?? 0)))
        return x.map((v, i) => v + (h / 6) * ((k1[i] ?? 0) + 2 * (k2[i] ?? 0) + 2 * (k3[i] ?? 0) + (k4[i] ?? 0)))
      },
    },
    midpoint: {
      description: 'the explicit midpoint rule: an Euler half step to evaluate the field at',
      properties: { order: 2 },
      space: dict({}),
      make: () => (f, x, h) => {
        const k1 = f(x)
        const k2 = f(x.map((v, i) => v + 0.5 * h * (k1[i] ?? 0)))
        return x.map((v, i) => v + h * (k2[i] ?? 0))
      },
    },
    euler: {
      description: 'the explicit Euler step',
      properties: { order: 1 },
      space: dict({}),
      make: () => (f, x, h) => {
        const k1 = f(x)
        return x.map((v, i) => v + h * (k1[i] ?? 0))
      },
    },
  },
  {
    default: 'rk4',
    label: 'integrator',
    description: 'How the replicator equation is advanced between time points.',
  },
)

function squareMatrix(a: TensorLike, what: string): number[][] {
  const t = asTensor(a)
  const [m, n] = t.shape as [number, number]
  if (t.shape.length !== 2 || m !== n || m === 0)
    throw new OmaiError('shape', `${what}: a non-empty square matrix is needed`)
  const flat = toFlat(t)
  return Array.from({ length: m }, (_, i) => flat.slice(i * n, (i + 1) * n))
}

/**
 * The replicator vector field of a symmetric game with payoff matrix A (nashpy's `get_derivative_of_fitness`):
 * `x' = (x * f) Q - x phi` with fitness `f = A x`, average fitness `phi = x . f` and an optional row-stochastic
 * mutation matrix Q (default the identity, the plain replicator equation `x_i' = x_i (f_i - phi)`).
 *
 * @param a - The payoff matrix of the symmetric game.
 * @param x - The population state (a probability vector).
 * @param mutation - The mutation matrix Q (default the identity).
 * @returns The time derivative of the state.
 * @concept replicator-dynamics
 * @example
 * ```ts
 * import { replicatorField } from 'omai-engine/games'
 * replicatorField([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], [0.5, 0.5, 0]) // => [-0.25, 0.25, 0]
 * ```
 */
export function replicatorField(a: TensorLike, x: readonly number[], mutation?: TensorLike): number[] {
  const m = squareMatrix(a, 'replicatorField')
  const q = mutation === undefined ? undefined : squareMatrix(mutation, 'replicatorField')
  return fieldOf(m, q)(x)
}

/** The fitness of each strategy and the mean fitness (one entry per population) at a state. */
type FitnessOf = (x: readonly number[]) => { readonly fitness: number[]; readonly mean: number[] }

function symmetricFitness(m: readonly (readonly number[])[]): FitnessOf {
  return (x) => {
    const f = m.map((r) => r.reduce((s, v, j) => s + v * (x[j] ?? 0), 0))
    let phi = 0
    for (const [i, v] of x.entries()) phi += v * (f[i] ?? 0)
    return { fitness: f, mean: [phi] }
  }
}

function fieldOf(m: readonly (readonly number[])[], q: readonly (readonly number[])[] | undefined): Field {
  const fitnessOf = symmetricFitness(m)
  return (x) => {
    const { fitness: f, mean } = fitnessOf(x)
    const phi = mean[0] ?? 0
    const g = x.map((v, i) => v * (f[i] ?? 0))
    const mutated = q === undefined ? g : x.map((_, j) => g.reduce((s, v, i) => s + v * (q[i]?.[j] ?? 0), 0))
    return mutated.map((v, i) => v - (x[i] ?? 0) * phi)
  }
}

/** The state of a replicator integration: the time point reached and the population state(s) there. */
export interface ReplicatorState {
  readonly index: number
  readonly time: number
  readonly x: readonly number[]
  /** The states at every time point so far. */
  readonly path: readonly (readonly number[])[]
}

/** Options for the replicator integrations. */
export interface ReplicatorOptions extends GameRunOptions {
  /** The times at which to report the state, increasing (default 1000 points evenly on [0, 10], as nashpy). */
  readonly timepoints?: readonly number[]
  /** The largest step between and within time points (default 0.01). */
  readonly maxStep?: number
  /** The integrator (default `'rk4'`, see {@link replicatorIntegrators}). */
  readonly integrator?: 'rk4' | 'midpoint' | 'euler'
}

/** The paths the replicator algorithms log, as a type. */
export type ReplicatorPath = 'replicator/x' | 'replicator/time' | 'replicator/fitness' | 'replicator/meanFitness'

/**
 * What the replicator algorithms log, by path: the population state at the time point reached (for the two-population
 * form the row population then the column population), the time, the fitness of every strategy there and the mean
 * fitness of each population.
 */
export const replicatorContract: Contract<unknown, ReplicatorPath> = defineContract({
  'replicator/x': tensor({
    role: 'state',
    shape: ['strategies'],
    description: 'the population state at this time point',
  }),
  'replicator/time': scalar({ description: 'the time point reached' }),
  'replicator/fitness': tensor({
    role: 'state',
    shape: ['strategies'],
    description: 'the fitness of each strategy at the state',
  }),
  'replicator/meanFitness': tensor({
    role: 'state',
    shape: ['populations'],
    description: 'the average fitness of each population at the state',
  }),
})

function integration(
  name: string,
  field: Field,
  fitnessOf: FitnessOf,
  x0: readonly number[],
  options: ReplicatorOptions,
): Algorithm<ReplicatorState, ReplicatorPath> {
  const times = options.timepoints ?? Array.from({ length: 1000 }, (_, i) => (10 * i) / 999)
  const maxStep = options.maxStep ?? 0.01
  if (!(maxStep > 0)) throw new OmaiError('value', `${name}: maxStep must be positive`)
  if (times.length === 0) throw new OmaiError('value', `${name}: at least one time point is needed`)
  for (let i = 1; i < times.length; i++)
    if ((times[i] ?? 0) < (times[i - 1] ?? 0)) throw new OmaiError('value', `${name}: time points must increase`)
  const integrator = options.integrator ?? 'rk4'
  const advance = replicatorIntegrators.make({ kind: integrator })
  return {
    name,
    config: { maxStep, points: times.length, integrator },
    init: () => ({ index: 0, time: times[0] ?? 0, x: x0, path: [[...x0]] }),
    step: (s, ctx) => {
      const target = times[s.index + 1] ?? s.time
      const span = target - s.time
      const steps = Math.max(1, Math.ceil(span / maxStep - 1e-9))
      let x = s.x
      for (let k = 0; k < steps; k++) x = advance(field, x, span / steps)
      if (ctx.wants('replicator/x')) ctx.log('replicator/x', Float64Array.from(x))
      if (ctx.wants('replicator/time')) ctx.log('replicator/time', target)
      if (ctx.wants('replicator/fitness') || ctx.wants('replicator/meanFitness')) {
        const { fitness, mean } = fitnessOf(x)
        if (ctx.wants('replicator/fitness')) ctx.log('replicator/fitness', Float64Array.from(fitness))
        if (ctx.wants('replicator/meanFitness')) ctx.log('replicator/meanFitness', Float64Array.from(mean))
      }
      return { index: s.index + 1, time: target, x, path: [...s.path, x] }
    },
    done: (s) => (s.index >= times.length - 1 ? 'completed' : undefined),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as ReplicatorState,
  }
}

/**
 * The replicator dynamics of a symmetric game as a steppable algorithm (see {@link replicatorDynamics}): each step
 * integrates the replicator, or replicator-mutator, equation to the next time point, logging the state, the time, the
 * strategies' fitness and the mean fitness.
 *
 * @param a - The payoff matrix of the symmetric game.
 * @param options - The initial state `x0` (default uniform), `mutation` matrix, time points, step size and integrator.
 * @returns The algorithm; it is done at the last time point.
 * @throws OmaiError with code `shape` for a non-square matrix or an initial state of the wrong length.
 * @concept replicator-dynamics
 * @example
 * ```ts
 * import { replicatorDynamicsAlgorithm } from 'omai-engine/games'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const alg = replicatorDynamicsAlgorithm([[3, 0], [5, 1]], { x0: [0.9, 0.1], timepoints: [0, 20] })
 * runAlgorithm(alg, { maxSteps: 1 }).state.x.map((v) => Math.round(v * 1e6) / 1e6) // => [0, 1]
 * ```
 */
export function replicatorDynamicsAlgorithm(
  a: TensorLike,
  options: ReplicatorOptions & { readonly x0?: readonly number[]; readonly mutation?: TensorLike } = {},
): Algorithm<ReplicatorState, ReplicatorPath> {
  const m = squareMatrix(a, 'replicatorDynamics')
  const x0 = options.x0 ?? m.map(() => 1 / m.length)
  if (x0.length !== m.length) throw new OmaiError('shape', 'replicatorDynamics: one share per strategy')
  const q = options.mutation === undefined ? undefined : squareMatrix(options.mutation, 'replicatorDynamics')
  return integration('replicator-dynamics', fieldOf(m, q), symmetricFitness(m), x0, options)
}

function trajectory(
  algorithm: Algorithm<ReplicatorState, ReplicatorPath>,
  points: number,
  options: GameRunOptions,
): number[][] {
  return runAlgorithm(algorithm, { ...options, maxSteps: points - 1 }).state.path.map((x) => [...x])
}

/**
 * The replicator dynamics of a symmetric game (nashpy's `replicator_dynamics`): the population state at each time
 * point, integrated from `x0` by fourth-order Runge-Kutta with steps of at most `maxStep` (nashpy uses odeint; the
 * two agree to about 1e-8). With a mutation matrix, the replicator-mutator equation.
 *
 * @param a - The payoff matrix of the symmetric game.
 * @param options - The initial state (default uniform), time points, step size, mutation matrix and tracing.
 * @returns One state per time point.
 * @throws OmaiError with code `shape` for a non-square matrix or an initial state of the wrong length.
 * @concept replicator-dynamics
 * @example
 * ```ts
 * import { replicatorDynamics } from 'omai-engine/games'
 * const xs = replicatorDynamics([[3, 0], [5, 1]], { x0: [0.9, 0.1], timepoints: [0, 20] })
 * xs[1].map((v) => Math.round(v * 1e6) / 1e6) // => [0, 1]
 * ```
 */
export function replicatorDynamics(
  a: TensorLike,
  options: ReplicatorOptions & { readonly x0?: readonly number[]; readonly mutation?: TensorLike } = {},
): number[][] {
  return trajectory(replicatorDynamicsAlgorithm(a, options), (options.timepoints ?? []).length || 1000, options)
}

/**
 * The two-population replicator dynamics of a bimatrix game as a steppable algorithm (see
 * {@link asymmetricReplicatorDynamics}): the state logged is the row population followed by the column population.
 *
 * @param game - A two-player game.
 * @param options - Initial states `x0` and `y0` (default uniform), time points, step size and integrator.
 * @returns The algorithm; it is done at the last time point.
 * @throws OmaiError with code `value` for a game with other than two players, `shape` for a wrong initial state.
 * @concept replicator-dynamics
 * @example
 * ```ts
 * import { asymmetricReplicatorDynamicsAlgorithm, prisonersDilemma } from 'omai-engine/games'
 * asymmetricReplicatorDynamicsAlgorithm(prisonersDilemma(), { timepoints: [0, 1] }).name // => 'asymmetric-replicator-dynamics'
 * ```
 */
export function asymmetricReplicatorDynamicsAlgorithm(
  game: NormalFormGame,
  options: ReplicatorOptions & { readonly x0?: readonly number[]; readonly y0?: readonly number[] } = {},
): Algorithm<ReplicatorState, ReplicatorPath> {
  if (game.shape.length !== 2) throw new OmaiError('value', `${game.name}: a two-player game is needed`)
  const [m, n] = game.shape as [number, number]
  const x0 = options.x0 ?? new Array<number>(m).fill(1 / m)
  const y0 = options.y0 ?? new Array<number>(n).fill(1 / n)
  if (x0.length !== m || y0.length !== n) throw new OmaiError('shape', `${game.name}: one share per action`)
  const fitnessOf: FitnessOf = (z) => {
    const x = z.slice(0, m)
    const y = z.slice(m)
    const f1 = actionValues(game, 0, [x, y])
    const f2 = actionValues(game, 1, [x, y])
    let p1 = 0
    for (const [i, v] of x.entries()) p1 += v * (f1[i] ?? 0)
    let p2 = 0
    for (const [j, v] of y.entries()) p2 += v * (f2[j] ?? 0)
    return { fitness: [...f1, ...f2], mean: [p1, p2] }
  }
  const field: Field = (z) => {
    const x = z.slice(0, m)
    const y = z.slice(m)
    const { fitness, mean } = fitnessOf(z)
    const [p1 = 0, p2 = 0] = mean
    return [...x.map((v, i) => v * ((fitness[i] ?? 0) - p1)), ...y.map((v, j) => v * ((fitness[m + j] ?? 0) - p2))]
  }
  return integration('asymmetric-replicator-dynamics', field, fitnessOf, [...x0, ...y0], options)
}

/**
 * The two-population (asymmetric) replicator dynamics of a bimatrix game (nashpy's `asymmetric_replicator_dynamics`):
 * `x_i' = x_i ((A y)_i - x.A y)` and `y_j' = y_j ((x B)_j - x B y)`, by fourth-order Runge-Kutta.
 *
 * @param game - A two-player game.
 * @param options - Initial states (default uniform), time points, step size and tracing.
 * @returns The row and column populations at each time point.
 * @throws OmaiError with code `value` for a game with other than two players.
 * @concept replicator-dynamics
 * @example
 * ```ts
 * import { asymmetricReplicatorDynamics, prisonersDilemma } from 'omai-engine/games'
 * const r = asymmetricReplicatorDynamics(prisonersDilemma(), { x0: [0.9, 0.1], y0: [0.9, 0.1], timepoints: [0, 20] })
 * r.column[1].map((v) => Math.round(v * 1e6) / 1e6) // => [0, 1]
 * ```
 */
export function asymmetricReplicatorDynamics(
  game: NormalFormGame,
  options: ReplicatorOptions & { readonly x0?: readonly number[]; readonly y0?: readonly number[] } = {},
): { row: number[][]; column: number[][] } {
  const [m] = game.shape as [number, number]
  const zs = trajectory(
    asymmetricReplicatorDynamicsAlgorithm(game, options),
    (options.timepoints ?? []).length || 1000,
    options,
  )
  return { row: zs.map((z) => z.slice(0, m)), column: zs.map((z) => z.slice(m)) }
}

/** Options for {@link projectedReplicatorDynamics}. */
export interface ProjectedReplicatorOptions extends GameRunOptions {
  /** Iterations (default 100000, as OpenSpiel). */
  readonly iterations?: number
  /** The Euler step (default 1e-3). */
  readonly dt?: number
  /** The floor every probability is projected above (default 1e-6). */
  readonly gamma?: number
  /** Starting strategies (default uniform). */
  readonly initialStrategies?: readonly (readonly number[])[]
}

/** The state of projected replicator dynamics. */
export interface ProjectedReplicatorState {
  readonly t: number
  readonly strategies: readonly (readonly number[])[]
  readonly sums: readonly (readonly number[])[]
}

/**
 * Projected replicator dynamics (Lanctot et al. 2017, OpenSpiel's `projected_replicator_dynamics`) for any number of
 * players: an Euler step of every player's replicator equation against the others' current strategies, projected
 * back onto the simplex with every probability at least `gamma`; the average of all iterates (the initial one
 * included) is the answer.
 *
 * @param game - The game.
 * @param options - Step size, floor and initial strategies.
 * @returns The algorithm; each step is one Euler step.
 * @concept replicator-dynamics
 * @example
 * ```ts
 * import { projectedReplicatorDynamicsAlgorithm, rockPaperScissors } from 'omai-engine/games'
 * import { runAlgorithm } from 'omai-engine/trace'
 * runAlgorithm(projectedReplicatorDynamicsAlgorithm(rockPaperScissors()), { maxSteps: 3 }).state.t // => 3
 * ```
 */
export function projectedReplicatorDynamicsAlgorithm(
  game: NormalFormGame,
  options: Omit<ProjectedReplicatorOptions, 'iterations'> = {},
): Algorithm<ProjectedReplicatorState> {
  const dt = options.dt ?? 1e-3
  const gamma = options.gamma ?? 1e-6
  if (game.shape.some((k) => k * gamma > 1))
    throw new OmaiError('value', 'projectedReplicatorDynamics: gamma too large')
  const start = game.shape.map((k, p) => [...(options.initialStrategies?.[p] ?? new Array<number>(k).fill(1 / k))])
  return {
    name: 'projected-replicator-dynamics',
    config: { dt, gamma },
    schema: channels({
      metrics: { nashConv: { description: 'NashConv of the average strategies', direction: 'lower' } },
      tensors: Object.fromEntries(
        game.shape.map((k, p) => [
          `strategy${p}`,
          { role: 'state' as const, description: `the current strategy of player ${p}`, shape: [k] },
        ]),
      ),
    }),
    init: () => ({ t: 0, strategies: start, sums: start.map((x) => [...x]) }),
    step: (s) => {
      const strategies = s.strategies.map((x, p) => {
        const v = actionValues(game, p, s.strategies)
        let mean = 0
        for (const [a, q] of x.entries()) mean += q * (v[a] ?? 0)
        const k = x.length
        // Project onto {y >= gamma, sum y = 1}: gamma plus the projection of (x - gamma) onto the simplex of mass 1 - k gamma.
        const moved = x.map((q, a) => q + dt * q * ((v[a] ?? 0) - mean) - gamma)
        return toFlat(projectSimplex(moved, 1 - k * gamma)).map((y) => Math.max(y + gamma, gamma))
      })
      return { t: s.t + 1, strategies, sums: s.sums.map((x, p) => x.map((v, a) => v + (strategies[p]?.[a] ?? 0))) }
    },
    observe: (s) => ({
      metrics: {
        nashConv: nashConv(
          game,
          s.sums.map((x) => x.map((v) => v / (s.t + 1))),
        ),
      },
      tensors: Object.fromEntries(s.strategies.map((x, p) => [`strategy${p}`, { role: 'state' as const, value: x }])),
    }),
  }
}

/**
 * Run projected replicator dynamics (see {@link projectedReplicatorDynamicsAlgorithm}).
 *
 * @param game - The game.
 * @param options - Iterations, step size, floor, initial strategies and tracing.
 * @returns The average strategies over all iterates, and the last ones.
 * @concept replicator-dynamics
 * @example
 * ```ts
 * import { projectedReplicatorDynamics, rockPaperScissors } from 'omai-engine/games'
 * const r = projectedReplicatorDynamics(rockPaperScissors(), { iterations: 2000, dt: 0.01 })
 * r.average[0].map((p) => Math.round(p * 1e3) / 1e3) // => [0.333, 0.333, 0.333]
 * ```
 */
export function projectedReplicatorDynamics(
  game: NormalFormGame,
  options: ProjectedReplicatorOptions = {},
): { average: number[][]; current: readonly (readonly number[])[] } {
  const { iterations = 100_000, ...rest } = options
  const { state } = runAlgorithm(projectedReplicatorDynamicsAlgorithm(game, rest), { ...rest, maxSteps: iterations })
  return { average: state.sums.map((x) => x.map((v) => v / (state.t + 1))), current: state.strategies }
}
