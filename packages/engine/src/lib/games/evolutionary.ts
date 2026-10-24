import { WhiteboxError } from '../base/index.ts'
import { projectSimplex } from '../optim/index.ts'
import { asTensor, type TensorLike, toFlat } from '../tensor/index.ts'
import { type Algorithm, channels, runAlgorithm } from '../trace/index.ts'
import { actionValues, type GameRunOptions, type NormalFormGame, nashConv } from './normal-form/index.ts'

/** A vector field on population states. */
type Field = (x: readonly number[]) => number[]

/** One classical Runge-Kutta step of size h. */
function rk4(f: Field, x: readonly number[], h: number): number[] {
  const k1 = f(x)
  const k2 = f(x.map((v, i) => v + 0.5 * h * (k1[i] ?? 0)))
  const k3 = f(x.map((v, i) => v + 0.5 * h * (k2[i] ?? 0)))
  const k4 = f(x.map((v, i) => v + h * (k3[i] ?? 0)))
  return x.map((v, i) => v + (h / 6) * ((k1[i] ?? 0) + 2 * (k2[i] ?? 0) + 2 * (k3[i] ?? 0) + (k4[i] ?? 0)))
}

function squareMatrix(a: TensorLike, what: string): number[][] {
  const t = asTensor(a)
  const [m, n] = t.shape as [number, number]
  if (t.shape.length !== 2 || m !== n || m === 0)
    throw new WhiteboxError('shape', `${what}: a non-empty square matrix is needed`)
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
 * import { replicatorField } from 'whitebox-engine/games'
 * replicatorField([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], [0.5, 0.5, 0]) // => [-0.25, 0.25, 0]
 * ```
 */
export function replicatorField(a: TensorLike, x: readonly number[], mutation?: TensorLike): number[] {
  const m = squareMatrix(a, 'replicatorField')
  const q = mutation === undefined ? undefined : squareMatrix(mutation, 'replicatorField')
  return fieldOf(m, q)(x)
}

function fieldOf(m: readonly (readonly number[])[], q: readonly (readonly number[])[] | undefined): Field {
  return (x) => {
    const f = m.map((r) => r.reduce((s, v, j) => s + v * (x[j] ?? 0), 0))
    let phi = 0
    for (const [i, v] of x.entries()) phi += v * (f[i] ?? 0)
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
  /** The largest Runge-Kutta step between and within time points (default 0.01). */
  readonly maxStep?: number
}

function integration(
  name: string,
  field: Field,
  x0: readonly number[],
  options: ReplicatorOptions,
): Algorithm<ReplicatorState> {
  const times = options.timepoints ?? Array.from({ length: 1000 }, (_, i) => (10 * i) / 999)
  const maxStep = options.maxStep ?? 0.01
  if (!(maxStep > 0)) throw new WhiteboxError('value', `${name}: maxStep must be positive`)
  if (times.length === 0) throw new WhiteboxError('value', `${name}: at least one time point is needed`)
  for (let i = 1; i < times.length; i++)
    if ((times[i] ?? 0) < (times[i - 1] ?? 0)) throw new WhiteboxError('value', `${name}: time points must increase`)
  return {
    name,
    config: { maxStep, points: times.length },
    schema: channels({
      metrics: { time: { description: 'the current time' } },
      tensors: { x: { role: 'state', description: 'the population state', shape: [x0.length] } },
    }),
    init: () => ({ index: 0, time: times[0] ?? 0, x: x0, path: [[...x0]] }),
    step: (s) => {
      const target = times[s.index + 1] ?? s.time
      const span = target - s.time
      const steps = Math.max(1, Math.ceil(span / maxStep - 1e-9))
      let x = s.x
      for (let k = 0; k < steps; k++) x = rk4(field, x, span / steps)
      return { index: s.index + 1, time: target, x, path: [...s.path, x] }
    },
    done: (s) => (s.index >= times.length - 1 ? 'completed' : undefined),
    observe: (s) => ({ metrics: { time: s.time }, tensors: { x: { role: 'state', value: s.x } } }),
  }
}

function trajectory(algorithm: Algorithm<ReplicatorState>, points: number, options: GameRunOptions): number[][] {
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
 * @throws WhiteboxError with code `shape` for a non-square matrix or an initial state of the wrong length.
 * @concept replicator-dynamics
 * @example
 * ```ts
 * import { replicatorDynamics } from 'whitebox-engine/games'
 * const xs = replicatorDynamics([[3, 0], [5, 1]], { x0: [0.9, 0.1], timepoints: [0, 20] })
 * xs[1].map((v) => Math.round(v * 1e6) / 1e6) // => [0, 1]
 * ```
 */
export function replicatorDynamics(
  a: TensorLike,
  options: ReplicatorOptions & { readonly x0?: readonly number[]; readonly mutation?: TensorLike } = {},
): number[][] {
  const m = squareMatrix(a, 'replicatorDynamics')
  const x0 = options.x0 ?? m.map(() => 1 / m.length)
  if (x0.length !== m.length) throw new WhiteboxError('shape', 'replicatorDynamics: one share per strategy')
  const q = options.mutation === undefined ? undefined : squareMatrix(options.mutation, 'replicatorDynamics')
  const algorithm = integration('replicator-dynamics', fieldOf(m, q), x0, options)
  return trajectory(algorithm, (options.timepoints ?? []).length || 1000, options)
}

/**
 * The two-population (asymmetric) replicator dynamics of a bimatrix game (nashpy's `asymmetric_replicator_dynamics`):
 * `x_i' = x_i ((A y)_i - x.A y)` and `y_j' = y_j ((x B)_j - x B y)`, by fourth-order Runge-Kutta.
 *
 * @param game - A two-player game.
 * @param options - Initial states (default uniform), time points, step size and tracing.
 * @returns The row and column populations at each time point.
 * @throws WhiteboxError with code `value` for a game with other than two players.
 * @concept replicator-dynamics
 * @example
 * ```ts
 * import { asymmetricReplicatorDynamics, prisonersDilemma } from 'whitebox-engine/games'
 * const r = asymmetricReplicatorDynamics(prisonersDilemma(), { x0: [0.9, 0.1], y0: [0.9, 0.1], timepoints: [0, 20] })
 * r.column[1].map((v) => Math.round(v * 1e6) / 1e6) // => [0, 1]
 * ```
 */
export function asymmetricReplicatorDynamics(
  game: NormalFormGame,
  options: ReplicatorOptions & { readonly x0?: readonly number[]; readonly y0?: readonly number[] } = {},
): { row: number[][]; column: number[][] } {
  if (game.shape.length !== 2) throw new WhiteboxError('value', `${game.name}: a two-player game is needed`)
  const [m, n] = game.shape as [number, number]
  const x0 = options.x0 ?? new Array<number>(m).fill(1 / m)
  const y0 = options.y0 ?? new Array<number>(n).fill(1 / n)
  if (x0.length !== m || y0.length !== n) throw new WhiteboxError('shape', `${game.name}: one share per action`)
  const field: Field = (z) => {
    const x = z.slice(0, m)
    const y = z.slice(m)
    const f1 = actionValues(game, 0, [x, y])
    const f2 = actionValues(game, 1, [x, y])
    let p1 = 0
    for (const [i, v] of x.entries()) p1 += v * (f1[i] ?? 0)
    let p2 = 0
    for (const [j, v] of y.entries()) p2 += v * (f2[j] ?? 0)
    return [...x.map((v, i) => v * ((f1[i] ?? 0) - p1)), ...y.map((v, j) => v * ((f2[j] ?? 0) - p2))]
  }
  const algorithm = integration('asymmetric-replicator-dynamics', field, [...x0, ...y0], options)
  const zs = trajectory(algorithm, (options.timepoints ?? []).length || 1000, options)
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
 * import { projectedReplicatorDynamicsAlgorithm, rockPaperScissors } from 'whitebox-engine/games'
 * import { runAlgorithm } from 'whitebox-engine/trace'
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
    throw new WhiteboxError('value', 'projectedReplicatorDynamics: gamma too large')
  const start = game.shape.map((k, p) => [...(options.initialStrategies?.[p] ?? new Array<number>(k).fill(1 / k))])
  return {
    name: 'projected-replicator-dynamics',
    config: { dt, gamma },
    schema: channels({
      metrics: { nashConv: { description: 'NashConv of the average strategies', direction: 'lower' } },
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
 * import { projectedReplicatorDynamics, rockPaperScissors } from 'whitebox-engine/games'
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
