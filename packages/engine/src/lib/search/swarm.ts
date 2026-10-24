import { argmaxOf, meanOf, WhiteboxError } from '../base/index.ts'
import { foldIn } from '../random/index.ts'
import { asTensor, type TensorLike, toFlat } from '../tensor/index.ts'
import { type Algorithm, channels, runAlgorithm } from '../trace/index.ts'
import { draws } from './_draw.ts'
import type { SearchRunOptions } from './problem.ts'

/** Options of {@link antColonyOptimization}. */
export interface AntColonyOptions extends SearchRunOptions {
  /** Ants per iteration (default the number of cities). */
  readonly ants?: number
  /** The weight of pheromone (default 1). */
  readonly alpha?: number
  /** The weight of the visibility 1 / distance (default 3). */
  readonly beta?: number
  /** The evaporation rate (default 0.5; 0.02 is usual for MAX-MIN). */
  readonly rho?: number
  /** The deposit constant Q (default 1). */
  readonly q?: number
  /**
   * The update rule: Ant System (every ant deposits Q / length), elitist (plus `elite` extra deposits on the best
   * tour so far), or MAX-MIN (only the best tour so far deposits, with pheromone clamped to [tau_min, tau_max]).
   */
  readonly variant?: 'ant-system' | 'elitist' | 'max-min'
  /** Extra deposits on the best tour in the elitist variant (default the number of cities). */
  readonly elite?: number
  /** The number of iterations (default 100). */
  readonly iterations?: number
}

/** The state of an ant colony: the pheromone matrix and the best tour so far. */
export interface AntColonyState {
  readonly pheromone: Float64Array
  readonly best: readonly number[]
  readonly bestLength: number
  /** The tours of the last iteration's ants. */
  readonly tours: readonly (readonly number[])[]
  readonly iteration: number
}

/** The result of an ant colony: the best tour, its length and the best length per iteration. */
export interface AntColonyResult {
  readonly tour: readonly number[]
  readonly length: number
  readonly iterations: number
  readonly history: readonly number[]
}

function distanceMatrix(distances: TensorLike): { n: number; d: number[] } {
  const t = asTensor(distances)
  const [n = 0, m = -1] = t.shape
  if (t.shape.length !== 2 || n !== m || n < 2) {
    throw new WhiteboxError(
      'shape',
      `antColonyOptimization: distances must be a square matrix of side >= 2, got [${t.shape}]`,
    )
  }
  const d = toFlat(t)
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (i !== j && !((d[i * n + j] ?? 0) > 0)) {
        throw new WhiteboxError('value', 'antColonyOptimization: distances between cities must be positive')
      }
    }
  }
  return { n, d }
}

function tourLength(d: readonly number[], n: number, tour: readonly number[]): number {
  let s = 0
  for (let i = 0; i < tour.length; i++) s += d[(tour[i] ?? 0) * n + (tour[(i + 1) % tour.length] ?? 0)] ?? 0
  return s
}

/**
 * Ant colony optimisation for the travelling salesman problem (Dorigo, Maniezzo and Colorni 1996; MAX-MIN, Stutzle
 * and Hoos 2000) as a traced algorithm: each step, every ant builds a tour city by city with probability proportional
 * to pheromone^alpha * (1 / distance)^beta, then pheromone evaporates and is deposited along good tours. Distances may
 * be asymmetric.
 *
 * @param distances - The n x n matrix of positive distances.
 * @param options - Ants, alpha, beta, rho, Q, the update rule and the elitist weight.
 * @returns The algorithm.
 * @throws WhiteboxError with code `shape` for a non-square matrix, or `value` for non-positive distances.
 * @concept ant-colony-optimisation
 * @example
 * ```ts
 * import { antColonyAlgorithm } from 'whitebox-engine/search'
 * antColonyAlgorithm([[0, 1], [1, 0]]).name // => 'ant-colony'
 * ```
 */
export function antColonyAlgorithm(distances: TensorLike, options: AntColonyOptions = {}): Algorithm<AntColonyState> {
  const { n, d } = distanceMatrix(distances)
  const ants = options.ants ?? n
  const alpha = options.alpha ?? 1
  const beta = options.beta ?? 3
  const variant = options.variant ?? 'ant-system'
  const rho = options.rho ?? (variant === 'max-min' ? 0.02 : 0.5)
  const q = options.q ?? 1
  const elite = options.elite ?? n
  if (!(rho > 0 && rho <= 1)) throw new WhiteboxError('value', `antColonyOptimization: rho ${rho} must be in (0, 1]`)
  const eta = d.map((x, k) => (k % (n + 1) === 0 ? 0 : (1 / x) ** beta))
  // A nearest-neighbour tour sets the pheromone scale (tau_0 = 1 / (rho * L_nn) for MAX-MIN, ants / L_nn otherwise).
  const nn = (() => {
    const tour = [0]
    const seen = new Uint8Array(n)
    seen[0] = 1
    for (let s = 1; s < n; s++) {
      const at = tour[tour.length - 1] ?? 0
      let next = -1
      for (let j = 0; j < n; j++)
        if (seen[j] === 0 && (next < 0 || (d[at * n + j] ?? 0) < (d[at * n + next] ?? 0))) next = j
      tour.push(next)
      seen[next] = 1
    }
    return tourLength(d, n, tour)
  })()
  const tauMax = (best: number): number => 1 / (rho * best)
  const tauMin = (best: number): number => tauMax(best) / (2 * n)
  const deposit = (tau: Float64Array, tour: readonly number[], amount: number): void => {
    for (let i = 0; i < tour.length; i++) {
      const a = tour[i] ?? 0
      const b = tour[(i + 1) % tour.length] ?? 0
      tau[a * n + b] = (tau[a * n + b] ?? 0) + amount
    }
  }
  return {
    name: 'ant-colony',
    config: { n, ants, alpha, beta, rho, q, variant, elite },
    schema: channels({
      metrics: {
        best: { description: 'best tour length so far', direction: 'lower' },
        mean: { description: 'mean tour length of the iteration', direction: 'lower' },
      },
      tensors: { pheromone: { role: 'state', description: 'pheromone on each edge', shape: ['n', 'n'] } },
    }),
    init: () => ({
      pheromone: new Float64Array(n * n).fill(variant === 'max-min' ? tauMax(nn) : ants / nn),
      best: [],
      bestLength: Number.POSITIVE_INFINITY,
      tours: [],
      iteration: 0,
    }),
    step: (s, ctx) => {
      const tau = s.pheromone
      const tours: number[][] = []
      for (let k = 0; k < ants; k++) {
        const dr = draws(foldIn(ctx.key, k))
        const tour = [dr.int(n)]
        const seen = new Uint8Array(n)
        seen[tour[0] ?? 0] = 1
        for (let step = 1; step < n; step++) {
          const at = tour[tour.length - 1] ?? 0
          let total = 0
          const w = new Float64Array(n)
          for (let j = 0; j < n; j++) {
            if (seen[j] === 1) continue
            w[j] = (tau[at * n + j] ?? 0) ** alpha * (eta[at * n + j] ?? 0)
            total += w[j] ?? 0
          }
          let u = dr.uniform() * total
          let next = -1
          for (let j = 0; j < n; j++) {
            if (seen[j] === 1) continue
            if (next < 0) next = j
            u -= w[j] ?? 0
            if (u <= 0 && total > 0) {
              next = j
              break
            }
          }
          tour.push(next)
          seen[next] = 1
        }
        tours.push(tour)
      }
      const lengths = tours.map((t) => tourLength(d, n, t))
      let best = s.best
      let bestLength = s.bestLength
      lengths.forEach((len, i) => {
        if (len < bestLength) {
          bestLength = len
          best = tours[i] ?? best
        }
      })
      const next = Float64Array.from(tau, (x) => (1 - rho) * x)
      if (variant === 'max-min') {
        deposit(next, best, 1 / bestLength)
        const hi = tauMax(bestLength)
        const lo = tauMin(bestLength)
        for (let i = 0; i < next.length; i++) next[i] = Math.min(hi, Math.max(lo, next[i] ?? 0))
      } else {
        tours.forEach((t, i) => {
          deposit(next, t, q / (lengths[i] ?? 1))
        })
        if (variant === 'elitist') deposit(next, best, (elite * q) / bestLength)
      }
      return { pheromone: next, best, bestLength, tours, iteration: s.iteration + 1 }
    },
    observe: (s) => ({
      metrics: {
        best: s.bestLength,
        mean: s.tours.length === 0 ? Number.NaN : s.tours.reduce((a, t) => a + tourLength(d, n, t), 0) / s.tours.length,
      },
      tensors: { pheromone: { role: 'state', value: s.pheromone, shape: [n, n] } },
    }),
  }
}

/**
 * Ant colony optimisation for the travelling salesman problem.
 *
 * @param distances - The n x n matrix of positive distances.
 * @param options - Ants, alpha, beta, rho, Q, the update rule, iterations (default 100), a seed or key, and tracing.
 * @returns The best tour (a permutation of the cities), its length and the best length per iteration.
 * @throws WhiteboxError with code `shape` for a non-square matrix, or `value` for non-positive distances.
 * @concept ant-colony-optimisation
 * @example
 * ```ts
 * import { antColonyOptimization } from 'whitebox-engine/search'
 * // Four cities on a unit square: the best tour goes round the edge.
 * const d = [[0, 1, Math.SQRT2, 1], [1, 0, 1, Math.SQRT2], [Math.SQRT2, 1, 0, 1], [1, Math.SQRT2, 1, 0]]
 * antColonyOptimization(d, { iterations: 20, seed: 0 }).length // => 4
 * ```
 */
export function antColonyOptimization(distances: TensorLike, options: AntColonyOptions = {}): AntColonyResult {
  const { iterations = 100, ants: _a, alpha: _al, beta: _b, rho: _r, q: _q, variant: _v, elite: _e, ...run } = options
  const history: number[] = []
  const algorithm = antColonyAlgorithm(distances, options)
  const tracked: Algorithm<AntColonyState> = {
    ...algorithm,
    step: (s, ctx) => {
      const next = algorithm.step(s, ctx)
      history.push(next.bestLength)
      return next
    },
  }
  const { state } = runAlgorithm(tracked, { ...run, maxSteps: iterations })
  return { tour: state.best, length: state.bestLength, iterations: state.iteration, history }
}

/** Options of {@link discreteParticleSwarm}. */
export interface ParticleSwarmOptions extends SearchRunOptions {
  /** The number of particles (default 30). */
  readonly particles?: number
  /** The inertia weight (default 0.9). */
  readonly inertia?: number
  /** The cognitive (personal best) coefficient (default 2). */
  readonly cognitive?: number
  /** The social (global best) coefficient (default 2). */
  readonly social?: number
  /** The velocity clamp (default 4, so bit probabilities stay within [0.018, 0.982]). */
  readonly vMax?: number
  /** The number of iterations (default 100). */
  readonly iterations?: number
  /** Stop once a fitness reaches this value. */
  readonly target?: number
}

/** The state of a binary particle swarm. */
export interface ParticleSwarmState {
  readonly positions: readonly (readonly number[])[]
  readonly velocities: readonly (readonly number[])[]
  readonly personal: readonly (readonly number[])[]
  readonly personalFitness: readonly number[]
  readonly best: readonly number[]
  readonly bestFitness: number
  readonly iteration: number
  readonly evaluations: number
}

/** The result of a particle swarm: the best bit string, its fitness and the best fitness per iteration. */
export interface ParticleSwarmResult {
  readonly best: readonly number[]
  readonly fitness: number
  readonly iterations: number
  readonly evaluations: number
  readonly history: readonly number[]
  readonly status: 'target' | 'budget'
}

/**
 * Binary particle swarm optimisation (Kennedy and Eberhart 1997) as a traced algorithm: each particle keeps a real
 * velocity per bit, pulled towards its own best and the swarm's best; a bit is set with probability sigmoid(velocity).
 * Fitness is maximised. (Continuous particle swarms are `particleSwarm` in `whitebox-engine/optim`.)
 *
 * @param fitness - The fitness of a bit string.
 * @param n - The number of bits.
 * @param options - Particles, inertia, coefficients, the velocity clamp and a target.
 * @returns The algorithm.
 * @throws WhiteboxError with code `value` when n or the number of particles is not a positive integer.
 * @concept particle-swarm-optimisation
 * @example
 * ```ts
 * import { discreteParticleSwarmAlgorithm } from 'whitebox-engine/search'
 * discreteParticleSwarmAlgorithm((b) => b.reduce((s, x) => s + x, 0), 8).name // => 'discrete-particle-swarm'
 * ```
 */
export function discreteParticleSwarmAlgorithm(
  fitness: (bits: readonly number[]) => number,
  n: number,
  options: ParticleSwarmOptions = {},
): Algorithm<ParticleSwarmState> {
  const particles = options.particles ?? 30
  const w = options.inertia ?? 0.9
  const c1 = options.cognitive ?? 2
  const c2 = options.social ?? 2
  const vMax = options.vMax ?? 4
  if (!(Number.isInteger(n) && n >= 1 && Number.isInteger(particles) && particles >= 1)) {
    throw new WhiteboxError('value', 'discreteParticleSwarm: n and particles must be positive integers')
  }
  const sigmoid = (v: number): number => 1 / (1 + Math.exp(-v))
  return {
    name: 'discrete-particle-swarm',
    config: { n, particles, inertia: w, cognitive: c1, social: c2, vMax },
    schema: channels({
      metrics: {
        best: { description: 'best fitness so far', direction: 'higher' },
        mean: { description: 'mean personal-best fitness', direction: 'higher' },
      },
    }),
    init: (key) => {
      const d = draws(key)
      const positions = Array.from({ length: particles }, () =>
        Array.from({ length: n }, () => (d.uniform() < 0.5 ? 0 : 1)),
      )
      const velocities = Array.from({ length: particles }, () =>
        Array.from({ length: n }, () => (2 * d.uniform() - 1) * vMax),
      )
      const f = positions.map(fitness)
      const b = argmaxOf(f)
      return {
        positions,
        velocities,
        personal: positions,
        personalFitness: f,
        best: positions[b] ?? [],
        bestFitness: f[b] ?? Number.NEGATIVE_INFINITY,
        iteration: 0,
        evaluations: particles,
      }
    },
    step: (s, ctx) => {
      const d = draws(ctx.key)
      const velocities = s.velocities.map((v, p) =>
        v.map((vi, i) => {
          const x = s.positions[p]?.[i] ?? 0
          const pull = c1 * d.uniform() * ((s.personal[p]?.[i] ?? 0) - x) + c2 * d.uniform() * ((s.best[i] ?? 0) - x)
          return Math.min(vMax, Math.max(-vMax, w * vi + pull))
        }),
      )
      const positions = velocities.map((v) => v.map((vi) => (d.uniform() < sigmoid(vi) ? 1 : 0)))
      const f = positions.map(fitness)
      const personal = s.personal.map((p, i) => ((f[i] ?? 0) > (s.personalFitness[i] ?? 0) ? (positions[i] ?? p) : p))
      const personalFitness = s.personalFitness.map((pf, i) => Math.max(pf, f[i] ?? pf))
      const b = argmaxOf(personalFitness)
      const improved = (personalFitness[b] ?? Number.NEGATIVE_INFINITY) > s.bestFitness
      return {
        positions,
        velocities,
        personal,
        personalFitness,
        best: improved ? (personal[b] ?? s.best) : s.best,
        bestFitness: improved ? (personalFitness[b] ?? s.bestFitness) : s.bestFitness,
        iteration: s.iteration + 1,
        evaluations: s.evaluations + particles,
      }
    },
    done: (s) => (options.target !== undefined && s.bestFitness >= options.target ? 'converged' : undefined),
    observe: (s) => ({
      metrics: {
        best: s.bestFitness,
        mean: meanOf(s.personalFitness),
      },
    }),
  }
}

/**
 * Binary particle swarm optimisation.
 *
 * @param fitness - The fitness of a bit string (maximised).
 * @param n - The number of bits.
 * @param options - Particles, inertia, coefficients, the velocity clamp, iterations (default 100), a target, a seed
 * or key, and tracing.
 * @returns The best bit string, its fitness and the best fitness per iteration.
 * @throws WhiteboxError with code `value` when n or the number of particles is not a positive integer.
 * @concept particle-swarm-optimisation
 * @example
 * ```ts
 * import { discreteParticleSwarm } from 'whitebox-engine/search'
 * // A 0/1 knapsack: values 6, 5, 4, 3 and weights 4, 3, 2, 1 with capacity 6; the best load is items 1, 2, 3 (12).
 * const v = [6, 5, 4, 3]
 * const wt = [4, 3, 2, 1]
 * const fit = (b: readonly number[]) => (b.reduce((s, x, i) => s + x * (wt[i] ?? 0), 0) <= 6 ? b.reduce((s, x, i) => s + x * (v[i] ?? 0), 0) : 0)
 * discreteParticleSwarm(fit, 4, { iterations: 50, seed: 0 }).fitness // => 12
 * ```
 */
export function discreteParticleSwarm(
  fitness: (bits: readonly number[]) => number,
  n: number,
  options: ParticleSwarmOptions = {},
): ParticleSwarmResult {
  const { particles: _p, inertia: _i, cognitive: _c, social: _s, vMax: _v, iterations = 100, target, ...run } = options
  const history: number[] = []
  const algorithm = discreteParticleSwarmAlgorithm(fitness, n, options)
  const tracked: Algorithm<ParticleSwarmState> = {
    ...algorithm,
    init: (k) => {
      const s = algorithm.init(k)
      history.push(s.bestFitness)
      return s
    },
    step: (s, ctx) => {
      const next = algorithm.step(s, ctx)
      history.push(next.bestFitness)
      return next
    },
  }
  const { state } = runAlgorithm(tracked, { ...run, maxSteps: iterations })
  return {
    best: state.best,
    fitness: state.bestFitness,
    iterations: state.iteration,
    evaluations: state.evaluations,
    history,
    status: target !== undefined && state.bestFitness >= target ? 'target' : 'budget',
  }
}
