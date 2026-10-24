import { type Json, meanOf, WhiteboxError } from '../../base/index.ts'
import { foldIn, type Key, permutation } from '../../random/index.ts'
import { type Algorithm, channels, runAlgorithm, type TraceSchema } from '../../trace/index.ts'
import { draws } from '../_draw.ts'
import type { SearchRunOptions } from '../problem.ts'
import {
  arithmeticCrossover,
  bitFlipMutation,
  blendCrossover,
  cycleCrossover,
  gaussianMutation,
  inversionMutation,
  onePointCrossover,
  orderCrossover,
  pmxCrossover,
  polynomialMutation,
  rankSelection,
  rouletteSelection,
  scrambleMutation,
  simulatedBinaryCrossover,
  stochasticUniversalSampling,
  swapMutation,
  tournamentSelection,
  twoPointCrossover,
  uniformCrossover,
} from './operators.ts'

/**
 * A genome representation for a genetic algorithm: how to draw a random genome, recombine two and mutate one.
 *
 * @concept evolutionary-algorithms
 */
export interface Representation<G> {
  random(key: Key): G
  crossover(key: Key, a: G, b: G): readonly [G, G]
  mutate(key: Key, genome: G): G
}

/**
 * Bit strings of length n: uniform random bits, one-point, two-point or uniform crossover, and bit-flip mutation.
 *
 * @param n - The length.
 * @param options - The crossover (default one-point) and the flip rate (default 1 / n).
 * @returns The representation.
 * @concept evolutionary-algorithms
 * @example
 * ```ts
 * import { key } from 'whitebox-engine/random'
 * import { bitStrings } from 'whitebox-engine/search'
 * bitStrings(5).random(key(0)).length // => 5
 * ```
 */
export function bitStrings(
  n: number,
  options: { readonly crossover?: 'one-point' | 'two-point' | 'uniform'; readonly rate?: number } = {},
): Representation<number[]> {
  const cross = options.crossover ?? 'one-point'
  const rate = options.rate ?? 1 / Math.max(1, n)
  return {
    random: (k) => {
      const d = draws(k)
      return Array.from({ length: n }, () => (d.uniform() < 0.5 ? 0 : 1))
    },
    crossover: (k, a, b) =>
      cross === 'one-point'
        ? onePointCrossover(k, a, b)
        : cross === 'two-point'
          ? twoPointCrossover(k, a, b)
          : uniformCrossover(k, a, b),
    mutate: (k, g) => bitFlipMutation(k, g, rate),
  }
}

/**
 * Permutations of 0 .. n - 1: uniform random permutations, order, PMX or cycle crossover, and swap, inversion or
 * scramble mutation.
 *
 * @param n - The size.
 * @param options - The crossover (default order) and mutation (default inversion).
 * @returns The representation.
 * @concept evolutionary-algorithms
 * @example
 * ```ts
 * import { key } from 'whitebox-engine/random'
 * import { permutations } from 'whitebox-engine/search'
 * [...permutations(4).random(key(0))].sort().join() // => '0,1,2,3'
 * ```
 */
export function permutations(
  n: number,
  options: {
    readonly crossover?: 'order' | 'pmx' | 'cycle'
    readonly mutation?: 'swap' | 'inversion' | 'scramble'
  } = {},
): Representation<number[]> {
  const cross = options.crossover ?? 'order'
  const mutation = options.mutation ?? 'inversion'
  return {
    random: (k) => Array.from(permutation(k, n).data),
    crossover: (k, a, b) =>
      cross === 'order' ? orderCrossover(k, a, b) : cross === 'pmx' ? pmxCrossover(k, a, b) : cycleCrossover(a, b),
    mutate: (k, g) =>
      mutation === 'swap'
        ? swapMutation(k, g)
        : mutation === 'inversion'
          ? inversionMutation(k, g)
          : scrambleMutation(k, g),
  }
}

/**
 * Real vectors in a box: uniform random points, blend, simulated binary or arithmetic crossover, and Gaussian or
 * polynomial mutation, kept within the bounds.
 *
 * @param lower - Lower bounds.
 * @param upper - Upper bounds (same length).
 * @param options - The crossover (default SBX), the mutation (default polynomial) and the Gaussian sigma as a fraction
 * of each range (default 0.1).
 * @returns The representation.
 * @throws WhiteboxError with code `shape` when the bounds differ in length, or `value` when a range is empty.
 * @concept evolutionary-algorithms
 * @example
 * ```ts
 * import { key } from 'whitebox-engine/random'
 * import { realVectors } from 'whitebox-engine/search'
 * realVectors([0, 0], [1, 2]).random(key(0)).every((x) => x >= 0 && x <= 2) // => true
 * ```
 */
export function realVectors(
  lower: readonly number[],
  upper: readonly number[],
  options: {
    readonly crossover?: 'blend' | 'sbx' | 'arithmetic'
    readonly mutation?: 'gaussian' | 'polynomial'
    readonly sigma?: number
  } = {},
): Representation<number[]> {
  if (lower.length !== upper.length) throw new WhiteboxError('shape', 'realVectors: bounds differ in length')
  if (lower.some((lo, i) => !((upper[i] ?? lo) >= lo))) throw new WhiteboxError('value', 'realVectors: empty range')
  const cross = options.crossover ?? 'sbx'
  const mutation = options.mutation ?? 'polynomial'
  const clip = (x: readonly number[]): number[] =>
    x.map((v, i) => Math.min(upper[i] as number, Math.max(lower[i] as number, v)))
  // Gaussian steps are a fraction of the mean range.
  const scale = (options.sigma ?? 0.1) * meanOf(upper.map((hi, i) => hi - (lower[i] as number)))
  return {
    random: (k) => {
      const d = draws(k)
      return lower.map((lo, i) => lo + d.uniform() * ((upper[i] as number) - lo))
    },
    crossover: (k, a, b) => {
      const [c, e] =
        cross === 'blend'
          ? blendCrossover(k, a, b)
          : cross === 'arithmetic'
            ? arithmeticCrossover(k, a, b)
            : simulatedBinaryCrossover(k, a, b, { lower, upper })
      return [clip(c), clip(e)]
    },
    mutate: (k, g) =>
      mutation === 'polynomial'
        ? polynomialMutation(k, g, lower, upper)
        : gaussianMutation(k, g, { sigma: scale, lower, upper, rate: 1 / Math.max(1, g.length) }),
  }
}

/** How a genetic algorithm picks parents. */
export type Selection = 'tournament' | 'roulette' | 'rank' | 'universal'

/** Options of {@link geneticAlgorithm}. */
export interface GeneticOptions extends SearchRunOptions {
  /** The population size (default 50). */
  readonly populationSize?: number
  /** The probability that a pair of parents is recombined (default 0.9). */
  readonly crossoverRate?: number
  /** The probability that a child is mutated (default 1). */
  readonly mutationRate?: number
  /** The number of best individuals copied unchanged into the next generation (default 1). */
  readonly elitism?: number
  /** Parent selection (default tournament). */
  readonly selection?: Selection
  /** The tournament size (default 3). */
  readonly tournamentSize?: number
  /** Stop once a fitness reaches this value. */
  readonly target?: number
  /** The number of generations (default 100). */
  readonly generations?: number
}

/** The state of a genetic algorithm: one generation. */
export interface GeneticState<G> {
  readonly population: readonly G[]
  readonly fitness: readonly number[]
  readonly best: G
  readonly bestFitness: number
  readonly generation: number
  readonly evaluations: number
}

/** The outcome of an evolutionary search. */
export interface EvolutionResult<G> {
  readonly best: G
  readonly fitness: number
  readonly generations: number
  readonly evaluations: number
  /** The best fitness of each generation (the initial one first). */
  readonly history: readonly number[]
  readonly status: 'target' | 'budget'
}

const POPULATION_SCHEMA: TraceSchema = channels({
  metrics: {
    best: { description: 'best fitness so far', direction: 'higher' },
    mean: { description: 'mean fitness of the population', direction: 'higher' },
    worst: { description: 'worst fitness of the population' },
    evaluations: { description: 'fitness evaluations so far' },
  },
})

function stats(fitness: readonly number[]): { mean: number; worst: number; bestAt: number } {
  let sum = 0
  let worst = Number.POSITIVE_INFINITY
  let bestAt = 0
  fitness.forEach((f, i) => {
    sum += f
    if (f < worst) worst = f
    if (f > (fitness[bestAt] ?? Number.NEGATIVE_INFINITY)) bestAt = i
  })
  return { mean: sum / Math.max(1, fitness.length), worst, bestAt }
}

function select(
  key: Key,
  fitness: readonly number[],
  n: number,
  selection: Selection,
  tournamentSize: number,
): number[] {
  if (selection === 'tournament') return tournamentSelection(key, fitness, n, tournamentSize)
  if (selection === 'rank') return rankSelection(key, fitness, n)
  // The wheels need non-negative weights: shift by the worst fitness.
  const { worst } = stats(fitness)
  const shifted = fitness.map((f) => f - worst)
  return selection === 'roulette' ? rouletteSelection(key, shifted, n) : stochasticUniversalSampling(key, shifted, n)
}

/**
 * A generational genetic algorithm (Holland 1975; Goldberg 1989) as a traced algorithm: each step breeds a generation
 * by selection, crossover and mutation, keeping the best `elitism` individuals. Fitness is maximised.
 *
 * @param representation - The genome representation.
 * @param fitness - The fitness of a genome.
 * @param options - Population size, rates, elitism, selection and the target.
 * @returns The algorithm.
 * @throws WhiteboxError with code `value` when the population or elitism is invalid.
 * @concept evolutionary-algorithms
 * @example
 * ```ts
 * import { bitStrings, geneticAlgorithmAlgorithm } from 'whitebox-engine/search'
 * geneticAlgorithmAlgorithm(bitStrings(8), (g) => g.reduce((a, b) => a + b, 0)).name // => 'genetic-algorithm'
 * ```
 */
export function geneticAlgorithmAlgorithm<G>(
  representation: Representation<G>,
  fitness: (genome: G) => number,
  options: GeneticOptions = {},
): Algorithm<GeneticState<G>> {
  const size = options.populationSize ?? 50
  const crossoverRate = options.crossoverRate ?? 0.9
  const mutationRate = options.mutationRate ?? 1
  const elitism = options.elitism ?? 1
  const selection = options.selection ?? 'tournament'
  const tournamentSize = options.tournamentSize ?? 3
  if (!(Number.isInteger(size) && size >= 2))
    throw new WhiteboxError('value', `geneticAlgorithm: population ${size} must be at least 2`)
  if (!(Number.isInteger(elitism) && elitism >= 0 && elitism < size)) {
    throw new WhiteboxError('value', `geneticAlgorithm: elitism ${elitism} must be in [0, population)`)
  }
  const target = options.target
  return {
    name: 'genetic-algorithm',
    config: { size, crossoverRate, mutationRate, elitism, selection, tournamentSize } as Json,
    schema: POPULATION_SCHEMA,
    init: (key) => {
      const population = Array.from({ length: size }, (_, i) => representation.random(foldIn(key, i)))
      const f = population.map(fitness)
      const { bestAt } = stats(f)
      return {
        population,
        fitness: f,
        best: population[bestAt] as G,
        bestFitness: f[bestAt] ?? Number.NEGATIVE_INFINITY,
        generation: 0,
        evaluations: size,
      }
    },
    step: (s, ctx) => {
      const d = draws(ctx.key)
      const elite = s.fitness
        .map((f, i) => ({ f, i }))
        .sort((a, b) => b.f - a.f || a.i - b.i)
        .slice(0, elitism)
        .map((x) => x.i)
      const parents = select(d.key(), s.fitness, size - elitism + 1, selection, tournamentSize)
      const children: G[] = elite.map((i) => s.population[i] as G)
      for (let p = 0; children.length < size; p += 2) {
        const a = s.population[parents[p % parents.length] ?? 0] as G
        const b = s.population[parents[(p + 1) % parents.length] ?? 0] as G
        const [c, e] = d.uniform() < crossoverRate ? representation.crossover(d.key(), a, b) : [a, b]
        for (const child of [c, e]) {
          if (children.length >= size) break
          children.push(d.uniform() < mutationRate ? representation.mutate(d.key(), child) : child)
        }
      }
      const f = children.map((g, i) => (i < elitism ? (s.fitness[elite[i] ?? 0] ?? fitness(g)) : fitness(g)))
      const { bestAt } = stats(f)
      const improved = (f[bestAt] ?? Number.NEGATIVE_INFINITY) > s.bestFitness
      return {
        population: children,
        fitness: f,
        best: improved ? (children[bestAt] as G) : s.best,
        bestFitness: improved ? (f[bestAt] as number) : s.bestFitness,
        generation: s.generation + 1,
        evaluations: s.evaluations + size - elitism,
      }
    },
    done: (s) => (target !== undefined && s.bestFitness >= target ? 'converged' : undefined),
    observe: (s) => {
      const { mean, worst } = stats(s.fitness)
      return { metrics: { best: s.bestFitness, mean, worst, evaluations: s.evaluations } }
    },
  }
}

function runPopulation<G>(
  algorithm: Algorithm<GeneticState<G>>,
  generations: number,
  target: number | undefined,
  run: SearchRunOptions,
): EvolutionResult<G> {
  const history: number[] = []
  const tracked: Algorithm<GeneticState<G>> = {
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
  const { state } = runAlgorithm(tracked, { ...run, maxSteps: generations })
  return {
    best: state.best,
    fitness: state.bestFitness,
    generations: state.generation,
    evaluations: state.evaluations,
    history,
    status: target !== undefined && state.bestFitness >= target ? 'target' : 'budget',
  }
}

function runOptions(options: SearchRunOptions): SearchRunOptions {
  const { tracer, policy, parent, runId, seed, key } = options
  return {
    ...(tracer === undefined ? {} : { tracer }),
    ...(policy === undefined ? {} : { policy }),
    ...(parent === undefined ? {} : { parent }),
    ...(runId === undefined ? {} : { runId }),
    ...(seed === undefined ? {} : { seed }),
    ...(key === undefined ? {} : { key }),
  }
}

/**
 * A generational genetic algorithm.
 *
 * @param representation - The genome representation ({@link bitStrings}, {@link permutations}, {@link realVectors} or
 * your own).
 * @param fitness - The fitness of a genome (maximised).
 * @param options - Population size, rates, elitism, selection, the target, the number of generations (default 100),
 * a seed or key, and tracing.
 * @returns The best genome, its fitness, the generations run and the best fitness per generation.
 * @throws WhiteboxError with code `value` when the population or elitism is invalid.
 * @concept evolutionary-algorithms
 * @example
 * ```ts
 * import { bitStrings, geneticAlgorithm } from 'whitebox-engine/search'
 * // OneMax: maximise the number of ones.
 * const r = geneticAlgorithm(bitStrings(20), (g) => g.reduce((a, b) => a + b, 0), { target: 20, generations: 200, seed: 0 })
 * [r.fitness, r.status] // => [20, 'target']
 * ```
 */
export function geneticAlgorithm<G>(
  representation: Representation<G>,
  fitness: (genome: G) => number,
  options: GeneticOptions = {},
): EvolutionResult<G> {
  return runPopulation(
    geneticAlgorithmAlgorithm(representation, fitness, options),
    options.generations ?? 100,
    options.target,
    runOptions(options),
  )
}

/** Options of {@link evolutionStrategy}. */
export interface EvolutionStrategyOptions extends SearchRunOptions {
  /** The number of parents mu (default 5). */
  readonly mu?: number
  /** The number of offspring lambda (default 30). */
  readonly lambda?: number
  /** (mu + lambda) selection from parents and offspring, or (mu, lambda) from offspring only (default false). */
  readonly plus?: boolean
  /** The initial step size (default 1). */
  readonly sigma?: number
  /** Recombination of two parents: intermediate (means), discrete (coordinate-wise picks) or none (default intermediate). */
  readonly recombination?: 'intermediate' | 'discrete' | 'none'
  /** The number of generations (default 200). */
  readonly generations?: number
  /** Stop once a fitness reaches this value. */
  readonly target?: number
}

/** An individual of an evolution strategy: a point and its own step size. */
export interface Strategist {
  readonly x: readonly number[]
  readonly sigma: number
}

/**
 * A self-adaptive evolution strategy (Rechenberg 1973, Schwefel 1977) as a traced algorithm: each offspring
 * recombines two parents, mutates its step size log-normally (tau = 1 / sqrt(2 n)) and then its point by that step
 * size; the mu fittest of the offspring (comma) or of offspring and parents (plus) become the parents. Fitness is
 * maximised. (CMA-ES, which adapts a full covariance, is `cmaes` in `whitebox-engine/optim`.)
 *
 * @param fitness - The fitness of a point.
 * @param x0 - The starting point.
 * @param options - mu, lambda, plus or comma selection, the initial sigma and recombination.
 * @returns The algorithm.
 * @throws WhiteboxError with code `value` when mu or lambda are invalid.
 * @concept evolution-strategies
 * @example
 * ```ts
 * import { evolutionStrategyAlgorithm } from 'whitebox-engine/search'
 * evolutionStrategyAlgorithm((x) => -x.reduce((a, v) => a + v * v, 0), [1, 1]).name // => 'evolution-strategy'
 * ```
 */
export function evolutionStrategyAlgorithm(
  fitness: (x: readonly number[]) => number,
  x0: readonly number[],
  options: EvolutionStrategyOptions = {},
): Algorithm<GeneticState<Strategist>> {
  const mu = options.mu ?? 5
  const lambda = options.lambda ?? 30
  const plus = options.plus ?? false
  const sigma0 = options.sigma ?? 1
  const recombination = options.recombination ?? 'intermediate'
  if (!(Number.isInteger(mu) && mu >= 1 && Number.isInteger(lambda) && lambda >= 1) || (!plus && lambda < mu)) {
    throw new WhiteboxError('value', `evolutionStrategy: need mu >= 1 and lambda >= ${plus ? 1 : 'mu'}`)
  }
  const n = x0.length
  const tau = 1 / Math.sqrt(2 * Math.max(1, n))
  return {
    name: 'evolution-strategy',
    config: { mu, lambda, plus, sigma0, recombination },
    schema: POPULATION_SCHEMA,
    init: (key) => {
      const d = draws(key)
      const population = Array.from({ length: mu }, (_, i) => ({
        x: i === 0 ? [...x0] : x0.map((v) => v + sigma0 * d.normal()),
        sigma: sigma0,
      }))
      const f = population.map((p) => fitness(p.x))
      const { bestAt } = stats(f)
      return {
        population,
        fitness: f,
        best: population[bestAt] as Strategist,
        bestFitness: f[bestAt] ?? Number.NEGATIVE_INFINITY,
        generation: 0,
        evaluations: mu,
      }
    },
    step: (s, ctx) => {
      const d = draws(ctx.key)
      const offspring = Array.from({ length: lambda }, () => {
        const a = s.population[d.int(s.population.length)] as Strategist
        const b = s.population[d.int(s.population.length)] as Strategist
        const base =
          recombination === 'none'
            ? a
            : recombination === 'intermediate'
              ? { x: a.x.map((v, i) => (v + (b.x[i] as number)) / 2), sigma: Math.sqrt(a.sigma * b.sigma) }
              : {
                  x: a.x.map((v, i) => (d.uniform() < 0.5 ? v : (b.x[i] as number))),
                  sigma: d.uniform() < 0.5 ? a.sigma : b.sigma,
                }
        const sigma = base.sigma * Math.exp(tau * d.normal())
        return { x: base.x.map((v) => v + sigma * d.normal()), sigma }
      })
      const fo = offspring.map((o) => fitness(o.x))
      const pool = plus
        ? [
            ...s.population.map((p, i) => ({ p, f: s.fitness[i] ?? 0 })),
            ...offspring.map((p, i) => ({ p, f: fo[i] ?? 0 })),
          ]
        : offspring.map((p, i) => ({ p, f: fo[i] ?? 0 }))
      const kept = pool
        .map((x, i) => ({ ...x, i }))
        .sort((a, b) => b.f - a.f || a.i - b.i)
        .slice(0, mu)
      const top = kept[0]
      const improved = top !== undefined && top.f > s.bestFitness
      return {
        population: kept.map((k) => k.p),
        fitness: kept.map((k) => k.f),
        best: improved ? top.p : s.best,
        bestFitness: improved ? top.f : s.bestFitness,
        generation: s.generation + 1,
        evaluations: s.evaluations + lambda,
      }
    },
    done: (s) => (options.target !== undefined && s.bestFitness >= options.target ? 'converged' : undefined),
    observe: (s) => {
      const { mean, worst } = stats(s.fitness)
      return { metrics: { best: s.bestFitness, mean, worst, evaluations: s.evaluations } }
    },
  }
}

/**
 * A self-adaptive (mu/2, lambda) or (mu/2 + lambda) evolution strategy.
 *
 * @param fitness - The fitness of a point (maximised).
 * @param x0 - The starting point.
 * @param options - mu, lambda, plus or comma, sigma, recombination, generations (default 200), the target, a seed or
 * key, and tracing.
 * @returns The best individual (point and step size), its fitness and the best fitness per generation.
 * @throws WhiteboxError with code `value` when mu or lambda are invalid.
 * @concept evolution-strategies
 * @example
 * ```ts
 * import { evolutionStrategy } from 'whitebox-engine/search'
 * const sphere = (x: readonly number[]) => -x.reduce((a, v) => a + v * v, 0)
 * evolutionStrategy(sphere, [3, -2, 1], { seed: 0 }).fitness > -1e-6 // => true
 * ```
 */
export function evolutionStrategy(
  fitness: (x: readonly number[]) => number,
  x0: readonly number[],
  options: EvolutionStrategyOptions = {},
): EvolutionResult<Strategist> {
  return runPopulation(
    evolutionStrategyAlgorithm(fitness, x0, options),
    options.generations ?? 200,
    options.target,
    runOptions(options),
  )
}
