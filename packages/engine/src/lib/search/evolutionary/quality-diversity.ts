/**
 * Quality-diversity search, which seeks many good solutions that behave differently rather than one best. MAP-Elites,
 * which keeps the fittest solution in each cell of a grid over behaviour descriptors, and novelty search, which rewards
 * behaviours far from those seen before, both as traced algorithms.
 */

import { meanOf, WhiteboxError } from '../../base/index.ts'
import { kNearest } from '../../numerics/spatial/index.ts'
import { foldIn, type Key } from '../../random/index.ts'
import { toFlat } from '../../tensor/index.ts'
import { type Algorithm, channels, runAlgorithm } from '../../trace/index.ts'
import { draws } from '../_draw.ts'
import type { SearchRunOptions } from '../problem.ts'

/** A genome with its fitness and behaviour descriptor. */
export interface Elite<G> {
  readonly genome: G
  readonly fitness: number
  readonly descriptor: readonly number[]
}

/** A quality-diversity problem: random genomes, variation, a fitness and a behaviour descriptor. */
export interface QualityDiversityProblem<G> {
  random(key: Key): G
  mutate(key: Key, genome: G): G
  /** Optional recombination of two elites before mutation. */
  crossover?(key: Key, a: G, b: G): G
  fitness(genome: G): number
  /** The behaviour of a genome: a point in descriptor space. */
  descriptor(genome: G): readonly number[]
}

/** Options of {@link mapElites}. */
export interface MapElitesOptions extends SearchRunOptions {
  /** Bins per descriptor dimension. */
  readonly bins: readonly number[]
  /** The [low, high] range of each descriptor dimension (values outside are clipped into the edge bins). */
  readonly bounds: readonly (readonly [number, number])[]
  /** Random genomes evaluated first (default 100). */
  readonly initial?: number
  /** Offspring per step (default 50). */
  readonly batch?: number
  /** The number of steps (default 100). */
  readonly iterations?: number
}

/** The state of MAP-Elites: the archive of elites by cell, and the effort. */
export interface MapElitesState<G> {
  readonly archive: ReadonlyMap<number, Elite<G>>
  readonly evaluations: number
  readonly iteration: number
}

/** The result of MAP-Elites: the filled cells with their elites, coverage and QD score. */
export interface MapElitesResult<G> {
  /** The elites by cell index (row-major over the bins). */
  readonly archive: ReadonlyMap<number, Elite<G>>
  /** The fraction of cells filled. */
  readonly coverage: number
  /** The sum of elite fitnesses (the QD score). */
  readonly qdScore: number
  readonly best: Elite<G> | null
  readonly evaluations: number
}

function cellOf(
  descriptor: readonly number[],
  bins: readonly number[],
  bounds: readonly (readonly [number, number])[],
): number {
  let cell = 0
  bins.forEach((b, i) => {
    const [lo, hi] = bounds[i] ?? [0, 1]
    const x = descriptor[i] ?? lo
    const k = Math.min(b - 1, Math.max(0, Math.floor(((x - lo) / (hi - lo)) * b)))
    cell = cell * b + k
  })
  return cell
}

function mapConfig(options: MapElitesOptions): { cells: number } {
  if (options.bins.length === 0 || options.bins.length !== options.bounds.length) {
    throw new WhiteboxError('value', 'mapElites: give bins and bounds for each descriptor dimension')
  }
  if (options.bins.some((b) => !(Number.isInteger(b) && b >= 1)) || options.bounds.some(([lo, hi]) => !(hi > lo))) {
    throw new WhiteboxError('value', 'mapElites: bins must be positive integers and bounds non-empty')
  }
  return { cells: options.bins.reduce((a, b) => a * b, 1) }
}

/**
 * MAP-Elites (Mouret and Clune 2015) as a traced algorithm: a grid over behaviour descriptors keeps the fittest genome
 * found in each cell; each step mutates (and optionally recombines) random elites and offers the offspring to their
 * cells. The result illuminates the best fitness achievable across behaviours.
 *
 * @param problem - The genomes, variation, fitness and descriptor.
 * @param options - Bins and bounds per descriptor dimension, the initial sample and the batch size.
 * @returns The algorithm.
 * @throws WhiteboxError with code `value` when the grid is malformed.
 * @concept quality-diversity
 * @example
 * ```ts
 * import { mapElitesAlgorithm } from 'whitebox-engine/search'
 * const p = { random: () => 0, mutate: (_k: unknown, x: number) => x, fitness: () => 0, descriptor: (x: number) => [x] }
 * mapElitesAlgorithm(p, { bins: [4], bounds: [[0, 1]] }).name // => 'map-elites'
 * ```
 */
export function mapElitesAlgorithm<G>(
  problem: QualityDiversityProblem<G>,
  options: MapElitesOptions,
): Algorithm<MapElitesState<G>> {
  const { cells } = mapConfig(options)
  const initial = options.initial ?? 100
  const batch = options.batch ?? 50
  const offer = (archive: Map<number, Elite<G>>, genome: G): void => {
    const e = { genome, fitness: problem.fitness(genome), descriptor: problem.descriptor(genome) }
    const c = cellOf(e.descriptor, options.bins, options.bounds)
    const old = archive.get(c)
    if (old === undefined || e.fitness > old.fitness) archive.set(c, e)
  }
  return {
    name: 'map-elites',
    config: { bins: [...options.bins], initial, batch },
    schema: channels({
      metrics: {
        coverage: { description: 'fraction of cells filled', direction: 'higher' },
        qdScore: { description: 'sum of elite fitnesses', direction: 'higher' },
        best: { description: 'best elite fitness', direction: 'higher' },
      },
    }),
    init: (key) => {
      const archive = new Map<number, Elite<G>>()
      for (let i = 0; i < initial; i++) offer(archive, problem.random(foldIn(key, i)))
      return { archive, evaluations: initial, iteration: 0 }
    },
    step: (s, ctx) => {
      const d = draws(ctx.key)
      const archive = new Map(s.archive)
      const elites = [...s.archive.values()]
      for (let i = 0; i < batch; i++) {
        if (elites.length === 0) {
          offer(archive, problem.random(d.key()))
          continue
        }
        const a = elites[d.int(elites.length)] as Elite<G>
        let child = a.genome
        if (problem.crossover !== undefined && elites.length > 1) {
          child = problem.crossover(d.key(), a.genome, (elites[d.int(elites.length)] as Elite<G>).genome)
        }
        offer(archive, problem.mutate(d.key(), child))
      }
      return { archive, evaluations: s.evaluations + batch, iteration: s.iteration + 1 }
    },
    observe: (s) => {
      let qd = 0
      let best = Number.NEGATIVE_INFINITY
      for (const e of s.archive.values()) {
        qd += e.fitness
        if (e.fitness > best) best = e.fitness
      }
      return { metrics: { coverage: s.archive.size / cells, qdScore: qd, best } }
    },
  }
}

/**
 * MAP-Elites.
 *
 * @param problem - The genomes, variation, fitness and descriptor.
 * @param options - Bins and bounds, the initial sample, batch size, iterations (default 100), a seed or key, and
 * tracing.
 * @returns The archive with its coverage, QD score and best elite.
 * @throws WhiteboxError with code `value` when the grid is malformed.
 * @concept quality-diversity
 * @example
 * ```ts
 * import { gaussianMutation, mapElites } from 'whitebox-engine/search'
 * import { uniform } from 'whitebox-engine/random'
 * // Points in the unit square, described by their coordinates: every cell can be filled.
 * const p = {
 *   random: (k: Parameters<typeof uniform>[0]) => Array.from(uniform(k, [2]).data),
 *   mutate: (k: Parameters<typeof uniform>[0], x: number[]) => gaussianMutation(k, x, { sigma: 0.2, lower: [0, 0], upper: [1, 1] }),
 *   fitness: (x: number[]) => -((x[0] ?? 0) ** 2),
 *   descriptor: (x: number[]) => x,
 * }
 * mapElites(p, { bins: [5, 5], bounds: [[0, 1], [0, 1]], iterations: 40, seed: 0 }).coverage // => 1
 * ```
 */
export function mapElites<G>(problem: QualityDiversityProblem<G>, options: MapElitesOptions): MapElitesResult<G> {
  const { bins: _b, bounds: _bo, initial: _i, batch: _ba, iterations = 100, ...run } = options
  const { cells } = mapConfig(options)
  const { state } = runAlgorithm(mapElitesAlgorithm(problem, options), { ...run, maxSteps: iterations })
  let qdScore = 0
  let best: Elite<G> | null = null
  for (const e of state.archive.values()) {
    qdScore += e.fitness
    if (best === null || e.fitness > best.fitness) best = e
  }
  return { archive: state.archive, coverage: state.archive.size / cells, qdScore, best, evaluations: state.evaluations }
}

/** Options of {@link noveltySearch}. */
export interface NoveltyOptions<G = unknown> extends SearchRunOptions {
  /** The population size (default 50). */
  readonly populationSize?: number
  /** Neighbours used for novelty (default 15). */
  readonly k?: number
  /** Novelty above which an individual enters the archive (default 1). */
  readonly threshold?: number
  /** The tournament size for selection by novelty (default 3). */
  readonly tournamentSize?: number
  /** The number of generations (default 100). */
  readonly generations?: number
  /** Stop when a genome passes this test (the goal novelty search looks for without aiming at it). */
  readonly isGoal?: (genome: G) => boolean
}

/** The state of novelty search. */
export interface NoveltyState<G> {
  readonly population: readonly G[]
  readonly behaviours: readonly (readonly number[])[]
  readonly novelty: readonly number[]
  readonly archive: readonly (readonly number[])[]
  readonly generation: number
  readonly evaluations: number
  /** The first genome to pass the goal test, if any. */
  readonly found: G | null
  /** The fittest genome seen (by `fitness`, when the problem has one). */
  readonly best: G
  readonly bestFitness: number
}

/** A novelty-search problem: random genomes, mutation and a behaviour; fitness is optional (only tracked). */
export interface NoveltyProblem<G> {
  random(key: Key): G
  mutate(key: Key, genome: G): G
  behaviour(genome: G): readonly number[]
  fitness?(genome: G): number
}

function noveltyOf(
  behaviours: readonly (readonly number[])[],
  archive: readonly (readonly number[])[],
  k: number,
): number[] {
  const pool = [...behaviours, ...archive]
  const m = Math.min(k, pool.length - 1)
  if (m < 1) return behaviours.map(() => 0)
  // Mean distance to the k nearest others in the population and archive (numerics' brute-force kNearest).
  const near = kNearest(pool as number[][], undefined, m, { excludeSelf: true })
  const dist = toFlat(near.distances)
  return behaviours.map((_, i) => {
    let s = 0
    for (let j = 0; j < m; j++) s += dist[i * m + j] ?? 0
    return s / m
  })
}

/**
 * Novelty search (Lehman and Stanley 2011) as a traced algorithm: selection rewards behaviours far from those seen
 * (mean distance to the k nearest in the population and an archive of novel behaviours), not fitness, which escapes
 * deceptive objectives. Each step breeds a generation by tournament on novelty and mutation.
 *
 * @param problem - Random genomes, mutation, the behaviour and an optional fitness to track.
 * @param options - Population size, k, the archive threshold, tournament size and a goal test.
 * @returns The algorithm.
 * @throws WhiteboxError with code `value` when the population is smaller than 2.
 * @concept quality-diversity
 * @example
 * ```ts
 * import { noveltySearchAlgorithm } from 'whitebox-engine/search'
 * const p = { random: () => 0, mutate: (_k: unknown, x: number) => x + 1, behaviour: (x: number) => [x] }
 * noveltySearchAlgorithm(p).name // => 'novelty-search'
 * ```
 */
export function noveltySearchAlgorithm<G>(
  problem: NoveltyProblem<G>,
  options: NoveltyOptions<G> = {},
): Algorithm<NoveltyState<G>> {
  const size = options.populationSize ?? 50
  const k = options.k ?? 15
  const threshold = options.threshold ?? 1
  const tournament = options.tournamentSize ?? 3
  const isGoal = options.isGoal
  if (!(Number.isInteger(size) && size >= 2))
    throw new WhiteboxError('value', 'noveltySearch: population must be at least 2')
  const score = (g: G): number => problem.fitness?.(g) ?? 0
  const evaluate = (population: readonly G[], archive: readonly (readonly number[])[], s: NoveltyState<G> | null) => {
    const behaviours = population.map((g) => problem.behaviour(g))
    const novelty = noveltyOf(behaviours, archive, k)
    const added = behaviours.filter((_, i) => (novelty[i] ?? 0) > threshold)
    let best = s?.best ?? (population[0] as G)
    let bestFitness = s?.bestFitness ?? Number.NEGATIVE_INFINITY
    for (const g of population) {
      const f = score(g)
      if (f > bestFitness) {
        best = g
        bestFitness = f
      }
    }
    const found = s?.found ?? (isGoal === undefined ? null : (population.find((g) => isGoal(g)) ?? null))
    return { behaviours, novelty, archive: [...archive, ...added], best, bestFitness, found }
  }
  return {
    name: 'novelty-search',
    config: { size, k, threshold, tournament },
    schema: channels({
      metrics: {
        novelty: { description: 'mean novelty of the population', direction: 'higher' },
        archive: { description: 'archived behaviours' },
        best: { description: 'best fitness seen', direction: 'higher' },
      },
    }),
    init: (key) => {
      const population = Array.from({ length: size }, (_, i) => problem.random(foldIn(key, i)))
      return { population, ...evaluate(population, [], null), generation: 0, evaluations: size }
    },
    step: (s, ctx) => {
      const d = draws(ctx.key)
      const children = Array.from({ length: size }, () => {
        let w = d.int(size)
        for (let t = 1; t < tournament; t++) {
          const c = d.int(size)
          if ((s.novelty[c] ?? 0) > (s.novelty[w] ?? 0)) w = c
        }
        return problem.mutate(d.key(), s.population[w] as G)
      })
      return {
        population: children,
        ...evaluate(children, s.archive, s),
        generation: s.generation + 1,
        evaluations: s.evaluations + size,
      }
    },
    done: (s) => (s.found !== null ? 'converged' : undefined),
    observe: (s) => ({
      metrics: {
        novelty: meanOf(s.novelty),
        archive: s.archive.length,
        best: s.bestFitness,
      },
    }),
  }
}

/**
 * Novelty search.
 *
 * @param problem - Random genomes, mutation, the behaviour and an optional fitness.
 * @param options - Population size, k, threshold, tournament size, generations (default 100), a goal test, a seed or
 * key, and tracing.
 * @returns The final state: population, archive, the genome that met the goal (if any) and the fittest seen.
 * @throws WhiteboxError with code `value` when the population is smaller than 2.
 * @concept quality-diversity
 * @example
 * ```ts
 * import { noveltySearch } from 'whitebox-engine/search'
 * import { normal } from 'whitebox-engine/random'
 * // A walker on a line rewarded by nothing: novelty alone drives it far from the start.
 * const p = { random: () => 0, mutate: (k: Parameters<typeof normal>[0], x: number) => x + (normal(k, []).data[0] ?? 0), behaviour: (x: number) => [x] }
 * noveltySearch(p, { generations: 60, isGoal: (x) => Math.abs(x) > 10, seed: 0 }).found !== null // => true
 * ```
 */
export function noveltySearch<G>(problem: NoveltyProblem<G>, options: NoveltyOptions<G> = {}): NoveltyState<G> {
  const {
    populationSize: _p,
    k: _k,
    threshold: _t,
    tournamentSize: _ts,
    generations = 100,
    isGoal: _g,
    ...run
  } = options
  return runAlgorithm(noveltySearchAlgorithm(problem, options), { ...run, maxSteps: generations }).state
}
