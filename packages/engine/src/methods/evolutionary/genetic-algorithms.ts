/**
 * Genetic algorithms on toy problems, played generation by generation from the core's genetic algorithm
 * (`whitebox-engine/search`): bit strings for OneMax and a random 0/1 knapsack, recorded as a board of genomes, and
 * real vectors minimising an objective on the plane, recorded over its contours.
 */

import { WhiteboxError } from '../../lib/base/index.ts'
import {
  type BoardSteps,
  boardSteps,
  type Explained,
  type Generation,
  type PopulationSteps,
  populationSteps,
} from '../../lib/exposition/index.ts'
import { type Key, key as makeKey, uniform } from '../../lib/random/index.ts'
import {
  bitStrings,
  type GeneticState,
  geneticAlgorithmAlgorithm,
  realVectors,
  type Selection,
} from '../../lib/search/index.ts'
import { type Algorithm, iterate } from '../../lib/trace/index.ts'
import { type PlaneFunctionName, planeFunction } from './_plane.ts'

/** A problem over bit strings: its length and the fitness (maximised) of a genome. */
export interface BitProblem {
  readonly name: string
  readonly n: number
  readonly fitness: (bits: readonly number[]) => number
  /** The best fitness possible, when known. */
  readonly optimum?: number
}

/**
 * OneMax: the fitness of a bit string is its number of ones.
 *
 * @param n - The length (default 32).
 * @returns The problem.
 */
export function oneMax(n = 32): BitProblem {
  return { name: 'OneMax', n, fitness: (b) => b.reduce((a, x) => a + x, 0), optimum: n }
}

/** A 0/1 knapsack: item weights and values, and the capacity. */
export interface Knapsack extends BitProblem {
  readonly weights: readonly number[]
  readonly values: readonly number[]
  readonly capacity: number
}

/**
 * A random 0/1 knapsack: items of weight 1 to 10 and value 1 to 10, the capacity a third of the total weight. A
 * packing over capacity scores the capacity minus its weight (negative), so feasible packings always score higher.
 *
 * @param key - The key.
 * @param n - The number of items (default 24).
 * @returns The problem.
 */
export function knapsack(key: Key, n = 24): Knapsack {
  const u = Array.from(uniform(key, [2, n]).data)
  const weights = Array.from({ length: n }, (_, i) => 1 + Math.floor((u[i] ?? 0) * 10))
  const values = Array.from({ length: n }, (_, i) => 1 + Math.floor((u[n + i] ?? 0) * 10))
  const capacity = Math.round(weights.reduce((a, b) => a + b, 0) / 3)
  return {
    name: 'knapsack',
    n,
    weights,
    values,
    capacity,
    fitness: (b) => {
      let w = 0
      let v = 0
      for (let i = 0; i < n; i++)
        if (b[i] === 1) {
          w += weights[i] ?? 0
          v += values[i] ?? 0
        }
      return w <= capacity ? v : capacity - w
    },
  }
}

/** Options of the genetic algorithms. */
export interface GeneticPlayOptions {
  readonly populationSize?: number
  readonly generations?: number
  readonly selection?: Selection
  readonly crossoverRate?: number
  /** Bit strings: the probability of flipping each bit (default 1 / n). */
  readonly bitFlipRate?: number
  readonly crossover?: 'one-point' | 'two-point' | 'uniform'
  readonly elitism?: number
  readonly key?: Key
}

/** What a genetic algorithm found. */
export interface GeneticPlay<G> {
  readonly best: G
  readonly fitness: number
  readonly generations: number
  readonly evaluations: number
  /** The best fitness of each generation. */
  readonly history: readonly number[]
}

function runGenerations<G>(
  algorithm: Algorithm<GeneticState<G>>,
  generations: number,
  key: Key,
  target?: number,
): GeneticState<G>[] {
  const out: GeneticState<G>[] = []
  for (const { state } of iterate(algorithm, { key, maxSteps: generations })) {
    out.push({ ...state, population: [...state.population], fitness: [...state.fitness] })
    if (target !== undefined && state.bestFitness >= target) break
  }
  return out
}

/**
 * A genetic algorithm on bit strings, recording every generation: the population as a board (one row per genome,
 * fittest first, one column per bit) and the fitness of every genome with the best so far.
 *
 * @param problem - The bit-string problem.
 * @param options - Population, generations, selection, crossover, mutation, elitism and key.
 * @returns The best genome and the history, with `boardSteps` and `populationSteps` records.
 */
export function evolveBitStrings(
  problem: BitProblem,
  options: GeneticPlayOptions = {},
): Explained<GeneticPlay<number[]>> {
  const size = options.populationSize ?? 30
  const generations = options.generations ?? 60
  const representation = bitStrings(problem.n, {
    crossover: options.crossover ?? 'one-point',
    rate: options.bitFlipRate ?? 1 / problem.n,
  })
  const algorithm = geneticAlgorithmAlgorithm(representation, problem.fitness, {
    populationSize: size,
    selection: options.selection ?? 'tournament',
    crossoverRate: options.crossoverRate ?? 0.9,
    elitism: options.elitism ?? 1,
  })
  const states = runGenerations(algorithm, generations, options.key ?? makeKey(0), problem.optimum)
  const last = states[states.length - 1]
  if (last === undefined) throw new WhiteboxError('value', 'evolveBitStrings: no generations')
  const order = (s: GeneticState<number[]>): number[] =>
    s.fitness
      .map((f, i) => [f, i] as const)
      .sort((a, b) => b[0] - a[0] || a[1] - b[1])
      .map(([, i]) => i)
  const board: BoardSteps = boardSteps({
    method: 'genetic algorithm',
    layout: { type: 'grid', rows: size, columns: problem.n },
    categories: ['0', '1'],
    steps: states.map((s) => ({
      cells: order(s).flatMap((i) => s.population[i] ?? []),
      caption: `best ${s.bestFitness}`,
    })),
    metrics: [{ name: 'best fitness', values: states.map((s) => s.bestFitness) }],
    stepName: 'generation',
    summary: `${problem.name}: best ${last.bestFitness} after ${last.generation} generations`,
  })
  const population: PopulationSteps = populationSteps({
    method: 'genetic algorithm',
    generations: states.map(
      (s): Generation => ({
        fitness: s.fitness,
        best: { fitness: s.bestFitness, label: s.best.join('') },
      }),
    ),
  })
  return {
    best: last.best,
    fitness: last.bestFitness,
    generations: last.generation,
    evaluations: last.evaluations,
    history: states.map((s) => s.bestFitness),
    exposition: [board, population],
  }
}

/** Options of {@link evolveOnPlane}. */
export interface PlaneGeneticOptions extends Omit<GeneticPlayOptions, 'bitFlipRate' | 'crossover'> {
  readonly crossover?: 'blend' | 'sbx' | 'arithmetic'
  readonly mutation?: 'gaussian' | 'polynomial'
}

/**
 * A real-coded genetic algorithm minimising an objective on the plane, recording every generation: the population
 * over the objective's contours, the best point, and the objective of every individual.
 *
 * @param name - The objective (`rastrigin`, `himmelblau`, `rosenbrock`, `ackley`).
 * @param options - Population, generations, selection, crossover, mutation, elitism and key.
 * @returns The best point and the history, with a `populationSteps` record.
 */
export function evolveOnPlane(
  name: PlaneFunctionName,
  options: PlaneGeneticOptions = {},
): Explained<GeneticPlay<number[]>> {
  const plane = planeFunction(name)
  const representation = realVectors(plane.lower, plane.upper, {
    crossover: options.crossover ?? 'sbx',
    mutation: options.mutation ?? 'gaussian',
  })
  const algorithm = geneticAlgorithmAlgorithm(representation, (x) => -plane.f(x), {
    populationSize: options.populationSize ?? 40,
    selection: options.selection ?? 'tournament',
    crossoverRate: options.crossoverRate ?? 0.9,
    elitism: options.elitism ?? 1,
  })
  const states = runGenerations(algorithm, options.generations ?? 40, options.key ?? makeKey(0))
  const last = states[states.length - 1]
  if (last === undefined) throw new WhiteboxError('value', 'evolveOnPlane: no generations')
  const record = populationSteps({
    method: 'genetic algorithm',
    lowerIsBetter: true,
    surface: { ...plane.surface, names: ['x1', 'x2'], optimum: plane.optimum },
    generations: states.map((s) => ({
      fitness: s.fitness.map((f) => -f),
      points: s.population.map((p) => [p[0] ?? 0, p[1] ?? 0] as const),
      best: { fitness: -s.bestFitness, point: [s.best[0] ?? 0, s.best[1] ?? 0] as const },
    })),
  })
  return {
    best: last.best,
    fitness: -last.bestFitness,
    generations: last.generation,
    evaluations: last.evaluations,
    history: states.map((s) => -s.bestFitness),
    exposition: [record],
  }
}
