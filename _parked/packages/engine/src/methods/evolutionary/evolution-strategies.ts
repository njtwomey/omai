/**
 * Evolution strategies and CMA-ES on objectives in the plane, played generation by generation: a self-adaptive (mu,
 * lambda) or (mu + lambda) strategy, and CMA-ES, which adapts a Gaussian search distribution's mean, step size and
 * covariance. Every generation records the offspring sampled, the search distribution as an ellipse and the best point
 * over the objective's contours.
 */

import { MdmlError } from '../../lib/base/index.ts'
import { type Explained, type Generation, type PopulationSteps, populationSteps } from '../../lib/exposition/index.ts'
import { type CmaesState, cmaesAlgorithm } from '../../lib/optim/index.ts'
import { type Key, key as makeKey } from '../../lib/random/index.ts'
import { evolutionStrategyAlgorithm } from '../../lib/search/index.ts'
import { tensor, toFlat, type Value } from '../../lib/tensor/index.ts'
import { iterate } from '../../lib/trace/index.ts'
import { type PlaneFunction, type PlaneFunctionName, planeFunction } from './_plane.ts'

export type { PlaneFunctionName } from './_plane.ts'

/** What an evolution strategy found on the plane. */
export interface PlaneEvolution {
  readonly best: readonly [number, number]
  readonly value: number
  readonly generations: number
  readonly evaluations: number
  /** The best value of each generation. */
  readonly history: readonly number[]
}

// The objective, counting and remembering every point it is asked about until the next generation is recorded.
function recorder(plane: PlaneFunction): {
  readonly f: (x: readonly number[]) => number
  take(): { points: [number, number][]; values: number[] }
} {
  let points: [number, number][] = []
  let values: number[] = []
  return {
    f: (x) => {
      const v = plane.f(x)
      points.push([x[0] ?? 0, x[1] ?? 0])
      values.push(v)
      return v
    },
    take: () => {
      const out = { points, values }
      points = []
      values = []
      return out
    },
  }
}

function record(plane: PlaneFunction, method: string, generations: readonly Generation[]): PopulationSteps {
  return populationSteps({
    method,
    lowerIsBetter: true,
    surface: { ...plane.surface, names: ['x1', 'x2'], optimum: plane.optimum },
    generations,
  })
}

/** Options of {@link evolutionStrategyOnPlane}. */
export interface PlaneStrategyOptions {
  /** Parents mu and offspring lambda (default 5 and 30). */
  readonly mu?: number
  readonly lambda?: number
  /** (mu + lambda) rather than (mu, lambda) selection (default false). */
  readonly plus?: boolean
  /** The initial step size (default 1). */
  readonly sigma?: number
  readonly generations?: number
  /** The start (default a corner of the box). */
  readonly x0?: readonly [number, number]
  readonly key?: Key
}

/**
 * A self-adaptive (mu, lambda) or (mu + lambda) evolution strategy minimising an objective on the plane, recording
 * every generation: all the offspring sampled, the parents kept and their mean, and the best point so far.
 *
 * @param name - The objective (`rastrigin`, `himmelblau`, `rosenbrock`, `ackley`).
 * @param options - mu, lambda, plus selection, step size, generations, start and key.
 * @returns The best point and the history, with a `populationSteps` record.
 */
export function evolutionStrategyOnPlane(
  name: PlaneFunctionName,
  options: PlaneStrategyOptions = {},
): Explained<PlaneEvolution> {
  const plane = planeFunction(name)
  const seen = recorder(plane)
  const x0 = options.x0 ?? cornerOf(plane)
  const algorithm = evolutionStrategyAlgorithm((x) => -seen.f(x), x0, {
    mu: options.mu ?? 5,
    lambda: options.lambda ?? 30,
    plus: options.plus ?? false,
    sigma: options.sigma ?? 1,
  })
  const gens: Generation[] = []
  let best: readonly number[] = x0
  let bestValue = Number.POSITIVE_INFINITY
  let evaluations = 0
  for (const { state } of iterate(algorithm, { key: options.key ?? makeKey(0), maxSteps: options.generations ?? 40 })) {
    const sampled = seen.take()
    evaluations = state.evaluations
    best = state.best.x
    bestValue = -state.bestFitness
    const parents = state.population.map((p) => p.x)
    const mean: [number, number] = [
      parents.reduce((a, p) => a + (p[0] ?? 0), 0) / Math.max(1, parents.length),
      parents.reduce((a, p) => a + (p[1] ?? 0), 0) / Math.max(1, parents.length),
    ]
    const sigma = state.population.reduce((a, p) => a + p.sigma, 0) / Math.max(1, state.population.length)
    gens.push({
      fitness: sampled.values,
      points: sampled.points,
      mean,
      covariance: [sigma * sigma, 0, sigma * sigma],
      best: { fitness: bestValue, point: [best[0] ?? 0, best[1] ?? 0] },
    })
  }
  if (gens.length === 0) throw new MdmlError('value', 'evolutionStrategyOnPlane: no generations')
  const method = `(${options.mu ?? 5} ${options.plus === true ? '+' : ','} ${options.lambda ?? 30})-ES`
  return {
    best: [best[0] ?? 0, best[1] ?? 0],
    value: bestValue,
    generations: gens.length - 1,
    evaluations,
    history: gens.map((g) => g.best.fitness),
    exposition: [record(plane, method, gens)],
  }
}

function cornerOf(plane: PlaneFunction): [number, number] {
  return [
    plane.lower[0] + 0.85 * (plane.upper[0] - plane.lower[0]),
    plane.lower[1] + 0.85 * (plane.upper[1] - plane.lower[1]),
  ]
}

/** Options of {@link cmaesOnPlane}. */
export interface PlaneCmaesOptions {
  /** Offspring per generation (default 4 + floor(3 ln 2) = 6). */
  readonly popSize?: number
  /** The initial step size (default 1). */
  readonly sigma0?: number
  readonly generations?: number
  readonly x0?: readonly [number, number]
  readonly key?: Key
}

/**
 * CMA-ES minimising an objective on the plane, recording every generation: the offspring sampled, the search
 * distribution (mean and sigma squared times the covariance, drawn as ellipses) adapting its shape and size, and the
 * best point so far.
 *
 * @param name - The objective (`rastrigin`, `himmelblau`, `rosenbrock`, `ackley`).
 * @param options - Population, step size, generations, start and key.
 * @returns The best point and the history, with a `populationSteps` record.
 */
export function cmaesOnPlane(name: PlaneFunctionName, options: PlaneCmaesOptions = {}): Explained<PlaneEvolution> {
  const plane = planeFunction(name)
  const seen = recorder(plane)
  const x0 = options.x0 ?? cornerOf(plane)
  const algorithm = cmaesAlgorithm((x) => tensor(seen.f(Array.from(toFlat(x as Value)))), [...x0], {
    sigma0: options.sigma0 ?? 1,
    ...(options.popSize === undefined ? {} : { popSize: options.popSize }),
    bounds: [
      [plane.lower[0], plane.upper[0]],
      [plane.lower[1], plane.upper[1]],
    ],
  })
  const gens: Generation[] = []
  let best: [number, number] = [x0[0], x0[1]]
  let bestValue = Number.POSITIVE_INFINITY
  let last: CmaesState | undefined
  for (const { state } of iterate(algorithm, { key: options.key ?? makeKey(0), maxSteps: options.generations ?? 40 })) {
    const sampled = seen.take()
    for (const [i, v] of sampled.values.entries())
      if (v < bestValue) {
        bestValue = v
        best = sampled.points[i] ?? best
      }
    const s2 = state.sigma * state.sigma
    gens.push({
      fitness: sampled.values,
      points: sampled.points,
      mean: [state.mean[0] ?? 0, state.mean[1] ?? 0],
      covariance: [s2 * (state.cov[0] ?? 0), s2 * (state.cov[1] ?? 0), s2 * (state.cov[3] ?? 0)],
      best: { fitness: bestValue, point: best },
    })
    last = state
  }
  if (last === undefined) throw new MdmlError('value', 'cmaesOnPlane: no generations')
  return {
    best,
    value: bestValue,
    generations: last.generation,
    evaluations: last.nfev,
    history: gens.map((g) => g.best.fitness),
    exposition: [record(plane, 'CMA-ES', gens)],
  }
}
