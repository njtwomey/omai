/**
 * Samplers, the strategies that propose a study's next configuration from its finished trials, Optuna style: grid
 * search, random search, quasi-random Sobol', Halton and Latin hypercube designs, CMA-ES and Nelder-Mead from `optim`
 * on the unit-cube encoding of the space, and an evolutionary sampler (NSGA-II for several objectives).
 */

import { type Json, MdmlError } from '../base/index.ts'
import { cmaesAlgorithm, crowdingDistance, nelderMeadAlgorithm, nonDominatedSort } from '../optim/index.ts'
import {
  foldIn,
  halton,
  type Key,
  latinHypercube,
  key as makeKey,
  permutation,
  scrambledSobol,
  sobol,
  uniform,
} from '../random/index.ts'
import { polynomialMutation, simulatedBinaryCrossover, tournamentSelection, uniformCrossover } from '../search/index.ts'
import { decode, encode, grid, type Space, type SpaceValue, sample } from '../space/index.ts'
import type { Value } from '../tensor/index.ts'
import type { Algorithm } from '../trace/index.ts'
import { equalMassCoordinate, fromCube, type Leaf, leavesOf } from './_leaves.ts'
import { replayAsk } from './_replay.ts'
import { lossMatrix, lossOf, type Study, type StudyDirection, type Trial } from './study.ts'

/**
 * The sampler contract: propose the next configuration of a study from its finished trials (Optuna's `BaseSampler`).
 * Samplers are pure functions of the study and a key, so a study can be resumed, replayed and run in slices without
 * sampler state. `ask` returns `undefined` when the sampler has nothing left to propose (a finished grid).
 *
 * @concept hyperparameter-optimisation
 */
export interface Sampler {
  readonly name: string
  readonly config: Json
  /** Whether the sampler handles several objectives. */
  readonly multiObjective: boolean
  ask(study: Study, key: Key): SpaceValue | undefined
  /** Whether nothing is left to propose (lets a study stop right after the last grid point). */
  exhausted?(study: Study): boolean
}

/**
 * The trials a sampler proposed, in order.
 *
 * @param study - The study.
 * @param name - The sampler's name.
 * @returns Its trials.
 * @example
 * ```ts
 * import { createStudy, ownTrials } from 'mdml-engine/tune'
 * import { real } from 'mdml-engine/space'
 * ownTrials(createStudy(real(0, 1)), 'random') // => []
 * ```
 */
export function ownTrials(study: Study, name: string): Trial[] {
  return study.trials.filter((t) => t.source === name)
}

/**
 * Exhaustive search over a grid (scikit-learn's `ParameterGrid` order: the last key of a dict varies fastest, so
 * listing keys alphabetically gives scikit-learn's order exactly): `points` values per continuous dimension (log
 * spaced for log dimensions), every option of choices. The study stops after the last configuration.
 *
 * @param options - Points per continuous dimension (default 5) and a key to visit the grid in random order.
 * @returns The sampler.
 * @concept grid-search
 * @example
 * ```ts
 * import { gridSampler, createStudy } from 'mdml-engine/tune'
 * import { dict, choice } from 'mdml-engine/space'
 * import { key } from 'mdml-engine/random'
 * gridSampler().ask(createStudy(dict({ a: choice([1, 2]), b: choice(['x', 'y']) })), key(0)) // => { a: 1, b: 'x' }
 * ```
 */
export function gridSampler(options: { readonly points?: number; readonly shuffle?: Key } = {}): Sampler {
  const points = options.points ?? 5
  const cache = new WeakMap<Space, SpaceValue[]>()
  const configs = (space: Space): SpaceValue[] => {
    const hit = cache.get(space)
    if (hit !== undefined) return hit
    const all = grid(space, points)
    const order =
      options.shuffle === undefined
        ? all
        : Array.from(permutation(options.shuffle, all.length).data, (i) => all[i] as SpaceValue)
    cache.set(space, order)
    return order
  }
  return {
    name: 'grid',
    config: { points, shuffle: options.shuffle !== undefined },
    multiObjective: true,
    ask: (study) => configs(study.space)[ownTrials(study, 'grid').length],
    exhausted: (study) => ownTrials(study, 'grid').length >= configs(study.space).length,
  }
}

/**
 * Independent uniform draws from the space (log-uniform on log dimensions; Optuna's `RandomSampler`, scikit-learn's
 * `ParameterSampler`).
 *
 * @returns The sampler.
 * @concept random-search
 * @example
 * ```ts
 * import { randomSampler, createStudy } from 'mdml-engine/tune'
 * import { real } from 'mdml-engine/space'
 * import { key } from 'mdml-engine/random'
 * const x = randomSampler().ask(createStudy(real(2, 3)), key(0)) as number
 * x >= 2 && x <= 3 // => true
 * ```
 */
export function randomSampler(): Sampler {
  return { name: 'random', config: {}, multiObjective: true, ask: (study, k) => sample(study.space, k) }
}

/**
 * Quasi-random search along the Sobol' sequence (Optuna's `QMCSampler(qmc_type='sobol')`): trial i is point
 * `skip + i` of the sequence in the space's unit-cube coordinates, read through equal-mass cells for integers and
 * options. Unscrambled by default (the points are a fixed sequence, as Optuna's default); with `scramble`, a keyed
 * linear-matrix scramble.
 *
 * @param options - The first index (default 0) and a scrambling key.
 * @returns The sampler.
 * @concept quasi-random
 * @example
 * ```ts
 * import { sobolSampler, createStudy } from 'mdml-engine/tune'
 * import { dict, real } from 'mdml-engine/space'
 * import { key } from 'mdml-engine/random'
 * sobolSampler({ skip: 1 }).ask(createStudy(dict({ x: real(0, 4), y: real(0, 4) })), key(0)) // => { x: 2, y: 2 }
 * ```
 */
export function sobolSampler(options: { readonly skip?: number; readonly scramble?: Key } = {}): Sampler {
  const skip = options.skip ?? 0
  return {
    name: 'sobol',
    config: { skip, scramble: options.scramble !== undefined },
    multiObjective: true,
    ask: (study) => {
      const leaves = leavesOf(study.space)
      const i = skip + ownTrials(study, 'sobol').length
      const d = Math.max(1, leaves.length)
      const u =
        options.scramble === undefined ? sobol(1, d, { skip: i }) : scrambledSobol(options.scramble, 1, d, { skip: i })
      return fromCube(study.space, leaves, u.data)
    },
  }
}

/**
 * Quasi-random search along the Halton sequence (Optuna's `QMCSampler(qmc_type='halton', scramble=False)`): trial i is
 * point `skip + i` (default skip 0, the origin first, as scipy's unscrambled Halton).
 *
 * @param options - The first index (default 0).
 * @returns The sampler.
 * @concept quasi-random
 * @example
 * ```ts
 * import { haltonSampler, createStudy } from 'mdml-engine/tune'
 * import { dict, real } from 'mdml-engine/space'
 * import { key } from 'mdml-engine/random'
 * haltonSampler({ skip: 1 }).ask(createStudy(dict({ x: real(0, 6), y: real(0, 6) })), key(0)) // => { x: 3, y: 2 }
 * ```
 */
export function haltonSampler(options: { readonly skip?: number } = {}): Sampler {
  const skip = options.skip ?? 0
  return {
    name: 'halton',
    config: { skip },
    multiObjective: true,
    ask: (study) => {
      const leaves = leavesOf(study.space)
      const i = skip + ownTrials(study, 'halton').length
      return fromCube(study.space, leaves, halton(1, Math.max(1, leaves.length), { skip: i }).data)
    },
  }
}

/**
 * Latin hypercube search: trials come in batches of `n`, each batch a Latin hypercube sample (every axis split into n
 * strata, one point per stratum) in the space's unit-cube coordinates, read through equal-mass cells.
 *
 * @param options - The batch size n (default 10) and the key of the batches (default `key(0)`).
 * @returns The sampler.
 * @concept quasi-random
 * @example
 * ```ts
 * import { latinHypercubeSampler, tune, functionObjective } from 'mdml-engine/tune'
 * import { real } from 'mdml-engine/space'
 * const study = tune(functionObjective((x) => x as number), real(0, 1), { sampler: latinHypercubeSampler({ n: 4 }), nTrials: 4 }).study
 * study.trials.map((t) => Math.floor((t.params as number) * 4)).sort() // => [0, 1, 2, 3]
 * ```
 */
export function latinHypercubeSampler(options: { readonly n?: number; readonly key?: Key } = {}): Sampler {
  const n = options.n ?? 10
  if (!(Number.isInteger(n) && n >= 1)) throw new MdmlError('value', 'latinHypercubeSampler: n must be positive')
  const base = options.key ?? makeKey(0)
  return {
    name: 'latinHypercube',
    config: { n },
    multiObjective: true,
    ask: (study) => {
      const leaves = leavesOf(study.space)
      const i = ownTrials(study, 'latinHypercube').length
      const d = Math.max(1, leaves.length)
      const batch = latinHypercube(foldIn(base, Math.floor(i / n)), n, d)
      const row = i % n
      return fromCube(study.space, leaves, batch.data.slice(row * d, (row + 1) * d))
    },
  }
}

/** The single-objective losses the optimiser-backed samplers are told: failures get a penalty worse than every value. */
function toldLosses(study: Study, trials: readonly Trial[]): number[] {
  const d = study.directions[0] as StudyDirection
  const losses = trials.map((t) => (t.state === 'complete' ? lossOf(d, t.values[0] ?? Number.NaN) : Number.NaN))
  const finite = losses.filter(Number.isFinite)
  const worst = finite.length === 0 ? 0 : finite.reduce((a, b) => Math.max(a, b))
  const best = finite.length === 0 ? 0 : finite.reduce((a, b) => Math.min(a, b))
  const penalty = worst + Math.max(1, worst - best)
  return losses.map((v) => (Number.isFinite(v) ? v : penalty))
}

function startOf(space: Space, x0: SpaceValue | undefined, n: number): number[] {
  return x0 === undefined ? new Array<number>(n).fill(0.5) : encode(space, x0)
}

function optimiserSampler(
  name: string,
  config: Json,
  make: (f: (x: unknown) => Value, x0: number[], restart: number) => Algorithm<unknown>,
  options: { readonly x0?: SpaceValue; readonly key?: Key },
): Sampler {
  const base = options.key ?? makeKey(0)
  return {
    name,
    config,
    multiObjective: false,
    ask: (study, k) => {
      const own = ownTrials(study, name)
      const n = Math.max(1, leavesOf(study.space).length)
      const x0 = startOf(study.space, options.x0, n)
      const x = replayAsk((f, restart) => make(f, x0, restart), toldLosses(study, own), base)
      return x === undefined ? sample(study.space, k) : decode(study.space, x)
    },
  }
}

/** Options of {@link cmaesSampler}. */
export interface CmaesSamplerOptions {
  /** The initial mean as a configuration (default the centre of the unit cube). */
  readonly x0?: SpaceValue
  /** The initial step size in unit-cube coordinates (default 1/6, Optuna's choice). */
  readonly sigma0?: number
  /** Offspring per generation (default 4 + floor(3 ln n)). */
  readonly popSize?: number
  /** Restarts double the population (IPOP-CMA-ES; default true). */
  readonly ipop?: boolean
  /** The key of the evolution strategy's draws (default `key(0)`). */
  readonly key?: Key
}

/**
 * CMA-ES as a sampler (Optuna's `CmaEsSampler`): the evolution strategy of `optim` searches the unit-cube encoding of
 * the space (in box bounds), each offspring one trial; the configuration is decoded by rounding. Failed and pruned
 * trials count as worse than every value. When the strategy converges it restarts from the centre (with a doubled
 * population under IPOP).
 *
 * @param options - Start, step size, population, IPOP restarts and key.
 * @returns The sampler (single objective).
 * @concept evolution-strategies
 * @example
 * ```ts
 * import { cmaesSampler, functionObjective, tune } from 'mdml-engine/tune'
 * import { box } from 'mdml-engine/space'
 * const sphere = functionObjective((x) => (x as number[]).reduce((s, v) => s + (v - 0.3) ** 2, 0))
 * (tune(sphere, box(-1, 1, [3]), { sampler: cmaesSampler(), nTrials: 300 }).best?.values[0] ?? 1) < 1e-4 // => true
 * ```
 */
export function cmaesSampler(options: CmaesSamplerOptions = {}): Sampler {
  const sigma0 = options.sigma0 ?? 1 / 6
  const ipop = options.ipop ?? true
  return optimiserSampler(
    'cmaes',
    { sigma0, popSize: options.popSize ?? null, ipop },
    (f, x0, restart) => {
      const n = x0.length
      const lambda = (options.popSize ?? 4 + Math.floor(3 * Math.log(n))) * (ipop ? 2 ** restart : 1)
      return cmaesAlgorithm(f, x0, {
        sigma0,
        popSize: lambda,
        bounds: { lower: 0, upper: 1 },
        tolFun: 1e-12,
        tolX: 1e-9,
      }) as never
    },
    options,
  )
}

/**
 * Nelder-Mead as a sampler: the simplex method of `optim` on the unit-cube encoding (clipped to the cube), starting
 * from a simplex of x0 and x0 plus a quarter of the range along each axis; each evaluation is one trial. It restarts
 * when the simplex collapses. Suited to a few continuous dimensions.
 *
 * @param options - Start (default the centre) and initial edge in unit-cube coordinates (default 0.25).
 * @returns The sampler (single objective).
 * @concept derivative-free-optimisation
 * @example
 * ```ts
 * import { nelderMeadSampler, functionObjective, tune } from 'mdml-engine/tune'
 * import { dict, real } from 'mdml-engine/space'
 * const f = functionObjective((p) => { const { x, y } = p as { x: number; y: number }; return (x - 1) ** 2 + (y + 1) ** 2 })
 * (tune(f, dict({ x: real(-3, 3), y: real(-3, 3) }), { sampler: nelderMeadSampler(), nTrials: 120 }).best?.values[0] ?? 1) < 1e-4 // => true
 * ```
 */
export function nelderMeadSampler(options: { readonly x0?: SpaceValue; readonly edge?: number } = {}): Sampler {
  const edge = options.edge ?? 0.25
  return optimiserSampler(
    'nelderMead',
    { edge },
    (f, x0, restart) => {
      const scale = edge / 2 ** restart
      const simplex = [
        x0,
        ...x0.map((_, i) => x0.map((v, j) => (j === i ? (v + scale <= 1 ? v + scale : v - scale) : v))),
      ]
      return nelderMeadAlgorithm(f, x0, {
        initialSimplex: simplex,
        bounds: { lower: 0, upper: 1 },
        xatol: 1e-6,
        fatol: 1e-10,
        maxEvaluations: Number.POSITIVE_INFINITY,
      }) as never
    },
    options,
  )
}

/** Options of {@link evolutionarySampler}. */
export interface EvolutionarySamplerOptions {
  /** Trials per generation (default 20). */
  readonly populationSize?: number
  /** `'uniform'` crossover (default, as Optuna's NSGA-II) or simulated binary crossover `'sbx'`. */
  readonly crossover?: 'uniform' | 'sbx'
  /** Probability of crossing two parents (default 0.9). */
  readonly crossoverProbability?: number
  /** Probability of taking each coordinate from the second parent in uniform crossover (default 0.5). */
  readonly swappingProbability?: number
  /** SBX distribution index (default 15). */
  readonly crossoverEta?: number
  /**
   * `'resample'` (default, as Optuna: a coordinate is redrawn uniformly over the space) or bounded `'polynomial'`
   * mutation.
   */
  readonly mutation?: 'resample' | 'polynomial'
  /** Polynomial mutation distribution index (default 20). */
  readonly mutationEta?: number
  /** Per-coordinate mutation rate (default 1 / dimension). */
  readonly mutationRate?: number
  /** Tournament size (default 2, binary tournaments). */
  readonly tournamentSize?: number
}

/**
 * A generational evolutionary sampler (Optuna's `NSGAIISampler` for several objectives, a real-coded genetic algorithm
 * for one), built from the operators of `search`: generation 0 is random; each later trial breeds two parents chosen by
 * tournament from the elite (the best `populationSize` trials of earlier generations: by value, or by non-dominated
 * rank and crowding distance) in the unit-cube encoding, by uniform crossover (or simulated binary crossover) and
 * mutation that redraws coordinates uniformly (or moves them polynomially).
 *
 * @param options - Population size and operator settings.
 * @returns The sampler (single or multi-objective).
 * @concept evolutionary-algorithms
 * @example
 * ```ts
 * import { evolutionarySampler, functionObjective, tune } from 'mdml-engine/tune'
 * import { box } from 'mdml-engine/space'
 * const sphere = functionObjective((x) => (x as number[]).reduce((s, v) => s + v * v, 0))
 * (tune(sphere, box(-2, 2, [2]), { sampler: evolutionarySampler({ populationSize: 10 }), nTrials: 200, seed: 0 }).best?.values[0] ?? 1) < 0.01 // => true
 * ```
 */
export function evolutionarySampler(options: EvolutionarySamplerOptions = {}): Sampler {
  const pop = options.populationSize ?? 20
  if (!(Number.isInteger(pop) && pop >= 2))
    throw new MdmlError('value', 'evolutionarySampler: populationSize must be an integer of at least 2')
  const crossover = options.crossover ?? 'uniform'
  const mutation = options.mutation ?? 'resample'
  const pc = options.crossoverProbability ?? 0.9
  const swap = options.swappingProbability ?? 0.5
  const etaC = options.crossoverEta ?? 15
  const etaM = options.mutationEta ?? 20
  const size = options.tournamentSize ?? 2
  return {
    name: 'evolutionary',
    config: {
      populationSize: pop,
      crossover,
      mutation,
      crossoverProbability: pc,
      swappingProbability: swap,
      crossoverEta: etaC,
      mutationEta: etaM,
      tournamentSize: size,
    },
    multiObjective: true,
    ask: (study, k) => {
      const own = ownTrials(study, 'evolutionary')
      const generation = Math.floor(own.length / pop)
      const earlier = own.slice(0, generation * pop).filter((t) => t.state === 'complete')
      if (generation === 0 || earlier.length < 2) return sample(study.space, k)
      const { elite, fitness } = selectElite(study, earlier, pop)
      const [a, b] = tournamentSelection(foldIn(k, 0), fitness, 2, size).map((i) =>
        encode(study.space, (elite[i] as Trial).params),
      )
      const pa = a as number[]
      const pb = b as number[]
      const zeros = pa.map(() => 0)
      const ones = pa.map(() => 1)
      const u = uniform(foldIn(k, 1), [1]).data[0] ?? 0
      let child = pa
      if (u < pc)
        child =
          crossover === 'uniform'
            ? uniformCrossover(foldIn(k, 2), pa, pb, swap)[0]
            : simulatedBinaryCrossover(foldIn(k, 2), pa, pb, { eta: etaC, lower: zeros, upper: ones })[0]
      const rate = options.mutationRate ?? 1 / pa.length
      if (mutation === 'polynomial')
        return decode(study.space, polynomialMutation(foldIn(k, 3), child, zeros, ones, { eta: etaM, rate }))
      const leaves = leavesOf(study.space)
      const draws = uniform(foldIn(k, 3), [2 * child.length]).data
      const mutated = child.map((v, i) =>
        (draws[2 * i] ?? 1) < rate ? equalMassCoordinate(leaves[i] as Leaf, draws[2 * i + 1] ?? 0.5) : v,
      )
      return decode(study.space, mutated)
    },
  }
}

/** The best `pop` trials (by loss, or NSGA-II rank then crowding) and tournament fitnesses (higher is better). */
function selectElite(study: Study, trials: readonly Trial[], pop: number): { elite: Trial[]; fitness: number[] } {
  const losses = lossMatrix(study, trials)
  if (study.directions.length === 1) {
    const order = trials.map((_, i) => i).sort((i, j) => (losses[i]?.[0] ?? 0) - (losses[j]?.[0] ?? 0) || i - j)
    const keep = order.slice(0, pop)
    return { elite: keep.map((i) => trials[i] as Trial), fitness: keep.map((i) => -(losses[i]?.[0] ?? 0)) }
  }
  const elite: Trial[] = []
  const fitness: number[] = []
  for (const [rank, front] of nonDominatedSort(losses).entries()) {
    const crowd = crowdingDistance(front.map((i) => losses[i] as number[]))
    const order = front.map((_, j) => j).sort((x, y) => (crowd[y] ?? 0) - (crowd[x] ?? 0) || x - y)
    for (const j of order) {
      if (elite.length >= pop) break
      const c = crowd[j] ?? 0
      elite.push(trials[front[j] as number] as Trial)
      fitness.push(-rank + (Number.isFinite(c) ? c / (1 + c) : 1))
    }
    if (elite.length >= pop) break
  }
  return { elite, fitness }
}
