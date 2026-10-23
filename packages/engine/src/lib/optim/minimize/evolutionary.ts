import { WhiteboxError } from '../../base/index.ts'
import { eigh } from '../../numerics/linalg/index.ts'
import { Generator, halton, latinHypercube } from '../../random/index.ts'
import { fromStorage, toFlat, type Value } from '../../tensor/index.ts'
import type { Algorithm, StepContext } from '../../trace/index.ts'
import { asPairs, boundArrays, clipInto, constraintValues, finiteBounds } from './_bounds.ts'
import { defaultLocal } from './_local.ts'
import { configOf, observeVector, runVector, type VectorState, vectorDone, vectorSchema } from './_shared.ts'
import { cobyla, type InequalityConstraint } from './cobyla.ts'
import type { Bounds } from './derivative-free.ts'
import type { LocalMethod } from './global.ts'
import {
  type FlatObjective,
  flatObjective,
  type MinimizeResult,
  type Objective,
  type Parameters,
  type VectorMethodOptions,
} from './objective.ts'

const EMPTY = new Float64Array(0)

const squaredDistance = (a: ArrayLike<number>, b: ArrayLike<number>): number => {
  let s = 0
  for (let i = 0; i < a.length; i++) s += ((a[i] ?? 0) - (b[i] ?? 0)) ** 2
  return s
}

// ---------------------------------------------------------------------------------------------------------------
// Differential evolution

/** A differential evolution strategy (scipy's names): base vector, number of differences and crossover. */
export type DifferentialEvolutionStrategy =
  | 'best1bin'
  | 'best1exp'
  | 'rand1bin'
  | 'rand1exp'
  | 'randtobest1bin'
  | 'randtobest1exp'
  | 'currenttobest1bin'
  | 'currenttobest1exp'
  | 'best2bin'
  | 'best2exp'
  | 'rand2bin'
  | 'rand2exp'

/** Options of {@link differentialEvolution}. */
export interface DifferentialEvolutionOptions extends VectorMethodOptions {
  /** Box bounds (required, finite): the search space. */
  readonly bounds: Bounds
  /** The strategy (default `best1bin`). */
  readonly strategy?: DifferentialEvolutionStrategy
  /** Population size as a multiple of the dimension (default 15; at least 5 members). */
  readonly popSize?: number
  /** The differential weight F, or a range [lo, hi) drawn once per generation (dithering; default [0.5, 1]). */
  readonly mutation?: number | readonly [number, number]
  /** The crossover probability CR (default 0.7). */
  readonly recombination?: number
  /** Stop when std(energies) <= atol + tol |mean(energies)| (defaults 0 and 0.01). */
  readonly tol?: number
  readonly atol?: number
  /** Initial population: `latinhypercube` (default), `random`, `halton`, or the points themselves. */
  readonly init?: 'latinhypercube' | 'random' | 'halton' | readonly (readonly number[])[]
  /** Replace the first member by x0 (scipy's x0 option; default false). */
  readonly includeX0?: boolean
  /** Polish the best member with a local method at the end (default true: bounded Nelder-Mead, or COBYLA with constraints). */
  readonly polish?: boolean | LocalMethod
  /** Replace members as soon as a better trial is found (`immediate`, default) or once per generation (`deferred`). */
  readonly updating?: 'immediate' | 'deferred'
  /** Inequality constraints c(x) >= 0, handled by Lampinen's feasibility rules. */
  readonly constraints?: readonly InequalityConstraint[]
  /** The evaluation budget (default unlimited). */
  readonly maxEvaluations?: number
}

/** State of a differential evolution run. */
export interface DifferentialEvolutionState extends VectorState {
  /** The population in the unit cube, best member first. */
  readonly population: readonly Float64Array[]
  readonly energies: Float64Array
  /** Constraint violations of each member (one value per constraint; empty without constraints). */
  readonly violations: readonly Float64Array[]
  /** std(energies) / |mean(energies)|, scipy's convergence measure. */
  readonly convergence: number
  readonly polished: boolean
}

function acceptTrial(eT: number, cvT: Float64Array, eO: number, cvO: Float64Array): boolean {
  const fT = cvT.every((v) => v <= 0)
  const fO = cvO.every((v) => v <= 0)
  if (fT && fO) return eT <= eO
  if (fT && !fO) return true
  if (!fT && cvT.every((v, i) => v <= (cvO[i] ?? 0))) return true
  return false
}

function promote(population: Float64Array[], energies: Float64Array, violations: Float64Array[]): void {
  let l = -1
  for (let i = 0; i < population.length; i++) {
    if (!(violations[i] as Float64Array).every((v) => v <= 0)) continue
    if (l < 0 || (energies[i] ?? 0) < (energies[l] ?? 0)) l = i
  }
  if (l < 0) {
    let best = Number.POSITIVE_INFINITY
    for (let i = 0; i < population.length; i++) {
      const total = (violations[i] as Float64Array).reduce((a, b) => a + b, 0)
      if (total < best) {
        best = total
        l = i
      }
    }
  }
  if (l <= 0) return
  ;[population[0], population[l]] = [population[l] as Float64Array, population[0] as Float64Array]
  ;[energies[0], energies[l]] = [energies[l] ?? 0, energies[0] ?? 0]
  ;[violations[0], violations[l]] = [violations[l] as Float64Array, violations[0] as Float64Array]
}

function stats(energies: Float64Array): { mean: number; std: number } {
  let mean = 0
  for (const e of energies) mean += e
  mean /= energies.length
  let v = 0
  for (const e of energies) v += (e - mean) ** 2
  return { mean, std: Math.sqrt(v / energies.length) }
}

/**
 * Differential evolution (Storn and Price 1997) as a traced algorithm, following scipy.optimize.differential_evolution:
 * a population in the unit cube scaled to the bounds, the twelve mutation strategies with binomial or exponential
 * crossover, dithered mutation, immediate or deferred updating, constraints by feasibility rules, the convergence test
 * std(E) <= atol + tol |mean(E)|, and a final polish by a local method as a child run. Each step is one generation.
 *
 * @param f - The objective.
 * @param x0 - A point fixing the parameter structure (and the first member with `includeX0`).
 * @param options - Bounds and the settings above.
 * @returns The algorithm.
 * @concept differential-evolution
 * @example
 * ```ts
 * import { differentialEvolutionAlgorithm } from 'whitebox-engine/optim'
 * import { square, sum, type Value } from 'whitebox-engine/tensor'
 * differentialEvolutionAlgorithm((x) => sum(square(x as Value)), [0, 0], { bounds: [[-1, 1], [-1, 1]] }).name // => 'differentialEvolution'
 * ```
 */
export function differentialEvolutionAlgorithm(
  f: Objective,
  x0: Parameters,
  options: DifferentialEvolutionOptions,
): Algorithm<DifferentialEvolutionState> {
  const obj = flatObjective(f, x0)
  const n = obj.n
  if (n === 0) throw new WhiteboxError('value', 'differentialEvolution: no parameters')
  const box = finiteBounds(options.bounds, n, 'differentialEvolution')
  const mid = Float64Array.from(box.lower, (l, i) => 0.5 * (l + (box.upper[i] ?? 0)))
  const span = Float64Array.from(box.lower, (l, i) => Math.abs(l - (box.upper[i] ?? 0)))
  const scale = (p: Float64Array): Float64Array =>
    Float64Array.from(p, (v, i) => (mid[i] ?? 0) + (v - 0.5) * (span[i] ?? 0))
  const unscale = (x: ArrayLike<number>): Float64Array =>
    Float64Array.from({ length: n }, (_, i) => ((x[i] ?? 0) - (mid[i] ?? 0)) / (span[i] ?? 1) + 0.5)
  const strategy = options.strategy ?? 'best1bin'
  const members = typeof options.init === 'object' ? options.init.length : Math.max(5, (options.popSize ?? 15) * n)
  const mutation = options.mutation ?? [0.5, 1]
  const cr = options.recombination ?? 0.7
  const tol = options.tol ?? 0.01
  const atol = options.atol ?? 0
  const maxfun = options.maxEvaluations ?? Number.POSITIVE_INFINITY
  const maxGenerations = options.maxIterations ?? 1000
  const constraints = options.constraints ?? []
  const deferred = options.updating === 'deferred'
  const binomial = strategy.endsWith('bin')
  const polishWith: LocalMethod | undefined =
    options.polish === false
      ? undefined
      : typeof options.polish === 'function'
        ? options.polish
        : constraints.length > 0
          ? (g, start, o = {}) =>
              cobyla(g, start, {
                ...o,
                constraints: constraints.map((c) => (v) => c(obj.unravel(toFlat(v as Value)))),
                rhobeg: 0.1,
                rhoend: 1e-9,
              })
          : defaultLocal
  if (members < (strategy.includes('2') ? 6 : 4))
    throw new WhiteboxError('value', 'differentialEvolution: the population is too small for the strategy')

  const violation = (x: Float64Array): Float64Array => {
    if (constraints.length === 0) return EMPTY
    const out: number[] = []
    for (const c of constraints) for (const v of constraintValues(c, obj, x)) out.push(Math.max(0, -v))
    return Float64Array.from(out)
  }
  const select = (rng: Generator, candidate: number, k: number): number[] => {
    // A random permutation (Fisher-Yates), excluding the candidate.
    const order = Array.from({ length: members }, (_, i) => i)
    const u = rng.uniform([members]).data
    for (let i = members - 1; i > 0; i--) {
      const j = Math.floor((u[i] ?? 0) * (i + 1))
      ;[order[i], order[j]] = [order[j] as number, order[i] as number]
    }
    return order.filter((i) => i !== candidate).slice(0, k)
  }
  const mutate = (rng: Generator, pop: readonly Float64Array[], candidate: number, F: number): Float64Array => {
    const fill = Math.min(n - 1, Math.floor((rng.uniform([1]).data[0] ?? 0) * n))
    const [r0 = 0, r1 = 0, r2 = 0, r3 = 0, r4 = 0] = select(rng, candidate, 5)
    const p = (i: number): Float64Array => pop[i] as Float64Array
    const best = p(0)
    const current = p(candidate)
    const bprime = Float64Array.from({ length: n }, (_, j) => {
      const at = (i: number): number => p(i)[j] ?? 0
      switch (strategy) {
        case 'best1bin':
        case 'best1exp':
          return (best[j] ?? 0) + F * (at(r0) - at(r1))
        case 'rand1bin':
        case 'rand1exp':
          return at(r0) + F * (at(r1) - at(r2))
        case 'randtobest1bin':
        case 'randtobest1exp': {
          let b = at(r0)
          b += F * ((best[j] ?? 0) - b)
          return b + F * (at(r1) - at(r2))
        }
        case 'currenttobest1bin':
        case 'currenttobest1exp':
          return (current[j] ?? 0) + F * ((best[j] ?? 0) - (current[j] ?? 0) + at(r0) - at(r1))
        case 'best2bin':
        case 'best2exp':
          return (best[j] ?? 0) + F * (at(r0) + at(r1) - at(r2) - at(r3))
        default:
          return at(r0) + F * (at(r1) + at(r2) - at(r3) - at(r4))
      }
    })
    const trial = Float64Array.from(current)
    const cross = rng.uniform([n]).data
    if (binomial) {
      for (let j = 0; j < n; j++) if ((cross[j] ?? 1) < cr || j === fill) trial[j] = bprime[j] ?? 0
    } else {
      let point = fill
      for (let i = 0; i < n && (i === 0 || (cross[i] ?? 1) < cr); i++) {
        trial[point] = bprime[point] ?? 0
        point = (point + 1) % n
      }
    }
    // Out-of-range coordinates are redrawn uniformly (scipy's _ensure_constraint).
    const fix = rng.uniform([n]).data
    for (let j = 0; j < n; j++) if ((trial[j] ?? 0) > 1 || (trial[j] ?? 0) < 0) trial[j] = fix[j] ?? 0.5
    return trial
  }

  const polishState = (s: DifferentialEvolutionState, ctx: StepContext): DifferentialEvolutionState => {
    if (polishWith === undefined) return s
    const best = scale(s.population[0] as Float64Array)
    const r = polishWith((v) => obj.ofValue(v as Value), fromStorage(best, [n], 'float64'), {
      bounds: asPairs(box),
      parent: ctx.run,
      key: ctx.key,
    })
    const x = Float64Array.from(toFlat(r.x as Value))
    const inside = x.every((v, i) => v >= (box.lower[i] ?? 0) && v <= (box.upper[i] ?? 0))
    const cv = violation(x)
    // A local constrained method ends on the boundary up to rounding: accept violations up to 1e-9.
    const feasible = cv.every((v) => v <= 1e-9)
    if (!(r.fun < s.value && r.status !== 'failed' && inside && feasible))
      return { ...s, nfev: s.nfev + r.nfev, polished: true }
    const population = [...s.population]
    population[0] = unscale(x)
    const energies = Float64Array.from(s.energies)
    energies[0] = r.fun
    const violations = [...s.violations]
    violations[0] = cv
    return { ...s, x, value: r.fun, population, energies, violations, nfev: s.nfev + r.nfev, polished: true }
  }

  const finish = (
    population: Float64Array[],
    energies: Float64Array,
    violations: Float64Array[],
    nfev: number,
    previous: Float64Array,
  ): Omit<DifferentialEvolutionState, 'stop'> => {
    const allFinite = energies.every(Number.isFinite)
    const { mean, std } = stats(energies)
    const x = scale(population[0] as Float64Array)
    return {
      x,
      value: energies[0] ?? Number.NaN,
      grad: EMPTY,
      nfev,
      ngev: 0,
      nhev: 0,
      step: Math.sqrt(squaredDistance(x, previous)),
      population,
      energies,
      violations,
      convergence: allFinite ? std / (Math.abs(mean) + 2.220446049250313e-16) : Number.POSITIVE_INFINITY,
      polished: false,
    }
  }

  return {
    name: 'differentialEvolution',
    config: configOf({
      strategy,
      members,
      recombination: cr,
      tol,
      atol,
      updating: deferred ? 'deferred' : 'immediate',
      mutation: typeof mutation === 'number' ? mutation : [...mutation],
      constraints: constraints.length,
    }),
    schema: vectorSchema({
      metrics: {
        convergence: { description: 'std / |mean| of the population energies' },
        meanEnergy: { description: 'mean objective of the population' },
      },
      tensors: { population: { role: 'state', description: 'the population, scaled to the bounds' } },
    }),
    init: (key) => {
      let unit: Float64Array[]
      const init = options.init ?? 'latinhypercube'
      if (typeof init !== 'string') {
        if (init.length < 5 || init.some((p) => p.length !== n))
          throw new WhiteboxError('shape', `differentialEvolution: init needs at least 5 points of length ${n}`)
        unit = init.map((p) => clipInto(unscale(p), { lower: new Float64Array(n), upper: new Float64Array(n).fill(1) }))
      } else {
        const t =
          init === 'random'
            ? new Generator(key).uniform([members, n])
            : init === 'halton'
              ? halton(members, n)
              : latinHypercube(key, members, n)
        unit = Array.from({ length: members }, (_, i) => t.data.slice(i * n, (i + 1) * n))
      }
      if (options.includeX0 === true)
        unit[0] = clipInto(unscale(obj.x0), { lower: new Float64Array(n), upper: new Float64Array(n).fill(1) })
      const violations = unit.map((p) => violation(scale(p)))
      let nfev = 0
      const energies = Float64Array.from(unit, (p, i) => {
        if (!(violations[i] as Float64Array).every((v) => v <= 0) || nfev >= maxfun) return Number.POSITIVE_INFINITY
        nfev += 1
        return obj.value(scale(p))
      })
      promote(unit, energies, violations)
      return finish(unit, energies, violations, nfev, scale(unit[0] as Float64Array))
    },
    step: (s, ctx) => {
      const rng = new Generator(ctx.key)
      const F =
        typeof mutation === 'number'
          ? mutation
          : mutation[0] + (rng.uniform([1]).data[0] ?? 0) * (mutation[1] - mutation[0])
      const population = [...s.population]
      const energies = Float64Array.from(s.energies)
      const violations = [...s.violations]
      let nfev = s.nfev
      let budget = false
      const evaluate = (trial: Float64Array): { e: number; cv: Float64Array } => {
        const x = scale(trial)
        const cv = violation(x)
        if (!cv.every((v) => v <= 0)) return { e: Number.POSITIVE_INFINITY, cv }
        nfev += 1
        return { e: obj.value(x), cv }
      }
      if (deferred) {
        const trials = population.map((_, c) => mutate(rng, population, c, F))
        const results = trials.map((t) =>
          nfev >= maxfun ? { e: Number.POSITIVE_INFINITY, cv: violation(scale(t)) } : evaluate(t),
        )
        for (const [c, t] of trials.entries()) {
          const r = results[c] as { e: number; cv: Float64Array }
          if (acceptTrial(r.e, r.cv, energies[c] ?? 0, violations[c] as Float64Array)) {
            population[c] = t
            energies[c] = r.e
            violations[c] = r.cv
          }
        }
        promote(population, energies, violations)
        budget = nfev >= maxfun
      } else {
        for (let c = 0; c < members; c++) {
          if (nfev >= maxfun) {
            budget = true
            break
          }
          const trial = mutate(rng, population, c, F)
          const r = evaluate(trial)
          if (acceptTrial(r.e, r.cv, energies[c] ?? 0, violations[c] as Float64Array)) {
            population[c] = trial
            energies[c] = r.e
            violations[c] = r.cv
            if (acceptTrial(r.e, r.cv, energies[0] ?? 0, violations[0] as Float64Array))
              promote(population, energies, violations)
          }
        }
      }
      const next = finish(population, energies, violations, nfev, s.x)
      const { mean, std } = stats(energies)
      const converged = energies.every(Number.isFinite) && std <= atol + tol * Math.abs(mean)
      if (converged || budget || ctx.t + 1 >= maxGenerations) {
        const polished = polishState(next as DifferentialEvolutionState, ctx)
        const feasible = (polished.violations[0] as Float64Array).every((v) => v <= 1e-9)
        return {
          ...polished,
          stop: !feasible
            ? { status: 'failed', message: 'no member satisfies the constraints' }
            : converged
              ? { status: 'converged', message: 'population energies converged (std <= atol + tol |mean|)' }
              : budget
                ? { status: 'failed', message: 'maximum number of function evaluations reached' }
                : { status: 'failed', message: 'maximum number of generations reached' },
        }
      }
      return next
    },
    // Infeasible members have infinite energy, so an infinite best value is not divergence here.
    done: (s) => s.stop?.status,
    observe: (s) => {
      const flat = new Float64Array(members * n)
      for (const [k, p] of s.population.entries()) flat.set(scale(p), k * n)
      return observeVector(s, {
        metrics: { convergence: s.convergence, meanEnergy: stats(s.energies).mean },
        tensors: { population: { role: 'state', value: flat, shape: [members, n] } },
      })
    },
  }
}

/**
 * Find a global minimum by differential evolution (scipy.optimize.differential_evolution). `maxIterations` counts
 * generations (default 1000).
 *
 * @param f - The objective.
 * @param x0 - A point fixing the parameter structure.
 * @param options - Bounds (required), strategy, population, mutation, crossover, tolerances, init, polish, updating,
 *   constraints, budget, key or seed and run options.
 * @returns The best member (polished).
 * @concept differential-evolution
 * @example
 * ```ts
 * import { differentialEvolution } from 'whitebox-engine/optim'
 * import { square, sub, sum, toFlat, type Value } from 'whitebox-engine/tensor'
 * const r = differentialEvolution((x) => sum(square(sub(x as Value, [1, -2]))), [0, 0], {
 *   bounds: [[-5, 5], [-5, 5]],
 *   seed: 0,
 * })
 * toFlat(r.x as Value).map((v) => Math.round(v * 1e6) / 1e6) // => [1, -2]
 * ```
 */
export function differentialEvolution(
  f: Objective,
  x0: Parameters,
  options: DifferentialEvolutionOptions,
): MinimizeResult {
  const obj = flatObjective(f, x0)
  return runVector(differentialEvolutionAlgorithm(f, x0, options), obj, options, 1000)
}

// ---------------------------------------------------------------------------------------------------------------
// CMA-ES

/** Options of {@link cmaes}. */
export interface CmaesOptions extends VectorMethodOptions {
  /** The initial step size (default 0.3 of the mean bound range, else 0.5). */
  readonly sigma0?: number
  /** Offspring per generation lambda (default 4 + floor(3 ln n)). */
  readonly popSize?: number
  /** Box bounds: offspring are repaired by clipping into the box. */
  readonly bounds?: Bounds
  /** Stop when the best values of recent generations and the current spread lie within this (default 1e-11). */
  readonly tolFun?: number
  /** Stop when sigma times every coordinate standard deviation falls below this (default 1e-11). */
  readonly tolX?: number
  /** The evaluation budget (default unlimited). */
  readonly maxEvaluations?: number
}

/** State of a CMA-ES run. */
export interface CmaesState extends VectorState {
  readonly mean: Float64Array
  readonly sigma: number
  /** The covariance matrix (n x n, row-major), its eigenvectors (columns) and the square roots of its eigenvalues. */
  readonly cov: Float64Array
  readonly basis: Float64Array
  readonly scales: Float64Array
  readonly pathSigma: Float64Array
  readonly pathC: Float64Array
  readonly generation: number
  /** Best value of each recent generation, for the tolFun test. */
  readonly recent: readonly number[]
  /** Spread (max - min) of the last generation's values. */
  readonly spread: number
}

/**
 * The covariance matrix adaptation evolution strategy, (mu/mu_w, lambda)-CMA-ES (Hansen and Ostermeier 2001; Hansen's
 * tutorial 2016), as a traced algorithm: offspring sampled from N(m, sigma^2 C), weighted recombination of the best mu,
 * cumulative step-size adaptation, rank-one and rank-mu covariance updates, eigendecomposition of C each generation.
 * Each step is one generation; the trace records sigma, the condition number of C and the mean.
 *
 * @param f - The objective.
 * @param x0 - The initial mean (any parameter tree).
 * @param options - Step size, population, bounds, tolerances and budget.
 * @returns The algorithm.
 * @concept evolution-strategies
 * @example
 * ```ts
 * import { cmaesAlgorithm } from 'whitebox-engine/optim'
 * import { square, sum, type Value } from 'whitebox-engine/tensor'
 * cmaesAlgorithm((x) => sum(square(x as Value)), [1, 2]).name // => 'cmaes'
 * ```
 */
export function cmaesAlgorithm(f: Objective, x0: Parameters, options: CmaesOptions = {}): Algorithm<CmaesState> {
  const obj = flatObjective(f, x0)
  const n = obj.n
  if (n === 0) throw new WhiteboxError('value', 'cmaes: no parameters')
  const box = options.bounds === undefined ? undefined : boundArrays(options.bounds, n, 'cmaes')
  const lambda = options.popSize ?? 4 + Math.floor(3 * Math.log(n))
  if (lambda < 2) throw new WhiteboxError('value', 'cmaes: popSize must be at least 2')
  const mu = Math.floor(lambda / 2)
  const raw = Array.from({ length: mu }, (_, i) => Math.log(mu + 0.5) - Math.log(i + 1))
  const total = raw.reduce((a, b) => a + b, 0)
  const weights = raw.map((w) => w / total)
  const mueff = 1 / weights.reduce((a, w) => a + w * w, 0)
  const cc = (4 + mueff / n) / (n + 4 + (2 * mueff) / n)
  const cs = (mueff + 2) / (n + mueff + 5)
  const c1 = 2 / ((n + 1.3) ** 2 + mueff)
  const cmu = Math.min(1 - c1, (2 * (mueff - 2 + 1 / mueff)) / ((n + 2) ** 2 + mueff))
  const damps = 1 + 2 * Math.max(0, Math.sqrt((mueff - 1) / (n + 1)) - 1) + cs
  const chiN = Math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
  const tolFun = options.tolFun ?? 1e-11
  const tolX = options.tolX ?? 1e-11
  const maxfun = options.maxEvaluations ?? Number.POSITIVE_INFINITY
  const history = 10 + Math.ceil((30 * n) / lambda)
  let range = 0
  if (box !== undefined) {
    let count = 0
    for (let i = 0; i < n; i++) {
      const r = (box.upper[i] ?? 0) - (box.lower[i] ?? 0)
      if (Number.isFinite(r)) {
        range += r
        count += 1
      }
    }
    range = count > 0 ? range / count : 0
  }
  const sigma0 = options.sigma0 ?? (range > 0 ? 0.3 * range : 0.5)
  const identity = (): Float64Array => Float64Array.from({ length: n * n }, (_, k) => (k % (n + 1) === 0 ? 1 : 0))

  return {
    name: 'cmaes',
    config: configOf({ lambda, mu, sigma0, tolFun, tolX, cc, cs, c1, cmu, damps }),
    schema: vectorSchema({
      metrics: {
        sigma: { description: 'global step size' },
        condition: { description: 'condition number of the covariance matrix' },
        spread: { description: 'range of the objective over the generation' },
      },
      tensors: { mean: { role: 'state', description: 'mean of the search distribution' } },
    }),
    init: () => {
      const m = box === undefined ? Float64Array.from(obj.x0) : clipInto(obj.x0, box)
      const fm = obj.value(m)
      return {
        x: m,
        value: fm,
        grad: EMPTY,
        nfev: 1,
        ngev: 0,
        nhev: 0,
        step: sigma0,
        mean: m,
        sigma: sigma0,
        cov: identity(),
        basis: identity(),
        scales: new Float64Array(n).fill(1),
        pathSigma: new Float64Array(n),
        pathC: new Float64Array(n),
        generation: 0,
        recent: [],
        spread: Number.POSITIVE_INFINITY,
      }
    },
    done: vectorDone,
    observe: (s) => {
      let lo = Number.POSITIVE_INFINITY
      let hi = 0
      for (const d of s.scales) {
        lo = Math.min(lo, d)
        hi = Math.max(hi, d)
      }
      return observeVector(s, {
        metrics: { sigma: s.sigma, condition: (hi / lo) ** 2, spread: s.spread },
        tensors: { mean: { role: 'state', value: s.mean } },
      })
    },
    step: (s: CmaesState, ctx: StepContext): CmaesState => {
      const rng = new Generator(ctx.key)
      const z = rng.normal([lambda, n]).data
      const offspring: { x: Float64Array; y: Float64Array; f: number }[] = []
      let nfev = s.nfev
      for (let k = 0; k < lambda; k++) {
        // y = B D z, x = m + sigma y (repaired into the box)
        const y = new Float64Array(n)
        for (let i = 0; i < n; i++) {
          let v = 0
          for (let j = 0; j < n; j++) v += (s.basis[i * n + j] ?? 0) * (s.scales[j] ?? 0) * (z[k * n + j] ?? 0)
          y[i] = v
        }
        let x: Float64Array = Float64Array.from(s.mean, (mi, i) => mi + s.sigma * (y[i] ?? 0))
        if (box !== undefined) {
          x = clipInto(x, box)
          for (let i = 0; i < n; i++) y[i] = ((x[i] ?? 0) - (s.mean[i] ?? 0)) / s.sigma
        }
        offspring.push({ x, y, f: obj.value(x) })
        nfev += 1
      }
      offspring.sort((a, b) => a.f - b.f)
      const bestChild = offspring[0] as { x: Float64Array; f: number }
      const mean = new Float64Array(n)
      const yw = new Float64Array(n)
      for (let k = 0; k < mu; k++) {
        const o = offspring[k] as { x: Float64Array; y: Float64Array }
        for (let i = 0; i < n; i++) {
          mean[i] = (mean[i] ?? 0) + (weights[k] ?? 0) * (o.x[i] ?? 0)
          yw[i] = (yw[i] ?? 0) + (weights[k] ?? 0) * (o.y[i] ?? 0)
        }
      }
      // C^{-1/2} yw = B D^{-1} B^T yw
      const bty = new Float64Array(n)
      for (let j = 0; j < n; j++) {
        let v = 0
        for (let i = 0; i < n; i++) v += (s.basis[i * n + j] ?? 0) * (yw[i] ?? 0)
        bty[j] = v / (s.scales[j] ?? 1)
      }
      const pathSigma = new Float64Array(n)
      const csn = Math.sqrt(cs * (2 - cs) * mueff)
      for (let i = 0; i < n; i++) {
        let v = 0
        for (let j = 0; j < n; j++) v += (s.basis[i * n + j] ?? 0) * (bty[j] ?? 0)
        pathSigma[i] = (1 - cs) * (s.pathSigma[i] ?? 0) + csn * v
      }
      const psNorm = Math.sqrt(pathSigma.reduce((a, v) => a + v * v, 0))
      const generation = s.generation + 1
      const hsig = psNorm / Math.sqrt(1 - (1 - cs) ** (2 * generation)) / chiN < 1.4 + 2 / (n + 1) ? 1 : 0
      const ccn = Math.sqrt(cc * (2 - cc) * mueff)
      const pathC = Float64Array.from(s.pathC, (p, i) => (1 - cc) * p + hsig * ccn * (yw[i] ?? 0))
      const cov = new Float64Array(n * n)
      const keep = 1 - c1 - cmu + (1 - hsig) * c1 * cc * (2 - cc)
      for (let i = 0; i < n; i++) {
        for (let j = 0; j <= i; j++) {
          let rankMu = 0
          for (let k = 0; k < mu; k++) {
            const o = offspring[k] as { y: Float64Array }
            rankMu += (weights[k] ?? 0) * (o.y[i] ?? 0) * (o.y[j] ?? 0)
          }
          const v = keep * (s.cov[i * n + j] ?? 0) + c1 * (pathC[i] ?? 0) * (pathC[j] ?? 0) + cmu * rankMu
          cov[i * n + j] = v
          cov[j * n + i] = v
        }
      }
      const sigma = s.sigma * Math.exp((cs / damps) * (psNorm / chiN - 1))
      const { eigenvalues, eigenvectors } = eigh(fromStorage(cov, [n, n], 'float64'))
      const values = toFlat(eigenvalues)
      const basis = Float64Array.from(toFlat(eigenvectors))
      const scales = Float64Array.from(values, (v) => Math.sqrt(Math.max(v, 1e-300)))
      const better = bestChild.f < s.value
      const recent = [...s.recent, bestChild.f].slice(-history)
      const spread = (offspring[lambda - 1] as { f: number }).f - bestChild.f
      let stop: CmaesState['stop']
      let recentRange = 0
      if (recent.length >= history) {
        let lo = Number.POSITIVE_INFINITY
        let hi = Number.NEGATIVE_INFINITY
        for (const v of recent) {
          lo = Math.min(lo, v)
          hi = Math.max(hi, v)
        }
        recentRange = hi - lo
      } else recentRange = Number.POSITIVE_INFINITY
      let sdMax = 0
      for (let i = 0; i < n; i++) sdMax = Math.max(sdMax, Math.sqrt(Math.max(cov[i * n + i] ?? 0, 0)))
      const pcMax = pathC.reduce((a, v) => Math.max(a, Math.abs(v)), 0)
      let dmin = Number.POSITIVE_INFINITY
      let dmax = 0
      for (const d of scales) {
        dmin = Math.min(dmin, d)
        dmax = Math.max(dmax, d)
      }
      if (Math.max(recentRange, spread) < tolFun)
        stop = { status: 'converged', message: 'tolFun: objective range below tolerance' }
      else if (sigma * Math.max(sdMax, pcMax) < tolX)
        stop = { status: 'converged', message: 'tolX: step sizes below tolerance' }
      else if ((dmax / dmin) ** 2 > 1e14) stop = { status: 'failed', message: 'condition number of C exceeds 1e14' }
      else if (nfev >= maxfun) stop = { status: 'failed', message: 'maximum number of function evaluations reached' }
      return {
        x: better ? bestChild.x : s.x,
        value: better ? bestChild.f : s.value,
        grad: EMPTY,
        nfev,
        ngev: 0,
        nhev: 0,
        step: sigma,
        mean,
        sigma,
        cov,
        basis,
        scales,
        pathSigma,
        pathC,
        generation,
        recent,
        spread,
        ...(stop === undefined ? {} : { stop }),
      }
    },
  }
}

/**
 * Minimise by CMA-ES. `maxIterations` counts generations (default 100 + 150 (n + 3)^2 / sqrt(lambda)).
 *
 * @param f - The objective.
 * @param x0 - The initial mean (any parameter tree).
 * @param options - Step size, population, bounds, tolerances, budget, key or seed and run options.
 * @returns The best point found.
 * @concept evolution-strategies
 * @example
 * ```ts
 * import { cmaes } from 'whitebox-engine/optim'
 * import { square, sub, sum, toFlat, type Value } from 'whitebox-engine/tensor'
 * const r = cmaes((x) => sum(square(sub(x as Value, [1, -2, 3]))), [0, 0, 0], { seed: 0 })
 * toFlat(r.x as Value).map((v) => Math.round(v * 1e4) / 1e4) // => [1, -2, 3]
 * ```
 */
export function cmaes(f: Objective, x0: Parameters, options: CmaesOptions = {}): MinimizeResult {
  const obj = flatObjective(f, x0)
  const lambda = options.popSize ?? 4 + Math.floor(3 * Math.log(obj.n))
  return runVector(
    cmaesAlgorithm(f, x0, options),
    obj,
    options,
    Math.ceil(100 + (150 * (obj.n + 3) ** 2) / Math.sqrt(lambda)),
  )
}

// ---------------------------------------------------------------------------------------------------------------
// Particle swarm

/** Options of {@link particleSwarm}. */
export interface ParticleSwarmOptions extends VectorMethodOptions {
  /** Box bounds (required, finite): the swarm starts uniformly in the box and stays in it. */
  readonly bounds: Bounds
  /** Number of particles (default 40, as SPSO 2011). */
  readonly swarmSize?: number
  /** Inertia weight w (default 0.7298, Clerc's constriction), or [start, end] for a linear decrease over the run. */
  readonly inertia?: number | readonly [number, number]
  /** Cognitive and social coefficients c1, c2 (default 1.49618 each). */
  readonly cognitive?: number
  readonly social?: number
  /** `global` best (default) or `ring` neighbourhoods of three particles. */
  readonly topology?: 'global' | 'ring'
  /** The largest velocity per coordinate, as a fraction of the bound range (default 0.5). */
  readonly velocityClamp?: number
  /** Stop when the best value improves by less than ftol over `patience` iterations (defaults 1e-10 and 50). */
  readonly ftol?: number
  readonly patience?: number
}

/** State of a particle swarm run. */
export interface ParticleSwarmState extends VectorState {
  readonly positions: readonly Float64Array[]
  readonly velocities: readonly Float64Array[]
  readonly personal: readonly Float64Array[]
  readonly personalValues: Float64Array
  /** Best value recorded `patience` iterations ago, and the iteration counter. */
  readonly history: readonly number[]
  readonly iteration: number
}

/**
 * Particle swarm optimisation (Kennedy and Eberhart 1995; Shi and Eberhart's inertia weight; Clerc's constriction
 * coefficients) as a traced algorithm: velocities pulled towards each particle's best and its neighbourhood's best,
 * clamped, with positions kept in the box (absorbing walls). Each step moves the whole swarm once.
 *
 * @param f - The objective.
 * @param x0 - A point fixing the parameter structure (it becomes the first particle).
 * @param options - Bounds, swarm size, coefficients, topology, velocity clamp and stagnation test.
 * @returns The algorithm.
 * @concept particle-swarm
 * @example
 * ```ts
 * import { particleSwarmAlgorithm } from 'whitebox-engine/optim'
 * import { square, sum, type Value } from 'whitebox-engine/tensor'
 * particleSwarmAlgorithm((x) => sum(square(x as Value)), [0, 0], { bounds: [[-1, 1], [-1, 1]] }).name // => 'particleSwarm'
 * ```
 */
export function particleSwarmAlgorithm(
  f: Objective,
  x0: Parameters,
  options: ParticleSwarmOptions,
): Algorithm<ParticleSwarmState> {
  const obj: FlatObjective = flatObjective(f, x0)
  const n = obj.n
  if (n === 0) throw new WhiteboxError('value', 'particleSwarm: no parameters')
  const box = finiteBounds(options.bounds, n, 'particleSwarm')
  const size = options.swarmSize ?? 40
  if (size < 2) throw new WhiteboxError('value', 'particleSwarm: swarmSize must be at least 2')
  const inertia = options.inertia ?? 0.7298
  const c1 = options.cognitive ?? 1.49618
  const c2 = options.social ?? 1.49618
  const ring = options.topology === 'ring'
  const vmax = Float64Array.from(box.upper, (u, i) => (options.velocityClamp ?? 0.5) * (u - (box.lower[i] ?? 0)))
  const ftol = options.ftol ?? 1e-10
  const patience = options.patience ?? 50
  const maxIterations = options.maxIterations ?? 1000
  const weightAt = (t: number): number =>
    typeof inertia === 'number'
      ? inertia
      : inertia[0] + ((inertia[1] - inertia[0]) * t) / Math.max(1, maxIterations - 1)

  const bestOf = (values: Float64Array): number => {
    let b = 0
    for (let i = 1; i < values.length; i++) if ((values[i] ?? 0) < (values[b] ?? 0)) b = i
    return b
  }

  return {
    name: 'particleSwarm',
    config: configOf({
      size,
      cognitive: c1,
      social: c2,
      topology: ring ? 'ring' : 'global',
      ftol,
      patience,
      inertia: typeof inertia === 'number' ? inertia : [...inertia],
    }),
    schema: vectorSchema({
      metrics: {
        meanValue: { description: 'mean objective over the particles' },
        diversity: { description: 'mean distance of the particles from their centroid' },
      },
      tensors: { positions: { role: 'state', description: 'particle positions' } },
    }),
    init: (key) => {
      const rng = new Generator(key)
      const u = rng.uniform([size, n]).data
      const w = rng.uniform([size, n]).data
      const range = (i: number): number => (box.upper[i] ?? 0) - (box.lower[i] ?? 0)
      const positions = Array.from({ length: size }, (_, k) =>
        k === 0
          ? clipInto(obj.x0, box)
          : Float64Array.from({ length: n }, (_, i) => (box.lower[i] ?? 0) + (u[k * n + i] ?? 0) * range(i)),
      )
      const velocities = Array.from({ length: size }, (_, k) =>
        Float64Array.from({ length: n }, (_, i) => 0.1 * range(i) * (2 * (w[k * n + i] ?? 0) - 1)),
      )
      const values = Float64Array.from(positions, (p) => obj.value(p))
      const b = bestOf(values)
      return {
        x: positions[b] as Float64Array,
        value: values[b] ?? Number.NaN,
        grad: EMPTY,
        nfev: size,
        ngev: 0,
        nhev: 0,
        step: 0,
        positions,
        velocities,
        personal: positions,
        personalValues: values,
        history: [values[b] ?? Number.NaN],
        iteration: 0,
      }
    },
    step: (s, ctx) => {
      const rng = new Generator(ctx.key)
      const r1 = rng.uniform([size, n]).data
      const r2 = rng.uniform([size, n]).data
      const w = weightAt(s.iteration)
      const globalBest = bestOf(s.personalValues)
      const positions: Float64Array[] = []
      const velocities: Float64Array[] = []
      for (let k = 0; k < size; k++) {
        let leader = globalBest
        if (ring) {
          const left = (k + size - 1) % size
          const right = (k + 1) % size
          leader = k
          for (const j of [left, right]) if ((s.personalValues[j] ?? 0) < (s.personalValues[leader] ?? 0)) leader = j
        }
        const x = s.positions[k] as Float64Array
        const v = s.velocities[k] as Float64Array
        const p = s.personal[k] as Float64Array
        const g = s.personal[leader] as Float64Array
        const nv = new Float64Array(n)
        const nx = new Float64Array(n)
        for (let i = 0; i < n; i++) {
          let vi =
            w * (v[i] ?? 0) +
            c1 * (r1[k * n + i] ?? 0) * ((p[i] ?? 0) - (x[i] ?? 0)) +
            c2 * (r2[k * n + i] ?? 0) * ((g[i] ?? 0) - (x[i] ?? 0))
          vi = Math.max(-(vmax[i] ?? 0), Math.min(vmax[i] ?? 0, vi))
          let xi = (x[i] ?? 0) + vi
          if (xi < (box.lower[i] ?? 0) || xi > (box.upper[i] ?? 0)) {
            xi = Math.min(Math.max(xi, box.lower[i] ?? 0), box.upper[i] ?? 0)
            vi = 0
          }
          nv[i] = vi
          nx[i] = xi
        }
        positions.push(nx)
        velocities.push(nv)
      }
      const values = Float64Array.from(positions, (p) => obj.value(p))
      const personal = s.personal.map((p, k) =>
        (values[k] ?? 0) < (s.personalValues[k] ?? 0) ? (positions[k] as Float64Array) : p,
      )
      const personalValues = Float64Array.from(s.personalValues, (pv, k) =>
        Math.min(pv, values[k] ?? Number.POSITIVE_INFINITY),
      )
      const b = bestOf(personalValues)
      const value = personalValues[b] ?? Number.NaN
      const history = [...s.history, value].slice(-(patience + 1))
      const stagnant = history.length > patience && (history[0] ?? 0) - value < ftol * Math.max(1, Math.abs(value))
      return {
        x: personal[b] as Float64Array,
        value,
        grad: EMPTY,
        nfev: s.nfev + size,
        ngev: 0,
        nhev: 0,
        step: Math.sqrt(squaredDistance(personal[b] as Float64Array, s.x)),
        positions,
        velocities,
        personal,
        personalValues,
        history,
        iteration: s.iteration + 1,
        ...(stagnant
          ? {
              stop: {
                status: 'converged' as const,
                message: `best value improved by less than ftol in ${patience} iterations`,
              },
            }
          : {}),
      }
    },
    done: vectorDone,
    observe: (s) => {
      const centroid = new Float64Array(n)
      for (const p of s.positions) for (let i = 0; i < n; i++) centroid[i] = (centroid[i] ?? 0) + (p[i] ?? 0) / size
      let diversity = 0
      let mean = 0
      for (const [k, p] of s.positions.entries()) {
        diversity += Math.sqrt(squaredDistance(p, centroid)) / size
        mean += (s.personalValues[k] ?? 0) / size
      }
      const flat = new Float64Array(size * n)
      for (const [k, p] of s.positions.entries()) flat.set(p, k * n)
      return observeVector(s, {
        metrics: { meanValue: mean, diversity },
        tensors: { positions: { role: 'state', value: flat, shape: [size, n] } },
      })
    },
  }
}

/**
 * Find a global minimum by particle swarm optimisation. `maxIterations` counts swarm moves (default 1000).
 *
 * @param f - The objective.
 * @param x0 - A point fixing the parameter structure (the first particle).
 * @param options - Bounds (required), swarm size, coefficients, topology, velocity clamp, stagnation test, key or
 *   seed and run options.
 * @returns The best position found.
 * @concept particle-swarm
 * @example
 * ```ts
 * import { particleSwarm } from 'whitebox-engine/optim'
 * import { square, sub, sum, toFlat, type Value } from 'whitebox-engine/tensor'
 * const r = particleSwarm((x) => sum(square(sub(x as Value, [1, -2]))), [0, 0], { bounds: [[-5, 5], [-5, 5]], seed: 0 })
 * toFlat(r.x as Value).map((v) => Math.round(v * 1e4) / 1e4) // => [1, -2]
 * ```
 */
export function particleSwarm(f: Objective, x0: Parameters, options: ParticleSwarmOptions): MinimizeResult {
  const obj = flatObjective(f, x0)
  return runVector(particleSwarmAlgorithm(f, x0, options), obj, options, 1000)
}
