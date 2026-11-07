/**
 * Differential evolution (Storn and Price), a population-based global optimiser that proposes candidates from scaled
 * differences between population members, following `scipy.optimize.differential_evolution` with its mutation
 * strategies, crossover, constraints and optional local polishing, as a traced algorithm.
 */

import { MdmlError } from '../../base/index.ts'
import { metricFunction } from '../../numerics/geometry/index.ts'
import { Generator, halton, latinHypercube } from '../../random/index.ts'
import { fromStorage, toFlat, type Value } from '../../tensor/index.ts'
import type { Algorithm, StepContext } from '../../trace/index.ts'
import { asPairs, clipInto, constraintValues, finiteBounds } from './_bounds.ts'
import { defaultLocal } from './_local.ts'
import { configOf, noParametersResult, observeVector, runVector, type VectorState, vectorSchema } from './_shared.ts'
import { cobyla, type InequalityConstraint } from './cobyla.ts'
import type { LocalMethod } from './global.ts'
import type { Bounds } from './objective.ts'
import {
  flatObjective,
  type MinimizeResult,
  type Objective,
  type Parameters,
  type VectorMethodOptions,
} from './objective.ts'

const EMPTY = new Float64Array(0)

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
 * import { differentialEvolutionAlgorithm } from 'mdml-engine/optim'
 * import { square, sum, type Value } from 'mdml-engine/tensor'
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
  if (n === 0) throw new MdmlError('value', 'differentialEvolution: no parameters')
  const euclid = metricFunction('euclidean', n)
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
    throw new MdmlError('value', 'differentialEvolution: the population is too small for the strategy')

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
      step: euclid(x, 0, previous, 0),
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
          throw new MdmlError('shape', `differentialEvolution: init needs at least 5 points of length ${n}`)
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
 * @index function Population-based global minimiser.
 * @example
 * ```ts
 * import { differentialEvolution } from 'mdml-engine/optim'
 * import { square, sub, sum, toFlat, type Value } from 'mdml-engine/tensor'
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
  if (obj.n === 0) return noParametersResult(obj)
  return runVector(differentialEvolutionAlgorithm(f, x0, options), obj, options, 1000)
}
