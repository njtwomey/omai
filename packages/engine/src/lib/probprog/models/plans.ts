/**
 * Method plans for the benchmark models: `nutsPlan`, `sviPlan`, `laplacePlan`, `importancePlan`, `temperedPlan`,
 * `gibbsPlan` and `exactPlan` build a `MethodPlan` (a kind, JSON knobs, the session's steps, what to expect and why)
 * with budgets that scale, so a model's `plans(scale)` reads as a list of what to run and what the result should be.
 * The numbers are modest on purpose: the unit tests run them at a small scale, the periodic campaign at scale 1.
 */

import type { Json } from '../../base/index.ts'
import type { AgreementTolerances } from '../validation/index.ts'
import type { MethodPlan } from './benchmark.ts'

/**
 * A budget scaled by the run's multiplier, rounded and kept above a floor.
 *
 * @param base - The budget at scale 1.
 * @param scale - The multiplier.
 * @param min - The smallest budget (default 5).
 * @returns The budget.
 * @example
 * ```ts
 * import { budget } from 'omai-engine/probprog/models'
 * [budget(500, 0.2), budget(500, 0.001), budget(500, 2)] // => [100, 5, 1000]
 * ```
 */
export function budget(base: number, scale: number, min = 5): number {
  return Math.max(min, Math.round(base * scale))
}

/** What every plan builder takes besides its budget. */
export interface PlanOptions {
  readonly expect?: 'agree' | 'differ' | 'report'
  readonly note?: string
  readonly quantities?: readonly string[]
  readonly tolerances?: AgreementTolerances
  /** Knobs beyond the budget (the algorithm, the target acceptance, a guide). */
  readonly config?: Readonly<Record<string, Json>>
}

function make(
  kind: string,
  config: Readonly<Record<string, Json>>,
  steps: number,
  options: PlanOptions,
  defaults: { readonly expect: 'agree' | 'differ' | 'report'; readonly note: string },
): MethodPlan {
  return {
    kind,
    config: { ...config, ...(options.config ?? {}) },
    steps,
    expect: options.expect ?? defaults.expect,
    note: options.note ?? defaults.note,
    ...(options.quantities === undefined ? {} : { quantities: options.quantities }),
    ...(options.tolerances === undefined ? {} : { tolerances: options.tolerances }),
  }
}

/**
 * NUTS with window adaptation: two chains, `warmup` and `draws` per chain (500 and 500 at scale 1 by default).
 *
 * @param scale - The budget multiplier.
 * @param options - Expectation, note, quantities, tolerances, extra knobs, and the warm-up and draws at scale 1.
 * @returns The plan.
 * @example
 * ```ts
 * import { nutsPlan } from 'omai-engine/probprog/models'
 * nutsPlan(0.2).config // => { numChains: 2, numWarmup: 100, numDraws: 100 }
 * ```
 */
export function nutsPlan(
  scale: number,
  options: PlanOptions & { readonly warmup?: number; readonly draws?: number; readonly chains?: number } = {},
): MethodPlan {
  return make(
    'nuts',
    {
      numChains: options.chains ?? 2,
      numWarmup: budget(options.warmup ?? 500, scale, 20),
      numDraws: budget(options.draws ?? 500, scale, 20),
    },
    1_000_000,
    options,
    { expect: 'agree', note: 'a consistent sampler: agrees within Monte Carlo error' },
  )
}

/**
 * Stochastic variational inference with the mean-field normal guide and Adam (`steps` steps of the optimiser).
 *
 * @param scale - The budget multiplier.
 * @param options - Expectation, note, quantities, tolerances, extra knobs and the steps at scale 1.
 * @returns The plan.
 * @example
 * ```ts
 * import { sviPlan } from 'omai-engine/probprog/models'
 * [sviPlan(1).kind, sviPlan(1).expect] // => ['svi', 'report']
 * ```
 */
export function sviPlan(scale: number, options: PlanOptions & { readonly steps?: number } = {}): MethodPlan {
  return make('svi', { learningRate: 0.05, numParticles: 4 }, budget(options.steps ?? 1500, scale, 100), options, {
    expect: 'report',
    note: 'a mean-field normal guide: the means are close, the spreads and skew are not',
  })
}

/**
 * The Laplace approximation: a mode and the inverse Hessian there, on the unconstrained scale.
 *
 * @param _scale - The budget multiplier (unused: the optimiser stops itself).
 * @param options - Expectation, note, quantities, tolerances and extra knobs.
 * @returns The plan.
 * @example
 * ```ts
 * import { laplacePlan } from 'omai-engine/probprog/models'
 * laplacePlan(1).kind // => 'laplace'
 * ```
 */
export function laplacePlan(_scale: number, options: PlanOptions = {}): MethodPlan {
  return make('laplace', {}, 400, options, {
    expect: 'report',
    note: 'a Gaussian at the mode of the unconstrained density: misses skew and the shape of a constrained scale',
  })
}

/**
 * Importance sampling from the prior, in batches.
 *
 * @param scale - The budget multiplier.
 * @param options - Expectation, note, quantities, tolerances, extra knobs and the draws at scale 1.
 * @returns The plan.
 * @example
 * ```ts
 * import { importancePlan } from 'omai-engine/probprog/models'
 * importancePlan(1).config // => { numSamples: 20000, batchSize: 2000 }
 * ```
 */
export function importancePlan(scale: number, options: PlanOptions & { readonly draws?: number } = {}): MethodPlan {
  const draws = budget(options.draws ?? 20000, scale, 200)
  const batch = Math.max(100, Math.round(draws / 10))
  return make('importance', { numSamples: draws, batchSize: batch }, Math.ceil(draws / batch) + 2, options, {
    expect: 'agree',
    note: 'weighted draws from the prior: unbiased, with an effective sample size that shrinks as the posterior concentrates',
  })
}

/**
 * Tempered SMC: particles moved from the prior to the posterior through a likelihood tempering, with MCMC moves.
 *
 * @param scale - The budget multiplier.
 * @param options - Expectation, note, quantities, tolerances, extra knobs and the particles at scale 1.
 * @returns The plan.
 * @example
 * ```ts
 * import { temperedPlan } from 'omai-engine/probprog/models'
 * temperedPlan(0.1).config // => { numParticles: 100 }
 * ```
 */
export function temperedPlan(scale: number, options: PlanOptions & { readonly particles?: number } = {}): MethodPlan {
  return make('tempered-smc', { numParticles: budget(options.particles ?? 1000, scale, 100) }, 200, options, {
    expect: 'agree',
    note: 'tempered particles with MCMC moves: consistent, with a noisier tail than NUTS',
  })
}

/**
 * Gibbs sampling over the conjugate and discrete conditionals (a slice step elsewhere).
 *
 * @param scale - The budget multiplier.
 * @param options - Expectation, note, quantities, tolerances, extra knobs and the draws at scale 1.
 * @returns The plan.
 * @example
 * ```ts
 * import { gibbsPlan } from 'omai-engine/probprog/models'
 * gibbsPlan(1).config // => { numWarmup: 300, numDraws: 2000 }
 * ```
 */
export function gibbsPlan(scale: number, options: PlanOptions & { readonly draws?: number } = {}): MethodPlan {
  return make(
    'gibbs',
    { numWarmup: budget(300, scale, 30), numDraws: budget(options.draws ?? 2000, scale, 100) },
    1_000_000,
    options,
    { expect: 'agree', note: 'draws each site from its conditional: exact conditionals, so consistent' },
  )
}

/**
 * An exact method (`conjugate`, `gaussian`, `enumeration`, `chain`): one step, the closed-form posterior.
 *
 * @param kind - The exact method's kind.
 * @param options - Expectation, note, quantities, tolerances and extra knobs.
 * @returns The plan.
 * @example
 * ```ts
 * import { exactPlan } from 'omai-engine/probprog/models'
 * exactPlan('conjugate').steps // => 3
 * ```
 */
export function exactPlan(kind: string, options: PlanOptions = {}): MethodPlan {
  return make(kind, {}, 3, options, { expect: 'agree', note: 'the exact posterior of this model: it is the truth' })
}

/**
 * The bootstrap particle filter over the `markov` loop of a program, its draws the genealogies of the final particles
 * (smoothing marginals that degenerate for the early steps of a long series).
 *
 * @param scale - The budget multiplier.
 * @param options - Expectation, note, quantities, tolerances, extra knobs and the particles at scale 1.
 * @returns The plan.
 * @example
 * ```ts
 * import { particleFilterPlan } from 'omai-engine/probprog/models'
 * particleFilterPlan(0.1).config // => { numParticles: 200 }
 * ```
 */
export function particleFilterPlan(
  scale: number,
  options: PlanOptions & { readonly particles?: number } = {},
): MethodPlan {
  return make('particle-filter', { numParticles: budget(options.particles ?? 2000, scale, 100) }, 200, options, {
    expect: 'report',
    note: 'a bootstrap filter: unbiased evidence, filtering marginals right, early smoothing marginals thinned by the genealogy',
  })
}
