/**
 * Bayesian optimisation with a Gaussian-process surrogate: the acquisition functions that decide where to evaluate next
 * by trading predicted value against uncertainty (expected improvement and its stable logarithm, probability of
 * improvement, confidence bounds), their maximisation over the unit cube, and `gpSampler`, which mirrors Optuna's
 * `GPSampler`.
 */

import { type Json, LOG_SQRT_2PI } from '../base/index.ts'
import { erfc, erfcx, logNdtr, ndtr, normalPdf } from '../numerics/special/index.ts'
import { lbfgsb } from '../optim/index.ts'
import { foldIn, type Key, scrambledSobol } from '../random/index.ts'
import { decode, sample } from '../space/index.ts'
import {
  add,
  asTensor,
  div,
  exp,
  less,
  log,
  mul,
  neg,
  square,
  sub,
  sum,
  type TensorLike,
  toFlat,
  type Value,
  where,
} from '../tensor/index.ts'
import { type Leaf, leafCoordinate, leavesOf, nativeOf } from './_leaves.ts'
import type { StudySampler } from './samplers.ts'
import { lossOf, type StudyDirection } from './study.ts'
import { fitGpSurrogate, type GpSurrogate } from './surrogate.ts'

const SQRT_HALF = Math.SQRT1_2
// ---------------------------------------------------------------------------------------------------------------
// Acquisition functions (minimisation: smaller objective values are better)

function standardised(mean: TensorLike, std: TensorLike, best: number, xi: number): Value {
  return div(sub(sub(best, mean), xi), std)
}

/**
 * Expected improvement below the best value, EI = (f* - mu - xi) Phi(z) + sigma phi(z) with
 * z = (f* - mu - xi) / sigma, for a Gaussian prediction N(mu, sigma^2) of a minimised objective.
 *
 * @param mean - Predicted means.
 * @param std - Predicted standard deviations (positive).
 * @param best - The best (smallest) value so far, f*.
 * @param xi - Exploration margin (default 0).
 * @returns EI, differentiable in the predictions.
 * @concept bayesian-optimisation
 * @reference Jones, Schonlau and Welch (1998), Efficient global optimization of expensive black-box functions.
 * @example
 * ```ts
 * import { expectedImprovement } from 'mdml-engine/tune'
 * import { toFlat } from 'mdml-engine/tensor'
 * toFlat(expectedImprovement([0], [1], 0)) // => [0.398942]
 * ```
 */
export function expectedImprovement(mean: TensorLike, std: TensorLike, best: number, xi = 0): Value {
  const z = standardised(mean, std, best, xi)
  return add(mul(mul(z, std), ndtr(z)), mul(std, normalPdf(z)))
}

/**
 * The logarithm of expected improvement, computed stably for very negative z (Ament et al. 2023, as Optuna's
 * `logei`): log(z Phi(z) + phi(z)) + log sigma, switching below z = -25 to
 * -z^2 / 2 - log sqrt(2 pi) + log(1 + sqrt(pi / 2) z erfcx(-z / sqrt 2)).
 *
 * @param mean - Predicted means.
 * @param std - Predicted standard deviations.
 * @param best - The best value so far.
 * @param xi - Exploration margin (default 0).
 * @returns log EI.
 * @concept bayesian-optimisation
 * @reference Ament et al. (2023), Unexpected improvements to expected improvement for Bayesian optimization.
 * @example
 * ```ts
 * import { logExpectedImprovement } from 'mdml-engine/tune'
 * import { toFlat } from 'mdml-engine/tensor'
 * (toFlat(logExpectedImprovement([40], [1], 0))[0] ?? Number.NaN) > -Infinity // => true
 * ```
 */
export function logExpectedImprovement(mean: TensorLike, std: TensorLike, best: number, xi = 0): Value {
  const z = standardised(mean, std, best, xi)
  const small = less(z, -25)
  // Each branch gets inputs from its own range, so the unused one never produces NaN gradients.
  const zl = where(small, -25, z)
  const zs = where(small, z, -26)
  const regular = log(
    add(mul(mul(zl, 0.5), erfc(mul(zl, -SQRT_HALF))), mul(exp(mul(square(zl), -0.5)), 1 / Math.sqrt(2 * Math.PI))),
  )
  const tail = add(
    sub(mul(square(zs), -0.5), LOG_SQRT_2PI),
    log(add(1, mul(mul(zs, Math.sqrt(Math.PI / 2)), erfcx(mul(zs, -SQRT_HALF))))),
  )
  return add(where(small, tail, regular), log(std))
}

/**
 * Probability of improvement below the best value, Phi((f* - mu - xi) / sigma).
 *
 * @param mean - Predicted means.
 * @param std - Predicted standard deviations.
 * @param best - The best value so far.
 * @param xi - Margin (default 0).
 * @returns PI.
 * @concept bayesian-optimisation
 * @example
 * ```ts
 * import { probabilityOfImprovement } from 'mdml-engine/tune'
 * import { toFlat } from 'mdml-engine/tensor'
 * toFlat(probabilityOfImprovement([0], [1], 0)) // => [0.5]
 * ```
 */
export function probabilityOfImprovement(mean: TensorLike, std: TensorLike, best: number, xi = 0): Value {
  return ndtr(standardised(mean, std, best, xi))
}

/**
 * log probability of improvement, accurate far in the tail.
 *
 * @param mean - Predicted means.
 * @param std - Predicted standard deviations.
 * @param best - The best value so far.
 * @param xi - Margin (default 0).
 * @returns log PI.
 * @concept bayesian-optimisation
 * @example
 * ```ts
 * import { logProbabilityOfImprovement } from 'mdml-engine/tune'
 * import { toFlat } from 'mdml-engine/tensor'
 * toFlat(logProbabilityOfImprovement([0], [1], 0)) // => [-0.693147]
 * ```
 */
export function logProbabilityOfImprovement(mean: TensorLike, std: TensorLike, best: number, xi = 0): Value {
  return logNdtr(standardised(mean, std, best, xi))
}

/**
 * The lower confidence bound mu - beta sigma of a minimised objective (GP-UCB of Srinivas et al. for maximisation is
 * the upper bound of the negated objective); smaller is more promising.
 *
 * @param mean - Predicted means.
 * @param std - Predicted standard deviations.
 * @param beta - Width in standard deviations (default 2).
 * @returns The bounds.
 * @concept bayesian-optimisation
 * @reference Srinivas et al. (2010), Gaussian process optimization in the bandit setting.
 * @example
 * ```ts
 * import { lowerConfidenceBound } from 'mdml-engine/tune'
 * import { toFlat } from 'mdml-engine/tensor'
 * toFlat(lowerConfidenceBound([1], [0.5], 2)) // => [0]
 * ```
 */
export function lowerConfidenceBound(mean: TensorLike, std: TensorLike, beta = 2): Value {
  return sub(mean, mul(std, beta))
}

/**
 * The upper confidence bound mu + beta sigma (GP-UCB for a maximised objective).
 *
 * @param mean - Predicted means.
 * @param std - Predicted standard deviations.
 * @param beta - Width in standard deviations (default 2).
 * @returns The bounds.
 * @concept bayesian-optimisation
 * @example
 * ```ts
 * import { upperConfidenceBound } from 'mdml-engine/tune'
 * import { toFlat } from 'mdml-engine/tensor'
 * toFlat(upperConfidenceBound([1], [0.5], 2)) // => [2]
 * ```
 */
export function upperConfidenceBound(mean: TensorLike, std: TensorLike, beta = 2): Value {
  return add(mean, mul(std, beta))
}

// ---------------------------------------------------------------------------------------------------------------
// Acquisition maximisation and the sampler

/** An acquisition: from predicted means and standard deviations (and the best value) to scores to maximise. */
export type Acquisition = (mean: Value, std: Value, best: number) => Value

/**
 * The acquisitions {@link gpSampler} knows by name, each maximised: log expected improvement, expected improvement,
 * log probability of improvement and the negated lower confidence bound (beta 2).
 *
 * @example
 * ```ts
 * import { acquisitions } from 'mdml-engine/tune'
 * import { asTensor, toFlat } from 'mdml-engine/tensor'
 * toFlat(acquisitions.LCB(asTensor([2]), asTensor([0.5]), 0)) // => [-1]
 * Object.keys(acquisitions) // => ['logEI', 'EI', 'PI', 'LCB']
 * ```
 */
export const acquisitions: Readonly<Record<'logEI' | 'EI' | 'PI' | 'LCB', Acquisition>> = {
  logEI: (m, s, best) => logExpectedImprovement(m, s, best),
  EI: (m, s, best) => expectedImprovement(m, s, best),
  PI: (m, s, best) => logProbabilityOfImprovement(m, s, best),
  LCB: (m, s) => neg(lowerConfidenceBound(m, s, 2)),
}

/** Options of {@link maximizeAcquisition}. */
export interface AcquisitionSearchOptions {
  /** Quasi-random candidates scored first (default 512). */
  readonly nCandidates?: number
  /** Best candidates refined by L-BFGS-B over the continuous columns (default 3). */
  readonly nLocal?: number
  /** L-BFGS-B iterations per refinement (default 30). */
  readonly maxIterations?: number
  /** Points to add to the candidates (the best observed configurations). */
  readonly seeds?: readonly (readonly number[])[]
}

/**
 * Maximise an acquisition over the unit cube (option indices for categorical columns), as Optuna's GP sampler does:
 * score scrambled Sobol' candidates (options read through equal cells) and seed points, then refine the best few by
 * L-BFGS-B over the continuous columns with the options fixed, gradients by automatic differentiation.
 *
 * @param surrogate - The surrogate.
 * @param acquisition - The acquisition (maximised).
 * @param best - The best observed value (passed to the acquisition).
 * @param key - The key of the candidates.
 * @param options - Candidates, refinements and seeds.
 * @returns The best point and its acquisition value.
 * @concept bayesian-optimisation
 * @example
 * ```ts
 * import { acquisitions, fitGpSurrogate, maximizeAcquisition } from 'mdml-engine/tune'
 * import { key } from 'mdml-engine/random'
 * const xs = [[0.1], [0.4], [0.9]]
 * const gp = fitGpSurrogate(xs, [1, 0, 1])
 * const { point } = maximizeAcquisition(gp, acquisitions.EI, 0, key(0))
 * point.length // => 1
 * ```
 */
export function maximizeAcquisition(
  surrogate: GpSurrogate,
  acquisition: Acquisition,
  best: number,
  key: Key,
  options: AcquisitionSearchOptions = {},
): { readonly point: number[]; readonly value: number } {
  const d = surrogate.categorical.length
  const sizes = surrogate.sizes
  const m = options.nCandidates ?? 512
  const u = scrambledSobol(key, m, Math.max(1, d)).data
  const candidates: number[][] = Array.from({ length: m }, (_, i) =>
    surrogate.categorical.map((c, j) => {
      const v = u[i * Math.max(1, d) + j] ?? 0.5
      return c ? Math.min((sizes[j] as number) - 1, Math.floor(v * (sizes[j] as number))) : v
    }),
  )
  for (const s of options.seeds ?? []) candidates.push([...s])
  const score = (rows: readonly (readonly number[])[]): number[] => {
    const p = surrogate.predict(rows)
    return toFlat(acquisition(asTensor(p.mean), asTensor(p.std), best))
  }
  const scores = score(candidates)
  const order = scores.map((_, i) => i).sort((a, b) => (scores[b] ?? 0) - (scores[a] ?? 0) || a - b)
  let bestPoint = candidates[order[0] ?? 0] as number[]
  let bestValue = scores[order[0] ?? 0] ?? Number.NEGATIVE_INFINITY
  const continuous = surrogate.categorical.flatMap((c, j) => (c ? [] : [j]))
  if (continuous.length > 0) {
    for (const i of order.slice(0, options.nLocal ?? 3)) {
      const start = candidates[i] as number[]
      const f = (z: unknown): Value => {
        const p = surrogate.predictAt(start, z as Value)
        return neg(sum(acquisition(p.mean, p.std, best)))
      }
      const r = lbfgsb(
        f,
        continuous.map((j) => start[j] as number),
        { bounds: { lower: 0, upper: 1 }, maxIterations: options.maxIterations ?? 30 },
      )
      if (-r.fun > bestValue) {
        const z = toFlat(r.x as Value)
        bestPoint = start.map((v, j) => (surrogate.categorical[j] === true ? v : (z[continuous.indexOf(j)] ?? v)))
        bestValue = -r.fun
      }
    }
  }
  return { point: bestPoint, value: bestValue }
}

/** Options of {@link gpSampler}. */
export interface GpSamplerOptions {
  /** Random trials before the surrogate is used (default 10). */
  readonly nStartupTrials?: number
  /** The acquisition (default `'logEI'`). */
  readonly acquisition?: keyof typeof acquisitions | Acquisition
  /** Acquisition search settings. */
  readonly search?: AcquisitionSearchOptions
  /** Hyperparameter fitting iterations (default 50). */
  readonly maxIterations?: number
}

/**
 * Bayesian optimisation with a Gaussian-process surrogate (Optuna's `GPSampler`): after `nStartupTrials` random
 * trials, fit {@link fitGpSurrogate} to the complete trials in the space's unit-cube coordinates (options by Hamming
 * distance), and propose the maximiser of the acquisition (log expected improvement by default) found by
 * {@link maximizeAcquisition}, seeded with the best trials so far. Single objective.
 * TODO(tune): multi-objective acquisition (log expected hypervolume improvement) as in Optuna's GPSampler.
 *
 * @param options - Startup trials, acquisition, search and fitting settings.
 * @returns The sampler.
 * @concept bayesian-optimisation
 * @reference Snoek, Larochelle and Adams (2012), Practical Bayesian optimization of machine learning algorithms.
 * @example
 * ```ts
 * import { gpSampler, functionObjective, tune } from 'mdml-engine/tune'
 * import { real } from 'mdml-engine/space'
 * const f = functionObjective((x) => ((x as number) - 0.7) ** 2)
 * (tune(f, real(0, 1), { sampler: gpSampler({ nStartupTrials: 4 }), nTrials: 10, seed: 0 }).best?.values[0] ?? 1) < 1e-3 // => true
 * ```
 */
export function gpSampler(options: GpSamplerOptions = {}): StudySampler {
  const nStartup = options.nStartupTrials ?? 10
  const acq = options.acquisition ?? 'logEI'
  const acquisition = typeof acq === 'function' ? acq : acquisitions[acq]
  const config: Json = {
    nStartupTrials: nStartup,
    acquisition: typeof acq === 'function' ? 'custom' : acq,
    maxIterations: options.maxIterations ?? 50,
  }
  return {
    name: 'gp',
    config,
    multiObjective: false,
    ask: (study, key) => {
      const complete = study.trials.filter((t) => t.state === 'complete' && Number.isFinite(t.values[0] ?? Number.NaN))
      if (complete.length < nStartup) return sample(study.space, key)
      const leaves = leavesOf(study.space)
      const d = study.directions[0] as StudyDirection
      const columns = (native: readonly number[]): number[] =>
        leaves.map((l, j) => columnOf(l, native[j] ?? Number.NaN))
      const x = complete.map((t) => columns(nativeOf(study.space, t.params)))
      const y = complete.map((t) => lossOf(d, t.values[0] ?? 0))
      const categorical = leaves.map((l) => l.kind === 'categorical')
      const sizes = leaves.map((l) => (l.kind === 'categorical' ? l.n : 1))
      const surrogate = fitGpSurrogate(x, y, {
        categorical,
        sizes,
        ...(options.maxIterations === undefined ? {} : { maxIterations: options.maxIterations }),
      })
      const best = y.reduce((a, b) => Math.min(a, b))
      const order = y.map((_, i) => i).sort((a, b) => (y[a] ?? 0) - (y[b] ?? 0))
      const seeds = order.slice(0, 3).map((i) => x[i] as number[])
      const { point } = maximizeAcquisition(surrogate, acquisition, best, foldIn(key, 0), { ...options.search, seeds })
      // Continuous columns are unit-cube coordinates already; options go through their coordinate.
      return decode(
        study.space,
        leaves.map((l, j) =>
          l.kind === 'categorical' ? leafCoordinate(l, Math.round(point[j] ?? 0)) : (point[j] ?? 0.5),
        ),
      )
    },
  }
}

/** A leaf's surrogate column: its unit-cube coordinate, or its option index. */
function columnOf(leaf: Leaf, native: number): number {
  if (leaf.kind === 'categorical') return Number.isNaN(native) ? -1 : native
  return leafCoordinate(leaf, native)
}
