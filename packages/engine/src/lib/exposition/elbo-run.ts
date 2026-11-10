/**
 * The exposition record of a variational run (`elboRun`): the bound the method maximises at each step (the ELBO of
 * stochastic variational inference, the best ELBO of Pathfinder) with its smoothing, the norm and variance of the
 * gradient, what the estimator reports about itself, the trajectories of the guide's parameters (location and scale per
 * site), the posterior beliefs the run's snapshots held at each step, Pathfinder's paths and Pareto k-hat, and, when the
 * truth is known, the final posterior over the truth's marginal. `elboRunFromTape` (`elbo-run-tape.ts`) reads it from a
 * tape or from anytime snapshots, so a run in progress has its charts and the same record serves the finished run.
 */

import { OmaiError } from '../base/index.ts'
import { seal } from './_check.ts'
import type { ComparedQuantity } from './posterior-comparison.ts'

/** Values at steps: `values[i]` was measured at `steps[i]`, the steps ascending. */
export interface RunSeries {
  readonly steps: readonly number[]
  readonly values: readonly number[]
}

/** A named series, with a label to show where the name is a tape path's last part. */
export interface NamedSeries extends RunSeries {
  readonly name: string
  readonly label?: string
}

/** One number of a parameter of the guide, over the run: the unconstrained scale the optimiser works on. */
export interface GuideParameter extends RunSeries {
  /** The element's name: `mu_auto_loc` for a scalar parameter, `w_auto_loc[2]` for the third of a vector's. */
  readonly name: string
  /** The site the parameter belongs to (`mu`, `w`), for the mean-field guides' `<site>_auto_loc` and `<site>_auto_scale`. */
  readonly site?: string
  /** What it is: a location, a scale (on the unconstrained scale, before the guide's positive transform), or another parameter. */
  readonly role: 'loc' | 'scale' | 'other'
}

/**
 * The belief about one quantity over the run: the mean and standard deviation of the variational posterior (on the
 * constrained scale, as the posterior summary reports it) at the steps a snapshot was taken.
 */
export interface QuantityBelief extends RunSeries {
  /** The quantity: `mu`, or `w[2]`. */
  readonly name: string
  /** `values` is the mean; the standard deviation and the interval at the same steps. */
  readonly sd: readonly number[]
  readonly lower?: readonly number[]
  readonly upper?: readonly number[]
}

/** The final variational posterior over the truth's marginal, per quantity (`ComparedQuantity`: the truth's CDF, the method's normal). */
export interface FinalPosterior {
  /** What the method is compared with (`oracle`). */
  readonly oracle: string
  readonly method: string
  readonly quantities: readonly ComparedQuantity[]
}

/**
 * A variational run: the bound over the steps, the gradient, the guide's parameters, the beliefs, and the final
 * posterior against the truth. `status` is `running` for a run in progress: every part is what is known so far.
 */
export interface ElboRun {
  readonly kind: 'elboRun'
  /** The method (`svi`, `pathfinder`), when the tape says. */
  readonly method?: string
  readonly status: 'running' | 'completed' | 'converged' | 'diverged' | 'stopped' | 'failed'
  /** The bound at each step (higher is better): the ELBO, or Pathfinder's best so far. */
  readonly elbo: RunSeries
  /** The bound smoothed (the tape's moving average, or the trailing mean over `smoothing` steps). */
  readonly smoothed?: RunSeries
  /** The window of the trailing mean `smoothed` was made with, when it was made here rather than read from the tape. */
  readonly smoothing?: number
  /** The norm of the gradient the optimiser was given, and its estimator's total variance (read every few steps). */
  readonly gradNorm?: RunSeries
  readonly gradVariance?: RunSeries
  /** What the estimator reports about itself (baselines, costs, weights) and other scalars of the method (a learning rate). */
  readonly diagnostics: readonly NamedSeries[]
  /** Pathfinder's paths: the latest ELBO of each over its iterations. */
  readonly paths: readonly NamedSeries[]
  /** The guide's parameters over the run, one element each. */
  readonly parameters: readonly GuideParameter[]
  /** The variational posterior's mean and sd per quantity at the steps a snapshot was taken. */
  readonly beliefs: readonly QuantityBelief[]
  /** Pathfinder's Pareto k-hat and the effective sample size of its importance weights, at the end. */
  readonly paretoK?: number
  readonly ess?: number
  /** The final posterior over the truth, when the truth's marginals were given. */
  readonly final?: FinalPosterior
}

function checkSeries(where: string, s: RunSeries): void {
  if (s.steps.length !== s.values.length)
    throw new OmaiError('shape', `elboRun: ${where} has ${s.steps.length} steps and ${s.values.length} values`)
  for (let i = 1; i < s.steps.length; i++)
    if ((s.steps[i] as number) < (s.steps[i - 1] as number))
      throw new OmaiError('value', `elboRun: ${where} is out of order`)
}

/**
 * A variational run, with its series checked: the steps ascend and match the values.
 *
 * @param record - The record without its `kind`.
 * @returns The record.
 * @throws OmaiError `shape` when a series has different numbers of steps and values (or a belief's sd or interval differs from its mean in length); `value` when steps go backwards.
 * @example
 * ```ts
 * import { elboRun } from 'omai-engine/exposition'
 * const r = elboRun({ status: 'running', elbo: { steps: [1, 2, 3], values: [-9, -5, -4] }, diagnostics: [], paths: [], parameters: [], beliefs: [] })
 * r.elbo.values.at(-1) // => -4
 * ```
 */
export function elboRun(record: Omit<ElboRun, 'kind'>): ElboRun {
  checkSeries('elbo', record.elbo)
  for (const [name, s] of [
    ['smoothed', record.smoothed],
    ['gradNorm', record.gradNorm],
    ['gradVariance', record.gradVariance],
  ] as const)
    if (s !== undefined) checkSeries(name, s)
  for (const s of [...record.diagnostics, ...record.paths, ...record.parameters]) checkSeries(s.name, s)
  for (const b of record.beliefs) {
    checkSeries(b.name, b)
    for (const [field, v] of [
      ['sd', b.sd],
      ['lower', b.lower],
      ['upper', b.upper],
    ] as const)
      if (v !== undefined && v.length !== b.values.length)
        throw new OmaiError('shape', `elboRun: ${b.name}.${field} has ${v.length} entries, expected ${b.values.length}`)
  }
  return seal({ kind: 'elboRun', ...record })
}

/**
 * The trailing mean of values over a window: element `i` is the mean of the last `window` values up to it (fewer at the
 * start). The smoothing of a noisy ELBO curve; a window of one is the values themselves.
 *
 * @param values - The series.
 * @param window - The window (at least one).
 * @returns The smoothed series, the same length.
 * @example
 * ```ts
 * import { trailingMean } from 'omai-engine/exposition'
 * trailingMean([1, 3, 5, 7], 2) // => [1, 2, 4, 6]
 * ```
 */
export function trailingMean(values: readonly number[], window: number): number[] {
  const w = Math.max(1, Math.floor(window))
  const out: number[] = []
  let acc = 0
  for (const [i, v] of values.entries()) {
    acc += v
    if (i >= w) acc -= values[i - w] as number
    out.push(acc / Math.min(w, i + 1))
  }
  return out
}
