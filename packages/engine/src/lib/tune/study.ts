/**
 * Studies and trials, the record of a parameter search: a `Study` holds a space, the direction of each objective and
 * its finished trials (complete, pruned or failed, with intermediate reports). Creation, losses, the best trial or
 * Pareto front of best trials, hypervolume, export as rows, and lossless JSON encoding for resuming a study later.
 */

import { type Decoded, type Encodable, fromJson, type Json, toJson, WhiteboxError } from '../base/index.ts'
import { hypervolume, nonDominatedSort } from '../optim/index.ts'
import type { Space, SpaceValue } from '../space/index.ts'

/** Which way an objective value is better. */
export type StudyDirection = 'minimize' | 'maximize'

/** How a trial ended: with values, stopped early by a pruner, or failed (an error or a NaN value). */
export type TrialState = 'complete' | 'pruned' | 'failed'

/** A value a trial reported part-way, at a step (an epoch, a fold, a number of samples). */
export interface IntermediateValue {
  readonly step: number
  readonly value: number
}

/**
 * One evaluation of the objective: its parameters, values (one per direction), intermediate reports and how it ended.
 * Trials are plain data (JSON, with non-finite numbers encoded by `studyToJson`).
 */
export interface Trial {
  readonly number: number
  readonly params: SpaceValue
  readonly state: TrialState
  /** The objective values (one per direction; empty unless complete). */
  readonly values: readonly number[]
  readonly intermediate: readonly IntermediateValue[]
  /** Who chose the parameters: a sampler's name, `'enqueued'` or `'promoted'` (by a schedule). */
  readonly source: string
  /** The resource the trial was given by a multi-fidelity schedule (epochs, samples, steps). */
  readonly budget?: number
  readonly bracket?: number
  readonly rung?: number
  /** The trial whose configuration a promotion re-ran. */
  readonly parent?: number
  /** Values at the rungs a successive-halving pruner let the trial pass (Optuna's `completed_rung_<i>`). */
  readonly rungValues?: readonly number[]
  /** The step at which a pruner stopped the trial. */
  readonly prunedAt?: number
  /** Why a trial failed or was pruned. */
  readonly message?: string
  /** Extra results returned by the objective (fold scores, fitted sizes). */
  readonly attrs?: Readonly<Record<string, Json>>
}

/** A study: a space, the directions of its objective values, and the finished trials in order. */
export interface Study {
  readonly name: string
  readonly directions: readonly StudyDirection[]
  readonly space: Space
  readonly trials: readonly Trial[]
  readonly metricNames?: readonly string[]
}

/**
 * Create an empty study (to fill with {@link tune}, or to hand to a sampler with a history of your own).
 *
 * @param space - The search space.
 * @param options - Name (default `'study'`), directions (default `['minimize']`), trials and metric names.
 * @returns The study.
 * @concept hyperparameter-optimisation
 * @example
 * ```ts
 * import { createStudy } from 'whitebox-engine/tune'
 * import { dict, real } from 'whitebox-engine/space'
 * createStudy(dict({ x: real(0, 1) })).directions // => ['minimize']
 * ```
 */
export function createStudy(
  space: Space,
  options: Partial<Pick<Study, 'name' | 'directions' | 'trials' | 'metricNames'>> = {},
): Study {
  const directions = options.directions ?? ['minimize']
  if (directions.length === 0) throw new WhiteboxError('value', 'createStudy: need at least one direction')
  return {
    name: options.name ?? 'study',
    directions,
    space,
    trials: options.trials ?? [],
    ...(options.metricNames === undefined ? {} : { metricNames: options.metricNames }),
  }
}

/**
 * The loss of a value: the value for a minimised objective, its negation for a maximised one.
 *
 * @param direction - The direction.
 * @param value - The value.
 * @returns The value to minimise.
 * @example
 * ```ts
 * import { lossOf } from 'whitebox-engine/tune'
 * lossOf('maximize', 0.9) // => -0.9
 * ```
 */
export function lossOf(direction: StudyDirection, value: number): number {
  return direction === 'minimize' ? value : -value
}

/**
 * The complete trials of a study, in order.
 *
 * @param study - The study.
 * @returns The trials whose state is `'complete'`.
 * @example
 * ```ts
 * import { completeTrials, createStudy } from 'whitebox-engine/tune'
 * import { real } from 'whitebox-engine/space'
 * completeTrials(createStudy(real(0, 1))).length // => 0
 * ```
 */
export function completeTrials(study: Study): Trial[] {
  return study.trials.filter((t) => t.state === 'complete')
}

/**
 * The best complete trial of a single-objective study (the first of equals).
 *
 * @param study - The study.
 * @returns The best trial, or `undefined` when none is complete.
 * @throws WhiteboxError with code `value` for a multi-objective study (use {@link bestTrials}).
 * @concept hyperparameter-optimisation
 * @example
 * ```ts
 * import { bestTrial, createStudy } from 'whitebox-engine/tune'
 * import { real } from 'whitebox-engine/space'
 * const trial = (number: number, value: number) =>
 *   ({ number, params: value, state: 'complete', values: [value], intermediate: [], source: 'x' }) as const
 * bestTrial(createStudy(real(0, 9), { trials: [trial(0, 3), trial(1, 1), trial(2, 2)] }))?.number // => 1
 * ```
 */
export function bestTrial(study: Study): Trial | undefined {
  if (study.directions.length !== 1)
    throw new WhiteboxError('value', 'bestTrial: a multi-objective study has a Pareto front; use bestTrials')
  const direction = study.directions[0] as StudyDirection
  let best: Trial | undefined
  for (const t of completeTrials(study)) {
    const v = lossOf(direction, t.values[0] ?? Number.NaN)
    if (best === undefined || v < lossOf(direction, best.values[0] ?? Number.NaN)) best = t
  }
  return best
}

/**
 * The loss vectors of trials: one row per trial, every objective turned into one to minimise.
 *
 * @param study - The study (for its directions).
 * @param trials - The trials.
 * @returns One row of losses per trial (NaN for missing values).
 * @example
 * ```ts
 * import { createStudy, lossMatrix } from 'whitebox-engine/tune'
 * import { real } from 'whitebox-engine/space'
 * const t = { number: 0, params: 0, state: 'complete', values: [1, 2], intermediate: [], source: 'x' } as const
 * lossMatrix(createStudy(real(0, 1), { directions: ['minimize', 'maximize'] }), [t]) // => [[1, -2]]
 * ```
 */
export function lossMatrix(study: Study, trials: readonly Trial[]): number[][] {
  return trials.map((t) => study.directions.map((d, j) => lossOf(d, t.values[j] ?? Number.NaN)))
}

/**
 * The best trials: for one objective, the trials sharing the best value; for several, the Pareto front of complete
 * trials (those no other complete trial dominates), in trial order, as Optuna's `study.best_trials`.
 *
 * @param study - The study.
 * @returns The best trials.
 * @concept multi-objective-optimisation
 * @example
 * ```ts
 * import { bestTrials, createStudy } from 'whitebox-engine/tune'
 * import { real } from 'whitebox-engine/space'
 * const trial = (number: number, values: number[]) =>
 *   ({ number, params: 0, state: 'complete', values, intermediate: [], source: 'x' }) as const
 * const study = createStudy(real(0, 1), {
 *   directions: ['minimize', 'minimize'],
 *   trials: [trial(0, [1, 3]), trial(1, [2, 2]), trial(2, [2, 3]), trial(3, [3, 1])],
 * })
 * bestTrials(study).map((t) => t.number) // => [0, 1, 3]
 * ```
 */
export function bestTrials(study: Study): Trial[] {
  const complete = completeTrials(study)
  if (complete.length === 0) return []
  if (study.directions.length === 1) {
    const best = bestTrial(study) as Trial
    return complete.filter((t) => t.values[0] === best.values[0])
  }
  const front = new Set(nonDominatedSort(lossMatrix(study, complete))[0] ?? [])
  return complete.filter((_, i) => front.has(i))
}

/**
 * The hypervolume dominated by the complete trials of a study with respect to a reference point (in the units of the
 * objective values; maximised objectives are negated with their reference coordinate).
 *
 * @param study - The study.
 * @param reference - The reference point, one coordinate per objective, worse than every trial.
 * @returns The hypervolume (0 when no trial is complete).
 * @concept multi-objective-optimisation
 * @example
 * ```ts
 * import { createStudy, studyHypervolume } from 'whitebox-engine/tune'
 * import { real } from 'whitebox-engine/space'
 * const trial = (number: number, values: number[]) =>
 *   ({ number, params: 0, state: 'complete', values, intermediate: [], source: 'x' }) as const
 * const study = createStudy(real(0, 1), { directions: ['minimize', 'minimize'], trials: [trial(0, [1, 2]), trial(1, [2, 1])] })
 * studyHypervolume(study, [3, 3]) // => 3
 * ```
 */
export function studyHypervolume(study: Study, reference: readonly number[]): number {
  const complete = completeTrials(study)
  if (complete.length === 0) return 0
  if (reference.length !== study.directions.length)
    throw new WhiteboxError('shape', 'studyHypervolume: one reference coordinate per objective')
  const ref = study.directions.map((d, j) => lossOf(d, reference[j] ?? 0))
  return hypervolume(lossMatrix(study, complete), ref)
}

/**
 * The trials of a study as rows of plain data (Optuna's `trials_dataframe`): number, state, values (`value`, or
 * `values_0`, `values_1`, ... or the metric names), each leaf parameter as `params_<path>`, the budget, bracket and
 * rung of multi-fidelity trials and the last intermediate value.
 *
 * @param study - The study.
 * @returns One row per trial.
 * @example
 * ```ts
 * import { createStudy, exportTrials } from 'whitebox-engine/tune'
 * import { dict, real } from 'whitebox-engine/space'
 * const study = createStudy(dict({ x: real(0, 1) }), {
 *   trials: [{ number: 0, params: { x: 0.5 }, state: 'complete', values: [0.25], intermediate: [], source: 'random' }],
 * })
 * exportTrials(study) // => [{ number: 0, state: 'complete', value: 0.25, params_x: 0.5, source: 'random' }]
 * ```
 */
export function exportTrials(study: Study): Readonly<Record<string, Json>>[] {
  const names =
    study.metricNames ?? (study.directions.length === 1 ? ['value'] : study.directions.map((_, j) => `values_${j}`))
  const flat = (prefix: string, v: SpaceValue, out: Record<string, Json>): void => {
    if (typeof v === 'object' && v !== null && !Array.isArray(v)) {
      for (const [k, c] of Object.entries(v as Readonly<Record<string, SpaceValue>>))
        flat(prefix === '' ? k : `${prefix}.${k}`, c, out)
    } else out[prefix === '' ? 'params' : `params_${prefix}`] = v as Json
  }
  return study.trials.map((t) => {
    const row: Record<string, Json> = { number: t.number, state: t.state }
    for (const [j, name] of names.entries()) if (t.state === 'complete') row[name] = t.values[j] ?? null
    flat('', t.params, row)
    if (t.budget !== undefined) row['budget'] = t.budget
    if (t.bracket !== undefined) row['bracket'] = t.bracket
    if (t.rung !== undefined) row['rung'] = t.rung
    const last = t.intermediate.at(-1)
    if (last !== undefined) row['lastIntermediate'] = last.value
    row['source'] = t.source
    return row
  })
}

/**
 * Encode a study as JSON without loss (non-finite values tagged), to store it and resume it later with
 * {@link studyFromJson} and `tune(objective, space, { study })`.
 *
 * @param study - The study.
 * @returns JSON.
 * @example
 * ```ts
 * import { createStudy, studyFromJson, studyToJson } from 'whitebox-engine/tune'
 * import { real } from 'whitebox-engine/space'
 * studyFromJson(studyToJson(createStudy(real(0, 1), { name: 'saved' }))).name // => 'saved'
 * ```
 */
export function studyToJson(study: Study): Json {
  return toJson(study as unknown as Encodable)
}

/**
 * Decode a study written by {@link studyToJson}.
 *
 * @param json - The JSON.
 * @returns The study.
 * @throws WhiteboxError with code `value` when the JSON is not a study.
 * @example
 * ```ts
 * import { createStudy, studyFromJson, studyToJson } from 'whitebox-engine/tune'
 * import { real } from 'whitebox-engine/space'
 * studyFromJson(studyToJson(createStudy(real(0, 1)))).trials // => []
 * ```
 */
export function studyFromJson(json: Json): Study {
  const s: Decoded = fromJson(json)
  const ok =
    typeof s === 'object' &&
    s !== null &&
    'name' in s &&
    'directions' in s &&
    'space' in s &&
    'trials' in s &&
    Array.isArray((s as { trials: unknown }).trials)
  if (!ok) throw new WhiteboxError('value', 'studyFromJson: not a study')
  return s as unknown as Study
}
