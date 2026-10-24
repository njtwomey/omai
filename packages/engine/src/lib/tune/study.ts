import { type Decoded, type Encodable, fromJson, type Json, toJson, WhiteboxError } from '../base/index.ts'
import { hypervolume, nonDominatedSort } from '../optim/index.ts'
import { foldIn, type Key, key as makeKey } from '../random/index.ts'
import type { Space, SpaceValue } from '../space/index.ts'
import {
  type Algorithm,
  channels,
  type RunHandle,
  type RunOptions,
  type RunStatus,
  runAlgorithm,
} from '../trace/index.ts'
import type { Objective, ObjectiveResult, TrialContext } from './objectives.ts'
import type { Pruner, RunningTrial } from './pruners.ts'
import { randomSampler, type Sampler } from './samplers.ts'
import { type Assignment, nextAssignment, type Schedule } from './schedules.ts'

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

// ---------------------------------------------------------------------------------------------------------------
// The study as a traced algorithm

/** Options of {@link tune} and {@link studyAlgorithm}. */
export interface TuneOptions extends Partial<Omit<RunOptions, 'maxSteps' | 'start'>> {
  /** The number of trials to run (default 100): the step budget of the run. */
  readonly nTrials?: number
  /** Stop once the trials' budgets (resource units from a schedule, or reported steps) sum to this. */
  readonly budget?: number
  /** The sampler (default `randomSampler()`). */
  readonly sampler?: Sampler
  /** A pruner deciding, after each intermediate report, whether to stop a trial. */
  readonly pruner?: Pruner
  /** A multi-fidelity schedule assigning budgets, brackets and promotions. */
  readonly schedule?: Schedule
  /** The directions of the objective values (default the objective's). */
  readonly directions?: readonly StudyDirection[]
  /** Configurations to evaluate first, in order. */
  readonly enqueue?: readonly SpaceValue[]
  /** A study to continue: its trials are kept and new trials are numbered after them. */
  readonly study?: Study
  /** The study's name (default `'study'`; Hyperband brackets hash it, as Optuna does). */
  readonly name?: string
  /** Stop after this many trials without improving the best value (single objective; default never). */
  readonly patience?: number
  /** Stop after this many failed trials in a row (default 20): the objective is broken. */
  readonly maxFailures?: number
  /** Stop when this many complete trials in a row, and every complete trial so far, gave one value (default 20). */
  readonly constantTrials?: number
  /** Record a failed trial when the objective throws (default true); `false` rethrows. */
  readonly catchErrors?: boolean
}

/** The state of a study run. */
export interface StudyState {
  readonly study: Study
  readonly failuresInRow: number
  readonly sinceImprovement: number
  readonly spent: number
  readonly stop?: { readonly status: RunStatus; readonly message: string }
}

/** The result of {@link tune}. */
export interface TuneResult {
  readonly study: Study
  /** The best trial of a single-objective study. */
  readonly best: Trial | undefined
  /** The best trials (the Pareto front for several objectives). */
  readonly bestTrials: readonly Trial[]
  readonly status: RunStatus
  readonly message: string
  readonly steps: number
  readonly runId: string
}

const SCHEMA = channels({
  metrics: {
    value: { description: 'first objective value of the trial (NaN unless complete)' },
    best: { description: 'best first objective value so far' },
    complete: { description: 'complete trials so far' },
    pruned: { description: 'pruned trials so far' },
    failed: { description: 'failed trials so far' },
  },
  messages: {
    trial: { description: 'a finished trial', payload: { number: 'number', params: 'value', state: 'string' } },
    stop: { description: 'why the study stopped' },
  },
})

const TRIAL_SCHEMA = channels({
  metrics: { intermediate: { description: 'an intermediate value reported by the objective' } },
  messages: {
    params: { description: 'the parameters of the trial' },
    decision: { description: 'a pruning decision after a report', payload: { step: 'number', prune: 'boolean' } },
  },
})

function asValues(result: ObjectiveResult): { values: number[]; attrs?: Readonly<Record<string, Json>> } {
  if (typeof result === 'number') return { values: [result] }
  if (Array.isArray(result)) return { values: [...(result as readonly number[])] }
  const r = result as { readonly values: number | readonly number[]; readonly attrs?: Readonly<Record<string, Json>> }
  return {
    values: typeof r.values === 'number' ? [r.values] : [...r.values],
    ...(r.attrs === undefined ? {} : { attrs: r.attrs }),
  }
}

interface Evaluated {
  readonly trial: Trial
  readonly spent: number
}

function evaluateTrial(
  objective: Objective,
  study: Study,
  params: SpaceValue,
  assignment: Omit<Trial, 'number' | 'params' | 'state' | 'values' | 'intermediate'>,
  key: Key,
  run: RunHandle,
  options: TuneOptions,
): Evaluated {
  const number = study.trials.length
  const trialRun = run.child({
    name: 'trial',
    seed: null,
    config: { number, params: params as Json },
    schema: TRIAL_SCHEMA,
  })
  trialRun.message(0, 'params', params as Json)
  const intermediate: IntermediateValue[] = []
  let rungValues: readonly number[] = []
  let prunedAt: number | undefined
  const running = (): RunningTrial => ({ number, params, intermediate, rungValues, ...assignment })
  const pruner = options.pruner
  const context: TrialContext = {
    number,
    key,
    run: trialRun,
    ...(assignment.budget === undefined ? {} : { budget: assignment.budget }),
    report: (step, value) => {
      if (prunedAt !== undefined) return true
      if (!Number.isFinite(step)) throw new WhiteboxError('value', 'report: the step must be a finite number')
      // A repeated step keeps its first value, as Optuna does.
      if (!intermediate.some((iv) => iv.step === step)) intermediate.push({ step, value })
      trialRun.metric(step, 'intermediate', value)
      if (pruner === undefined) return false
      const decision = pruner.prune(study, running())
      if (decision.rungValues !== undefined) rungValues = decision.rungValues
      trialRun.message(step, 'decision', { step, prune: decision.prune, pruner: pruner.name })
      if (decision.prune) prunedAt = step
      return decision.prune
    },
    shouldPrune: () => prunedAt !== undefined,
  }
  let state: TrialState = 'complete'
  let values: number[] = []
  let attrs: Readonly<Record<string, Json>> | undefined
  let message: string | undefined
  try {
    const out = asValues(objective.evaluate(params, context))
    values = out.values
    attrs = out.attrs
    if (prunedAt !== undefined) state = 'pruned'
    else if (values.length !== study.directions.length) {
      state = 'failed'
      message = `the objective returned ${values.length} values for ${study.directions.length} directions`
    } else if (values.some(Number.isNaN)) {
      state = 'failed'
      message = 'the objective returned NaN'
    }
  } catch (error) {
    if (options.catchErrors === false) throw error
    state = 'failed'
    message = error instanceof Error ? error.message : String(error)
  }
  if (state === 'pruned') message = `pruned at step ${prunedAt}`
  const lastStep = intermediate.reduce((m, iv) => Math.max(m, iv.step), 0)
  const trial: Trial = {
    number,
    params,
    state,
    values: state === 'complete' ? values : [],
    intermediate: [...intermediate].sort((a, b) => a.step - b.step),
    ...assignment,
    ...(rungValues.length === 0 ? {} : { rungValues }),
    ...(prunedAt === undefined ? {} : { prunedAt }),
    ...(message === undefined ? {} : { message }),
    ...(attrs === undefined ? {} : { attrs }),
  }
  trialRun.end(state === 'complete' ? 'completed' : state === 'pruned' ? 'stopped' : 'failed', intermediate.length, {
    ...(state === 'complete' ? { value: values[0] ?? Number.NaN } : {}),
  })
  return { trial, spent: assignment.budget ?? (intermediate.length === 0 ? 1 : lastStep) }
}

function choose(
  study: Study,
  sampler: Sampler,
  options: TuneOptions,
  key: Key,
): { readonly params: SpaceValue; readonly assignment: Assignment | undefined; readonly source: string } | undefined {
  const enqueue = options.enqueue ?? []
  const used = study.trials.filter((t) => t.source === 'enqueued').length
  const schedule = options.schedule
  const assignment = schedule === undefined ? undefined : nextAssignment(schedule, study)
  if (schedule !== undefined && assignment === undefined) return undefined
  if (assignment?.kind === 'promote') return { params: assignment.params, assignment, source: 'promoted' }
  if (used < enqueue.length) return { params: enqueue[used] as SpaceValue, assignment, source: 'enqueued' }
  const params = sampler.ask(study, key)
  return params === undefined ? undefined : { params, assignment, source: sampler.name }
}

function improved(study: Study, trial: Trial): boolean {
  if (trial.state !== 'complete' || study.directions.length !== 1) return false
  const d = study.directions[0] as StudyDirection
  const v = lossOf(d, trial.values[0] ?? Number.NaN)
  return study.trials.every((t) => t.state !== 'complete' || lossOf(d, t.values[0] ?? Number.NaN) > v)
}

function constantRun(study: Study, n: number): boolean {
  const complete = completeTrials(study)
  if (complete.length < n) return false
  const tail = study.trials.slice(-n)
  if (tail.some((t) => t.state !== 'complete')) return false
  const first = complete[0]?.values ?? []
  return complete.every((t) => t.values.every((v, j) => v === first[j]))
}

/**
 * A study as a traced algorithm: each step is one trial. The trial's parameters come from the enqueued
 * configurations, then from a multi-fidelity schedule (promotions) or the sampler; the objective runs as a child run
 * (`trial`) that records intermediate reports (metric `intermediate`) and pruning decisions (message `decision`); the
 * study run records each finished trial (message `trial`) and the value, best value and trial counts. Keys: the
 * sampler draws from `foldIn(stepKey, 0)` and the objective from `foldIn(stepKey, 1)`, so a study resumed at trial t
 * continues exactly as an uninterrupted one would. It stops when the trial budget or resource budget is spent, the
 * sampler or schedule is exhausted (a finished grid), `maxFailures` trials fail in a row (a NaN or throwing
 * objective), the objective is constant over `constantTrials` trials, or `patience` trials pass without improvement.
 *
 * @param objective - The objective.
 * @param space - The search space (default the objective's).
 * @param options - Sampler, pruner, schedule, directions, enqueued configurations, a study to resume and stopping
 *   rules.
 * @returns The algorithm.
 * @throws WhiteboxError with code `value` when there is no space or the directions do not match.
 * @concept hyperparameter-optimisation
 * @example
 * ```ts
 * import { functionObjective, studyAlgorithm } from 'whitebox-engine/tune'
 * import { real } from 'whitebox-engine/space'
 * studyAlgorithm(functionObjective((x) => (x as number) ** 2), real(-1, 1)).name // => 'study'
 * ```
 */
export function studyAlgorithm(objective: Objective, space?: Space, options: TuneOptions = {}): Algorithm<StudyState> {
  const searchSpace = space ?? objective.space ?? options.study?.space
  if (searchSpace === undefined) throw new WhiteboxError('value', 'tune: give a space (the objective has none)')
  const directions = options.directions ?? options.study?.directions ?? objective.directions
  if (directions.length !== objective.directions.length)
    throw new WhiteboxError('value', 'tune: the directions do not match the objective')
  const sampler = options.sampler ?? randomSampler()
  if (directions.length > 1 && !sampler.multiObjective)
    throw new WhiteboxError('value', `tune: the ${sampler.name} sampler handles one objective only`)
  const maxFailures = options.maxFailures ?? 20
  const constantTrials = options.constantTrials ?? 20
  const initial = (): Study =>
    options.study ??
    createStudy(searchSpace, {
      name: options.name ?? 'study',
      directions,
      ...(objective.metricNames === undefined ? {} : { metricNames: objective.metricNames }),
    })
  return {
    name: 'study',
    config: {
      objective: objective.name,
      sampler: sampler.name,
      samplerConfig: sampler.config,
      pruner: options.pruner?.name ?? null,
      schedule: options.schedule?.kind ?? null,
      directions: [...directions],
    },
    schema: SCHEMA,
    init: () => ({ study: initial(), failuresInRow: 0, sinceImprovement: 0, spent: 0 }),
    step: (s, ctx) => {
      const picked = choose(s.study, sampler, options, foldIn(ctx.key, 0))
      if (picked === undefined)
        return { ...s, stop: { status: 'completed', message: 'the sampler or schedule has no configurations left' } }
      const a = picked.assignment
      const assignment = {
        source: picked.source,
        ...(a === undefined ? {} : { budget: a.budget, bracket: a.bracket, rung: a.rung }),
        ...(a?.kind === 'promote' ? { parent: a.parent } : {}),
      }
      const { trial, spent } = evaluateTrial(
        objective,
        s.study,
        picked.params,
        assignment,
        foldIn(ctx.key, 1),
        ctx.run,
        options,
      )
      ctx.run.message(ctx.t, 'trial', {
        number: trial.number,
        params: trial.params as Json,
        state: trial.state,
        values: trial.values.map((v) => (Number.isFinite(v) ? v : String(v))),
      })
      const better = improved(s.study, trial)
      const study = { ...s.study, trials: [...s.study.trials, trial] }
      const failuresInRow = trial.state === 'failed' ? s.failuresInRow + 1 : 0
      const sinceImprovement = better ? 0 : s.sinceImprovement + 1
      const next = { study, failuresInRow, sinceImprovement, spent: s.spent + spent }
      let stop: StudyState['stop']
      if (failuresInRow >= maxFailures)
        stop = { status: 'failed', message: `${failuresInRow} trials failed in a row: ${trial.message ?? ''}` }
      else if (constantRun(study, constantTrials))
        stop = { status: 'converged', message: `the objective was constant over ${constantTrials} trials` }
      else if (options.patience !== undefined && sinceImprovement >= options.patience)
        stop = { status: 'converged', message: `no improvement in ${options.patience} trials` }
      else if (options.budget !== undefined && next.spent >= options.budget)
        stop = { status: 'completed', message: 'the resource budget is spent' }
      else if (sampler.exhausted?.(study) === true && options.schedule === undefined)
        stop = { status: 'completed', message: 'the sampler has no configurations left' }
      if (stop !== undefined) ctx.run.message(ctx.t + 1, 'stop', stop.message)
      return stop === undefined ? next : { ...next, stop }
    },
    done: (s) => s.stop?.status,
    observe: (s) => {
      const last = s.study.trials.at(-1)
      const counts = { complete: 0, pruned: 0, failed: 0 }
      for (const t of s.study.trials) counts[t.state] += 1
      const best = directions.length === 1 ? bestTrial(s.study) : undefined
      return {
        metrics: {
          value: last?.state === 'complete' ? (last.values[0] ?? Number.NaN) : Number.NaN,
          best: best?.values[0] ?? Number.NaN,
          ...counts,
        },
      }
    },
    serialise: (s) => toJson(s as unknown as Encodable),
    deserialise: (json) => fromJson(json) as unknown as StudyState,
  }
}

/**
 * Select parameters: run a study of `nTrials` trials of an objective over a space (Optuna's `study.optimize`, Ray
 * Tune's `Tuner.fit`), traced as a run whose steps are trials. Pass `study` to resume a study: its trials are kept, new
 * ones continue the numbering, and the trials drawn are those an uninterrupted run would have drawn.
 *
 * @param objective - The objective (any `Objective`; see `functionObjective`, `estimatorObjective`,
 *   `trainingObjective`, `agentObjective`).
 * @param space - The search space (default the objective's, as an estimator's `hyperSpace`).
 * @param options - Trials, sampler, pruner, schedule, directions, enqueued configurations, a study to resume, stopping
 *   rules, and the run options (key or seed, tracer, policy, parent).
 * @returns The study, its best trial(s), and how the run ended.
 * @concept hyperparameter-optimisation
 * @example
 * ```ts
 * import { functionObjective, tune, gridSampler } from 'whitebox-engine/tune'
 * import { dict, int } from 'whitebox-engine/space'
 * const objective = functionObjective((p) => ((p as { k: number }).k - 3) ** 2)
 * const result = tune(objective, dict({ k: int(0, 5) }), { sampler: gridSampler({ points: 6 }), nTrials: 10 })
 * result.best?.params // => { k: 3 }
 * result.study.trials.length // => 6
 * ```
 */
export function tune(objective: Objective, space?: Space, options: TuneOptions = {}): TuneResult {
  const algorithm = studyAlgorithm(objective, space, options)
  const nTrials = options.nTrials ?? 100
  const {
    nTrials: _n,
    budget: _b,
    sampler: _s,
    pruner: _p,
    schedule: _sc,
    directions: _d,
    enqueue: _e,
    study: resumed,
    name: _name,
    patience: _pa,
    maxFailures: _mf,
    constantTrials: _ct,
    catchErrors: _ce,
    ...run
  } = options
  // A resumed study starts at step `trials.length`, so its step keys continue those of the earlier run.
  const result = runAlgorithm(algorithm, {
    ...run,
    maxSteps: nTrials,
    ...(resumed === undefined ? {} : { start: { state: algorithm.init(makeKey(0)), step: resumed.trials.length } }),
  })
  const study = result.state.study
  return {
    study,
    best: study.directions.length === 1 ? bestTrial(study) : undefined,
    bestTrials: bestTrials(study),
    status: result.status,
    message: result.state.stop?.message ?? 'the trial budget is spent',
    steps: result.steps,
    runId: result.runId,
  }
}
