/**
 * Running a parameter search: `tune` runs a study of an objective over a space for a number of trials with a sampler,
 * an optional pruner and multi-fidelity schedule, traced as a run whose steps are trials (Optuna's `study.optimize`,
 * Ray Tune's `Tuner.fit`), and `studyAlgorithm` exposes the same loop as an `Algorithm`.
 */

import { type Encodable, fromJson, type Json, toJson, WhiteboxError } from '../base/index.ts'
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
import type { Sampler } from './samplers.ts'
import { type Assignment, nextAssignment, type Schedule } from './schedules.ts'
import {
  bestTrial,
  bestTrials,
  completeTrials,
  createStudy,
  type IntermediateValue,
  lossOf,
  type Study,
  type StudyDirection,
  type Trial,
  type TrialState,
} from './study.ts'
import { tpeSampler } from './tpe.ts'

/** Options of {@link tune} and {@link studyAlgorithm}. */
export interface TuneOptions extends Partial<Omit<RunOptions, 'maxSteps' | 'start'>> {
  /** The number of trials to run (default 100): the step budget of the run. */
  readonly nTrials?: number
  /** Stop once the trials' budgets (resource units from a schedule, or reported steps) sum to this. */
  readonly budget?: number
  /** The sampler (default `tpeSampler()`, sequential model-based optimisation as Optuna's default). */
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
  const sampler = options.sampler ?? tpeSampler()
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
