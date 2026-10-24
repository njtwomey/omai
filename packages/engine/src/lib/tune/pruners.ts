import { type Json, quantileOf, WhiteboxError } from '../base/index.ts'
import { crc32 } from '../information/index.ts'
import type { SpaceValue } from '../space/index.ts'
import type { IntermediateValue, Study, StudyDirection, Trial } from './study.ts'

/** The trial a pruner decides about: its reports so far (in report order) and the rungs it has passed. */
export interface RunningTrial {
  readonly number: number
  readonly params: SpaceValue
  readonly intermediate: readonly IntermediateValue[]
  /** Values at the successive-halving rungs the trial has passed so far. */
  readonly rungValues: readonly number[]
  readonly budget?: number
  readonly bracket?: number
}

/** A pruning decision, and the rung values the trial has recorded (successive halving keeps them on the trial). */
export interface PruneDecision {
  readonly prune: boolean
  readonly rungValues?: readonly number[]
}

/**
 * The pruner contract: after each intermediate report, decide from the finished trials of the study and the running
 * trial's reports whether to stop the trial (Optuna's `BasePruner.prune`). Pruners are pure: everything they remember
 * (successive halving's rung values) lives on the trials.
 *
 * @concept early-stopping
 */
export interface Pruner {
  readonly name: string
  readonly config: Json
  prune(study: Study, trial: RunningTrial): PruneDecision
}

const KEEP: PruneDecision = { prune: false }
const PRUNE: PruneDecision = { prune: true }

function direction(study: Study, name: string): StudyDirection {
  if (study.directions.length !== 1) throw new WhiteboxError('value', `${name}: pruning needs a single objective`)
  return study.directions[0] as StudyDirection
}

function lastStep(intermediate: readonly IntermediateValue[]): number | undefined {
  if (intermediate.length === 0) return undefined
  return intermediate.reduce((m, iv) => Math.max(m, iv.step), Number.NEGATIVE_INFINITY)
}

function valueAt(intermediate: readonly IntermediateValue[], step: number): number | undefined {
  return intermediate.find((iv) => iv.step === step)?.value
}

function nonNegativeInteger(x: number, what: string, name: string): void {
  if (!(Number.isInteger(x) && x >= 0))
    throw new WhiteboxError('value', `${name}: ${what} must be a non-negative integer`)
}

/** Whether `step` is the first report at or after the last pruning checkpoint (Optuna's `_is_first_in_interval_step`). */
function firstInInterval(step: number, steps: readonly number[], warmup: number, interval: number): boolean {
  const nearestLower = Math.floor((step - warmup) / interval) * interval + warmup
  let secondLast = -1
  for (const s of steps) if (s > secondLast && s !== step) secondLast = s
  return secondLast < nearestLower
}

/** Options of {@link percentilePruner} and {@link medianPruner}. */
export interface PercentilePrunerOptions {
  /** Complete trials needed before pruning starts (default 5). */
  readonly nStartupTrials?: number
  /** Steps of a trial before it may be pruned (default 0). */
  readonly nWarmupSteps?: number
  /** Check every this many steps after the warm-up (default 1). */
  readonly intervalSteps?: number
  /** Complete trials reporting at a step needed to compare against (default 1). */
  readonly nMinTrials?: number
}

/**
 * Prune a trial whose best intermediate value so far is worse than the given percentile of the complete trials'
 * values at the same step (Optuna's `PercentilePruner`), after `nStartupTrials` complete trials and `nWarmupSteps`
 * steps, every `intervalSteps` steps. A NaN best value is pruned.
 *
 * @param percentile - The percentile in [0, 100] (25 keeps the top quarter of minimised values).
 * @param options - Startup trials, warm-up steps, interval and the minimum number of trials compared.
 * @returns The pruner.
 * @throws WhiteboxError with code `value` for a percentile outside [0, 100] or invalid counts.
 * @concept early-stopping
 * @example
 * ```ts
 * import { percentilePruner } from 'whitebox-engine/tune'
 * percentilePruner(25).name // => 'percentile'
 * ```
 */
export function percentilePruner(percentile: number, options: PercentilePrunerOptions = {}): Pruner {
  if (!(percentile >= 0 && percentile <= 100))
    throw new WhiteboxError('value', `percentilePruner: percentile must be in [0, 100], got ${percentile}`)
  const nStartup = options.nStartupTrials ?? 5
  const warmup = options.nWarmupSteps ?? 0
  const interval = options.intervalSteps ?? 1
  const nMin = options.nMinTrials ?? 1
  nonNegativeInteger(nStartup, 'nStartupTrials', 'percentilePruner')
  nonNegativeInteger(warmup, 'nWarmupSteps', 'percentilePruner')
  if (!(Number.isInteger(interval) && interval >= 1))
    throw new WhiteboxError('value', 'percentilePruner: intervalSteps must be at least 1')
  if (!(Number.isInteger(nMin) && nMin >= 1))
    throw new WhiteboxError('value', 'percentilePruner: nMinTrials must be at least 1')
  return {
    name: percentile === 50 ? 'median' : 'percentile',
    config: { percentile, nStartupTrials: nStartup, nWarmupSteps: warmup, intervalSteps: interval, nMinTrials: nMin },
    prune: (study, trial) => {
      const complete = study.trials.filter((t) => t.state === 'complete')
      if (complete.length === 0 || complete.length < nStartup) return KEEP
      const step = lastStep(trial.intermediate)
      if (step === undefined || step < warmup) return KEEP
      if (
        !firstInInterval(
          step,
          trial.intermediate.map((iv) => iv.step),
          warmup,
          interval,
        )
      )
        return KEEP
      const d = direction(study, 'percentilePruner')
      const own = trial.intermediate.map((iv) => iv.value).filter((v) => !Number.isNaN(v))
      if (own.length === 0) return PRUNE
      const best = d === 'maximize' ? own.reduce((a, b) => Math.max(a, b)) : own.reduce((a, b) => Math.min(a, b))
      const others = complete.flatMap((t) => {
        const v = valueAt(t.intermediate, step)
        return v === undefined ? [] : [v]
      })
      if (others.length < nMin) return KEEP
      const finite = others.filter((v) => !Number.isNaN(v))
      if (finite.length === 0) return KEEP
      const q = (d === 'maximize' ? 100 - percentile : percentile) / 100
      const p = quantileOf(finite, q)
      return { prune: d === 'maximize' ? best < p : best > p }
    },
  }
}

/**
 * Prune a trial whose best intermediate value is worse than the median of the complete trials at the same step
 * (Optuna's `MedianPruner`, the percentile pruner at 50).
 *
 * @param options - Startup trials (default 5), warm-up steps, interval and minimum trials.
 * @returns The pruner.
 * @concept early-stopping
 * @example
 * ```ts
 * import { medianPruner } from 'whitebox-engine/tune'
 * medianPruner({ nStartupTrials: 2 }).config // => { percentile: 50, nStartupTrials: 2, nWarmupSteps: 0, intervalSteps: 1, nMinTrials: 1 }
 * ```
 */
export function medianPruner(options: PercentilePrunerOptions = {}): Pruner {
  return percentilePruner(50, options)
}

/**
 * Prune a trial whose latest intermediate value leaves `[lower, upper]` or is NaN (Optuna's `ThresholdPruner`): a
 * guard against divergence.
 *
 * @param options - Bounds (at least one), warm-up steps and interval.
 * @returns The pruner.
 * @throws WhiteboxError with code `value` when neither bound is given or lower exceeds upper.
 * @concept early-stopping
 * @example
 * ```ts
 * import { thresholdPruner, createStudy } from 'whitebox-engine/tune'
 * import { real } from 'whitebox-engine/space'
 * const pruner = thresholdPruner({ upper: 10 })
 * pruner.prune(createStudy(real(0, 1)), { number: 0, params: 0, intermediate: [{ step: 0, value: 12 }], rungValues: [] }).prune // => true
 * ```
 */
export function thresholdPruner(
  options: {
    readonly lower?: number
    readonly upper?: number
    readonly nWarmupSteps?: number
    readonly intervalSteps?: number
  } = {},
): Pruner {
  if (options.lower === undefined && options.upper === undefined)
    throw new WhiteboxError('value', 'thresholdPruner: give lower or upper')
  const lower = options.lower ?? Number.NEGATIVE_INFINITY
  const upper = options.upper ?? Number.POSITIVE_INFINITY
  if (lower > upper) throw new WhiteboxError('value', 'thresholdPruner: lower must not exceed upper')
  const warmup = options.nWarmupSteps ?? 0
  const interval = options.intervalSteps ?? 1
  nonNegativeInteger(warmup, 'nWarmupSteps', 'thresholdPruner')
  if (!(Number.isInteger(interval) && interval >= 1))
    throw new WhiteboxError('value', 'thresholdPruner: intervalSteps must be at least 1')
  return {
    name: 'threshold',
    config: {
      lower: Number.isFinite(lower) ? lower : null,
      upper: Number.isFinite(upper) ? upper : null,
      nWarmupSteps: warmup,
      intervalSteps: interval,
    },
    prune: (_study, trial) => {
      const step = lastStep(trial.intermediate)
      if (step === undefined || step < warmup) return KEEP
      if (
        !firstInInterval(
          step,
          trial.intermediate.map((iv) => iv.step),
          warmup,
          interval,
        )
      )
        return KEEP
      const latest = valueAt(trial.intermediate, step) ?? Number.NaN
      return { prune: Number.isNaN(latest) || latest < lower || latest > upper }
    },
  }
}

function nanExtreme(values: readonly number[], max: boolean): number {
  const finite = values.filter((v) => !Number.isNaN(v))
  if (finite.length === 0) return Number.NaN
  return max ? finite.reduce((a, b) => Math.max(a, b)) : finite.reduce((a, b) => Math.min(a, b))
}

/**
 * Prune a trial that has not improved on its best value before the last `patience` + 1 reports by more than
 * `minDelta` (Optuna's `PatientPruner`); with a wrapped pruner, such trials are pruned only when it agrees.
 *
 * @param wrapped - A pruner to consult once patience runs out, or `undefined` to prune then.
 * @param patience - Reports without improvement tolerated.
 * @param minDelta - The improvement that counts (default 0).
 * @returns The pruner.
 * @throws WhiteboxError with code `value` for a negative patience or delta.
 * @concept early-stopping
 * @example
 * ```ts
 * import { patientPruner, createStudy } from 'whitebox-engine/tune'
 * import { real } from 'whitebox-engine/space'
 * const reports = [3, 2, 2.5, 2.4].map((value, step) => ({ step, value }))
 * patientPruner(undefined, 1).prune(createStudy(real(0, 1)), { number: 0, params: 0, intermediate: reports, rungValues: [] }).prune // => true
 * ```
 */
export function patientPruner(wrapped: Pruner | undefined, patience: number, minDelta = 0): Pruner {
  nonNegativeInteger(patience, 'patience', 'patientPruner')
  if (!(minDelta >= 0)) throw new WhiteboxError('value', 'patientPruner: minDelta must be non-negative')
  return {
    name: 'patient',
    config: { patience, minDelta, wrapped: wrapped?.name ?? null },
    prune: (study, trial) => {
      const steps = trial.intermediate.map((iv) => iv.step).sort((a, b) => a - b)
      if (steps.length <= patience + 1) return KEEP
      const cut = steps.length - patience - 1
      const value = (s: number): number => valueAt(trial.intermediate, s) ?? Number.NaN
      const before = steps.slice(0, cut).map(value)
      const after = steps.slice(cut).map(value)
      const maybe =
        direction(study, 'patientPruner') === 'minimize'
          ? nanExtreme(before, false) + minDelta < nanExtreme(after, false)
          : nanExtreme(before, true) - minDelta > nanExtreme(after, true)
      if (!maybe) return KEEP
      return wrapped === undefined ? PRUNE : wrapped.prune(study, trial)
    },
  }
}

/** Options of {@link successiveHalvingPruner}. */
export interface SuccessiveHalvingPrunerOptions {
  /** Steps before the first rung, or `'auto'` (default): 1 / 100 of the longest complete trial, at least 1. */
  readonly minResource?: number | 'auto'
  /** eta: a trial passes a rung when in the top 1 / eta of the values recorded there (default 4). */
  readonly reductionFactor?: number
  /** s: the first rung is at minResource * eta^s (default 0). */
  readonly minEarlyStoppingRate?: number
  /** Trials that must reach a rung before any passes it (default 0). */
  readonly bootstrapCount?: number
}

/**
 * The longest complete trial (its last reported step) as Optuna's 'auto' settings see it: estimated once, at the first
 * prune call after a complete trial with reports exists, and frozen from then on. In a sequential study that call is
 * made by the first later trial that reports (possibly the running one), so the estimate is a function of the trials.
 */
function frozenLongest(study: Study, number: number): number | undefined {
  const firstComplete = study.trials.find((t) => t.state === 'complete' && t.intermediate.length > 0)
  if (firstComplete === undefined) return undefined
  const estimator = study.trials.find((t) => t.number > firstComplete.number && t.intermediate.length > 0)
  const at = estimator?.number ?? number
  let longest = Number.NEGATIVE_INFINITY
  for (const t of study.trials)
    if (t.number < at && t.state === 'complete' && t.intermediate.length > 0)
      longest = Math.max(longest, lastStep(t.intermediate) ?? 0)
  return longest
}

/** Successive-halving decisions against a set of competing trials (the bracket's, for Hyperband). */
function halving(
  trials: readonly Trial[],
  trial: RunningTrial,
  d: StudyDirection,
  minResource: number,
  eta: number,
  rate: number,
  bootstrap: number,
): PruneDecision {
  const step = lastStep(trial.intermediate)
  if (step === undefined) return KEEP
  const value = valueAt(trial.intermediate, step) ?? Number.NaN
  const rungs = [...trial.rungValues]
  for (;;) {
    const rung = rungs.length
    if (step < minResource * eta ** (rate + rung)) return { prune: false, rungValues: rungs }
    if (Number.isNaN(value)) return { prune: true, rungValues: rungs }
    rungs.push(value)
    const competing = trials.flatMap((t) => {
      const v = t.rungValues?.[rung]
      return v === undefined ? [] : [v]
    })
    competing.push(value)
    if (competing.length <= bootstrap) return { prune: true, rungValues: rungs }
    competing.sort((a, b) => a - b)
    let idx = Math.floor(competing.length / eta) - 1
    if (idx === -1) idx = 0
    const promotable =
      d === 'maximize' ? value >= (competing.at(-(idx + 1)) as number) : value <= (competing[idx] as number)
    if (!promotable) return { prune: true, rungValues: rungs }
  }
}

/**
 * Asynchronous successive halving (ASHA; Optuna's `SuccessiveHalvingPruner`): rung k sits at
 * minResource * eta^(s + k) steps; a trial reaching a rung records its value there and continues only if that value is
 * in the top 1 / eta of all values recorded at the rung so far. Decisions never wait for other trials.
 *
 * @param options - Minimum resource (or `'auto'`), reduction factor eta, minimum early-stopping rate s and bootstrap
 *   count.
 * @returns The pruner.
 * @throws WhiteboxError with code `value` for invalid settings.
 * @concept successive-halving
 * @reference Li et al. (2020), A system for massively parallel hyperparameter tuning (ASHA).
 * @example
 * ```ts
 * import { successiveHalvingPruner } from 'whitebox-engine/tune'
 * successiveHalvingPruner({ minResource: 1, reductionFactor: 3 }).name // => 'successiveHalving'
 * ```
 */
export function successiveHalvingPruner(options: SuccessiveHalvingPrunerOptions = {}): Pruner {
  const minResource = options.minResource ?? 'auto'
  const eta = options.reductionFactor ?? 4
  const rate = options.minEarlyStoppingRate ?? 0
  const bootstrap = options.bootstrapCount ?? 0
  if (minResource !== 'auto' && !(Number.isInteger(minResource) && minResource >= 1))
    throw new WhiteboxError('value', 'successiveHalvingPruner: minResource must be a positive integer or auto')
  if (!(Number.isInteger(eta) && eta >= 2))
    throw new WhiteboxError('value', 'successiveHalvingPruner: reductionFactor must be an integer of at least 2')
  nonNegativeInteger(rate, 'minEarlyStoppingRate', 'successiveHalvingPruner')
  nonNegativeInteger(bootstrap, 'bootstrapCount', 'successiveHalvingPruner')
  if (bootstrap > 0 && minResource === 'auto')
    throw new WhiteboxError('value', 'successiveHalvingPruner: bootstrapCount needs an explicit minResource')
  return {
    name: 'successiveHalving',
    config: { minResource, reductionFactor: eta, minEarlyStoppingRate: rate, bootstrapCount: bootstrap },
    prune: (study, trial) => {
      const longest = minResource === 'auto' ? frozenLongest(study, trial.number) : undefined
      if (minResource === 'auto' && longest === undefined) return KEEP
      const r = minResource === 'auto' ? Math.max(Math.floor((longest ?? 0) / 100), 1) : minResource
      return halving(study.trials, trial, direction(study, 'successiveHalvingPruner'), r, eta, rate, bootstrap)
    },
  }
}

/** The brackets of a Hyperband pruner: trial budgets per bracket and their total. */
function hyperbandBrackets(maxResource: number, minResource: number, eta: number): number[] {
  const n = Math.floor(Math.log(maxResource / minResource) / Math.log(eta)) + 1
  return Array.from({ length: n }, (_, b) => {
    const s = n - 1 - b
    return Math.ceil((n * eta ** s) / (s + 1))
  })
}

/**
 * The Hyperband bracket of a trial, as Optuna assigns it: CRC-32 of `<study name>_<trial number>` modulo the total
 * trial budget, walked through the brackets' budgets.
 *
 * @param studyName - The study's name.
 * @param number - The trial number.
 * @param budgets - Trial budgets per bracket.
 * @returns The bracket index.
 * @example
 * ```ts
 * import { hyperbandBracketOf } from 'whitebox-engine/tune'
 * hyperbandBracketOf('study', 0, [3, 2, 1]) // => 0
 * ```
 */
export function hyperbandBracketOf(studyName: string, number: number, budgets: readonly number[]): number {
  const total = budgets.reduce((a, b) => a + b, 0)
  let n = crc32(`${studyName}_${number}`) % total
  for (const [b, budget] of budgets.entries()) {
    n -= budget
    if (n < 0) return b
  }
  return budgets.length - 1
}

/**
 * Hyperband (Optuna's `HyperbandPruner`): several successive-halving brackets with minimum early-stopping rates
 * 0, 1, ..., each trial assigned to one bracket (by a hash of the study name and trial number, weighted by the
 * brackets' trial budgets) and judged only against trials of its bracket. It hedges between aggressive and cautious
 * early stopping.
 *
 * @param options - Minimum resource (default 1), maximum resource (or `'auto'`: the longest complete trial plus one),
 *   reduction factor (default 3) and bootstrap count.
 * @returns The pruner.
 * @throws WhiteboxError with code `value` for invalid settings.
 * @concept successive-halving
 * @reference Li et al. (2018), Hyperband: a novel bandit-based approach to hyperparameter optimization.
 * @example
 * ```ts
 * import { hyperbandPruner } from 'whitebox-engine/tune'
 * hyperbandPruner({ maxResource: 27 }).config // => { minResource: 1, maxResource: 27, reductionFactor: 3, bootstrapCount: 0 }
 * ```
 */
export function hyperbandPruner(
  options: {
    readonly minResource?: number
    readonly maxResource?: number | 'auto'
    readonly reductionFactor?: number
    readonly bootstrapCount?: number
  } = {},
): Pruner {
  const minResource = options.minResource ?? 1
  const maxResource = options.maxResource ?? 'auto'
  const eta = options.reductionFactor ?? 3
  const bootstrap = options.bootstrapCount ?? 0
  if (maxResource !== 'auto' && !(Number.isInteger(maxResource) && maxResource >= minResource))
    throw new WhiteboxError('value', 'hyperbandPruner: maxResource must be an integer of at least minResource, or auto')
  if (!(Number.isInteger(minResource) && minResource >= 1))
    throw new WhiteboxError('value', 'hyperbandPruner: minResource must be a positive integer')
  if (!(Number.isInteger(eta) && eta >= 2))
    throw new WhiteboxError('value', 'hyperbandPruner: reductionFactor must be an integer of at least 2')
  if (bootstrap > 0 && maxResource === 'auto')
    throw new WhiteboxError('value', 'hyperbandPruner: bootstrapCount needs an explicit maxResource')
  return {
    name: 'hyperband',
    config: { minResource, maxResource, reductionFactor: eta, bootstrapCount: bootstrap },
    prune: (study, trial) => {
      const longest = maxResource === 'auto' ? frozenLongest(study, trial.number) : undefined
      if (maxResource === 'auto' && longest === undefined) return KEEP
      const max = maxResource === 'auto' ? (longest ?? 0) + 1 : maxResource
      const budgets = hyperbandBrackets(max, minResource, eta)
      const bracket = hyperbandBracketOf(study.name, trial.number, budgets)
      const peers = study.trials.filter((t) => hyperbandBracketOf(study.name, t.number, budgets) === bracket)
      return halving(peers, trial, direction(study, 'hyperbandPruner'), minResource, eta, bracket, bootstrap)
    },
  }
}

/**
 * A pruner that never prunes (Optuna's `NopPruner`), for studies that report intermediate values only to record them.
 *
 * @returns The pruner.
 * @example
 * ```ts
 * import { nopPruner, createStudy } from 'whitebox-engine/tune'
 * import { real } from 'whitebox-engine/space'
 * nopPruner().prune(createStudy(real(0, 1)), { number: 0, params: 0, intermediate: [], rungValues: [] }).prune // => false
 * ```
 */
export function nopPruner(): Pruner {
  return { name: 'nop', config: {}, prune: () => KEEP }
}
