import { WhiteboxError } from '../base/index.ts'
import type { SpaceValue } from '../space/index.ts'
import { lossOf, type Study, type StudyDirection, type Trial } from './study.ts'

/** One rung of a synchronous bracket: how many configurations run, and with how much resource each. */
export interface Rung {
  readonly rung: number
  readonly nConfigs: number
  readonly resource: number
}

/** A bracket of synchronous successive halving: rungs of shrinking size and growing resource. */
export interface Bracket {
  readonly bracket: number
  readonly rungs: readonly Rung[]
}

/** Synchronous multi-fidelity schedules as data: one bracket (successive halving) or several (Hyperband). */
export interface SynchronousSchedule {
  readonly kind: 'successiveHalving' | 'hyperband'
  readonly reductionFactor: number
  readonly minResource: number
  readonly maxResource: number
  readonly brackets: readonly Bracket[]
  /** Start again with the first bracket once the last is done (default false: the schedule then ends the study). */
  readonly repeat: boolean
}

/** Asynchronous successive halving as data: the rung resources; promotions happen whenever a rung allows one. */
export interface AshaSchedule {
  readonly kind: 'asha'
  readonly reductionFactor: number
  readonly minResource: number
  readonly maxResource: number
  /** Resource of each rung, minResource * eta^(s + k), the last capped at maxResource. */
  readonly rungs: readonly number[]
}

/** A multi-fidelity schedule. */
export type Schedule = SynchronousSchedule | AshaSchedule

/** What a schedule asks for next: a new configuration, or a promotion of an earlier one, with its resource. */
export type Assignment =
  | { readonly kind: 'sample'; readonly bracket: number; readonly rung: number; readonly budget: number }
  | {
      readonly kind: 'promote'
      readonly bracket: number
      readonly rung: number
      readonly budget: number
      readonly params: SpaceValue
      readonly parent: number
    }

function checkEta(eta: number, name: string): void {
  if (!(Number.isInteger(eta) && eta >= 2))
    throw new WhiteboxError('value', `${name}: reductionFactor must be an integer of at least 2`)
}

function checkResources(min: number, max: number, name: string): void {
  if (!(min > 0 && max >= min))
    throw new WhiteboxError('value', `${name}: need 0 < minResource <= maxResource, got ${min} and ${max}`)
}

/** floor(log_eta(x)), corrected for rounding when x is an exact power of eta. */
function floorLog(x: number, eta: number): number {
  let s = Math.floor(Math.log(x) / Math.log(eta))
  while (eta ** (s + 1) <= x * (1 + 1e-12)) s += 1
  while (s > 0 && eta ** s > x * (1 + 1e-12)) s -= 1
  return s
}

/**
 * Synchronous successive halving (Jamieson and Talwalkar 2016) as data: `nConfigs` configurations at minResource,
 * then the best 1 / eta of them at eta times the resource, and so on up to maxResource.
 *
 * @param options - Configurations in the first rung, minimum and maximum resource and the reduction factor eta
 *   (default 3).
 * @returns The schedule (one bracket).
 * @throws WhiteboxError with code `value` for invalid resources or eta.
 * @concept successive-halving
 * @example
 * ```ts
 * import { successiveHalvingSchedule } from 'whitebox-engine/tune'
 * successiveHalvingSchedule({ nConfigs: 9, minResource: 1, maxResource: 9 }).brackets[0]?.rungs
 * // => [{ rung: 0, nConfigs: 9, resource: 1 }, { rung: 1, nConfigs: 3, resource: 3 }, { rung: 2, nConfigs: 1, resource: 9 }]
 * ```
 */
export function successiveHalvingSchedule(options: {
  readonly nConfigs: number
  readonly minResource?: number
  readonly maxResource: number
  readonly reductionFactor?: number
  readonly repeat?: boolean
}): SynchronousSchedule {
  const eta = options.reductionFactor ?? 3
  const min = options.minResource ?? 1
  const max = options.maxResource
  checkEta(eta, 'successiveHalvingSchedule')
  checkResources(min, max, 'successiveHalvingSchedule')
  if (!(Number.isInteger(options.nConfigs) && options.nConfigs >= 1))
    throw new WhiteboxError('value', 'successiveHalvingSchedule: nConfigs must be a positive integer')
  const s = floorLog(max / min, eta)
  const rungs: Rung[] = []
  for (let i = 0; i <= s; i++) {
    const n = Math.floor(options.nConfigs / eta ** i)
    if (n < 1) break
    rungs.push({ rung: i, nConfigs: n, resource: Math.min(max, min * eta ** i) })
  }
  return {
    kind: 'successiveHalving',
    reductionFactor: eta,
    minResource: min,
    maxResource: max,
    brackets: [{ bracket: 0, rungs }],
    repeat: options.repeat ?? false,
  }
}

/**
 * Hyperband's brackets (Li et al. 2018, Algorithm 1) as data: with s_max = floor(log_eta(R / r_min)), bracket s
 * (s = s_max, ..., 0) starts n = ceil((s_max + 1) eta^s / (s + 1)) configurations at r = R eta^-s and halves them by
 * eta per rung. For R = 81, eta = 3 this is the paper's table: (81, 1), (34, 3), (15, 9), (8, 27), (5, 81).
 *
 * @param options - Maximum resource R, minimum resource (default 1), eta (default 3), and whether to repeat.
 * @returns The schedule.
 * @throws WhiteboxError with code `value` for invalid resources or eta.
 * @concept successive-halving
 * @reference Li, Jamieson, DeSalvo, Rostamizadeh and Talwalkar (2018), Hyperband, JMLR 18.
 * @example
 * ```ts
 * import { hyperbandSchedule } from 'whitebox-engine/tune'
 * hyperbandSchedule({ maxResource: 81 }).brackets.map((b) => [b.rungs[0]?.nConfigs, b.rungs[0]?.resource])
 * // => [[81, 1], [34, 3], [15, 9], [8, 27], [5, 81]]
 * ```
 */
export function hyperbandSchedule(options: {
  readonly maxResource: number
  readonly minResource?: number
  readonly reductionFactor?: number
  readonly repeat?: boolean
}): SynchronousSchedule {
  const eta = options.reductionFactor ?? 3
  const min = options.minResource ?? 1
  const max = options.maxResource
  checkEta(eta, 'hyperbandSchedule')
  checkResources(min, max, 'hyperbandSchedule')
  const sMax = floorLog(max / min, eta)
  const brackets: Bracket[] = []
  for (let s = sMax; s >= 0; s--) {
    const n = Math.ceil(((sMax + 1) * eta ** s) / (s + 1))
    const r = max * eta ** -s
    const rungs = Array.from({ length: s + 1 }, (_, i) => ({
      rung: i,
      nConfigs: Math.floor(n * eta ** -i + 1e-9),
      resource: r * eta ** i,
    }))
    brackets.push({ bracket: sMax - s, rungs })
  }
  return {
    kind: 'hyperband',
    reductionFactor: eta,
    minResource: min,
    maxResource: max,
    brackets,
    repeat: options.repeat ?? false,
  }
}

/**
 * Asynchronous successive halving (ASHA, Li et al. 2020) as data: rung k gets minResource * eta^(s + k), capped at
 * maxResource. Run as a schedule, a configuration is promoted to rung k + 1 as soon as it is in the top 1 / eta of the
 * configurations finished at rung k; otherwise a new configuration starts at the bottom rung.
 *
 * @param options - Minimum and maximum resource, eta (default 3) and the minimum early-stopping rate s (default 0).
 * @returns The schedule.
 * @throws WhiteboxError with code `value` for invalid resources or eta.
 * @concept successive-halving
 * @reference Li et al. (2020), A system for massively parallel hyperparameter tuning, MLSys.
 * @example
 * ```ts
 * import { ashaSchedule } from 'whitebox-engine/tune'
 * ashaSchedule({ minResource: 1, maxResource: 10 }).rungs // => [1, 3, 9, 10]
 * ```
 */
export function ashaSchedule(options: {
  readonly minResource?: number
  readonly maxResource: number
  readonly reductionFactor?: number
  readonly minEarlyStoppingRate?: number
}): AshaSchedule {
  const eta = options.reductionFactor ?? 3
  const min = options.minResource ?? 1
  const max = options.maxResource
  const s = options.minEarlyStoppingRate ?? 0
  checkEta(eta, 'ashaSchedule')
  checkResources(min, max, 'ashaSchedule')
  const rungs: number[] = []
  for (let k = 0; ; k++) {
    const r = min * eta ** (s + k)
    if (r >= max) {
      rungs.push(max)
      break
    }
    rungs.push(r)
  }
  return { kind: 'asha', reductionFactor: eta, minResource: min, maxResource: max, rungs }
}

/** The first-objective loss of a finished trial (Infinity unless complete), so failures rank last. */
function rankLoss(direction: StudyDirection, t: Trial): number {
  if (t.state !== 'complete') return Number.POSITIVE_INFINITY
  const v = lossOf(direction, t.values[0] ?? Number.NaN)
  return Number.isNaN(v) ? Number.POSITIVE_INFINITY : v
}

/** The complete trials of a rung, best first (ties by trial number). */
function ranked(study: Study, trials: readonly Trial[]): Trial[] {
  const d = study.directions[0] as StudyDirection
  return trials
    .filter((t) => t.state === 'complete')
    .sort((a, b) => rankLoss(d, a) - rankLoss(d, b) || a.number - b.number)
}

function promotion(trial: Trial, bracket: number, rung: number, budget: number): Assignment {
  return { kind: 'promote', bracket, rung, budget, params: trial.params, parent: trial.number }
}

function nextSynchronous(schedule: SynchronousSchedule, study: Study): Assignment | undefined {
  const nb = schedule.brackets.length
  for (let k = 0; schedule.repeat || k < nb; k++) {
    const bracket = schedule.brackets[k % nb] as Bracket
    const inBracket = study.trials.filter((t) => t.bracket === k)
    // A repeated schedule visits each round's brackets in turn; stop scanning past the first unfinished one.
    for (const r of bracket.rungs) {
      const here = inBracket.filter((t) => t.rung === r.rung)
      if (r.rung === 0) {
        if (here.length < r.nConfigs) return { kind: 'sample', bracket: k, rung: 0, budget: r.resource }
        continue
      }
      const promoted = new Set(here.map((t) => t.parent))
      const below = ranked(
        study,
        inBracket.filter((t) => t.rung === r.rung - 1),
      ).slice(0, r.nConfigs)
      const next = below.find((t) => !promoted.has(t.number))
      if (next !== undefined) return promotion(next, k, r.rung, r.resource)
    }
    if (schedule.repeat && k >= study.trials.length + nb) return undefined
  }
  return undefined
}

function nextAsha(schedule: AshaSchedule, study: Study): Assignment {
  const eta = schedule.reductionFactor
  const rungs = schedule.rungs
  for (let k = rungs.length - 2; k >= 0; k--) {
    const here = ranked(
      study,
      study.trials.filter((t) => t.rung === k),
    )
    const all = study.trials.filter((t) => t.rung === k).length
    const top = here.slice(0, Math.floor(all / eta))
    const promoted = new Set(study.trials.filter((t) => t.rung === k + 1).map((t) => t.parent))
    const next = top.find((t) => !promoted.has(t.number))
    if (next !== undefined) return promotion(next, 0, k + 1, rungs[k + 1] as number)
  }
  return { kind: 'sample', bracket: 0, rung: 0, budget: rungs[0] as number }
}

/**
 * What a schedule asks for next, given the finished trials of a study (each tagged with its bracket and rung): a new
 * configuration at a rung's resource, or the promotion of the best configuration of the rung below that has not been
 * promoted yet. Synchronous schedules fill each rung before the next; ASHA promotes whenever a configuration is in the
 * top 1 / eta of its rung. Failed and pruned trials are never promoted.
 *
 * @param schedule - The schedule.
 * @param study - The study so far.
 * @returns The assignment, or `undefined` when a synchronous schedule has finished.
 * @concept successive-halving
 * @example
 * ```ts
 * import { createStudy, nextAssignment, successiveHalvingSchedule } from 'whitebox-engine/tune'
 * import { real } from 'whitebox-engine/space'
 * nextAssignment(successiveHalvingSchedule({ nConfigs: 4, maxResource: 4, reductionFactor: 2 }), createStudy(real(0, 1)))
 * // => { kind: 'sample', bracket: 0, rung: 0, budget: 1 }
 * ```
 */
export function nextAssignment(schedule: Schedule, study: Study): Assignment | undefined {
  if (study.directions.length !== 1) throw new WhiteboxError('value', 'schedules rank trials by a single objective')
  return schedule.kind === 'asha' ? nextAsha(schedule, study) : nextSynchronous(schedule, study)
}

/** The iterations of scikit-learn's successive-halving search, as data. */
export interface HalvingPlan {
  readonly nIterations: number
  readonly nRequiredIterations: number
  readonly nPossibleIterations: number
  readonly minResources: number
  readonly maxResources: number
  /** Candidates evaluated in each iteration. */
  readonly nCandidates: readonly number[]
  /** Resource of each iteration. */
  readonly nResources: readonly number[]
}

/**
 * The iterations of scikit-learn's `HalvingGridSearchCV` and `HalvingRandomSearchCV` as data: the number of
 * iterations needed to reach one candidate (1 + floor(log_factor(nCandidates))) and possible within the resources
 * (1 + floor(log_factor(max // min))), the resource of iteration i (factor^i min, capped; with aggressive elimination
 * the first iterations repeat the minimum), and ceil(n / factor) candidates kept after each. With `exhaust`, the
 * minimum resource is raised so the last iteration uses as much of the maximum as possible.
 *
 * @param options - Candidates, minimum and maximum resources, factor (default 3), exhaustion and aggressive
 *   elimination.
 * @returns The plan (the `n_iterations_`, `n_resources_`, `n_candidates_` and related attributes of scikit-learn).
 * @throws WhiteboxError with code `value` when the minimum exceeds the maximum.
 * @concept successive-halving
 * @example
 * ```ts
 * import { halvingPlan } from 'whitebox-engine/tune'
 * const plan = halvingPlan({ nCandidates: 27, minResources: 20, maxResources: 1000, exhaust: true })
 * [plan.nCandidates, plan.nResources] // => [[27, 9, 3, 1], [37, 111, 333, 999]]
 * ```
 */
export function halvingPlan(options: {
  readonly nCandidates: number
  readonly minResources: number
  readonly maxResources: number
  readonly factor?: number
  readonly exhaust?: boolean
  readonly aggressiveElimination?: boolean
}): HalvingPlan {
  const factor = options.factor ?? 3
  const max = options.maxResources
  if (!(options.minResources >= 1 && options.minResources <= max))
    throw new WhiteboxError('value', `halvingPlan: min resources ${options.minResources} exceed max ${max}`)
  if (!(factor > 1)) throw new WhiteboxError('value', 'halvingPlan: factor must exceed 1')
  // scikit-learn's floor(log(x, factor)) in floating point, reproduced as it computes it.
  const logf = (x: number): number => Math.floor(Math.log(x) / Math.log(factor))
  const nRequired = 1 + logf(options.nCandidates)
  const min =
    options.exhaust === true
      ? Math.max(options.minResources, Math.floor(max / factor ** (nRequired - 1)))
      : options.minResources
  const nPossible = 1 + logf(Math.floor(max / min))
  const aggressive = options.aggressiveElimination === true
  const nIterations = aggressive ? nRequired : Math.min(nPossible, nRequired)
  const nCandidates: number[] = []
  const nResources: number[] = []
  let n = options.nCandidates
  for (let itr = 0; itr < nIterations; itr++) {
    const power = aggressive ? Math.max(0, itr - nRequired + nPossible) : itr
    nResources.push(Math.min(Math.trunc(factor ** power * min), max))
    nCandidates.push(n)
    n = Math.ceil(n / factor)
  }
  return {
    nIterations,
    nRequiredIterations: nRequired,
    nPossibleIterations: nPossible,
    minResources: min,
    maxResources: max,
    nCandidates,
    nResources,
  }
}
