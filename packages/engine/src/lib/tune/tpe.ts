/**
 * The tree-structured Parzen estimator (TPE, Bergstra et al. 2011; Optuna's `TPESampler`): split the finished trials
 * into a good group and the rest, fit a Parzen estimator to each, and propose the candidate with the largest density
 * ratio. Also Optuna's split for several objectives, with ties in a front broken by greedy hypervolume subset
 * selection.
 */

import { argmaxOf, type Json } from '../base/index.ts'
import { hypervolume, nonDominatedSort } from '../optim/index.ts'
import { foldIn } from '../random/index.ts'
import { sample } from '../space/index.ts'
import { assemble, type Leaf, leavesOf, nativeOf } from './_leaves.ts'
import {
  defaultGamma,
  defaultGammaMultiObjective,
  type ParzenOptions,
  parzenEstimator,
  parzenSample,
  tpeAcquisition,
} from './parzen.ts'
import type { StudySampler } from './samplers.ts'
import { lossMatrix, lossOf, type Study, type StudyDirection, type Trial } from './study.ts'

const EPS = 1e-12

/** The good and bad groups of trials for TPE. */
export interface TpeSplit {
  readonly below: readonly Trial[]
  readonly above: readonly Trial[]
}

function prunedScore(t: Trial, d: StudyDirection): [number, number] {
  if (t.intermediate.length === 0) return [1, 0]
  const last = t.intermediate.reduce((a, b) => (b.step > a.step ? b : a))
  if (Number.isNaN(last.value)) return [-last.step, Number.POSITIVE_INFINITY]
  return [-last.step, lossOf(d, last.value)]
}

/**
 * The greedy hypervolume subset selection Optuna uses to break ties in a non-dominated front: repeatedly take the
 * point adding the most hypervolume (first in lexicographic order on ties), over the distinct points.
 *
 * @param losses - The front's loss vectors.
 * @param size - How many to select.
 * @param reference - The reference point.
 * @returns Indices into `losses`.
 * @concept hypervolume
 * @example
 * ```ts
 * import { hypervolumeSubsetSelection } from 'omai-engine/tune'
 * hypervolumeSubsetSelection([[1, 4], [2, 2], [4, 1]], 1, [5, 5]) // => [1]
 * ```
 */
export function hypervolumeSubsetSelection(
  losses: readonly (readonly number[])[],
  size: number,
  reference: readonly number[],
): number[] {
  if (size >= losses.length) return losses.map((_, i) => i)
  // Distinct rows in lexicographic order, each by its first index (numpy.unique(axis=0, return_index=True)).
  const keyOf = (r: readonly number[]): string => JSON.stringify(r)
  const first = new Map<string, number>()
  for (const [i, r] of losses.entries()) if (!first.has(keyOf(r))) first.set(keyOf(r), i)
  const unique = [...first.values()].sort((a, b) => {
    const ra = losses[a] as readonly number[]
    const rb = losses[b] as readonly number[]
    for (let j = 0; j < ra.length; j++) if (ra[j] !== rb[j]) return (ra[j] ?? 0) - (rb[j] ?? 0)
    return 0
  })
  if (unique.length < size) {
    const chosen = new Set(unique)
    for (const [i] of losses.entries()) if (chosen.size < size && !chosen.has(i)) chosen.add(i)
    return losses.map((_, i) => i).filter((i) => chosen.has(i))
  }
  if (!reference.every(Number.isFinite)) return unique.slice(0, size)
  const selected: number[] = []
  let remaining = [...unique]
  let hv = 0
  while (selected.length < size) {
    const gains = remaining.map(
      (i) =>
        hypervolume(
          [...selected, i].map((s) => losses[s] as number[]),
          reference,
        ) - hv,
    )
    const best = argmaxOf(gains)
    hv += gains[best] ?? 0
    selected.push(remaining[best] as number)
    remaining = remaining.filter((_, j) => j !== best)
  }
  return selected
}

/**
 * Optuna's split of finished trials into the good group (`nBelow` of them) and the rest: complete trials by value
 * (for several objectives by non-dominated rank, ties in the last front broken by hypervolume subset selection), then
 * pruned trials by the step they reached and their last value.
 *
 * @param study - The study.
 * @param trials - Complete and pruned trials.
 * @param nBelow - Size of the good group.
 * @returns Both groups, each in trial order.
 * @concept tree-structured-parzen-estimator
 * @example
 * ```ts
 * import { createStudy, splitTrials } from 'omai-engine/tune'
 * import { real } from 'omai-engine/space'
 * const trial = (number: number, value: number) =>
 *   ({ number, params: value, state: 'complete', values: [value], intermediate: [], source: 'x' }) as const
 * const study = createStudy(real(0, 9), { trials: [trial(0, 5), trial(1, 1), trial(2, 3)] })
 * splitTrials(study, study.trials, 2).below.map((t) => t.number) // => [1, 2]
 * ```
 */
export function splitTrials(study: Study, trials: readonly Trial[], nBelow: number): TpeSplit {
  const complete = trials.filter((t) => t.state === 'complete')
  const pruned = trials.filter((t) => t.state === 'pruned')
  const nComplete = Math.min(nBelow, complete.length)
  let belowComplete: Trial[]
  if (study.directions.length === 1) {
    const d = study.directions[0] as StudyDirection
    const order = complete
      .map((t, i) => ({ t, i, v: lossOf(d, t.values[0] ?? Number.NaN) }))
      .sort((a, b) => a.v - b.v || a.i - b.i)
    belowComplete = order.slice(0, nComplete).map((o) => o.t)
  } else belowComplete = splitMultiObjective(study, complete, nComplete)
  let rest = Math.max(0, nBelow - belowComplete.length)
  const d = study.directions[0] as StudyDirection
  const prunedOrder = pruned
    .map((t, i) => ({ t, i, s: prunedScore(t, d) }))
    .sort((a, b) => a.s[0] - b.s[0] || a.s[1] - b.s[1] || a.i - b.i)
  const belowPruned =
    study.directions.length === 1 ? prunedOrder.slice(0, Math.min(rest, pruned.length)).map((o) => o.t) : []
  rest = Math.max(0, rest - belowPruned.length)
  const below = new Set([...belowComplete, ...belowPruned])
  const byNumber = (a: Trial, b: Trial): number => a.number - b.number
  return { below: [...below].sort(byNumber), above: trials.filter((t) => !below.has(t)).sort(byNumber) }
}

function splitMultiObjective(study: Study, trials: readonly Trial[], nBelow: number): Trial[] {
  if (nBelow === 0) return []
  if (nBelow >= trials.length) return [...trials]
  const losses = lossMatrix(study, trials)
  const chosen: number[] = []
  for (const front of nonDominatedSort(losses)) {
    if (chosen.length + front.length <= nBelow) {
      chosen.push(...front)
      if (chosen.length === nBelow) break
      continue
    }
    const rows = front.map((i) => losses[i] as number[])
    const worst = rows[0]?.map((_, j) => rows.reduce((m, r) => Math.max(m, r[j] ?? 0), Number.NEGATIVE_INFINITY)) ?? []
    const reference = worst.map((w) => {
      const r = Math.max(1.1 * w, 0.9 * w)
      return r === 0 ? EPS : r
    })
    chosen.push(...hypervolumeSubsetSelection(rows, nBelow - chosen.length, reference).map((k) => front[k] as number))
    break
  }
  const keep = new Set(chosen)
  return trials.filter((_, i) => keep.has(i))
}

/** Options of {@link tpeSampler}. */
export interface TpeOptions extends Omit<ParzenOptions, 'predeterminedWeights'> {
  /** Random trials before the model is used (default 10). */
  readonly nStartupTrials?: number
  /** Candidates drawn from the good density per trial (default 24). */
  readonly nEiCandidates?: number
  /** The size of the good group for n finished trials (default {@link defaultGamma}, or ceil(0.1 n) for several). */
  readonly gamma?: (n: number) => number
  /** Model the leaves jointly (default: for one objective; independently for several, as Optuna). */
  readonly multivariate?: boolean
}

function observationsOf(study: Study, trials: readonly Trial[], columns: readonly number[]): number[][] {
  return trials.flatMap((t) => {
    const native = nativeOf(study.space, t.params)
    const row = columns.map((c) => native[c] as number)
    return row.some(Number.isNaN) ? [] : [row]
  })
}

/**
 * The tree-structured Parzen estimator (Optuna's `TPESampler`): after `nStartupTrials` random trials, split the
 * finished trials into a good group (gamma(n) best) and the rest, fit a Parzen estimator l(x) to the good and g(x) to
 * the rest, draw `nEiCandidates` candidates from l and take the one maximising log l(x) - log g(x) (the expected
 * improvement ratio). Multivariate by default for one objective (one joint estimator over the leaves every trial
 * has; conditional leaves independently), independent per leaf for several objectives (MOTPE, with
 * hypervolume-based tie breaking).
 *
 * @param options - Startup trials, candidates, gamma, weights, prior weight, bandwidth clipping and multivariate.
 * @returns The sampler.
 * @concept tree-structured-parzen-estimator
 * @reference Bergstra et al. (2011); Ozaki et al. (2020), Multiobjective tree-structured Parzen estimator.
 * @example
 * ```ts
 * import { tpeSampler, functionObjective, tune } from 'omai-engine/tune'
 * import { dict, real } from 'omai-engine/space'
 * const f = functionObjective((p) => { const { x, y } = p as { x: number; y: number }; return (x - 1) ** 2 + (y + 2) ** 2 })
 * (tune(f, dict({ x: real(-5, 5), y: real(-5, 5) }), { sampler: tpeSampler(), nTrials: 80, seed: 0 }).best?.values[0] ?? 9) < 0.5 // => true
 * ```
 */
export function tpeSampler(options: TpeOptions = {}): StudySampler {
  const nStartup = options.nStartupTrials ?? 10
  const nCandidates = options.nEiCandidates ?? 24
  const parzen: ParzenOptions = {
    ...(options.priorWeight === undefined ? {} : { priorWeight: options.priorWeight }),
    ...(options.considerMagicClip === undefined ? {} : { considerMagicClip: options.considerMagicClip }),
    ...(options.considerEndpoints === undefined ? {} : { considerEndpoints: options.considerEndpoints }),
    ...(options.weights === undefined ? {} : { weights: options.weights }),
  }
  const config: Json = {
    nStartupTrials: nStartup,
    nEiCandidates: nCandidates,
    priorWeight: options.priorWeight ?? 1,
    considerMagicClip: options.considerMagicClip ?? true,
    considerEndpoints: options.considerEndpoints ?? false,
    multivariate: options.multivariate ?? null,
  }
  return {
    name: 'tpe',
    config,
    multiObjective: true,
    ask: (study, key) => {
      const finished = study.trials.filter((t) => t.state !== 'failed')
      if (finished.length < nStartup) return sample(study.space, key)
      const multi = study.directions.length > 1
      const gamma = options.gamma ?? (multi ? defaultGammaMultiObjective : defaultGamma)
      const split = splitTrials(study, finished, gamma(finished.length))
      const leaves = leavesOf(study.space)
      const natives = finished.map((t) => nativeOf(study.space, t.params))
      const varying = leaves.map((l) => l.low < l.high)
      const everywhere = leaves.map(
        (_, j) => varying[j] === true && natives.every((r) => !Number.isNaN(r[j] ?? Number.NaN)),
      )
      const multivariate = options.multivariate ?? !multi
      const groups: number[][] = []
      const joint = leaves.flatMap((_, j) => (multivariate && everywhere[j] === true ? [j] : []))
      if (joint.length > 0) groups.push(joint)
      for (const [j] of leaves.entries()) if (varying[j] === true && !joint.includes(j)) groups.push([j])
      const out = leaves.map((l) => (l.low < l.high ? Number.NaN : l.low))
      for (const [g, columns] of groups.entries()) {
        const covered = columns.map((c) => leaves[c] as Leaf)
        const below = observationsOf(study, split.below, columns)
        const above = observationsOf(study, split.above, columns)
        const l = parzenEstimator(
          covered,
          below,
          multi ? { ...parzen, predeterminedWeights: below.map(() => 1) } : parzen,
        )
        const gEst = parzenEstimator(covered, above, parzen)
        const candidates = parzenSample(l, foldIn(key, g), nCandidates)
        const best = candidates[argmaxOf(tpeAcquisition(l, gEst, candidates))] as number[]
        for (const [i, c] of columns.entries()) out[c] = best[i] as number
      }
      return assemble(study.space, leaves, out)
    },
  }
}
