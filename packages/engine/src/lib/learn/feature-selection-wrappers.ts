/**
 * Wrapper feature selection, which chooses features by refitting a model on candidate subsets: recursive feature
 * elimination (`rfe`, and `rfecv` with the count chosen by cross-validation), greedy forward or backward sequential
 * selection, and stability selection, which keeps the features chosen on most of many random half-samples.
 *
 * @index function RFE, sequential selection and stability selection around any estimator.
 */

import { type Json, MdmlError, meanOf, varianceOf } from '../base/index.ts'
import { foldIn, type Key, subsample } from '../random/index.ts'
import { dict, int, type SpaceValue } from '../space/index.ts'
import { type TensorLike, toFlat } from '../tensor/index.ts'
import {
  type Algorithm,
  type Contract,
  channels,
  defineContract,
  runAlgorithm,
  scalar,
  stateFromJson,
  stateToJson,
  tensor,
} from '../trace/index.ts'
import { innerFitOptions, keyFor, runOptionsFor } from './_shared.ts'
import { type IterativeEstimator, iterative } from './bases/index.ts'
import {
  defineEstimator,
  type Estimator,
  type FitOptions,
  type Fitted,
  type LearnData,
  type Verbs,
} from './estimator.ts'
import { type SelectorVerbs, selectorVerbs } from './feature-selection.ts'
import { eliminationSchedules } from './feature-selection-options.ts'
import { scoresFromJson, scoresToJson } from './model-selection/_state.ts'
import { type CvOption, checkCv, crossValScore, splitInputOf } from './model-selection/cross-validation.ts'
import { splitIndices } from './model-selection/splitters.ts'
import { type Scoring, scorersOf } from './scoring.ts'
import { fromView, matrixView, takeColumns, takeData } from './validation.ts'

function columns(data: LearnData, support: readonly boolean[]): LearnData {
  const m = matrixView(data.x, 'featureSelection', { allowNaN: true })
  return {
    ...data,
    x: fromView(
      takeColumns(
        m,
        support.flatMap((s, j) => (s ? [j] : [])),
      ),
    ),
  }
}

function importancesOf(fitted: Fitted, who: string): readonly number[] {
  const imp = fitted.featureImportances
  if (imp === undefined) throw new MdmlError('value', `${who}: ${fitted.estimator} reports no featureImportances`)
  return imp
}

/** The verbs of a fitted wrapper selector: the selection, and the final estimator's verbs on the selected features. */
export type WrapperVerbs = SelectorVerbs & Verbs

function delegate(final: Fitted | undefined, sel: SelectorVerbs, on: SelectorVerbs = sel): WrapperVerbs {
  const out: Record<string, unknown> = { ...sel }
  if (final !== undefined) {
    for (const verb of ['predict', 'predictProba', 'decisionFunction'] as const) {
      const f = final[verb]
      if (typeof f === 'function') out[verb] = (x: TensorLike) => f(on.transform(x))
    }
    const score = final.score
    if (score !== undefined) out['score'] = (data: LearnData) => score({ ...data, x: on.transform(data.x) })
  }
  return out as unknown as WrapperVerbs
}

/** Options of {@link rfe}. */
export interface RfeOptions {
  readonly estimator: Estimator
  /** How many features to keep (default half; a fraction below 1 of the features). */
  readonly nFeaturesToSelect?: number
  /** Features removed per round (an integer, or a fraction below 1 of the features). */
  readonly step?: number
  /** The schedule of the bite, a value of `eliminationSchedules.space` (takes precedence over `step`). */
  readonly schedule?: SpaceValue
}

/** Parameters of a fitted {@link rfe}. */
export type RfeParams = {
  readonly support: boolean[]
  readonly ranking: number[]
  readonly estimator: Json
  /** The features the estimator was fitted on (the support itself once the elimination is done). */
  readonly estimatorSupport?: boolean[]
}

/** The paths {@link rfeAlgorithm} logs. */
export type RfePath = 'rfe/remaining' | 'rfe/dropped' | 'rfe/ranking' | 'rfe/score'

/** What a stepped elimination logs, by path: features left, features dropped, the ranking and the round's score. */
export const rfeContract: Contract<unknown, RfePath> = defineContract({
  'rfe/remaining': scalar({ description: 'features still in after the round' }),
  'rfe/dropped': scalar({ description: 'features dropped in the round' }),
  'rfe/ranking': tensor({
    role: 'output',
    shape: ['features'],
    description: '1 for kept features, else the rounds survived plus one',
  }),
  'rfe/score': scalar({
    description: 'the round’s model scored on the evaluation data (when given)',
    direction: 'higher',
  }),
})

/**
 * The state of {@link rfeAlgorithm}: the support and ranking so far, and for each round the number of features the
 * model was fitted on and its score on the evaluation data (NaN without any). `estimator` holds the parameters of the
 * last model fitted; once finished, that is the model on the final support.
 */
export interface RfeState {
  readonly support: readonly boolean[]
  readonly ranking: readonly number[]
  /** The features each round's model was fitted on. */
  readonly counts: readonly number[]
  readonly scores: readonly number[]
  readonly finished: boolean
  readonly estimator: Json
  /** The features `estimator` was fitted on. */
  readonly fittedOn: readonly boolean[]
}

/** What {@link rfeAlgorithm} eliminates down to and how. */
export interface RfeSettings {
  /** How many features to keep (default half; a fraction below 1 of the features). */
  readonly nFeaturesToSelect?: number
  /** Features removed per round (an integer, or a fraction below 1 of the features; default 1). */
  readonly step?: number
  /** The schedule of the bite, a value of `eliminationSchedules.space` (takes precedence over `step`). */
  readonly schedule?: SpaceValue
  /** Score each round's model on this data (restricted to the round's features). */
  readonly evaluation?: LearnData
  readonly scoring?: Scoring
}

function stepSize(step: number, d: number, who: string): number {
  if (step > 0 && step < 1) return Math.max(1, Math.floor(step * d))
  if (!(Number.isInteger(step) && step >= 1))
    throw new MdmlError('value', `${who}: step must be an integer >= 1 or a fraction`)
  return step
}

function targetCount(n: number | undefined, d: number): number {
  if (n === undefined) return Math.floor(d / 2)
  if (n > 0 && n < 1) return Math.floor(n * d)
  return Math.min(n, d)
}

/**
 * Recursive feature elimination (Guyon et al. 2002) as a stepped algorithm: each round fits the estimator on the
 * features still in, optionally scores it on evaluation data, and drops the least important by the fitted
 * `featureImportances`, until the target number remain (the last round fits on the final features and drops none). The
 * ranking is 1 for kept features and counts the rounds survived otherwise. The state serialises.
 *
 * @param estimator - The estimator; it must report `featureImportances`.
 * @param data - The data.
 * @param settings - nFeaturesToSelect, step or schedule, evaluation data and scoring.
 * @returns The algorithm, one step per round.
 * @throws MdmlError with code `value` for a step that is neither an integer >= 1 nor a fraction.
 * @concept feature-selection
 * @example
 * ```ts
 * import { defineEstimator, rfeAlgorithm } from 'mdml-engine/learn'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * import { sum, toFlat } from 'mdml-engine/tensor'
 * const byColumnSum = defineEstimator(
 *   { name: 'sums', kind: 'regressor', fit: (d) => ({ imp: toFlat(sum(d.x, 0)) }), methods: (p) => ({ featureImportances: p.imp }) },
 *   {},
 * )
 * const alg = rfeAlgorithm(byColumnSum, { x: [[1, 5, 2], [1, 5, 2]], y: [0, 0] }, { nFeaturesToSelect: 1 })
 * runAlgorithm(alg, { maxSteps: 10 }).state.ranking // => [3, 1, 2]
 * ```
 */
export function rfeAlgorithm(
  estimator: Estimator,
  data: LearnData,
  settings: RfeSettings = {},
  who = 'rfe',
): Algorithm<RfeState, RfePath> {
  const d = matrixView(data.x, who, { allowNaN: true }).d
  const target = targetCount(settings.nFeaturesToSelect, d)
  const schedule =
    settings.schedule === undefined
      ? (
          (k: number) => (remaining: number, goal: number) =>
            Math.min(k, remaining - goal)
        )(stepSize(settings.step ?? 1, d, who))
      : eliminationSchedules.make(settings.schedule)
  const scorer = settings.evaluation === undefined ? undefined : Object.values(scorersOf(settings.scoring))[0]
  return {
    name: who,
    config: {
      estimator: estimator.name,
      nFeatures: d,
      target,
      schedule: eliminationSchedules.optionOf(settings.schedule).name,
    },
    schema: channels({ metrics: { nFeatures: { description: 'features remaining' } } }),
    init: () => ({
      support: new Array<boolean>(d).fill(true),
      ranking: new Array<number>(d).fill(1),
      counts: [],
      scores: [],
      finished: false,
      estimator: null,
      fittedOn: new Array<boolean>(d).fill(true),
    }),
    step: (s, ctx) => {
      const fitted = estimator.fit(columns(data, s.support), innerFitOptions({}, foldIn(ctx.key, 0), ctx.run))
      const remaining = s.support.filter(Boolean).length
      const score =
        scorer === undefined || settings.evaluation === undefined
          ? Number.NaN
          : scorer.score(fitted, columns(settings.evaluation, s.support))
      const counts = [...s.counts, remaining]
      const scores = [...s.scores, score]
      if (!Number.isNaN(score)) ctx.log('rfe/score', score)
      if (remaining <= target) {
        ctx.log('rfe/remaining', remaining)
        ctx.log('rfe/dropped', 0)
        if (ctx.wants('rfe/ranking')) ctx.log('rfe/ranking', Int32Array.from(s.ranking))
        return { ...s, counts, scores, finished: true, estimator: fitted.params, fittedOn: s.support }
      }
      const features = s.support.flatMap((on, j) => (on ? [j] : []))
      const imp = importancesOf(fitted, who)
      const order = imp.map((v, r) => ({ v, r })).sort((a, b) => a.v - b.v || a.r - b.r)
      const drop = Math.max(1, Math.min(schedule(remaining, target), remaining - target))
      const support = [...s.support]
      for (const { r } of order.slice(0, drop)) support[features[r] ?? 0] = false
      const ranking = s.ranking.map((rank, j) => (support[j] === true ? rank : rank + 1))
      ctx.log('rfe/remaining', remaining - drop)
      ctx.log('rfe/dropped', drop)
      if (ctx.wants('rfe/ranking')) ctx.log('rfe/ranking', Int32Array.from(ranking))
      return { support, ranking, counts, scores, finished: false, estimator: fitted.params, fittedOn: s.support }
    },
    done: (s) => (s.finished ? 'completed' : undefined),
    observe: (s) => ({ metrics: { nFeatures: s.support.filter(Boolean).length } }),
    serialise: (s) => stateToJson({ ...s, scores: scoresToJson(s.scores) }),
    deserialise: (j) => {
      const o = stateFromJson(j) as Omit<RfeState, 'scores'> & { readonly scores: never }
      return { ...o, scores: scoresFromJson(o.scores) }
    },
  }
}

/** Run recursive elimination down to `target` features, fitting once per round (and once at the end). */
function eliminate(
  estimator: Estimator,
  data: LearnData,
  settings: RfeSettings,
  options: FitOptions,
  who: string,
): RfeState {
  const d = matrixView(data.x, who, { allowNaN: true }).d
  return runAlgorithm(rfeAlgorithm(estimator, data, settings, who), { ...runOptionsFor(options, who), maxSteps: d + 1 })
    .state
}

const rfeVerbs = (p: RfeParams, h: { readonly estimator: Estimator }, who: string): WrapperVerbs =>
  delegate(
    p.estimator === null ? undefined : h.estimator.revive(p.estimator),
    selectorVerbs(p.support, who),
    selectorVerbs(p.estimatorSupport ?? p.support, who),
  )

/**
 * Recursive feature elimination (Guyon et al. 2002; sklearn.feature_selection.RFE) on the iterative base: fit, drop
 * the `step` least important features (by the fitted `featureImportances`), and repeat until `nFeaturesToSelect`
 * remain; `ranking` is 1 for kept features and counts the rounds survived otherwise. One step per round
 * ({@link rfeAlgorithm}); the fitted value at step `t` is the selection so far, and at the end it also predicts with
 * the final estimator on the kept features.
 *
 * @param options - estimator, nFeaturesToSelect and step (default 1) or schedule.
 * @returns The estimator.
 * @example
 * ```ts
 * import { defineEstimator, rfe } from 'mdml-engine/learn'
 * import { sum, toFlat } from 'mdml-engine/tensor'
 * const byColumnSum = defineEstimator(
 *   { name: 'sums', kind: 'regressor', fit: (d) => ({ imp: toFlat(sum(d.x, 0)) }), methods: (p) => ({ featureImportances: p.imp }) },
 *   {},
 * )
 * rfe({ estimator: byColumnSum, nFeaturesToSelect: 1 }).fit({ x: [[1, 5, 2], [1, 5, 2]], y: [0, 0] }).params.ranking // => [3, 1, 2]
 * ```
 */
export function rfe(options: RfeOptions): IterativeEstimator<RfeOptions, RfeState, RfeParams, WrapperVerbs, RfePath> {
  return iterative<RfeOptions, RfeState, RfeParams, WrapperVerbs, RfePath>({
    name: 'rfe',
    kind: options.estimator.kind,
    defaults: options,
    hyperSpace: (h) =>
      dict({
        ...(h.nFeaturesToSelect === undefined || !Number.isInteger(h.nFeaturesToSelect)
          ? {}
          : { nFeaturesToSelect: int(1, 1000, { default: h.nFeaturesToSelect, label: 'features to keep' }) }),
        ...(h.step === undefined || !Number.isInteger(h.step)
          ? {}
          : { step: int(1, 100, { default: h.step, label: 'features per round' }) }),
      }),
    algorithm: (data, h) =>
      rfeAlgorithm(h.estimator, data, {
        ...(h.nFeaturesToSelect === undefined ? {} : { nFeaturesToSelect: h.nFeaturesToSelect }),
        ...(h.step === undefined ? {} : { step: h.step }),
        ...(h.schedule === undefined ? {} : { schedule: h.schedule }),
      }),
    maxSteps: (_, data) => matrixView(data.x, 'rfe', { allowNaN: true }).d + 1,
    // Before the first round there is no model yet: the estimator fitted on every feature stands in, so the fitted value
    // can predict at every step.
    params: (s, data, h) => ({
      support: [...s.support],
      ranking: [...s.ranking],
      estimator: s.estimator ?? h.estimator.fit(data).params,
      estimatorSupport: [...s.fittedOn],
    }),
    verbs: (p, h) => rfeVerbs(p, h, 'rfe'),
    contract: () => rfeContract as Contract<RfeState, RfePath>,
  })
}

/** Options of {@link rfecv}. */
export interface RfecvOptions {
  readonly estimator: Estimator
  readonly step?: number
  readonly minFeaturesToSelect?: number
  readonly cv?: CvOption
  readonly scoring?: Scoring
  readonly groups?: ArrayLike<number>
  /** The schedule of the bite, a value of `eliminationSchedules.space` (takes precedence over `step`). */
  readonly schedule?: SpaceValue
}

/** Parameters of a fitted {@link rfecv}. */
export type RfecvParams = RfeParams & {
  /** The feature counts evaluated, increasing. */
  readonly nFeatures: number[]
  /** Mean and standard deviation of the test score at each count. */
  readonly meanTestScore: number[]
  readonly stdTestScore: number[]
  /** splitScores[fold][count]. */
  readonly splitScores: number[][]
}

/** The paths {@link rfecvAlgorithm} logs. */
export type RfecvPath = 'rfecv/fold' | 'rfecv/best' | 'rfecv/remaining'

/** What a stepped cross-validated elimination logs, by path. */
export const rfecvContract: Contract<unknown, RfecvPath> = defineContract({
  'rfecv/fold': scalar({ description: 'the fold whose elimination path was just scored (from 0)' }),
  'rfecv/best': scalar({ description: 'the feature count with the best total score, once all folds are done' }),
  'rfecv/remaining': scalar({ description: 'features kept by the final elimination' }),
})

/**
 * The state of {@link rfecvAlgorithm}: the folds scored (each fold's score at every feature count, decreasing in the
 * count), and, after the last step, the final elimination to the best count.
 */
export interface RfecvState {
  readonly done: number
  /** `foldScores[fold][round]`: the test score of the fold's model on the features of that round (counts fall). */
  readonly foldScores: readonly (readonly number[])[]
  readonly foldCounts: readonly (readonly number[])[]
  readonly final: RfeState | null
}

/**
 * RFE with the number of features chosen by cross-validation (sklearn RFECV) as a stepped algorithm: each of the first
 * steps eliminates on one fold's training rows, scoring every round's model on the fold's test rows; the last step
 * chooses the count with the best total score (the fewest features on ties) and eliminates on all the data down to it.
 * The state serialises.
 *
 * @param estimator - The estimator; it must report `featureImportances`.
 * @param data - The data.
 * @param options - step or schedule, minFeaturesToSelect, cv, scoring and groups.
 * @returns The algorithm, one step per fold and a last one.
 * @concept feature-selection
 * @example
 * ```ts
 * import { defineEstimator, rfecvAlgorithm } from 'mdml-engine/learn'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * import { full } from 'mdml-engine/tensor'
 * const constant = defineEstimator(
 *   { name: 'c', kind: 'regressor', fit: (d) => ({ d: d.x.shape[1] }), methods: (p) => ({ featureImportances: Array.from({ length: p.d }, (_, j) => j), predict: (x) => full([x.shape[0]], 0) }) },
 *   {},
 * )
 * const data = { x: [[1, 2], [3, 4], [5, 6], [7, 8]], y: [0, 0, 0, 0] }
 * runAlgorithm(rfecvAlgorithm(constant, data, { cv: 2, scoring: 'neg_mean_squared_error' }), { maxSteps: 5 }).steps // => 3
 * ```
 */
export function rfecvAlgorithm(
  estimator: Estimator,
  data: LearnData,
  options: Omit<RfecvOptions, 'estimator'> = {},
  key: Key = keyFor({}, 'rfecv'),
): Algorithm<RfecvState, RfecvPath> {
  const d = matrixView(data.x, 'rfecv', { allowNaN: true }).d
  const minFeatures = Math.min(options.minFeaturesToSelect ?? 1, d)
  const scorer = Object.values(scorersOf(options.scoring))[0]
  if (scorer === undefined) throw new MdmlError('value', 'rfecv: no scorer')
  const folds = splitIndices(checkCv(options.cv, estimator), splitInputOf(data, options.groups), foldIn(key, 0))
  const settings = (target: number): RfeSettings => ({
    nFeaturesToSelect: target,
    ...(options.step === undefined ? {} : { step: options.step }),
    ...(options.schedule === undefined ? {} : { schedule: options.schedule }),
  })
  // The counts and total scores over the folds, in increasing count order, and the count they favour.
  const choose = (
    s: RfecvState,
  ): { readonly counts: number[]; readonly splitScores: number[][]; readonly n: number } => {
    const counts = [...(s.foldCounts[0] ?? [])].reverse()
    const splitScores = s.foldScores.map((row) => [...row].reverse())
    const totals = counts.map((_, c) => splitScores.reduce((t, row) => t + (row[c] ?? 0), 0))
    let best = 0
    for (const [c, t] of totals.entries()) if (t > (totals[best] ?? Number.NEGATIVE_INFINITY)) best = c
    return { counts, splitScores, n: counts[best] ?? minFeatures }
  }
  const algorithm: Algorithm<RfecvState, RfecvPath> = {
    name: 'rfecv',
    config: { estimator: estimator.name, folds: folds.length, minFeatures },
    schema: channels({ metrics: { nFeatures: { description: 'the feature count chosen' } } }),
    init: () => ({ done: 0, foldScores: [], foldCounts: [], final: null }),
    step: (s, ctx) => {
      const fold = folds[s.done]
      if (fold === undefined) {
        const { n } = choose(s)
        const final = eliminate(estimator, data, settings(n), innerFitOptions({}, foldIn(key, 0), ctx.run), 'rfecv')
        ctx.log('rfecv/best', n)
        ctx.log('rfecv/remaining', final.support.filter(Boolean).length)
        return { ...s, done: s.done + 1, final }
      }
      const path = eliminate(
        estimator,
        takeData(data, fold.train),
        {
          ...settings(minFeatures),
          evaluation: takeData(data, fold.test),
          ...(options.scoring === undefined ? {} : { scoring: options.scoring }),
        },
        innerFitOptions({}, foldIn(key, 1 + s.done), ctx.run),
        'rfecv',
      )
      ctx.log('rfecv/fold', s.done)
      return {
        done: s.done + 1,
        foldScores: [...s.foldScores, path.scores],
        foldCounts: [...s.foldCounts, path.counts],
        final: null,
      }
    },
    done: (s) => (s.final !== null ? 'completed' : undefined),
    observe: (s) => ({ metrics: s.final === null ? {} : { nFeatures: s.final.support.filter(Boolean).length } }),
    serialise: (s) =>
      stateToJson({
        done: s.done,
        foldScores: s.foldScores.map((r) => scoresToJson(r)),
        foldCounts: s.foldCounts,
        final: s.final === null ? null : { ...s.final, scores: scoresToJson(s.final.scores) },
      }),
    deserialise: (j) => {
      const o = stateFromJson(j) as {
        readonly done: number
        readonly foldScores: readonly never[]
        readonly foldCounts: readonly (readonly number[])[]
        readonly final: (Omit<RfeState, 'scores'> & { readonly scores: never }) | null
      }
      return {
        done: o.done,
        foldScores: o.foldScores.map((r) => scoresFromJson(r)),
        foldCounts: o.foldCounts,
        final: o.final === null ? null : { ...o.final, scores: scoresFromJson(o.final.scores) },
      }
    },
  }
  return algorithm
}

/**
 * Recursive feature elimination with the number of features chosen by cross-validation (sklearn RFECV) on the iterative
 * base: on each fold, eliminate on the training rows scoring every intermediate subset on the test rows; choose the
 * count with the best total score (the fewest features on ties), then eliminate on all the data down to it. One step per
 * fold and a last one ({@link rfecvAlgorithm}).
 *
 * @param options - estimator, step or schedule, minFeaturesToSelect (default 1), cv, scoring and groups.
 * @returns The estimator.
 * @example
 * ```ts
 * import { rfecv } from 'mdml-engine/learn'
 * import { defineEstimator } from 'mdml-engine/learn'
 * import { full } from 'mdml-engine/tensor'
 * const constant = defineEstimator(
 *   { name: 'c', kind: 'regressor', fit: (d) => ({ d: d.x.shape[1] }), methods: (p) => ({ featureImportances: Array.from({ length: p.d }, (_, j) => j), predict: (x) => full([x.shape[0]], 0) }) },
 *   {},
 * )
 * rfecv({ estimator: constant, cv: 2, scoring: 'neg_mean_squared_error' }).fit({ x: [[1, 2], [3, 4], [5, 6], [7, 8]], y: [0, 0, 0, 0] }).params.nFeatures // => [1, 2]
 * ```
 */
export function rfecv(
  options: RfecvOptions,
): IterativeEstimator<RfecvOptions, RfecvState, RfecvParams, WrapperVerbs, RfecvPath> {
  return iterative<RfecvOptions, RfecvState, RfecvParams, WrapperVerbs, RfecvPath>({
    name: 'rfecv',
    kind: options.estimator.kind,
    defaults: options,
    hyperSpace: (h) =>
      dict({
        ...(h.step === undefined || !Number.isInteger(h.step)
          ? {}
          : { step: int(1, 100, { default: h.step, label: 'features per round' }) }),
        ...(h.minFeaturesToSelect === undefined
          ? {}
          : { minFeaturesToSelect: int(1, 1000, { default: h.minFeaturesToSelect, label: 'fewest features' }) }),
      }),
    algorithm: (data, h, key) => rfecvAlgorithm(h.estimator, data, h, key),
    maxSteps: (_, data) => matrixView(data.x, 'rfecv', { allowNaN: true }).n + 2,
    params: (s, data, h) => {
      const d = matrixView(data.x, 'rfecv', { allowNaN: true }).d
      const minFeatures = Math.min(h.minFeaturesToSelect ?? 1, d)
      const counts = [...(s.foldCounts[0] ?? [])].reverse()
      const splitScores = s.foldScores.map((row) => [...row].reverse())
      const final = s.final
      return {
        support: final === null ? new Array<boolean>(d).fill(true) : [...final.support],
        ranking: final === null ? new Array<number>(d).fill(1) : [...final.ranking],
        estimator: final === null ? h.estimator.fit(data).params : final.estimator,
        estimatorSupport: final === null ? new Array<boolean>(d).fill(true) : [...final.fittedOn],
        nFeatures: counts.length === 0 ? [minFeatures] : counts,
        meanTestScore: counts.map((_, c) => meanOf(splitScores.map((r) => r[c] ?? 0))),
        stdTestScore: counts.map((_, c) =>
          Math.sqrt(
            varianceOf(
              splitScores.map((r) => r[c] ?? 0),
              0,
            ),
          ),
        ),
        splitScores,
      }
    },
    verbs: (p, h) => rfeVerbs(p, h, 'rfecv'),
    contract: () => rfecvContract as Contract<RfecvState, RfecvPath>,
  })
}

/** Options of {@link sequentialFeatureSelector}. */
export interface SequentialOptions {
  readonly estimator: Estimator
  /** How many features to keep, or `'auto'` (half, or with `tol`, until the gain drops below it). */
  readonly nFeaturesToSelect?: number | 'auto'
  /** With `'auto'`: stop when adding (or removing) a feature improves the score by less than this. */
  readonly tol?: number
  /** Add features from none (`'forward'`, default) or remove them from all (`'backward'`). */
  readonly direction?: 'forward' | 'backward'
  readonly cv?: CvOption
  readonly scoring?: Scoring
}

/** Parameters of a fitted {@link sequentialFeatureSelector}. */
export type SequentialParams = { readonly support: boolean[]; readonly order: number[]; readonly scores: number[] }

/**
 * Greedy sequential feature selection (sklearn.feature_selection.SequentialFeatureSelector): at each step add (or
 * remove) the feature whose inclusion (or removal) gives the best mean cross-validated score. A traced algorithm with
 * one step per decision.
 *
 * @param options - estimator, nFeaturesToSelect, tol, direction, cv and scoring.
 * @returns The estimator.
 * @example
 * ```ts
 * import { sequentialFeatureSelector } from 'mdml-engine/learn'
 * import { defineEstimator } from 'mdml-engine/learn'
 * import { tensor } from 'mdml-engine/tensor'
 * const firstColumn = defineEstimator(
 *   { name: 'first', kind: 'regressor', fit: () => ({}), methods: () => ({ predict: (x) => tensor(Array.from({ length: x.shape[0] }, (_, i) => x.data[i * x.shape[1]])) }) },
 *   {},
 * )
 * const x = [[0, 5], [1, 4], [2, 9], [3, 1]]
 * sequentialFeatureSelector({ estimator: firstColumn, nFeaturesToSelect: 1, cv: 2, scoring: 'neg_mean_squared_error' }).fit({ x, y: [0, 1, 2, 3] }).support // => [true, false]
 * ```
 */
export function sequentialFeatureSelector(
  options: SequentialOptions,
): Estimator<SequentialOptions, Fitted<SequentialParams> & SelectorVerbs> {
  return defineEstimator(
    {
      name: 'sequential-feature-selector',
      kind: 'transformer',
      fit: (data, h, fitOptions) => {
        const d = matrixView(data.x, 'sequentialFeatureSelector', { allowNaN: true }).d
        const forward = (h.direction ?? 'forward') === 'forward'
        const auto = h.nFeaturesToSelect === undefined || h.nFeaturesToSelect === 'auto'
        const wanted = auto
          ? h.tol === undefined
            ? Math.floor(d / 2)
            : d - 1
          : (h.nFeaturesToSelect as number) > 0 && (h.nFeaturesToSelect as number) < 1
            ? Math.floor((h.nFeaturesToSelect as number) * d)
            : (h.nFeaturesToSelect as number)
        if (!(Number.isInteger(wanted) && wanted >= 1 && wanted < d)) {
          throw new MdmlError('value', `sequentialFeatureSelector: nFeaturesToSelect must be in [1, ${d - 1}]`)
        }
        const iterations = auto || forward ? wanted : d - wanted
        const key = keyFor(fitOptions, 'sequentialFeatureSelector')
        type State = {
          readonly mask: boolean[]
          readonly order: number[]
          readonly scores: number[]
          readonly stop: boolean
        }
        const algorithm: Algorithm<State> = {
          name: 'sequential-feature-selector',
          config: { estimator: h.estimator.name, direction: forward ? 'forward' : 'backward', iterations },
          schema: channels({
            metrics: { score: { description: 'the best mean CV score of the step', direction: 'higher' } },
          }),
          init: () => ({ mask: new Array<boolean>(d).fill(false), order: [], scores: [], stop: false }),
          step: (s, ctx) => {
            let bestJ = -1
            let bestScore = Number.NEGATIVE_INFINITY
            for (let j = 0; j < d; j++) {
              if (s.mask[j] === true) continue
              const candidate = s.mask.map((v, t) => v || t === j)
              const support = forward ? candidate : candidate.map((v) => !v)
              const score = meanOf(
                crossValScore(h.estimator, columns(data, support), {
                  ...innerFitOptions(fitOptions, foldIn(key, 0), ctx.run),
                  ...(h.cv === undefined ? {} : { cv: h.cv }),
                  ...(h.scoring === undefined ? {} : { scoring: h.scoring }),
                }),
              )
              if (score > bestScore) {
                bestScore = score
                bestJ = j
              }
            }
            const previous = s.scores[s.scores.length - 1] ?? Number.NEGATIVE_INFINITY
            if (auto && h.tol !== undefined && bestScore - previous < h.tol) return { ...s, stop: true }
            const mask = [...s.mask]
            mask[bestJ] = true
            return { mask, order: [...s.order, bestJ], scores: [...s.scores, bestScore], stop: false }
          },
          done: (s) => (s.stop || s.order.length >= iterations ? 'completed' : undefined),
          observe: (s) => ({ metrics: s.scores.length === 0 ? {} : { score: s.scores[s.scores.length - 1] ?? 0 } }),
        }
        const { state } = runAlgorithm(algorithm, {
          ...runOptionsFor(fitOptions, 'sequentialFeatureSelector'),
          maxSteps: iterations,
        })
        return { support: forward ? state.mask : state.mask.map((v) => !v), order: state.order, scores: state.scores }
      },
      methods: (p) => selectorVerbs(p.support, 'sequentialFeatureSelector'),
    },
    options,
  )
}

/** Options of {@link stabilitySelection}. */
export interface StabilityOptions {
  /** A selector (its fitted `support` is read) or an estimator (features with importance above `eps` count as selected). */
  readonly selector: Estimator
  /** Subsamples (default 100), each of half the rows without replacement. */
  readonly nSubsamples?: number
  /** Keep features selected in at least this fraction of the subsamples (default 0.6). */
  readonly threshold?: number
  /** Importances at or below this do not count as selected (default 1e-10). */
  readonly eps?: number
}

/** Parameters of a fitted {@link stabilitySelection}. */
export type StabilityParams = { readonly support: boolean[]; readonly frequencies: number[] }

/**
 * Stability selection (Meinshausen and Buehlmann 2010): run a selector (a sparse model or any selector) on many
 * half-size subsamples and keep the features selected in at least `threshold` of them, which controls the expected
 * number of false selections. A traced algorithm with one step per subsample.
 *
 * @param options - selector, nSubsamples, threshold and eps.
 * @returns The estimator.
 * @example
 * ```ts
 * import { selectKBest, fRegression, stabilitySelection } from 'mdml-engine/learn'
 * const x = Array.from({ length: 40 }, (_, i) => [i, Math.sin(7 * i), (i * 13) % 7])
 * const y = x.map(([a]) => a)
 * stabilitySelection({ selector: selectKBest({ scoreFunc: fRegression, k: 1 }), nSubsamples: 20 }).fit({ x, y }).support // => [true, false, false]
 * ```
 */
export function stabilitySelection(
  options: StabilityOptions,
): Estimator<StabilityOptions, Fitted<StabilityParams> & SelectorVerbs> {
  return defineEstimator(
    {
      name: 'stability-selection',
      kind: 'transformer',
      fit: (data, h, fitOptions) => {
        const m = matrixView(data.x, 'stabilitySelection', { allowNaN: true })
        const b = h.nSubsamples ?? 100
        if (!(Number.isInteger(b) && b >= 1))
          throw new MdmlError('value', 'stabilitySelection: nSubsamples must be >= 1')
        const half = Math.floor(m.n / 2)
        if (half < 2) throw new MdmlError('value', 'stabilitySelection: needs at least 4 samples')
        const key = keyFor(fitOptions, 'stabilitySelection')
        const counts = new Float64Array(m.d)
        const algorithm: Algorithm<number> = {
          name: 'stability-selection',
          config: { selector: h.selector.name, nSubsamples: b },
          init: () => 0,
          step: (s, ctx) => {
            const rows = toFlat(subsample(foldIn(key, 1 + s), m.n, half))
            const fitted = h.selector.fit(
              takeData(data, rows),
              innerFitOptions(fitOptions, foldIn(ctx.key, 0), ctx.run),
            )
            const support = (fitted as Partial<SelectorVerbs>).support
            const chosen =
              support ?? importancesOf(fitted, 'stabilitySelection').map((v) => Math.abs(v) > (h.eps ?? 1e-10))
            chosen.forEach((on, j) => {
              if (on) counts[j] = (counts[j] ?? 0) + 1
            })
            return s + 1
          },
          done: (s) => (s >= b ? 'completed' : undefined),
        }
        runAlgorithm(algorithm, { ...runOptionsFor(fitOptions, 'stabilitySelection'), maxSteps: b })
        const frequencies = Array.from(counts, (c) => c / b)
        return { support: frequencies.map((f) => f >= (h.threshold ?? 0.6)), frequencies }
      },
      methods: (p) => selectorVerbs(p.support, 'stabilitySelection'),
    },
    options,
  )
}
