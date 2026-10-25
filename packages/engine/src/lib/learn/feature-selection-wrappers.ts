/**
 * Wrapper feature selection, which chooses features by refitting a model on candidate subsets: recursive feature
 * elimination (`rfe`, and `rfecv` with the count chosen by cross-validation), greedy forward or backward sequential
 * selection, and stability selection, which keeps the features chosen on most of many random half-samples.
 */

import { type Json, MdmlError, meanOf, varianceOf } from '../base/index.ts'
import { foldIn, subsample } from '../random/index.ts'
import { type TensorLike, toFlat } from '../tensor/index.ts'
import { type Algorithm, channels, runAlgorithm } from '../trace/index.ts'
import { innerFitOptions, keyFor, runOptionsFor } from './_shared.ts'
import {
  defineEstimator,
  type Estimator,
  type FitOptions,
  type Fitted,
  type LearnData,
  type Verbs,
} from './estimator.ts'
import { type SelectorVerbs, selectorVerbs } from './feature-selection.ts'
import { type CvOption, checkCv, crossValScore, splitInputOf } from './model-selection/cross-validation.ts'
import { type Fold, splitIndices } from './model-selection/splitters.ts'
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

function delegate(final: Fitted | undefined, sel: SelectorVerbs): WrapperVerbs {
  const out: Record<string, unknown> = { ...sel }
  if (final !== undefined) {
    for (const verb of ['predict', 'predictProba', 'decisionFunction'] as const) {
      const f = final[verb]
      if (typeof f === 'function') out[verb] = (x: TensorLike) => f(sel.transform(x))
    }
    const score = final.score
    if (score !== undefined) out['score'] = (data: LearnData) => score({ ...data, x: sel.transform(data.x) })
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
}

/** Parameters of a fitted {@link rfe}. */
export type RfeParams = { readonly support: boolean[]; readonly ranking: number[]; readonly estimator: Json }

/** One elimination path: the support before each round and after the last. */
interface RfePath {
  readonly support: boolean[]
  readonly ranking: number[]
  readonly steps: readonly { readonly support: boolean[]; readonly fitted: Fitted }[]
}

function stepSize(step: number, d: number, who: string): number {
  if (step > 0 && step < 1) return Math.max(1, Math.floor(step * d))
  if (!(Number.isInteger(step) && step >= 1))
    throw new MdmlError('value', `${who}: step must be an integer >= 1 or a fraction`)
  return step
}

/** Run recursive elimination down to `target` features, fitting once per round (and once at the end). */
function eliminate(
  estimator: Estimator,
  data: LearnData,
  target: number,
  step: number,
  options: FitOptions,
  who: string,
): RfePath {
  const d = matrixView(data.x, who, { allowNaN: true }).d
  const k = stepSize(step, d, who)
  const algorithm: Algorithm<RfePath> = {
    name: who,
    config: { estimator: estimator.name, nFeatures: d, target, step: k },
    schema: channels({ metrics: { nFeatures: { description: 'features remaining' } } }),
    init: () => ({ support: new Array<boolean>(d).fill(true), ranking: new Array<number>(d).fill(1), steps: [] }),
    step: (s, ctx) => {
      const fitted = estimator.fit(columns(data, s.support), innerFitOptions(options, foldIn(ctx.key, 0), ctx.run))
      const remaining = s.support.filter(Boolean).length
      if (remaining <= target) return { ...s, steps: [...s.steps, { support: s.support, fitted }] }
      const features = s.support.flatMap((on, j) => (on ? [j] : []))
      const imp = importancesOf(fitted, who)
      const order = imp.map((v, r) => ({ v, r })).sort((a, b) => a.v - b.v || a.r - b.r)
      const drop = Math.min(k, remaining - target)
      const support = [...s.support]
      for (const { r } of order.slice(0, drop)) support[features[r] ?? 0] = false
      const ranking = s.ranking.map((rank, j) => (support[j] === true ? rank : rank + 1))
      return { support, ranking, steps: [...s.steps, { support: s.support, fitted }] }
    },
    done: (s) => {
      const last = s.steps[s.steps.length - 1]
      return last !== undefined && last.support.filter(Boolean).length <= target ? 'completed' : undefined
    },
    observe: (s) => ({ metrics: { nFeatures: s.support.filter(Boolean).length } }),
  }
  return runAlgorithm(algorithm, { ...runOptionsFor(options, who), maxSteps: d + 1 }).state
}

function targetCount(n: number | undefined, d: number): number {
  if (n === undefined) return Math.floor(d / 2)
  if (n > 0 && n < 1) return Math.floor(n * d)
  return Math.min(n, d)
}

/**
 * Recursive feature elimination (Guyon et al. 2002; sklearn.feature_selection.RFE): fit, drop the `step` least
 * important features (by the fitted `featureImportances`), and repeat until `nFeaturesToSelect` remain; `ranking` is
 * 1 for kept features and counts the rounds survived otherwise. A traced algorithm with one step per round; the
 * fitted value also predicts with the final estimator on the kept features.
 *
 * @param options - estimator, nFeaturesToSelect and step (default 1).
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
export function rfe(options: RfeOptions): Estimator<RfeOptions, Fitted<RfeParams> & WrapperVerbs> {
  return defineEstimator(
    {
      name: 'rfe',
      kind: options.estimator.kind,
      fit: (data, h, fitOptions) => {
        const d = matrixView(data.x, 'rfe', { allowNaN: true }).d
        const path = eliminate(h.estimator, data, targetCount(h.nFeaturesToSelect, d), h.step ?? 1, fitOptions, 'rfe')
        const final = path.steps[path.steps.length - 1]?.fitted
        return { support: path.support, ranking: path.ranking, estimator: final?.params ?? null }
      },
      methods: (p, h) =>
        delegate(p.estimator === null ? undefined : h.estimator.revive(p.estimator), selectorVerbs(p.support, 'rfe')),
    },
    options,
  )
}

/** Options of {@link rfecv}. */
export interface RfecvOptions {
  readonly estimator: Estimator
  readonly step?: number
  readonly minFeaturesToSelect?: number
  readonly cv?: CvOption
  readonly scoring?: Scoring
  readonly groups?: ArrayLike<number>
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

/**
 * Recursive feature elimination with the number of features chosen by cross-validation (sklearn RFECV): on each fold,
 * eliminate on the training rows scoring every intermediate subset on the test rows; choose the count with the best
 * total score (the fewest features on ties), then eliminate on all the data down to it.
 *
 * @param options - estimator, step, minFeaturesToSelect (default 1), cv, scoring and groups.
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
export function rfecv(options: RfecvOptions): Estimator<RfecvOptions, Fitted<RfecvParams> & WrapperVerbs> {
  return defineEstimator(
    {
      name: 'rfecv',
      kind: options.estimator.kind,
      fit: (data, h, fitOptions) => {
        const d = matrixView(data.x, 'rfecv', { allowNaN: true }).d
        const minFeatures = Math.min(h.minFeaturesToSelect ?? 1, d)
        const scorer = Object.values(scorersOf(h.scoring))[0]
        if (scorer === undefined) throw new MdmlError('value', 'rfecv: no scorer')
        const key = keyFor(fitOptions, 'rfecv')
        const folds = splitIndices(checkCv(h.cv, h.estimator), splitInputOf(data, h.groups), foldIn(key, 0))
        const perFold = folds.map((fold: Fold, f) => {
          const train = takeData(data, fold.train)
          const test = takeData(data, fold.test)
          const path = eliminate(
            h.estimator,
            train,
            minFeatures,
            h.step ?? 1,
            innerFitOptions(fitOptions, foldIn(key, 1 + f), fitOptions.parent),
            'rfecv',
          )
          return path.steps.map((s) => ({
            n: s.support.filter(Boolean).length,
            score: scorer.score(s.fitted, columns(test, s.support)),
          }))
        })
        const counts = (perFold[0] ?? []).map((e) => e.n).reverse()
        const splitScores = perFold.map((p) => p.map((e) => e.score).reverse())
        const totals = counts.map((_, c) => splitScores.reduce((s, row) => s + (row[c] ?? 0), 0))
        let best = 0
        totals.forEach((t, c) => {
          if (t > (totals[best] ?? Number.NEGATIVE_INFINITY)) best = c
        })
        const n = counts[best] ?? minFeatures
        const path = eliminate(
          h.estimator,
          data,
          n,
          h.step ?? 1,
          innerFitOptions(fitOptions, foldIn(key, 0), fitOptions.parent),
          'rfecv',
        )
        const final = path.steps[path.steps.length - 1]?.fitted
        return {
          support: path.support,
          ranking: path.ranking,
          estimator: final?.params ?? null,
          nFeatures: counts,
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
      methods: (p, h) =>
        delegate(p.estimator === null ? undefined : h.estimator.revive(p.estimator), selectorVerbs(p.support, 'rfecv')),
    },
    options,
  )
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
