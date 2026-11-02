/**
 * Voting and stacking over a set of base estimators of the library, on the iterative base, one base estimator per
 * step. A voting ensemble combines the bases' predictions directly (hard: a weighted majority; soft: the weighted mean
 * of the probabilities; for regression the weighted mean). A stacking ensemble feeds the bases' out-of-fold predictions
 * (from cross-validation, so the meta-learner never sees a prediction made on a row the base was fitted to) to a final
 * estimator, in a last step. The tape shows the bases' own scores, how much they disagree and what the combination
 * learnt to trust.
 */

import type { Json } from '../base/index.ts'
import { argmaxOf, MdmlError } from '../base/index.ts'
import { categorical, type Distribution } from '../dists/index.ts'
import type { Exposition } from '../exposition/index.ts'
import { type IterativeEstimator, iterative } from '../learn/bases/iterative.ts'
import {
  type Estimator,
  type Fitted,
  type FittedFacts,
  type LearnData,
  type Verbs,
  withDefaultScore,
} from '../learn/estimator.ts'
import { crossValPredict } from '../learn/model-selection/cross-validation.ts'
import { encodeLabels, matrixView, rowsOf, targetVector } from '../learn/validation.ts'
import { foldIn } from '../random/index.ts'
import { dict, int } from '../space/index.ts'
import { fromStorage, type Tensor, type TensorLike, toFlat } from '../tensor/index.ts'
import {
  type Algorithm,
  type Contract,
  defineContract,
  derived,
  Log,
  scalar,
  stateFromJson,
  stateToJson,
  tensor,
} from '../trace/index.ts'
import { decisionTreeClassifier, decisionTreeRegressor } from '../trees/decision-tree.ts'
import { memberProba, stagesFromRun } from './_stages.ts'

/** A named base estimator. */
export type NamedEstimator = readonly [string, Estimator]

/** Hyperparameters of the voting and stacking ensembles. */
export interface StackingHyper {
  readonly estimators: readonly NamedEstimator[]
  /** `voting`: combine the bases directly; `stacking`: train `final` on their out-of-fold predictions. */
  readonly mode: 'voting' | 'stacking'
  /** Voting classifiers: majority of labels (`hard`) or mean of probabilities (`soft`). */
  readonly voting: 'hard' | 'soft'
  /** Voting weights per base (default equal). */
  readonly weights?: readonly number[]
  /** The meta-learner of a stack (default a depth-2 tree, until the linear estimators arrive). */
  readonly final: Estimator
  /** Folds for the out-of-fold predictions of a stack (default 5). */
  readonly cv: number
  /** Give the final estimator the original features beside the predictions. */
  readonly passthrough: boolean
}

/** How a base's predictions enter the combination. */
type Method = 'predictProba' | 'decisionFunction' | 'predict'

/** The state of the voting and stacking algorithm. */
export interface StackingState {
  /** The bases fitted so far (parameters), in order. */
  readonly bases: readonly Json[]
  /** Each base's out-of-fold (stacking) or in-sample (voting) predictions on the training rows: `[n * width]`. */
  readonly meta: readonly Float64Array[]
  /** Each base's labels on those rows (classifiers) or values (regressors). */
  readonly labels: readonly Float64Array[]
  /** Each base's training score. */
  readonly baseScores: readonly number[]
  /** The ensemble's score after each step. */
  readonly scores: readonly number[]
  /** The final estimator of a stack, once fitted. */
  readonly final: Json | null
}

/** Parameters of a fitted voting or stacking ensemble. */
export type StackingParams = {
  readonly bases: Json[]
  readonly final: Json | null
  readonly classes: number[]
  readonly nFeatures: number
  readonly mode: 'voting' | 'stacking'
  /** The coefficients of the final estimator if it has any, else the voting weights. */
  readonly weights: number[]
}

/** Verbs and facts of a fitted voting or stacking ensemble. */
export interface StackingVerbs extends Verbs, FittedFacts {
  readonly predict: (x: TensorLike) => Tensor
  readonly predictProba?: (x: TensorLike) => Tensor
  readonly predictDistribution?: (x: TensorLike) => Distribution
  readonly nFeaturesIn: number
}

/**
 * The paths of voting and stacking: what each base adds, the training score (`eval/accuracy` for classifiers,
 * `eval/r2` for regressors) and the derived `model/diversity`.
 *
 * @example
 * ```ts
 * import type { StackingPath } from 'mdml-engine/ensembles'
 * const path: StackingPath = 'ensemble/baseScores'
 * path // => 'ensemble/baseScores'
 * ```
 */
export type StackingPath =
  | 'ensemble/baseScores'
  | 'ensemble/weights'
  | 'ensemble/diversity'
  | 'ensemble/weight'
  | 'ensemble/trainScore'
  | 'eval/accuracy'
  | 'eval/r2'
  | 'model/diversity'

/** A voting or stacking estimator. */
export type StackingEstimator = IterativeEstimator<
  StackingHyper,
  StackingState,
  StackingParams,
  StackingVerbs,
  StackingPath
>

/** How a base is read: probabilities when it has them, else decision scores, else predictions. */
function methodOf(f: Fitted, classifier: boolean): Method {
  if (!classifier) return 'predict'
  if (f.predictProba !== undefined) return 'predictProba'
  if (f.decisionFunction !== undefined) return 'decisionFunction'
  return 'predict'
}

/** A base's output on rows as columns `[m * width]`; binary probabilities keep only the positive class. */
function outputsOf(
  f: Fitted,
  method: Method,
  x: TensorLike,
  k: number,
): { readonly values: Float64Array; readonly width: number } {
  const fn = f[method] as (x: TensorLike) => Tensor
  const t = fn(x)
  const flat = Float64Array.from(toFlat(t))
  const m = t.shape[0] ?? 0
  const width = flat.length / Math.max(m, 1)
  if (method === 'predictProba' && k === 2) {
    const out = new Float64Array(m)
    for (let i = 0; i < m; i++) out[i] = flat[i * 2 + 1] ?? 0
    return { values: out, width: 1 }
  }
  return { values: flat, width }
}

/** The meta-feature matrix `[m, total]` from the bases' outputs (and the original features when passing through). */
function metaMatrix(parts: readonly (readonly [Float64Array, number])[], m: number, x?: TensorLike): Tensor {
  const extra = x === undefined ? undefined : matrixView(x, 'stacking', { allowNaN: true })
  const total = parts.reduce((a, [, w]) => a + w, 0) + (extra?.d ?? 0)
  const out = new Float64Array(m * total)
  for (let i = 0; i < m; i++) {
    let col = 0
    for (const [v, w] of parts) {
      for (let c = 0; c < w; c++) out[i * total + col++] = v[i * w + c] ?? 0
    }
    if (extra !== undefined) for (let c = 0; c < extra.d; c++) out[i * total + col++] = extra.data[i * extra.d + c] ?? 0
  }
  return fromStorage(out, [m, total], 'float64')
}

function stackingEstimator(name: string, classifier: boolean, defaults: StackingHyper): StackingEstimator {
  const classesOf = (data: LearnData): readonly number[] => (classifier ? encodeLabels(data.y ?? [], name).classes : [])
  const algorithmOf = (data: LearnData, h: StackingHyper): Algorithm<StackingState, StackingPath> => {
    if (h.estimators.length === 0) throw new MdmlError('value', `${name}: needs at least one base estimator`)
    if (h.weights !== undefined && h.weights.length !== h.estimators.length)
      throw new MdmlError('value', `${name}: ${h.weights.length} weights for ${h.estimators.length} estimators`)
    const n = rowsOf(data.x)
    const classes = classesOf(data)
    const k = classes.length
    const y = classifier ? Float64Array.from(encodeLabels(data.y ?? [], name).codes) : targetVector(data.y ?? [], name)
    const stack = h.mode === 'stacking'
    const nBases = h.estimators.length
    /** The ensemble's training score from the bases' outputs so far (a vote, or the stacked model). */
    const scoreOfVote = (state: readonly Fitted[], weights: readonly number[]): number => {
      const pred = voteOf(state, weights, data.x, h, classes)
      let right = 0
      let sse = 0
      let sst = 0
      const mean = Array.from(y).reduce((a, b) => a + b, 0) / n
      for (let i = 0; i < n; i++) {
        if (classifier) right += pred[i] === (classes[y[i] ?? 0] ?? 0) ? 1 : 0
        else {
          sse += ((pred[i] ?? 0) - (y[i] ?? 0)) ** 2
          sst += ((y[i] ?? 0) - mean) ** 2
        }
      }
      return classifier ? right / n : sst > 0 ? 1 - sse / sst : 1
    }
    return {
      name,
      config: { mode: h.mode, bases: h.estimators.map(([id]) => id), voting: h.voting },
      init: () => ({ bases: [], meta: [], labels: [], baseScores: [], scores: [], final: null }),
      step: (s, ctx) => {
        const b = s.bases.length
        if (b < nBases) {
          const [, est] = h.estimators[b] as NamedEstimator
          const fitted = est.fit(data, { key: foldIn(ctx.key, 1), explain: false })
          const method = methodOf(fitted, classifier)
          const insample = outputsOf(fitted, method, data.x, k)
          // A stack trains on out-of-fold predictions; a vote shows the bases as fitted.
          const meta = stack
            ? (() => {
                const folded = crossValPredict(est, data, { cv: h.cv, method, key: foldIn(ctx.key, 0) })
                const flat = Float64Array.from(toFlat(folded))
                if (method === 'predictProba' && k === 2)
                  return Float64Array.from({ length: n }, (_, i) => flat[i * 2 + 1] ?? 0)
                return flat
              })()
            : insample.values
          const labels = Float64Array.from(toFlat(fitted.predict?.(data.x) as Tensor))
          const baseScore = fitted.score?.(data) ?? Number.NaN
          const bases = [...s.bases, fitted.params]
          const fits = bases.map((p, i) => (h.estimators[i] as NamedEstimator)[1].revive(p))
          const weights = (h.weights ?? new Array<number>(nBases).fill(1)).slice(0, bases.length)
          const score = scoreOfVote(fits, weights)
          if (ctx.wants('ensemble/baseScores')) ctx.log('ensemble/baseScores', [...s.baseScores, baseScore])
          if (ctx.wants(classifier ? 'eval/accuracy' : 'eval/r2'))
            ctx.log(classifier ? 'eval/accuracy' : 'eval/r2', score)
          if (ctx.wants('ensemble/trainScore')) ctx.log('ensemble/trainScore', score)
          if (ctx.wants('ensemble/weight')) ctx.log('ensemble/weight', h.weights?.[b] ?? 1)
          const allLabels = [...s.labels, labels]
          if (ctx.wants('ensemble/diversity')) ctx.log('ensemble/diversity', disagreement(allLabels))
          if (ctx.wants('ensemble/weights'))
            ctx.log('ensemble/weights', h.weights === undefined ? new Array<number>(bases.length).fill(1) : weights)
          return {
            bases,
            meta: [...s.meta, meta],
            labels: allLabels,
            baseScores: [...s.baseScores, baseScore],
            scores: [...s.scores, score],
            final: null,
          }
        }
        // The final estimator of a stack, on the out-of-fold predictions.
        const widths = s.meta.map((v) => v.length / n)
        const features = metaMatrix(
          s.meta.map((v, i) => [v, widths[i] ?? 1] as const),
          n,
          h.passthrough ? data.x : undefined,
        )
        const final = h.final.fit({ ...data, x: features }, { key: foldIn(ctx.key, 2), explain: false })
        const fits = s.bases.map((p, i) => (h.estimators[i] as NamedEstimator)[1].revive(p))
        const methods = fits.map((f) => methodOf(f, classifier))
        const parts = fits.map((f, i) => {
          const o = outputsOf(f, methods[i] as Method, data.x, k)
          return [o.values, o.width] as const
        })
        const predicted = Float64Array.from(
          toFlat(final.predict?.(metaMatrix(parts, n, h.passthrough ? data.x : undefined)) as Tensor),
        )
        let right = 0
        let sse = 0
        let sst = 0
        const mean = Array.from(y).reduce((a, b) => a + b, 0) / n
        for (let i = 0; i < n; i++) {
          if (classifier) right += predicted[i] === (classes[y[i] ?? 0] ?? 0) ? 1 : 0
          else {
            sse += ((predicted[i] ?? 0) - (y[i] ?? 0)) ** 2
            sst += ((y[i] ?? 0) - mean) ** 2
          }
        }
        const score = classifier ? right / n : sst > 0 ? 1 - sse / sst : 1
        if (ctx.wants(classifier ? 'eval/accuracy' : 'eval/r2'))
          ctx.log(classifier ? 'eval/accuracy' : 'eval/r2', score)
        if (ctx.wants('ensemble/trainScore')) ctx.log('ensemble/trainScore', score)
        if (ctx.wants('ensemble/weights'))
          ctx.log('ensemble/weights', weightsOf(final.params, final.featureImportances, s.bases.length))
        return { ...s, scores: [...s.scores, score], final: final.params }
      },
      done: (s) => (s.bases.length >= nBases && (!stack || s.final !== null) ? 'completed' : undefined),
      serialise: (s) => stateToJson(s),
      deserialise: (j) => stateFromJson(j) as StackingState,
    }
  }
  return iterative<StackingHyper, StackingState, StackingParams, StackingVerbs, StackingPath>(
    {
      name,
      kind: classifier ? 'classifier' : 'regressor',
      defaults,
      hyperSpace: (h) => dict({ cv: int(2, Math.max(10, h.cv), { default: h.cv, label: 'folds' }) }),
      algorithm: (data, h) => algorithmOf(data, h),
      params: (state, data, h) => ({
        bases: [...state.bases],
        final: state.final,
        classes: [...classesOf(data)],
        nFeatures: matrixView(data.x, name, { allowNaN: true }).d,
        mode: h.mode,
        weights:
          state.final !== null
            ? weightsOf(state.final, undefined, state.bases.length)
            : (h.weights ?? new Array<number>(state.bases.length).fill(1)).slice(0, state.bases.length),
      }),
      verbs: (p, h) => {
        const fits = p.bases.map((b, i) => (h.estimators[i] as NamedEstimator)[1].revive(b))
        const final = p.final === null ? undefined : h.final.revive(p.final)
        const k = p.classes.length
        const metaOf = (x: TensorLike): Tensor => {
          const m = matrixView(x, name, { allowNaN: true }).n
          const parts = fits.map((f) => {
            const o = outputsOf(f, methodOf(f, classifier), x, k)
            return [o.values, o.width] as const
          })
          return metaMatrix(parts, m, h.passthrough ? x : undefined)
        }
        const weights = h.weights ?? new Array<number>(fits.length).fill(1)
        const probaOf = (x: TensorLike): Tensor => {
          if (final?.predictProba !== undefined) return final.predictProba(metaOf(x))
          const m = matrixView(x, name, { allowNaN: true }).n
          const mean = new Float64Array(m * k)
          let total = 0
          for (const [i, f] of fits.entries()) {
            const pr = memberProba(f, x, p.classes)
            total += weights[i] ?? 1
            for (let j = 0; j < mean.length; j++) mean[j] = (mean[j] ?? 0) + (weights[i] ?? 1) * (pr[j] ?? 0)
          }
          return fromStorage(
            mean.map((v) => v / (total || 1)),
            [m, k],
            'float64',
          )
        }
        if (classifier) {
          const hasProba = final !== undefined ? final.predictProba !== undefined : h.voting === 'soft'
          return withDefaultScore('classifier', {
            predict: (x: TensorLike) => {
              if (final !== undefined) return final.predict?.(metaOf(x)) as Tensor
              if (h.voting === 'soft') {
                const pr = probaOf(x)
                const m = pr.shape[0] ?? 0
                return fromStorage(
                  Float64Array.from(
                    { length: m },
                    (_, i) => p.classes[argmaxOf(pr.data.subarray(i * k, (i + 1) * k))] ?? 0,
                  ),
                  [m],
                  'float64',
                )
              }
              const pred = voteOf(fits, weights, x, h, p.classes)
              return fromStorage(pred, [pred.length], 'float64')
            },
            ...(hasProba
              ? { predictProba: probaOf, predictDistribution: (x: TensorLike) => categorical({ probs: probaOf(x) }) }
              : {}),
            classes: p.classes,
            nFeaturesIn: p.nFeatures,
          })
        }
        return withDefaultScore('regressor', {
          predict: (x: TensorLike) => {
            if (final !== undefined) return final.predict?.(metaOf(x)) as Tensor
            const pred = voteOf(fits, weights, x, h, [])
            return fromStorage(pred, [pred.length], 'float64')
          },
          nFeaturesIn: p.nFeatures,
        })
      },
      maxSteps: (h) => h.estimators.length + (h.mode === 'stacking' ? 1 : 0),
      contract: (): Contract<StackingState, StackingPath> => {
        const fixed = defineContract({
          'ensemble/baseScores': tensor({
            role: 'output',
            shape: ['bases'],
            description: 'each base estimator’s training score',
          }),
          'ensemble/weights': tensor({
            role: 'output',
            shape: ['bases'],
            description: 'the vote weights, or what the meta-learner gave each base',
          }),
          'ensemble/diversity': tensor({
            role: 'output',
            shape: ['bases', 'bases'],
            description: 'how often each pair of bases disagree on the training rows',
          }),
          'ensemble/weight': scalar({ description: 'the new base’s voting weight' }),
          'ensemble/trainScore': scalar({
            description: 'the ensemble’s training accuracy (or R²)',
            direction: 'higher',
          }),
          'model/diversity': derived<StackingState>({
            description: 'the pairwise disagreement of the bases fitted so far',
            compute: (state) => disagreement(state.labels),
          }),
        })
        // The training score is one path or the other by the task; the type holds both.
        return {
          ...fixed,
          [classifier ? 'eval/accuracy' : 'eval/r2']: scalar({
            description: 'the ensemble’s training accuracy (or R²) so far',
            direction: 'higher',
          }),
        } as Contract<StackingState, StackingPath>
      },
      projections: {
        'ensemble/stages': (run) =>
          stagesFromRun(run, {
            method: name,
            scoreName: classifier ? 'training accuracy' : 'training R²',
            higherIsBetter: true,
            weight: 'ensemble/weight',
            train: 'ensemble/trainScore',
          }) as Exposition | undefined,
      },
      logs: {
        'ensemble/baseScores': Log(),
        'ensemble/weights': Log({ detail: 'full' }),
        'ensemble/diversity': Log(),
        'ensemble/weight': Log(),
        'ensemble/trainScore': Log(),
        [classifier ? 'eval/accuracy' : 'eval/r2']: Log(),
      },
    },
    defaults,
  )
}

/** The ensemble's predictions from its bases: the weighted majority of labels, the mean of probabilities, or values. */
function voteOf(
  fits: readonly Fitted[],
  weights: readonly number[],
  x: TensorLike,
  h: Pick<StackingHyper, 'mode' | 'voting'>,
  classes: readonly number[],
): Float64Array {
  const m = matrixView(x, 'voting', { allowNaN: true }).n
  const k = classes.length
  const w = (i: number): number => weights[i] ?? 1
  if (k === 0) {
    const mean = new Float64Array(m)
    let total = 0
    for (const [i, f] of fits.entries()) {
      const pred = toFlat((f.predict as NonNullable<Verbs['predict']>)(x))
      total += w(i)
      for (let r = 0; r < m; r++) mean[r] = (mean[r] ?? 0) + w(i) * (pred[r] ?? 0)
    }
    return mean.map((v) => v / (total || 1))
  }
  const soft = h.mode === 'voting' ? h.voting === 'soft' : fits.every((f) => f.predictProba !== undefined)
  const score = new Float64Array(m * k)
  for (const [i, f] of fits.entries()) {
    if (soft) {
      const pr = memberProba(f, x, classes)
      for (let j = 0; j < score.length; j++) score[j] = (score[j] ?? 0) + w(i) * (pr[j] ?? 0)
    } else {
      const pred = toFlat((f.predict as NonNullable<Verbs['predict']>)(x))
      for (let r = 0; r < m; r++) {
        const c = classes.indexOf(pred[r] ?? Number.NaN)
        if (c >= 0) score[r * k + c] = (score[r * k + c] ?? 0) + w(i)
      }
    }
  }
  return Float64Array.from({ length: m }, (_, r) => classes[argmaxOf(score.subarray(r * k, (r + 1) * k))] ?? 0)
}

/** How often each pair of bases differ on the training rows (labels), or their mean squared difference (values). */
function disagreement(labels: readonly Float64Array[]): number[][] {
  return labels.map((a) =>
    labels.map((b) => {
      const n = a.length
      let differ = 0
      let discrete = true
      for (let i = 0; i < n; i++) {
        if (!Number.isInteger(a[i] ?? 0) || !Number.isInteger(b[i] ?? 0)) discrete = false
      }
      for (let i = 0; i < n; i++)
        differ += discrete ? ((a[i] ?? 0) === (b[i] ?? 0) ? 0 : 1) : ((a[i] ?? 0) - (b[i] ?? 0)) ** 2
      return differ / Math.max(n, 1)
    }),
  )
}

/** What a stack's final estimator gave each base: its coefficients, else its importances, else equal weights. */
function weightsOf(params: Json, importances: readonly number[] | undefined, nBases: number): number[] {
  if (typeof params === 'object' && params !== null && !Array.isArray(params)) {
    const record = params as { readonly [key: string]: Json }
    const coef = record['coef'] ?? record['coefficients']
    if (Array.isArray(coef)) return coef.flat(2).filter((v): v is number => typeof v === 'number')
  }
  return importances !== undefined && importances.length > 0 ? [...importances] : new Array<number>(nBases).fill(1)
}

const stackingDefaults = (
  hyper: Partial<StackingHyper> & Pick<StackingHyper, 'estimators'>,
  mode: 'voting' | 'stacking',
  final: Estimator,
): StackingHyper => ({
  estimators: hyper.estimators,
  mode,
  voting: hyper.voting ?? 'hard',
  ...(hyper.weights === undefined ? {} : { weights: hyper.weights }),
  final: hyper.final ?? final,
  cv: hyper.cv ?? 5,
  passthrough: hyper.passthrough ?? false,
})

/**
 * A voting classifier (sklearn.ensemble.VotingClassifier): the bases are fitted one per step and their predictions
 * combined, by a weighted majority of labels (`hard`) or the weighted mean of probabilities (`soft`). The session logs
 * `ensemble/baseScores` (each base's training score), `ensemble/diversity` (pairwise disagreement), `ensemble/weights`
 * and `eval/accuracy` of the vote so far.
 *
 * @param hyper - The named bases, `voting` (hard) and `weights`.
 * @returns The estimator.
 * @concept ensembles
 * @example
 * ```ts
 * import { votingClassifier, baggingClassifier } from 'mdml-engine/ensembles'
 * import { decisionTreeClassifier } from 'mdml-engine/trees'
 * const x = Array.from({ length: 40 }, (_, i) => [i / 40])
 * const y = x.map(([a = 0]) => (a < 0.5 ? 0 : 1))
 * const bases = [['a', decisionTreeClassifier({ maxDepth: 1 })], ['b', decisionTreeClassifier({ maxDepth: 2 })]]
 * votingClassifier({ estimators: bases, voting: 'soft' }).fit({ x, y }).params.bases.length // => 2
 * ```
 */
export function votingClassifier(hyper: Partial<StackingHyper> & Pick<StackingHyper, 'estimators'>): StackingEstimator {
  return stackingEstimator(
    'voting-classifier',
    true,
    stackingDefaults(hyper, 'voting', decisionTreeClassifier({ maxDepth: 2, record: 0 })),
  )
}

/**
 * A voting regressor (sklearn.ensemble.VotingRegressor): the weighted mean of the bases' predictions.
 *
 * @param hyper - The named bases and `weights`.
 * @returns The estimator.
 * @concept ensembles
 * @example
 * ```ts
 * import { votingRegressor } from 'mdml-engine/ensembles'
 * import { decisionTreeRegressor } from 'mdml-engine/trees'
 * import { toFlat } from 'mdml-engine/tensor'
 * const x = Array.from({ length: 20 }, (_, i) => [i])
 * const f = votingRegressor({ estimators: [['a', decisionTreeRegressor({ maxDepth: 1 })], ['b', decisionTreeRegressor({ maxDepth: 2 })]] })
 * toFlat(f.fit({ x, y: x.map(([a = 0]) => a) }).predict([[3]])).length // => 1
 * ```
 */
export function votingRegressor(hyper: Partial<StackingHyper> & Pick<StackingHyper, 'estimators'>): StackingEstimator {
  return stackingEstimator(
    'voting-regressor',
    false,
    stackingDefaults(hyper, 'voting', decisionTreeRegressor({ maxDepth: 2, record: 0 })),
  )
}

/**
 * A stacking classifier (sklearn.ensemble.StackingClassifier): each base is fitted and cross-validated in turn (one per
 * step), and a last step fits the final estimator on the bases' out-of-fold probabilities (decision scores or labels
 * for bases without), so it learns how far to trust each. `passthrough` adds the original features.
 *
 * @param hyper - The named bases, the `final` estimator (default a shallow tree), `cv` folds (5) and `passthrough`.
 * @returns The estimator.
 * @concept ensembles
 * @index estimator Stacked generalisation; voting ensembles sit beside it.
 * @example
 * ```ts
 * import { stackingClassifier } from 'mdml-engine/ensembles'
 * import { decisionTreeClassifier } from 'mdml-engine/trees'
 * const x = Array.from({ length: 60 }, (_, i) => [i / 60, ((i * 7) % 11) / 11])
 * const y = x.map(([a = 0]) => (a < 0.5 ? 0 : 1))
 * const bases = [['stump', decisionTreeClassifier({ maxDepth: 1 })], ['deep', decisionTreeClassifier()]]
 * stackingClassifier({ estimators: bases, cv: 3 }).fit({ x, y }).params.final === null // => false
 * ```
 */
export function stackingClassifier(
  hyper: Partial<StackingHyper> & Pick<StackingHyper, 'estimators'>,
): StackingEstimator {
  return stackingEstimator(
    'stacking-classifier',
    true,
    stackingDefaults(hyper, 'stacking', decisionTreeClassifier({ maxDepth: 2, record: 0 })),
  )
}

/**
 * A stacking regressor (sklearn.ensemble.StackingRegressor): the final estimator is fitted on the bases' out-of-fold
 * predictions.
 *
 * @param hyper - As {@link stackingClassifier}.
 * @returns The estimator.
 * @concept ensembles
 * @example
 * ```ts
 * import { stackingRegressor } from 'mdml-engine/ensembles'
 * import { decisionTreeRegressor } from 'mdml-engine/trees'
 * const x = Array.from({ length: 30 }, (_, i) => [i])
 * const f = stackingRegressor({ estimators: [['a', decisionTreeRegressor({ maxDepth: 2 })]], cv: 3 })
 * f.fit({ x, y: x.map(([a = 0]) => a) }).params.mode // => 'stacking'
 * ```
 */
export function stackingRegressor(
  hyper: Partial<StackingHyper> & Pick<StackingHyper, 'estimators'>,
): StackingEstimator {
  return stackingEstimator(
    'stacking-regressor',
    false,
    stackingDefaults(hyper, 'stacking', decisionTreeRegressor({ maxDepth: 2, record: 0 })),
  )
}
