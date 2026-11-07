/**
 * AdaBoost (Freund and Schapire 1997; Zhu et al. 2009 for the multiclass forms; Drucker 1997 for regression) over any
 * base estimator, a decision stump by default: weak learners fitted in sequence, each on rows reweighted towards those
 * the ensemble so far gets wrong, one round per step. SAMME votes with each member's weight log((1 - err) / err) +
 * log(K - 1); SAMME.R uses the members' class probabilities and weights none; AdaBoost.R2 fits regression trees to
 * weighted resamples and predicts the weighted median. The search for the best weighted decision stump over a pool of
 * feature columns is shared with the Haar-feature cascade of `vision`, which is the same algorithm over a different
 * pool of features.
 */

import type { Json } from '../base/index.ts'
import { argmaxOf, MdmlError, maxOf } from '../base/index.ts'
import { categorical, type Distribution } from '../dists/index.ts'
import type { Exposition } from '../exposition/index.ts'
import { type IterativeEstimator, iterative } from '../learn/bases/iterative.ts'
import { type Estimator, type FittedFacts, type LearnData, type Verbs, withDefaultScore } from '../learn/estimator.ts'
import { encodeLabels, matrixView, rowsOf, sampleWeightsOf, takeData, targetVector } from '../learn/validation.ts'
import { choice as drawRows, foldIn } from '../random/index.ts'
import { choice, dict, int, real } from '../space/index.ts'
import { fromStorage, type Tensor, type TensorLike, toFlat } from '../tensor/index.ts'
import {
  type Algorithm,
  type Contract,
  defineContract,
  derived,
  event,
  Log,
  scalar,
  stateFromJson,
  stateToJson,
  tensor,
} from '../trace/index.ts'
import { decisionTreeClassifier, decisionTreeRegressor } from '../trees/decision-tree.ts'
import { memberProba, stagesFromRun } from './_stages.ts'

/** A decision stump on one feature column: votes 1 when `polarity * value < polarity * threshold`. */
export interface Stump {
  readonly feature: number
  readonly threshold: number
  readonly polarity: 1 | -1
  /** The weighted error of the stump on the labels it was searched against. */
  readonly error: number
}

/**
 * The decision stump of least weighted error over a pool of feature columns, both polarities, thresholds between
 * consecutive sorted values (and beyond both ends). Ties go to the first feature, then to the earlier threshold, then to
 * polarity 1. The weak-learner search of Viola and Jones's cascade and of AdaBoost over stumps.
 *
 * @param values - `values[j][i]` is feature j of row i.
 * @param order - `order[j]` lists the rows by increasing `values[j]` (sort once, reuse each round).
 * @param positive - Whether row i is a positive example.
 * @param weights - The row weights (they need not sum to one; the error is in their units).
 * @returns The best stump and its weighted error.
 * @concept boosting
 * @example
 * ```ts
 * import { bestStump } from 'mdml-engine/ensembles'
 * const values = [[0.1, 0.4, 0.6, 0.9]]
 * const stump = bestStump(values, [[0, 1, 2, 3]], (i) => i >= 2, [0.25, 0.25, 0.25, 0.25])
 * [stump.threshold, stump.polarity, stump.error] // => [0.5, -1, 0]
 * ```
 */
export function bestStump(
  values: readonly ArrayLike<number>[],
  order: readonly (readonly number[])[],
  positive: (row: number) => boolean,
  weights: ArrayLike<number>,
): Stump {
  const n = weights.length
  let tPos = 0
  let tNeg = 0
  for (let i = 0; i < n; i++) {
    if (positive(i)) tPos += weights[i] ?? 0
    else tNeg += weights[i] ?? 0
  }
  let best: Stump = { error: Number.POSITIVE_INFINITY, feature: 0, threshold: 0, polarity: 1 }
  for (const [j, idx] of order.entries()) {
    const v = values[j] as ArrayLike<number>
    let sPos = 0
    let sNeg = 0
    for (let q = 0; q <= n; q++) {
      // Split before sorted position q: below are the first q rows.
      if (q === 0 || q === n || (v[idx[q] ?? 0] ?? 0) !== (v[idx[q - 1] ?? 0] ?? 0)) {
        const threshold =
          q === 0
            ? (v[idx[0] ?? 0] ?? 0) - 1
            : q === n
              ? (v[idx[n - 1] ?? 0] ?? 0) + 1
              : ((v[idx[q] ?? 0] ?? 0) + (v[idx[q - 1] ?? 0] ?? 0)) / 2
        // Polarity 1 votes positive below the threshold; -1 above.
        const e1 = sNeg + (tPos - sPos)
        const e2 = sPos + (tNeg - sNeg)
        if (e1 < best.error) best = { error: e1, feature: j, threshold, polarity: 1 }
        if (e2 < best.error) best = { error: e2, feature: j, threshold, polarity: -1 }
      }
      if (q < n) {
        const i = idx[q] ?? 0
        if (positive(i)) sPos += weights[i] ?? 0
        else sNeg += weights[i] ?? 0
      }
    }
  }
  return best
}

/**
 * The rows ordered by increasing value, for {@link bestStump} (a stable sort).
 *
 * @param values - One feature column.
 * @returns The row indices, smallest value first.
 * @example
 * ```ts
 * import { sortedRows } from 'mdml-engine/ensembles'
 * sortedRows([0.3, 0.1, 0.2]) // => [1, 2, 0]
 * ```
 */
export function sortedRows(values: ArrayLike<number>): number[] {
  return Array.from({ length: values.length }, (_, i) => i).sort((a, b) => (values[a] ?? 0) - (values[b] ?? 0))
}

/** Which AdaBoost: SAMME votes, SAMME.R averages log probabilities. */
export type AdaBoostAlgorithm = 'SAMME' | 'SAMME.R'

/** Hyperparameters of the AdaBoost estimators. */
export interface AdaBoostHyper {
  /** The weak learner (default a stump for classification, a depth-3 tree for regression). */
  readonly estimator: Estimator
  readonly nEstimators: number
  readonly learningRate: number
  /** Classification: SAMME (default) or SAMME.R (the base must give probabilities). */
  readonly algorithm: AdaBoostAlgorithm
  /** Regression: the loss of AdaBoost.R2. */
  readonly loss: 'linear' | 'square' | 'exponential'
}

/** The state of the AdaBoost algorithms. */
export interface AdaBoostState {
  /** The row weights the next member will be fitted to. */
  readonly sampleWeight: Float64Array
  readonly members: readonly Json[]
  /** Each member's weight in the vote (all 1 for SAMME.R), and its weighted error. */
  readonly weights: readonly number[]
  readonly errors: readonly number[]
  /** The row weights each member was fitted to. */
  readonly fittedTo: readonly Float64Array[]
  /** Classification: the running decision over the training rows, `[n * k]`. */
  readonly decision: Float64Array
  /** After each member: the ensemble's training accuracy (classification) or R^2 (regression). */
  readonly scores: readonly number[]
  readonly stopped: boolean
}

/** Parameters of a fitted AdaBoost ensemble. */
export type AdaBoostParams = {
  readonly members: Json[]
  readonly weights: number[]
  readonly classes: number[]
  readonly nFeatures: number
  readonly algorithm: AdaBoostAlgorithm
}

/** Verbs and facts of a fitted AdaBoost ensemble. */
export interface AdaBoostVerbs extends Verbs, FittedFacts {
  readonly predict: (x: TensorLike) => Tensor
  readonly predictProba?: (x: TensorLike) => Tensor
  readonly predictDistribution?: (x: TensorLike) => Distribution
  readonly nFeaturesIn: number
}

/**
 * The paths of AdaBoost: the member's error and weight, the ensemble's training error and score, the row weights the
 * member was fitted to, the stop event, and the derived `boost/weights`, `boost/stump` and `model/margins`.
 *
 * @example
 * ```ts
 * import type { AdaBoostPath } from 'mdml-engine/ensembles'
 * const path: AdaBoostPath = 'boost/alpha'
 * path // => 'boost/alpha'
 * ```
 */
export type AdaBoostPath =
  | 'boost/error'
  | 'boost/alpha'
  | 'boost/trainingError'
  | 'boost/stumps'
  | 'ensemble/trainScore'
  | 'ensemble/target'
  | 'boost/stopped'
  | 'boost/weights'
  | 'boost/stump'
  | 'model/margins'

/** An AdaBoost estimator. */
export type AdaBoostEstimator = IterativeEstimator<
  AdaBoostHyper,
  AdaBoostState,
  AdaBoostParams,
  AdaBoostVerbs,
  AdaBoostPath
>

const EPS = Number.EPSILON

/** SAMME.R's per-class contribution of one member's class probabilities: (K - 1) (log p_c - mean log p). */
function realDecision(proba: Float64Array, k: number): Float64Array {
  const out = new Float64Array(proba.length)
  for (let i = 0; i < proba.length / k; i++) {
    let mean = 0
    for (let c = 0; c < k; c++) mean += Math.log(Math.max(proba[i * k + c] ?? 0, EPS)) / k
    for (let c = 0; c < k; c++) out[i * k + c] = (k - 1) * (Math.log(Math.max(proba[i * k + c] ?? 0, EPS)) - mean)
  }
  return out
}

/** A one-hot vote in SAMME's coding: 1 for the voted class, -1 / (K - 1) for the others. */
function codedVote(vote: Float64Array, k: number): Float64Array {
  return vote.map((v) => (v > 0 ? 1 : -1 / (k - 1)))
}

/** The decision of a SAMME or SAMME.R ensemble: the summed votes, over the summed weights for SAMME. */
function decisionOf(
  members: readonly (readonly [Float64Array, number])[],
  k: number,
  algorithm: AdaBoostAlgorithm,
): Float64Array {
  const len = members[0]?.[0].length ?? 0
  const sum = new Float64Array(len)
  let total = 0
  for (const [out, w] of members) {
    total += w
    const contribution = algorithm === 'SAMME.R' ? realDecision(out, k) : codedVote(out, k)
    for (let i = 0; i < len; i++) sum[i] = (sum[i] ?? 0) + (algorithm === 'SAMME.R' ? 1 : w) * (contribution[i] ?? 0)
  }
  if (algorithm === 'SAMME') for (let i = 0; i < len; i++) sum[i] = (sum[i] ?? 0) / (total || 1)
  else for (let i = 0; i < len; i++) sum[i] = (sum[i] ?? 0) / (members.length || 1)
  return sum
}

/** Softmax rows of the decision divided by K - 1 (binary SAMME: of [-d, d] / 2), as scikit-learn's predict_proba. */
function probaOf(decision: Float64Array, k: number): Float64Array {
  const m = decision.length / k
  const out = new Float64Array(m * k)
  for (let i = 0; i < m; i++) {
    const row = Array.from({ length: k }, (_, c) => (decision[i * k + c] ?? 0) / (k - 1))
    const z = k === 2 ? [((row[0] ?? 0) - (row[1] ?? 0)) / 2, ((row[1] ?? 0) - (row[0] ?? 0)) / 2] : row
    const max = maxOf(z)
    const e = z.map((v) => Math.exp(v - max))
    const s = e.reduce((a, b) => a + b, 0)
    for (let c = 0; c < k; c++) out[i * k + c] = (e[c] ?? 0) / s
  }
  return out
}

/** The weighted median of member predictions (AdaBoost.R2's combination). */
function weightedMedianPredict(preds: readonly Float64Array[], weights: readonly number[]): Float64Array {
  const m = preds[0]?.length ?? 0
  const total = weights.reduce((a, b) => a + b, 0)
  return Float64Array.from({ length: m }, (_, i) => {
    const order = preds.map((_, s) => s).sort((a, b) => (preds[a]?.[i] ?? 0) - (preds[b]?.[i] ?? 0))
    let acc = 0
    for (const s of order) {
      acc += weights[s] ?? 0
      if (acc >= 0.5 * total) return preds[s]?.[i] ?? 0
    }
    return preds[order[order.length - 1] ?? 0]?.[i] ?? 0
  })
}

function adaBoostEstimator(name: string, classifier: boolean, defaults: AdaBoostHyper): AdaBoostEstimator {
  const classesOf = (data: LearnData): readonly number[] => (classifier ? encodeLabels(data.y ?? [], name).classes : [])
  const algorithmOf = (data: LearnData, h: AdaBoostHyper): Algorithm<AdaBoostState, AdaBoostPath> => {
    const n = rowsOf(data.x)
    const classes = classesOf(data)
    const k = classes.length
    const y = classifier ? encodeLabels(data.y ?? [], name).codes : targetVector(data.y ?? [], name)
    const w0 = sampleWeightsOf(data.weights, n, name)
    const total0 = w0.reduce((a, b) => a + b, 0)
    if (classifier && k < 2) throw new MdmlError('value', `${name}: needs two classes`)
    /** The member's outputs on the training rows: labels as one-hot (SAMME), probabilities (SAMME.R), values. */
    const outputsOf = (params: Json): Float64Array => {
      const f = h.estimator.revive(params)
      if (!classifier) return Float64Array.from(toFlat((f.predict as NonNullable<Verbs['predict']>)(data.x)))
      if (h.algorithm === 'SAMME.R') {
        if (f.predictProba === undefined)
          throw new MdmlError('value', `${name}: SAMME.R needs a base with predictProba`)
        return memberProba(f, data.x, classes)
      }
      const pred = toFlat((f.predict as NonNullable<Verbs['predict']>)(data.x))
      const out = new Float64Array(n * k)
      pred.forEach((v, i) => {
        out[i * k + classes.indexOf(v)] = 1
      })
      return out
    }
    return {
      name,
      config: {
        nEstimators: h.nEstimators,
        learningRate: h.learningRate,
        algorithm: classifier ? h.algorithm : h.loss,
      },
      init: () => ({
        sampleWeight: Float64Array.from(w0, (w) => w / total0),
        members: [],
        weights: [],
        errors: [],
        fittedTo: [],
        decision: new Float64Array(n * Math.max(k, 1)),
        scores: [],
        stopped: false,
      }),
      step: (s, ctx) => {
        const sw = s.sampleWeight
        const record = {
          fittedTo: [...s.fittedTo, sw],
        }
        let fitted: Json
        let err: number
        let weight: number
        let next: Float64Array
        let contribution: Float64Array
        let stopNow = false
        if (classifier) {
          fitted = h.estimator.fit({ ...data, weights: sw }, { key: foldIn(ctx.key, 0), explain: false }).params
          const out = outputsOf(fitted)
          const samme = h.algorithm === 'SAMME'
          // The member's labels, for its weighted error.
          const labels = Array.from({ length: n }, (_, i) => argmaxOf(out.subarray(i * k, (i + 1) * k)))
          const wrong: number[] = labels.map((l, i) => (l === y[i] ? 0 : 1))
          err = wrong.reduce((a, v, i) => a + v * (sw[i] ?? 0), 0) / sw.reduce((a, b) => a + b, 0)
          contribution = out
          if (err <= 0) {
            weight = 1
            next = sw
            stopNow = true
          } else if (samme && err >= 1 - 1 / k) {
            if (s.members.length === 0)
              throw new MdmlError('value', `${name}: the first weak learner is no better than chance`)
            ctx.emit('boost/stopped', { round: s.members.length, reason: 'no better than chance', error: err })
            return { ...s, stopped: true }
          } else if (samme) {
            weight = h.learningRate * (Math.log((1 - err) / err) + Math.log(k - 1))
            next = Float64Array.from(sw, (v, i) => (wrong[i] && v > 0 ? v * Math.exp(weight) : v))
          } else {
            // SAMME.R: reweight by the member's log probabilities of the true class.
            weight = 1
            next = Float64Array.from(sw, (v, i) => {
              let sum = 0
              for (let c = 0; c < k; c++) {
                const code = c === y[i] ? 1 : -1 / (k - 1)
                sum += code * Math.log(Math.max(out[i * k + c] ?? 0, EPS))
              }
              const exponent = -1 * h.learningRate * ((k - 1) / k) * sum
              return v * Math.exp(exponent * (v > 0 || exponent < 0 ? 1 : 0))
            })
          }
        } else {
          const rows = Array.from(
            toFlat(drawRows(foldIn(ctx.key, 0), n, n, { replace: true, weights: Array.from(sw) })),
          )
          fitted = h.estimator.fit(takeData(data, rows), { key: foldIn(ctx.key, 1), explain: false }).params
          const pred = outputsOf(fitted)
          contribution = pred
          const abs = Array.from(pred, (p, i) => Math.abs(p - (y[i] ?? 0)))
          const max = maxOf(abs)
          const loss = abs.map((e) => {
            const r = max > 0 ? e / max : 0
            return h.loss === 'square' ? r * r : h.loss === 'exponential' ? 1 - Math.exp(-r) : r
          })
          err = loss.reduce((a, l, i) => a + l * (sw[i] ?? 0), 0)
          if (err <= 0) {
            weight = 1
            next = sw
            stopNow = true
          } else if (err >= 0.5) {
            if (s.members.length === 0) throw new MdmlError('value', `${name}: the first weak learner is too weak`)
            ctx.emit('boost/stopped', { round: s.members.length, reason: 'error of one half or more', error: err })
            return { ...s, stopped: true }
          } else {
            const beta = err / (1 - err)
            weight = h.learningRate * Math.log(1 / beta)
            next = Float64Array.from(sw, (v, i) => v * beta ** ((1 - (loss[i] ?? 0)) * h.learningRate))
          }
        }
        const total = next.reduce((a, b) => a + b, 0)
        const normalised = total > 0 ? next.map((v) => v / total) : next
        // The ensemble so far, on the training rows.
        let decision = s.decision
        let score: number
        const members = [...s.members, fitted]
        const weights = [...s.weights, weight]
        if (classifier) {
          decision = Float64Array.from(s.decision)
          const add = h.algorithm === 'SAMME.R' ? realDecision(contribution, k) : codedVote(contribution, k)
          for (let i = 0; i < decision.length; i++)
            decision[i] = (decision[i] ?? 0) + (h.algorithm === 'SAMME.R' ? 1 : weight) * (add[i] ?? 0)
          let right = 0
          for (let i = 0; i < n; i++) if (argmaxOf(decision.subarray(i * k, (i + 1) * k)) === y[i]) right++
          score = right / n
        } else {
          const preds = members.map((m) => outputsOf(m))
          const med = weightedMedianPredict(preds, weights)
          let sse = 0
          let sst = 0
          const mean = Array.from(y).reduce((a, b) => a + b, 0) / n
          for (let i = 0; i < n; i++) {
            sse += ((y[i] ?? 0) - (med[i] ?? 0)) ** 2
            sst += ((y[i] ?? 0) - mean) ** 2
          }
          score = sst > 0 ? 1 - sse / sst : 1
        }
        if (stopNow) ctx.emit('boost/stopped', { round: members.length, reason: 'a perfect member', error: err })
        if (ctx.wants('boost/error')) ctx.log('boost/error', err)
        if (ctx.wants('boost/alpha')) ctx.log('boost/alpha', weight)
        if (ctx.wants('boost/trainingError')) ctx.log('boost/trainingError', 1 - score)
        if (ctx.wants('boost/stumps')) ctx.log('boost/stumps', members.length)
        if (ctx.wants('ensemble/trainScore')) ctx.log('ensemble/trainScore', score)
        if (ctx.wants('ensemble/target')) ctx.log('ensemble/target', sw)
        return {
          sampleWeight: stopNow ? sw : Float64Array.from(normalised),
          members,
          weights,
          errors: [...s.errors, err],
          fittedTo: record.fittedTo,
          decision,
          scores: [...s.scores, score],
          stopped: stopNow,
        }
      },
      done: (s) => (s.stopped || s.members.length >= h.nEstimators ? 'completed' : undefined),
      serialise: (s) => stateToJson(s),
      deserialise: (j) => stateFromJson(j) as AdaBoostState,
    }
  }
  return iterative<AdaBoostHyper, AdaBoostState, AdaBoostParams, AdaBoostVerbs, AdaBoostPath>(
    {
      name,
      kind: classifier ? 'classifier' : 'regressor',
      defaults,
      hyperSpace: (h) =>
        dict({
          nEstimators: int(1, Math.max(300, h.nEstimators), { default: h.nEstimators, label: 'rounds' }),
          learningRate: real(0.01, Math.max(2, h.learningRate), {
            label: 'learning rate',
            description: 'How much each weak learner’s vote counts.',
            log: true,
            default: h.learningRate,
          }),
          ...(classifier
            ? { algorithm: choice(['SAMME', 'SAMME.R'] as const, { default: h.algorithm }) }
            : { loss: choice(['linear', 'square', 'exponential'] as const, { default: h.loss }) }),
        }),
      algorithm: (data, h) => algorithmOf(data, h),
      params: (state, data, h) => ({
        members: [...state.members],
        weights: [...state.weights],
        classes: [...classesOf(data)],
        nFeatures: matrixView(data.x, name, { allowNaN: true }).d,
        algorithm: h.algorithm,
      }),
      verbs: (p, h) => {
        const fitted = p.members.map((m) => h.estimator.revive(m))
        const k = p.classes.length
        if (!classifier) {
          return withDefaultScore('regressor', {
            predict: (x: TensorLike) => {
              const v = weightedMedianPredict(
                fitted.map((f) => Float64Array.from(toFlat((f.predict as NonNullable<Verbs['predict']>)(x)))),
                p.weights,
              )
              return fromStorage(v, [v.length], 'float64')
            },
            nFeaturesIn: p.nFeatures,
          })
        }
        const decision = (x: TensorLike): Float64Array => {
          const outs = fitted.map((f, i) => {
            if (p.algorithm === 'SAMME.R') return [memberProba(f, x, p.classes), 1] as const
            const pred = toFlat((f.predict as NonNullable<Verbs['predict']>)(x))
            const out = new Float64Array(pred.length * k)
            pred.forEach((v, r) => {
              out[r * k + p.classes.indexOf(v)] = 1
            })
            return [out, p.weights[i] ?? 0] as const
          })
          return decisionOf(outs, k, p.algorithm)
        }
        const proba = (x: TensorLike): Tensor => {
          const d = decision(x)
          return fromStorage(probaOf(d, k), [d.length / k, k], 'float64')
        }
        return withDefaultScore('classifier', {
          predict: (x: TensorLike) => {
            const d = decision(x)
            const m = d.length / k
            return fromStorage(
              Float64Array.from({ length: m }, (_, i) => p.classes[argmaxOf(d.subarray(i * k, (i + 1) * k))] ?? 0),
              [m],
              'float64',
            )
          },
          predictProba: proba,
          predictDistribution: (x: TensorLike) => categorical({ probs: proba(x) }),
          decisionFunction: (x: TensorLike) => {
            const d = decision(x)
            const m = d.length / k
            // Binary problems give one score per row: the margin of the second class over the first.
            return k === 2
              ? fromStorage(
                  Float64Array.from({ length: m }, (_, i) => (d[i * 2 + 1] ?? 0) - (d[i * 2] ?? 0)),
                  [m],
                  'float64',
                )
              : fromStorage(d, [m, k], 'float64')
          },
          classes: p.classes,
          nFeaturesIn: p.nFeatures,
        })
      },
      maxSteps: (h) => h.nEstimators,
      contract: (data, h): Contract<AdaBoostState, AdaBoostPath> =>
        defineContract({
          'boost/error': scalar({ description: 'the weighted error of the member just added', direction: 'lower' }),
          'boost/alpha': scalar({ description: 'the member’s weight in the vote' }),
          'boost/trainingError': scalar({
            description: 'the ensemble’s training error (1 - accuracy, or 1 - R²)',
            direction: 'lower',
          }),
          'boost/stumps': scalar({ description: 'members so far' }),
          'ensemble/trainScore': scalar({
            description: 'the ensemble’s training accuracy (or R²)',
            direction: 'higher',
          }),
          'ensemble/target': tensor({
            role: 'output',
            shape: ['n'],
            description: 'the row weights the member was fitted to',
          }),
          'boost/stopped': event({
            description: 'boosting ended early: a perfect member, or one no better than chance',
          }),
          'boost/weights': derived<AdaBoostState>({
            description: 'the row weights the next member will be fitted to (concentrating on the misclassified)',
            compute: (state) => Array.from(state.sampleWeight),
          }),
          'boost/stump': derived<AdaBoostState>({
            description: 'the parameters of one member (args: member)',
            compute: (state, args) => {
              const a = (args ?? {}) as { readonly member?: number }
              return state.members[Math.min(a.member ?? state.members.length - 1, state.members.length - 1)] ?? null
            },
          }),
          'model/margins': derived<AdaBoostState>({
            description:
              'the margin of each training row: the vote for its class minus the strongest vote against, in [-1, 1]',
            compute: (state) => {
              const k = classesOf(data).length
              if (k === 0 || state.members.length === 0) return []
              const y = encodeLabels(data.y ?? [], name).codes
              const scale =
                h.algorithm === 'SAMME'
                  ? (state.weights.reduce((a, b) => a + b, 0) * k) / (k - 1)
                  : 2 * Math.max(maxOf(state.decision.map(Math.abs)), EPS)
              return Array.from(y, (c, i) => {
                let against = Number.NEGATIVE_INFINITY
                for (let j = 0; j < k; j++) if (j !== c) against = Math.max(against, state.decision[i * k + j] ?? 0)
                return ((state.decision[i * k + c] ?? 0) - against) / (scale || 1)
              })
            },
          }),
        }),
      projections: {
        'ensemble/stages': (run) =>
          stagesFromRun(run, {
            method: name,
            scoreName: classifier ? 'training accuracy' : 'training R²',
            higherIsBetter: true,
            weight: 'boost/alpha',
            train: 'ensemble/trainScore',
            target: 'ensemble/target',
          }) as Exposition | undefined,
      },
      logs: {
        'boost/error': Log(),
        'boost/alpha': Log(),
        'boost/trainingError': Log(),
        'boost/stumps': Log(),
        'ensemble/trainScore': Log(),
        'ensemble/target': Log({ detail: 'full' }),
      },
    },
    defaults,
  )
}

function adaDefaults(hyper: Partial<AdaBoostHyper>, estimator: Estimator): AdaBoostHyper {
  return {
    estimator: hyper.estimator ?? estimator,
    nEstimators: hyper.nEstimators ?? 50,
    learningRate: hyper.learningRate ?? 1,
    algorithm: hyper.algorithm ?? 'SAMME',
    loss: hyper.loss ?? 'linear',
  }
}

/**
 * AdaBoost for classification (sklearn.ensemble.AdaBoostClassifier) over any base estimator that takes sample
 * weights (a stump by default), SAMME or SAMME.R, one boosting round per step. The session logs `boost/error`,
 * `boost/alpha`, `boost/trainingError` and `boost/stumps`, and answers the probes `boost/weights` (the row weights the
 * next member will see), `boost/stump` (a member's parameters) and `model/margins` (the margin distribution, which keeps
 * widening after the training error reaches zero); its fit records the stages with the row weights each member was
 * fitted to.
 *
 * @param hyper - Base estimator, rounds (default 50), learning rate (1) and algorithm (SAMME).
 * @returns The estimator.
 * @concept boosting
 * @index estimator AdaBoost (SAMME) over any base estimator.
 * @example
 * ```ts
 * import { adaBoostClassifier } from 'mdml-engine/ensembles'
 * import { toFlat } from 'mdml-engine/tensor'
 * const x = Array.from({ length: 40 }, (_, i) => [i / 40, ((i * 7) % 11) / 11])
 * const y = x.map(([a = 0, b = 0]) => (a + b > 0.9 ? 1 : 0))
 * const fitted = adaBoostClassifier({ nEstimators: 20 }).fit({ x, y })
 * toFlat(fitted.predict([[0.05, 0.05], [0.9, 0.9]])) // => [0, 1]
 * ```
 */
export function adaBoostClassifier(hyper: Partial<AdaBoostHyper> = {}): AdaBoostEstimator {
  return adaBoostEstimator(
    'ada-boost-classifier',
    true,
    adaDefaults(hyper, decisionTreeClassifier({ maxDepth: 1, record: 0 })),
  )
}

/**
 * AdaBoost.R2 for regression (sklearn.ensemble.AdaBoostRegressor): base regressors on keyed weighted resamples,
 * weighted by log(1 / beta), combined by the weighted median. Logging as {@link adaBoostClassifier}.
 *
 * @param hyper - Base estimator (default a depth-3 regression tree), rounds, learning rate and loss.
 * @returns The estimator.
 * @concept boosting
 * @example
 * ```ts
 * import { adaBoostRegressor } from 'mdml-engine/ensembles'
 * import { key } from 'mdml-engine/random'
 * const x = Array.from({ length: 30 }, (_, i) => [i / 30])
 * adaBoostRegressor({ nEstimators: 4 }).fit({ x, y: x.map(([a = 0]) => a * a) }, { key: key(0) }).params.classes // => []
 * ```
 */
export function adaBoostRegressor(hyper: Partial<AdaBoostHyper> = {}): AdaBoostEstimator {
  return adaBoostEstimator(
    'ada-boost-regressor',
    false,
    adaDefaults(hyper, decisionTreeRegressor({ maxDepth: 3, record: 0 })),
  )
}
