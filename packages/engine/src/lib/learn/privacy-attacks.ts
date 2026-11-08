/**
 * Privacy attacks on a trained model, the measuring side of differential privacy: membership inference (is this record
 * in the training set?) by a loss threshold (Yeom et al.), by shadow models whose behaviour calibrates the threshold
 * (Shokri et al.), and by the likelihood-ratio attack (LiRA, Carlini et al.); attribute inference (a hidden feature of a
 * record from the model's confidence in its label, Fredrikson et al.); model inversion by gradient ascent on a class's
 * confidence, as a stepped algorithm; gradient leakage (deep leakage from gradients, Zhu et al., and the closed-form
 * leak of a first linear layer); and model extraction by querying. Models are logits functions, as in `nn/adversarial`.
 */

import { grad } from '../autodiff/index.ts'
import { maxOf, OmaiError } from '../base/index.ts'
import { crossEntropy } from '../losses/index.ts'
import { rocAucScore, rocCurve } from '../metrics/index.ts'
import type { LogitsFunction, Params } from '../nn/index.ts'
import { adam, applyUpdates, type GradientTransformation } from '../optim/index.ts'
import { foldIn, type Key, normal, permutation } from '../random/index.ts'
import {
  add,
  argmax,
  asTensor,
  clip,
  fromStorage,
  item,
  logSoftmax,
  mul,
  primalOf,
  reshape,
  sub,
  sum,
  type Tensor,
  type TensorLike,
  toFlat,
  treeFlatten,
  type Value,
} from '../tensor/index.ts'
import {
  type Algorithm,
  type Contract,
  defineContract,
  scalar,
  stateFromJson,
  stateToJson,
  tensor as tensorSource,
} from '../trace/index.ts'

/** Per-example cross-entropy of a classifier on labelled rows. */
function exampleLosses(model: LogitsFunction, x: TensorLike, y: TensorLike): number[] {
  return toFlat(primalOf(crossEntropy(model(asTensor(x)), asTensor(y), { reduction: 'none' })))
}

/**
 * The per-example losses of a classifier, the quantity most membership attacks look at.
 *
 * @param model - Inputs to logits.
 * @param x - Inputs.
 * @param y - Labels.
 * @returns One cross-entropy per row.
 * @concept membership-inference
 * @example
 * ```ts
 * import { exampleLoss } from 'omai-engine/learn'
 * exampleLoss((x) => x, [[0, 0]], [1])[0]?.toFixed(6) // => '0.693147'
 * ```
 */
export function exampleLoss(model: LogitsFunction, x: TensorLike, y: TensorLike): number[] {
  return exampleLosses(model, x, y)
}

/** How well an attack separates members from non-members. */
export interface MembershipReport {
  /** Area under the ROC curve of the membership scores (0.5 is chance). */
  readonly auc: number
  /** The best accuracy over thresholds, with balanced classes (0.5 is chance). */
  readonly accuracy: number
  /** The best TPR - FPR (Yeom et al.'s membership advantage; 0 is chance). */
  readonly advantage: number
  /** The largest true-positive rate at a false-positive rate of at most `fprLevel`. */
  readonly tprAtFpr: number
  readonly fprLevel: number
  /** The score above which a record is called a member at the best accuracy. */
  readonly threshold: number
}

/**
 * Score membership attacks: given the scores (higher means more likely a member) of known members and non-members, the
 * AUC, best balanced accuracy, advantage and the true-positive rate at a low false-positive rate, which Carlini et al.
 * argue is the measure that matters (an attack right on a few records with confidence beats one barely above chance on
 * all).
 *
 * @param members - Scores of training records.
 * @param nonMembers - Scores of records outside the training set.
 * @param options - The false-positive rate level (default 0.01).
 * @returns The report.
 * @concept membership-inference
 * @example
 * ```ts
 * import { membershipReport } from 'omai-engine/learn'
 * membershipReport([3, 4, 5], [0, 1, 2]).auc // => 1
 * ```
 */
export function membershipReport(
  members: readonly number[],
  nonMembers: readonly number[],
  options: { readonly fprLevel?: number } = {},
): MembershipReport {
  const fprLevel = options.fprLevel ?? 0.01
  const y = [...members.map(() => 1), ...nonMembers.map(() => 0)]
  const s = [...members, ...nonMembers]
  const auc = rocAucScore(y, s).estimate.value
  const c = rocCurve(y, s, { dropIntermediate: false })
  let accuracy = 0
  let advantage = 0
  let threshold = Number.POSITIVE_INFINITY
  let tprAtFpr = 0
  const tprs = toFlat(c.tpr)
  const thresholds = toFlat(c.thresholds)
  toFlat(c.fpr).forEach((fpr, i) => {
    const tpr = tprs[i] ?? 0
    const acc = (tpr + 1 - fpr) / 2
    if (acc > accuracy) {
      accuracy = acc
      threshold = thresholds[i] ?? threshold
    }
    advantage = Math.max(advantage, tpr - fpr)
    if (fpr <= fprLevel) tprAtFpr = Math.max(tprAtFpr, tpr)
  })
  return { auc, accuracy, advantage, tprAtFpr, fprLevel, threshold }
}

/**
 * The loss-threshold attack (Yeom et al. 2018): call a record a member when the model's loss on it is low, since a model
 * fits what it was trained on better than what it was not. The score of a record is minus its loss; the report needs no
 * threshold to be chosen (AUC) or reports the best one.
 *
 * @param memberLosses - Losses on training records.
 * @param nonMemberLosses - Losses on held-out records.
 * @param options - The false-positive level.
 * @returns The attack's report.
 * @concept membership-inference
 * @example
 * ```ts
 * import { lossThresholdAttack } from 'omai-engine/learn'
 * lossThresholdAttack([0.01, 0.02, 0.05], [1, 2, 0.8]).auc // => 1
 * ```
 */
export function lossThresholdAttack(
  memberLosses: readonly number[],
  nonMemberLosses: readonly number[],
  options: { readonly fprLevel?: number } = {},
): MembershipReport {
  return membershipReport(
    memberLosses.map((l) => -l),
    nonMemberLosses.map((l) => -l),
    options,
  )
}

/** Trains a classifier on labelled rows and returns its logits function. */
export type ClassifierTrainer = (x: Tensor, y: Tensor, key: Key) => LogitsFunction

const rowsOf = (t: Tensor, picked: readonly number[]): Tensor => {
  const n = t.shape[0] ?? 0
  const d = n === 0 ? 0 : t.data.length / n
  const out = new Float64Array(picked.length * d)
  for (const [i, r] of picked.entries()) out.set(t.data.subarray(r * d, (r + 1) * d), i * d)
  return fromStorage(out, [picked.length, ...t.shape.slice(1)], 'float64')
}

/** The rows of the pool a shadow model trains on (a random half), by index. */
function halfOf(key: Key, n: number): { readonly inside: number[]; readonly outside: number[] } {
  const order = toFlat(permutation(key, n))
  const half = Math.floor(n / 2)
  return { inside: order.slice(0, half), outside: order.slice(half) }
}

/** Options of {@link shadowModelAttack}. */
export interface ShadowAttackOptions {
  /** Data from the target's distribution that the attacker holds (disjoint from the target's training set). */
  readonly pool: { readonly x: TensorLike; readonly y: TensorLike }
  /** How the target was trained, which the attacker replicates for its shadows. */
  readonly trainer: ClassifierTrainer
  readonly shadows: number
  readonly key: Key
  readonly fprLevel?: number
}

/** The result of {@link shadowModelAttack}. */
export interface ShadowAttackResult extends MembershipReport {
  /** The loss below which the attack calls a record a member, calibrated on the shadows. */
  readonly calibratedThreshold: number
  /** The attack's accuracy on the target at that threshold, with equal weight on members and non-members. */
  readonly calibratedAccuracy: number
  /** The shadows' own members and non-members' mean losses. */
  readonly shadowLosses: { readonly members: number; readonly nonMembers: number }
}

/**
 * The shadow-model attack (Shokri et al. 2017, with a threshold as the attack model): train `shadows` models on random
 * halves of data the attacker holds, using the same training recipe as the target, record their losses on their own
 * training halves (members) and on the other halves (non-members), and calibrate the loss threshold that best tells
 * them apart; then apply that threshold to the target's losses on its members and on non-members, which the attacker
 * does not use for anything but the final score.
 *
 * @param target - The target classifier.
 * @param members - Rows the target trained on.
 * @param nonMembers - Rows it did not.
 * @param options - The attacker's pool, the trainer to replicate, the number of shadows and a key.
 * @returns The target's report (by loss, so the AUC is threshold-free), the calibrated threshold and accuracy.
 * @concept membership-inference
 * @example
 * ```ts
 * import { shadowModelAttack } from 'omai-engine/learn'
 * import { key } from 'omai-engine/random'
 * import { tensor } from 'omai-engine/tensor'
 * const trainer = () => (x) => x
 * const pool = { x: tensor([[2, 0], [0, 2], [3, 0], [0, 3]]), y: tensor([0, 1, 0, 1]) }
 * shadowModelAttack((x) => x, pool, pool, { pool, trainer, shadows: 2, key: key(0) }).auc >= 0 // => true
 * ```
 */
export function shadowModelAttack(
  target: LogitsFunction,
  members: { readonly x: TensorLike; readonly y: TensorLike },
  nonMembers: { readonly x: TensorLike; readonly y: TensorLike },
  options: ShadowAttackOptions,
): ShadowAttackResult {
  const px = asTensor(options.pool.x) as Tensor
  const py = asTensor(options.pool.y) as Tensor
  const n = px.shape[0] ?? 0
  const inLosses: number[] = []
  const outLosses: number[] = []
  for (let s = 0; s < options.shadows; s++) {
    const split = halfOf(foldIn(options.key, 2 * s), n)
    const model = options.trainer(rowsOf(px, split.inside), rowsOf(py, split.inside), foldIn(options.key, 2 * s + 1))
    inLosses.push(...exampleLosses(model, rowsOf(px, split.inside), rowsOf(py, split.inside)))
    outLosses.push(...exampleLosses(model, rowsOf(px, split.outside), rowsOf(py, split.outside)))
  }
  const calibration = membershipReport(
    inLosses.map((l) => -l),
    outLosses.map((l) => -l),
  )
  const calibratedThreshold = -calibration.threshold
  const memberLosses = exampleLosses(target, members.x, members.y)
  const nonMemberLosses = exampleLosses(target, nonMembers.x, nonMembers.y)
  const tpr = memberLosses.filter((l) => l <= calibratedThreshold).length / Math.max(1, memberLosses.length)
  const fpr = nonMemberLosses.filter((l) => l <= calibratedThreshold).length / Math.max(1, nonMemberLosses.length)
  const mean = (v: readonly number[]): number => v.reduce((a, b) => a + b, 0) / Math.max(1, v.length)
  return {
    ...lossThresholdAttack(
      memberLosses,
      nonMemberLosses,
      options.fprLevel === undefined ? {} : { fprLevel: options.fprLevel },
    ),
    calibratedThreshold,
    calibratedAccuracy: (tpr + 1 - fpr) / 2,
    shadowLosses: { members: mean(inLosses), nonMembers: mean(outLosses) },
  }
}

/** The log-odds of the label under a model, log p_y - log (1 - p_y), the statistic LiRA models (stable from logits). */
function labelLogit(model: LogitsFunction, x: Tensor, y: Tensor): number[] {
  const logp = logSoftmax(model(x), -1)
  const lp = toFlat(primalOf(logp))
  const c = (logp.shape[logp.shape.length - 1] ?? 1) as number
  return toFlat(y).map((label, i) => {
    const own = lp[i * c + label] as number
    // log (1 - p_y) = log sum_{j != y} p_j.
    let rest = Number.NEGATIVE_INFINITY
    for (let j = 0; j < c; j++) {
      if (j === label) continue
      const v = lp[i * c + j] as number
      rest = Math.max(rest, v) + Math.log1p(Math.exp(-Math.abs(rest - v)))
    }
    return own - rest
  })
}

/**
 * The confidence statistic of LiRA: the logit of the model's probability of the true label, log p_y - log (1 - p_y),
 * which is close to normally distributed over models trained with and without a record.
 *
 * @param model - Inputs to logits.
 * @param x - Inputs.
 * @param y - Labels.
 * @returns One value per row.
 * @concept membership-inference
 * @example
 * ```ts
 * import { confidenceLogit } from 'omai-engine/learn'
 * confidenceLogit((x) => x, [[0, Math.log(3)]], [1])[0]?.toFixed(6) // => '1.098612'
 * ```
 */
export function confidenceLogit(model: LogitsFunction, x: TensorLike, y: TensorLike): number[] {
  return labelLogit(model, asTensor(x) as Tensor, asTensor(y) as Tensor)
}

/** Options of {@link likelihoodRatioAttack}. */
export interface LiraOptions {
  /** Labelled data from the target's distribution, plus the queried records, from which shadows train on random halves. */
  readonly pool: { readonly x: TensorLike; readonly y: TensorLike }
  readonly trainer: ClassifierTrainer
  /** Shadow models (the more, the better the per-record Gaussians; default 16). */
  readonly shadows?: number
  readonly key: Key
  /** One variance for all records instead of one per record and side (better with few shadows; default false). */
  readonly globalVariance?: boolean
  readonly fprLevel?: number
}

/** The result of {@link likelihoodRatioAttack}. */
export interface LiraResult extends MembershipReport {
  /** The log likelihood ratio per queried record (larger means member). */
  readonly scores: readonly number[]
}

const logNormal = (x: number, m: number, v: number): number => -0.5 * (Math.log(2 * Math.PI * v) + (x - m) ** 2 / v)

/**
 * The likelihood-ratio attack, online version (LiRA, Carlini et al. 2022): train shadow models on random halves of a pool
 * that contains the queried records, so that every record is in some shadows and out of others; for each record fit a
 * Gaussian to the confidence logit among the shadows that saw it and another among those that did not; then score the
 * target's confidence logit on the record by the log likelihood ratio of the two. Per-record calibration makes the
 * attack far stronger than a global loss threshold at low false-positive rates, where easy and hard records differ.
 *
 * @param target - The target classifier.
 * @param queries - The records to score (members and non-members), which must also be rows of the pool, in order.
 * @param isMember - Which queries the target trained on (for the report).
 * @param options - The pool (the queries come first), the trainer, shadow count and a key.
 * @returns The scores and the report.
 * @concept membership-inference
 * @example
 * ```ts
 * import { likelihoodRatioAttack } from 'omai-engine/learn'
 * import { key } from 'omai-engine/random'
 * import { matmul, tensor } from 'omai-engine/tensor'
 * const pool = { x: tensor([[0], [1], [2], [3]]), y: tensor([0, 1, 0, 1]) }
 * const logits = (x) => matmul(x, tensor([[-1, 1]]))
 * likelihoodRatioAttack(logits, 4, [true, true, false, false], { pool, trainer: () => logits, shadows: 4, key: key(0) }).scores.length // => 4
 * ```
 */
export function likelihoodRatioAttack(
  target: LogitsFunction,
  queries: number,
  isMember: readonly boolean[],
  options: LiraOptions,
): LiraResult {
  const px = asTensor(options.pool.x) as Tensor
  const py = asTensor(options.pool.y) as Tensor
  const n = px.shape[0] ?? 0
  if (queries > n) throw new OmaiError('value', 'likelihoodRatioAttack: the queries must be the first rows of the pool')
  const shadows = options.shadows ?? 16
  const qx = rowsOf(
    px,
    Array.from({ length: queries }, (_, i) => i),
  )
  const qy = rowsOf(
    py,
    Array.from({ length: queries }, (_, i) => i),
  )
  const inside: number[][] = Array.from({ length: queries }, () => [])
  const outside: number[][] = Array.from({ length: queries }, () => [])
  for (let s = 0; s < shadows; s++) {
    const split = halfOf(foldIn(options.key, 2 * s), n)
    const model = options.trainer(rowsOf(px, split.inside), rowsOf(py, split.inside), foldIn(options.key, 2 * s + 1))
    const phi = labelLogit(model, qx, qy)
    const members = new Set(split.inside)
    phi.forEach((v, i) => {
      ;(members.has(i) ? inside : outside)[i]?.push(v)
    })
  }
  const stats = (v: readonly number[]): { m: number; var: number } => {
    const m = v.reduce((a, b) => a + b, 0) / Math.max(1, v.length)
    return { m, var: v.length < 2 ? Number.NaN : v.reduce((a, b) => a + (b - m) ** 2, 0) / (v.length - 1) }
  }
  // No query with two shadows on either side gives no pooled variance (NaN): the floor then, as Math.max would give NaN.
  const pooledVar = stats([
    ...inside.flatMap((v) => v.map((q) => q - stats(v).m)),
    ...outside.flatMap((v) => v.map((q) => q - stats(v).m)),
  ]).var
  const pooled = Math.max(1e-6, Number.isNaN(pooledVar) ? 0 : pooledVar)
  const observed = labelLogit(target, qx, qy)
  const scores = observed.map((phi, i) => {
    const a = stats(inside[i] ?? [])
    const b = stats(outside[i] ?? [])
    const va = options.globalVariance === true || Number.isNaN(a.var) ? pooled : Math.max(a.var, 1e-6)
    const vb = options.globalVariance === true || Number.isNaN(b.var) ? pooled : Math.max(b.var, 1e-6)
    return logNormal(phi, a.m, va) - logNormal(phi, b.m, vb)
  })
  const mem = scores.filter((_, i) => isMember[i] === true)
  const non = scores.filter((_, i) => isMember[i] !== true)
  return { ...membershipReport(mem, non, options.fprLevel === undefined ? {} : { fprLevel: options.fprLevel }), scores }
}

/** The result of {@link attributeInference}. */
export interface AttributeInferenceResult {
  /** The inferred value of the hidden attribute per record. */
  readonly predicted: readonly number[]
  /** Accuracy against the true values, if given. */
  readonly accuracy: number
  /** The accuracy of always guessing the most common value (the attacker's prior). */
  readonly baseline: number
}

/**
 * Attribute inference (Fredrikson et al. 2014, as a Bayes attack): for a record whose other features and label the
 * attacker knows, try each candidate value of the hidden feature and keep the one under which the model is most
 * confident in the known label, weighted by a prior over the values. A model that leans on the attribute leaks it;
 * the gain over the prior is the leak.
 *
 * @param model - Inputs (with the hidden feature in column `column`) to logits.
 * @param x - Records `[N, d]` with any value in the hidden column (it is overwritten).
 * @param y - Their labels.
 * @param column - The hidden feature's column.
 * @param candidates - The values it can take.
 * @param options - A prior over the candidates (default uniform) and the true values for scoring.
 * @returns The inferred values, their accuracy and the baseline.
 * @concept attribute-inference
 * @example
 * ```ts
 * import { attributeInference } from 'omai-engine/learn'
 * // The logit of class 0 is the hidden column itself, so label 1 means the column is low and label 0 that it is high.
 * attributeInference((x) => x, [[0, 0], [0, 0]], [1, 0], 0, [0, 1], { truth: [0, 1] }).predicted // => [0, 1]
 * ```
 */
export function attributeInference(
  model: LogitsFunction,
  x: TensorLike,
  y: TensorLike,
  column: number,
  candidates: readonly number[],
  options: { readonly prior?: readonly number[]; readonly truth?: readonly number[] } = {},
): AttributeInferenceResult {
  const xs = asTensor(x) as Tensor
  const labels = toFlat(asTensor(y))
  const n = xs.shape[0] ?? 0
  const d = xs.shape[1] ?? 0
  const prior = options.prior ?? candidates.map(() => 1 / candidates.length)
  const logLik = candidates.map((v) => {
    const data = Float64Array.from(xs.data)
    for (let i = 0; i < n; i++) data[i * d + column] = v
    const lp = toFlat(primalOf(logSoftmax(model(fromStorage(data, [n, d], 'float64')), -1)))
    const c = lp.length / n
    return labels.map((label, i) => lp[i * c + label] as number)
  })
  const predicted = Array.from({ length: n }, (_, i) => {
    let best = 0
    let bestScore = Number.NEGATIVE_INFINITY
    candidates.forEach((_, k) => {
      const score = (logLik[k]?.[i] as number) + Math.log(prior[k] as number)
      if (score > bestScore) {
        best = k
        bestScore = score
      }
    })
    return candidates[best] as number
  })
  const truth = options.truth
  const accuracy = truth === undefined ? Number.NaN : predicted.filter((p, i) => p === truth[i]).length / Math.max(1, n)
  const counts = new Map<number, number>()
  for (const t of truth ?? []) counts.set(t, (counts.get(t) ?? 0) + 1)
  const baseline = truth === undefined ? Number.NaN : Math.max(0, maxOf([...counts.values()])) / Math.max(1, n)
  return { predicted, accuracy, baseline }
}

/** The paths of {@link modelInversionAlgorithm}. */
export type InversionPath = 'inversion/step' | 'inversion/input' | 'inversion/confidence'

/** What a model-inversion run logs. */
export const inversionContract: Contract<InversionState, InversionPath> = defineContract({
  'inversion/step': scalar({ description: 'the gradient step' }),
  'inversion/input': tensorSource({ role: 'state', description: 'the reconstructed input' }),
  'inversion/confidence': scalar({ description: 'the model’s probability of the class on the reconstructed input' }),
})

/** The state of a model-inversion run. */
export interface InversionState {
  readonly x: Tensor
  readonly step: number
  readonly confidence: number
}

/**
 * Model inversion (Fredrikson, Jha and Ristenpart 2015): reconstruct a representative input of a class by gradient
 * ascent on the model's log probability of that class, from a blank start, optionally with an l2 penalty and a clip to
 * the data range. One gradient step per step; the input and the model's confidence in it are logged.
 *
 * @param model - Inputs to logits.
 * @param targetClass - The class to reconstruct.
 * @param inputShape - The shape of one input (without the batch axis).
 * @param options - Steps, learning rate, l2 penalty, range.
 * @returns The algorithm.
 * @concept model-inversion
 * @example
 * ```ts
 * import { modelInversionAlgorithm } from 'omai-engine/learn'
 * import { runAlgorithm } from 'omai-engine/trace'
 * import { matmul } from 'omai-engine/tensor'
 * const model = (x) => matmul(x, [[1, -1], [1, -1]])
 * runAlgorithm(modelInversionAlgorithm(model, 0, [2], { steps: 30 }), { maxSteps: 30 }).state.confidence > 0.9 // => true
 * ```
 */
export function modelInversionAlgorithm(
  model: LogitsFunction,
  targetClass: number,
  inputShape: readonly number[],
  options: {
    readonly steps: number
    readonly learningRate?: number
    readonly l2?: number
    readonly range?: readonly [number, number]
  },
): Algorithm<InversionState, InversionPath> {
  const lr = options.learningRate ?? 0.1
  const confidenceOf = (x: Value): number => {
    const lp = toFlat(primalOf(logSoftmax(model(reshape(x, [1, ...inputShape])), -1)))
    return Math.exp(lp[targetClass] as number)
  }
  return {
    name: 'modelInversion',
    config: { targetClass, steps: options.steps, learningRate: lr, l2: options.l2 ?? 0 },
    init: () => {
      const x = fromStorage(new Float64Array(inputShape.reduce((a, b) => a * b, 1)), [...inputShape], 'float64')
      return { x, step: 0, confidence: confidenceOf(x) }
    },
    step: (s, ctx) => {
      const objective = (v: Value): Value => {
        const lp = logSoftmax(model(reshape(v, [1, ...inputShape])), -1)
        const own = sum(mul(lp, oneAt(lp.shape[lp.shape.length - 1] ?? 0, targetClass)))
        return sub(own, mul(sum(mul(v, v)), (options.l2 ?? 0) / 2))
      }
      let x: Value = add(s.x, mul(grad(objective)(s.x), lr))
      if (options.range !== undefined) x = clip(x, options.range[0], options.range[1])
      const next = { x: asTensor(x) as Tensor, step: s.step + 1, confidence: confidenceOf(x) }
      ctx.log('inversion/step', next.step)
      ctx.log('inversion/input', next.x)
      ctx.log('inversion/confidence', next.confidence)
      return next
    },
    done: (s) => (s.step >= options.steps ? 'completed' : undefined),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as InversionState,
    observe: (s) => ({ metrics: { confidence: s.confidence } }),
  }
}

const oneAt = (n: number, i: number): Value =>
  fromStorage(
    Float64Array.from({ length: n }, (_, k) => (k === i ? 1 : 0)),
    [n],
    'float64',
  )

/**
 * The closed-form leak of a first linear layer (Phong et al. 2018): for one example, the gradient of the weight is the
 * outer product of the output-side error and the input, and the gradient of the bias is the error, so the input is the
 * weight's gradient row over the bias's gradient. Exact, for any network whose first layer is linear with a bias, from
 * the gradient of a single example.
 *
 * @param weightGrad - Gradient of the first layer's weight `[out, in]`.
 * @param biasGrad - Gradient of its bias `[out]`.
 * @returns The input.
 * @concept gradient-leakage
 * @example
 * ```ts
 * import { analyticGradientLeak } from 'omai-engine/learn'
 * analyticGradientLeak([[2, 4, 6], [0, 0, 0]], [2, 0]).data // => Float64Array.of(1, 2, 3)
 * ```
 */
export function analyticGradientLeak(weightGrad: TensorLike, biasGrad: TensorLike): Tensor {
  const w = asTensor(weightGrad) as Tensor
  const b = toFlat(asTensor(biasGrad))
  const cols = w.shape[1] ?? 0
  let best = 0
  b.forEach((v, i) => {
    if (Math.abs(v) > Math.abs(b[best] as number)) best = i
  })
  if (Math.abs(b[best] ?? 0) < 1e-300) throw new OmaiError('value', 'analyticGradientLeak: every bias gradient is zero')
  return fromStorage(
    Float64Array.from(w.data.subarray(best * cols, (best + 1) * cols), (v) => v / (b[best] as number)),
    [cols],
    'float64',
  )
}

/** The paths of {@link gradientLeakageAlgorithm}. */
export type LeakagePath = 'leak/step' | 'leak/match' | 'leak/input'

/** What a gradient-leakage run logs. */
export const leakageContract: Contract<LeakageState, LeakagePath> = defineContract({
  'leak/step': scalar({ description: 'the optimisation step' }),
  'leak/match': scalar({ description: 'the squared distance between the dummy input’s gradient and the leaked one' }),
  'leak/input': tensorSource({ role: 'state', description: 'the reconstructed input' }),
})

/** The state of a gradient-leakage run. */
export interface LeakageState {
  readonly x: Tensor
  readonly optState: unknown
  readonly step: number
  readonly match: number
}

/**
 * Deep leakage from gradients (Zhu, Liu and Han 2019, with the label known as in iDLG): given the parameters of a
 * network and the gradient of the loss on one example, find the input whose gradient matches, by Adam on the squared
 * distance between the dummy input's parameter gradient and the observed one. The network must be smooth for the
 * gradient to carry second-order information (tanh and sigmoid, not ReLU). One optimiser step per step.
 *
 * @param model - Inputs to logits (as a function of the parameters).
 * @param params - The parameters the gradient was taken at.
 * @param leaked - The observed gradient (same tree as the parameters).
 * @param label - The example's label.
 * @param inputShape - The shape of one input.
 * @param options - Steps, learning rate and the optimiser.
 * @returns The algorithm.
 * @concept gradient-leakage
 * @example
 * ```ts
 * import { gradientLeakageAlgorithm } from 'omai-engine/learn'
 * import { Linear } from 'omai-engine/nn'
 * const m = new Linear(2, 2)
 * gradientLeakageAlgorithm((p, x) => m.apply(p, x), m.params, m.params, 0, [2], { steps: 3 }).name // => 'gradientLeakage'
 * ```
 */
export function gradientLeakageAlgorithm(
  model: (params: Params, x: Value) => Value,
  params: Params,
  leaked: Params,
  label: number,
  inputShape: readonly number[],
  options: { readonly steps: number; readonly learningRate?: number; readonly optimizer?: GradientTransformation },
): Algorithm<LeakageState, LeakagePath> {
  const optimizer = options.optimizer ?? adam({ learningRate: options.learningRate ?? 0.1 })
  const target = treeFlatten(leaked).leaves
  const y = fromStorage(Float64Array.of(label), [1], 'float64')
  const match = (x: Value): Value => {
    const g = treeFlatten(grad((p: Params) => crossEntropy(model(p, reshape(x, [1, ...inputShape])), y))(params)).leaves
    return g.reduce<Value>(
      (acc, leaf, i) => add(acc, sum(mul(sub(leaf, target[i] as Value), sub(leaf, target[i] as Value)))),
      fromStorage(Float64Array.of(0), [], 'float64'),
    )
  }
  return {
    name: 'gradientLeakage',
    config: { label, steps: options.steps, optimizer: optimizer.name },
    init: (key) => {
      const x = normal(key, [...inputShape])
      return { x, optState: optimizer.init(x), step: 0, match: Number.NaN }
    },
    step: (s, ctx) => {
      const g = grad(match)(s.x)
      const r = optimizer.update(g, s.optState, s.x)
      const x = asTensor(applyUpdates(s.x, r.updates) as Value) as Tensor
      const m = item(primalOf(match(x)))
      ctx.log('leak/step', s.step + 1)
      ctx.log('leak/match', m)
      ctx.log('leak/input', x)
      return { x, optState: r.state, step: s.step + 1, match: m }
    },
    done: (s) => (s.step >= options.steps ? 'completed' : undefined),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as LeakageState,
    observe: (s) => ({ metrics: Number.isNaN(s.match) ? {} : { match: s.match } }),
  }
}

/**
 * A model-extraction attack (Tramer et al. 2016): query the target on inputs the attacker chooses, fit a surrogate to
 * its answers, and measure how often the surrogate and the target agree on fresh inputs.
 *
 * @param victim - The target (queried for its predicted class only).
 * @param queries - The attacker's inputs.
 * @param fit - Fits a surrogate to inputs and the victim's labels.
 * @param test - Fresh inputs to measure agreement on.
 * @returns The surrogate and the share of test inputs on which it agrees with the victim.
 * @concept model-extraction
 * @example
 * ```ts
 * import { extractionAttack } from 'omai-engine/learn'
 * const victim = (x) => x
 * extractionAttack(victim, [[1, 0]], () => victim, [[0, 1]]).agreement // => 1
 * ```
 */
export function extractionAttack(
  victim: LogitsFunction,
  queries: TensorLike,
  fit: (x: Tensor, labels: Tensor) => LogitsFunction,
  test: TensorLike,
): { readonly surrogate: LogitsFunction; readonly agreement: number } {
  const q = asTensor(queries) as Tensor
  const t = asTensor(test) as Tensor
  const surrogate = fit(q, argmax(victim(q), -1) as Tensor)
  const a = toFlat(argmax(victim(t), -1))
  const b = toFlat(argmax(surrogate(t), -1))
  return { surrogate, agreement: a.filter((v, i) => v === b[i]).length / Math.max(1, a.length) }
}
