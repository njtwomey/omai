/**
 * The objectives as option sets: the losses of regression and of binary classification, each the library's loss from
 * `losses` with its knobs bound and its properties (the task, convexity, smoothness, whether it is a likelihood), and
 * the penalties (none, ridge, lasso, elastic net) with their strength, their value on parameters, their proximal
 * operator where they have one, and their properties (smooth, has a prox, sparse). A method's objective slot is a
 * loss and a penalty from these tables; its validity rules read their properties (a proximal solver needs a prox,
 * IRLS a likelihood).
 */

import { proxElasticNet, proxL1, proxSquaredL2 } from '../optim/proximal.ts'
import { choice, dict, type Option, type OptionTable, optionTable, real } from '../space/index.ts'
import { add, asTensor, broadcastTo, concat, div, eye, mul, type TensorLike, type Value } from '../tensor/index.ts'
import type { Reduction } from './_reduce.ts'
import {
  backwardCorrected,
  binaryCrossEntropyWithLogits,
  biTemperedLogistic,
  bootstrapped,
  categoricalFocal,
  crossEntropy,
  focal,
  forwardCorrected,
  generalisedCrossEntropy,
  probabilityMae,
  symmetricCrossEntropy,
  truncatedCrossEntropy,
} from './classification.ts'
import { exponential, hinge, logistic, modifiedHuber, perceptron, squaredHinge } from './margin.ts'
import { bradleyTerry, dpo, ipo, kto, orpo, type PreferenceBatch, simpo } from './preference.ts'
import { expectile, huber, logCosh, mae, mse, pinball, smoothL1 } from './regression.ts'
import { elasticNet as elasticNetPenalty, l1Penalty, l2Penalty, type Penalised } from './regularisers.ts'

/** A loss with its knobs bound: the prediction (or score), the target (or label), the reduction and weights, as `mse`. */
export type Loss = typeof mse

/** What a loss is, for validity rules and selectors. */
export interface LossProperties {
  readonly task: 'regression' | 'classification'
  readonly convex: boolean
  readonly smooth: boolean
  /** The negative log-likelihood of a family (squared: Gaussian; logistic: Bernoulli), so IRLS applies. */
  readonly likelihood: boolean
}

const regression = (
  properties: Omit<LossProperties, 'task'>,
  description: string,
  space: Option<Loss, LossProperties>['space'],
  make: Option<Loss, LossProperties>['make'],
): Option<Loss, LossProperties> => ({ description, space, properties: { task: 'regression', ...properties }, make })

const classification = (
  properties: Omit<LossProperties, 'task'>,
  description: string,
  space: Option<Loss, LossProperties>['space'],
  make: Option<Loss, LossProperties>['make'],
): Option<Loss, LossProperties> => ({ description, space, properties: { task: 'classification', ...properties }, make })

/** The regression losses as one slot: squared (default), absolute, Huber, log-cosh, pinball, expectile, smooth L1. */
export const regressionLosses: OptionTable<Loss, LossProperties> = optionTable(
  {
    squared: regression({ convex: true, smooth: true, likelihood: true }, 'the squared error', dict({}), () => mse),
    absolute: regression({ convex: true, smooth: false, likelihood: true }, 'the absolute error', dict({}), () => mae),
    huber: regression(
      { convex: true, smooth: true, likelihood: false },
      'squared near zero, absolute beyond delta',
      dict({ delta: real(0.01, 10, { log: true, default: 1, label: 'delta' }) }),
      (k) => (p, t, o) => huber(p, t, { ...o, delta: k['delta'] as number }),
    ),
    logCosh: regression(
      { convex: true, smooth: true, likelihood: false },
      'log cosh of the residual',
      dict({}),
      () => logCosh,
    ),
    pinball: regression(
      { convex: true, smooth: false, likelihood: false },
      'the quantile loss at tau',
      dict({ tau: real(0.01, 0.99, { default: 0.5, label: 'quantile τ' }) }),
      (k) => (p, t, o) => pinball(p, t, { ...o, tau: k['tau'] as number }),
    ),
    expectile: regression(
      { convex: true, smooth: true, likelihood: false },
      'the asymmetric squared error at tau',
      dict({ tau: real(0.01, 0.99, { default: 0.5, label: 'expectile τ' }) }),
      (k) => (p, t, o) => expectile(p, t, { ...o, tau: k['tau'] as number }),
    ),
    smoothL1: regression(
      { convex: true, smooth: true, likelihood: false },
      'the Huber loss scaled by beta',
      dict({ beta: real(0.01, 10, { log: true, default: 1, label: 'beta' }) }),
      (k) => (p, t, o) => smoothL1(p, t, { ...o, beta: k['beta'] as number }),
    ),
  },
  { label: 'loss', description: 'the regression loss' },
)

/** The binary classification losses as one slot: logistic (default), hinge, squared hinge, exponential, modified Huber, perceptron. */
export const classificationLosses: OptionTable<Loss, LossProperties> = optionTable(
  {
    logistic: classification(
      { convex: true, smooth: true, likelihood: true },
      'the logistic loss',
      dict({}),
      () => logistic,
    ),
    hinge: classification(
      { convex: true, smooth: false, likelihood: false },
      'the hinge loss at a margin',
      dict({ margin: real(0.1, 5, { default: 1, label: 'margin' }) }),
      (k) => (s, y, o) => hinge(s, y, { ...o, margin: k['margin'] as number }),
    ),
    squaredHinge: classification(
      { convex: true, smooth: true, likelihood: false },
      'the squared hinge loss',
      dict({}),
      () => squaredHinge,
    ),
    exponential: classification(
      { convex: true, smooth: true, likelihood: false },
      'the exponential loss',
      dict({}),
      () => exponential,
    ),
    modifiedHuber: classification(
      { convex: true, smooth: true, likelihood: false },
      'the modified Huber loss',
      dict({}),
      () => modifiedHuber,
    ),
    perceptron: classification(
      { convex: true, smooth: false, likelihood: false },
      'the perceptron loss',
      dict({}),
      () => perceptron,
    ),
    focal: classification(
      { convex: false, smooth: true, likelihood: false },
      'the logistic loss scaled by (1 - p_t)^gamma, which down-weights the examples already well classified',
      dict({ gamma: real(0, 5, { default: 2, label: 'focusing γ' }) }),
      (k) => (s, y, o) => focal(s, div(add(y, 1), 2), { ...o, alpha: -1, gamma: k['gamma'] as number }),
    ),
    smoothedLogistic: classification(
      { convex: true, smooth: true, likelihood: false },
      'the logistic loss with the labels mixed with the uniform distribution (label smoothing), which caps the confidence',
      dict({ epsilon: real(0, 0.5, { default: 0.1, label: 'smoothing ε' }) }),
      (k) => (s, y, o) =>
        binaryCrossEntropyWithLogits(s, div(add(y, 1), 2), { ...o, labelSmoothing: k['epsilon'] as number }),
    ),
  },
  { label: 'loss', description: 'the classification loss' },
)

/** A penalty with its knobs bound: its strength, its value on parameters, and its proximal operator where it has one. */
export interface Penalty {
  readonly strength: number
  /** The penalty on parameters (already scaled by the strength). */
  value(params: Penalised): Value
  /** The proximal operator of t times the penalty, when it has a closed form. */
  prox?(x: TensorLike, t: number): Value
}

/** What a penalty is, for validity rules and selectors. */
export interface PenaltyProperties {
  readonly smooth: boolean
  readonly hasProx: boolean
  /** Whether it drives coefficients to exactly zero. */
  readonly sparse: boolean
}

const alpha = real(1e-6, 10, { log: true, default: 1e-2, label: 'strength α' })

/** The penalties as one slot: none (default), ridge (l2), lasso (l1) and elastic net. */
export const penalties: OptionTable<Penalty, PenaltyProperties> = optionTable(
  {
    none: {
      description: 'no penalty',
      space: dict({}),
      properties: { smooth: true, hasProx: true, sparse: false },
      make: () => ({ strength: 0, value: () => asTensor(0), prox: (x) => asTensor(x) }),
    },
    l2: {
      description: 'ridge: half the squared norm times the strength',
      space: dict({ alpha }),
      properties: { smooth: true, hasProx: true, sparse: false },
      make: (k) => {
        const strength = k['alpha'] as number
        return {
          strength,
          value: (p) => l2Penalty(p, { strength }),
          prox: (x, t) => proxSquaredL2(x, t * strength),
        }
      },
    },
    l1: {
      description: 'lasso: the absolute norm times the strength',
      space: dict({ alpha }),
      properties: { smooth: false, hasProx: true, sparse: true },
      make: (k) => {
        const strength = k['alpha'] as number
        return { strength, value: (p) => l1Penalty(p, { strength }), prox: (x, t) => proxL1(x, t * strength) }
      },
    },
    elasticNet: {
      description: 'elastic net: a share of lasso and the rest ridge',
      space: dict({ alpha, l1Ratio: real(0, 1, { default: 0.5, label: 'l1 share' }) }),
      properties: { smooth: false, hasProx: true, sparse: true },
      make: (k) => {
        const strength = k['alpha'] as number
        const l1Ratio = k['l1Ratio'] as number
        return {
          strength,
          value: (p) => elasticNetPenalty(p, { alpha: strength, l1Ratio }),
          prox: (x, t) => proxElasticNet(x, t, { l1: strength * l1Ratio, l2: strength * (1 - l1Ratio) }),
        }
      },
    },
  },
  { label: 'penalty', description: 'the penalty on the parameters' },
)

/** A multi-class loss with its knobs bound: logits `[N, C]` and class indices `[N]` (torch.nn.functional.cross_entropy style). */
export type ClassLoss = (logits: TensorLike, target: TensorLike, options?: { readonly reduction?: Reduction }) => Value

/** What a multi-class loss is, for validity rules and selectors. */
export interface ClassLossProperties extends LossProperties {
  /** Whether the loss is bounded above, so no single mislabelled example can dominate the sum. */
  readonly bounded: boolean
  /** Whether it is designed for noisy labels (or provably tolerant of symmetric noise). */
  readonly noiseRobust: boolean
  /** Whether it is a proper scoring rule, so the minimiser's probabilities are calibrated. */
  readonly proper: boolean
  /** Whether it needs the label-noise transition matrix (the table builds a symmetric one from the noise rate). */
  readonly needsTransition: boolean
}

const multiclass = (
  properties: Omit<ClassLossProperties, 'task'>,
  description: string,
  space: Option<ClassLoss, ClassLossProperties>['space'],
  make: Option<ClassLoss, ClassLossProperties>['make'],
): Option<ClassLoss, ClassLossProperties> => ({
  description,
  space,
  properties: { task: 'classification', ...properties },
  make,
})

/** The row-stochastic transition matrix of symmetric label noise at rate `noise` over `c` classes. */
const symmetricTransition = (c: number, noise: number): Value =>
  add(mul(eye(c), 1 - noise - noise / (c - 1)), noise / (c - 1))

const num = (v: unknown): number => v as number

/**
 * The multi-class classification losses as one slot (logits and class indices): cross-entropy (default), label
 * smoothing, focal, and the losses made for noisy labels: generalised, truncated and symmetric cross-entropy,
 * bootstrapping, the probability MAE, the forward and backward corrected losses (for symmetric noise at a given rate)
 * and the bi-tempered logistic loss.
 */
export const multiclassLosses: OptionTable<ClassLoss, ClassLossProperties> = optionTable(
  {
    crossEntropy: multiclass(
      {
        convex: true,
        smooth: true,
        likelihood: true,
        bounded: false,
        noiseRobust: false,
        proper: true,
        needsTransition: false,
      },
      'the softmax cross-entropy, the negative log-likelihood of the categorical',
      dict({}),
      () => (z, y, o) => crossEntropy(z, y, o),
    ),
    labelSmoothing: multiclass(
      {
        convex: true,
        smooth: true,
        likelihood: false,
        bounded: false,
        noiseRobust: false,
        proper: false,
        needsTransition: false,
      },
      'the cross-entropy against labels mixed with the uniform distribution, which caps the confidence at 1 - ε (1 - 1/C)',
      dict({ epsilon: real(0, 0.5, { default: 0.1, label: 'smoothing ε' }) }),
      (k) => (z, y, o) => crossEntropy(z, y, { ...o, labelSmoothing: num(k['epsilon']) }),
    ),
    focal: multiclass(
      {
        convex: false,
        smooth: true,
        likelihood: false,
        bounded: false,
        noiseRobust: false,
        proper: false,
        needsTransition: false,
      },
      'the cross-entropy scaled by (1 - p_y)^gamma, which down-weights the examples already well classified',
      dict({ gamma: real(0, 5, { default: 2, label: 'focusing γ' }) }),
      (k) => (z, y, o) => categoricalFocal(z, y, { ...o, gamma: num(k['gamma']) }),
    ),
    generalised: multiclass(
      {
        convex: false,
        smooth: true,
        likelihood: false,
        bounded: true,
        noiseRobust: true,
        proper: false,
        needsTransition: false,
      },
      'generalised cross-entropy (1 - p_y^q) / q: the cross-entropy at q to 0, the MAE at q = 1',
      dict({ q: real(0.01, 1, { default: 0.7, label: 'q' }) }),
      (k) => (z, y, o) => generalisedCrossEntropy(z, y, { ...o, q: num(k['q']) }),
    ),
    truncated: multiclass(
      {
        convex: false,
        smooth: false,
        likelihood: false,
        bounded: true,
        noiseRobust: true,
        proper: false,
        needsTransition: false,
      },
      'generalised cross-entropy with the target probability floored at k: the examples below it are ignored as noise',
      dict({
        q: real(0.01, 1, { default: 0.7, label: 'q' }),
        k: real(0.01, 0.99, { default: 0.5, label: 'threshold k' }),
      }),
      (kn) => (z, y, o) => truncatedCrossEntropy(z, y, { ...o, q: num(kn['q']), k: num(kn['k']) }),
    ),
    symmetric: multiclass(
      {
        convex: false,
        smooth: true,
        likelihood: false,
        bounded: false,
        noiseRobust: true,
        proper: false,
        needsTransition: false,
      },
      'symmetric cross-entropy: α times the cross-entropy plus β times the reverse cross-entropy',
      dict({
        alpha: real(0.01, 10, { log: true, default: 0.1, label: 'α' }),
        beta: real(0.01, 10, { log: true, default: 1, label: 'β' }),
      }),
      (k) => (z, y, o) => symmetricCrossEntropy(z, y, { ...o, alpha: num(k['alpha']), beta: num(k['beta']) }),
    ),
    bootstrapped: multiclass(
      {
        convex: false,
        smooth: true,
        likelihood: false,
        bounded: false,
        noiseRobust: true,
        proper: false,
        needsTransition: false,
      },
      "cross-entropy against a mix of the label and the model's own belief (soft or hard bootstrapping)",
      dict({
        beta: real(0.1, 1, { default: 0.8, label: 'trust in the label β' }),
        mode: choice(['soft', 'hard'], { default: 'soft' }),
      }),
      (k) => (z, y, o) => bootstrapped(z, y, { ...o, beta: num(k['beta']), mode: k['mode'] as 'soft' | 'hard' }),
    ),
    mae: multiclass(
      {
        convex: false,
        smooth: true,
        likelihood: false,
        bounded: true,
        noiseRobust: true,
        proper: false,
        needsTransition: false,
      },
      'the mean absolute error of the probabilities, tolerant of symmetric noise and slow to fit',
      dict({}),
      () => (z, y, o) => probabilityMae(z, y, o),
    ),
    forwardCorrected: multiclass(
      {
        convex: false,
        smooth: true,
        likelihood: false,
        bounded: false,
        noiseRobust: true,
        proper: false,
        needsTransition: true,
      },
      'the cross-entropy of the probabilities pushed through the label-noise transition (symmetric noise at a given rate)',
      dict({ noise: real(0, 0.9, { default: 0.2, label: 'noise rate' }) }),
      (k) => (z, y, o) => {
        const c = asTensor(z).shape.at(-1) ?? 2
        return forwardCorrected(z, y, symmetricTransition(c, num(k['noise'])), o)
      },
    ),
    backwardCorrected: multiclass(
      {
        convex: false,
        smooth: true,
        likelihood: false,
        bounded: false,
        noiseRobust: true,
        proper: false,
        needsTransition: true,
      },
      'the per-class loss vector mapped through the inverse transition (an unbiased estimate of the clean loss)',
      dict({ noise: real(0, 0.9, { default: 0.2, label: 'noise rate' }) }),
      (k) => (z, y, o) => {
        const c = asTensor(z).shape.at(-1) ?? 2
        return backwardCorrected(z, y, symmetricTransition(c, num(k['noise'])), o)
      },
    ),
    biTempered: multiclass(
      {
        convex: false,
        smooth: true,
        likelihood: false,
        bounded: true,
        noiseRobust: true,
        proper: false,
        needsTransition: false,
      },
      'the bi-tempered logistic loss: bounded (t1 < 1) with heavy-tailed probabilities (t2 > 1)',
      dict({ t1: real(0.1, 1, { default: 0.8, label: 't1' }), t2: real(1, 4, { default: 1.2, label: 't2' }) }),
      (k) => (z, y, o) => biTemperedLogistic(z, y, { ...o, t1: num(k['t1']), t2: num(k['t2']) }),
    ),
  },
  { label: 'loss', description: 'the multi-class classification loss' },
)

/** A preference objective with its knobs bound: the log-probabilities of chosen and rejected responses to a scalar loss. */
export type PreferenceLoss = (batch: PreferenceBatch, options?: { readonly reduction?: Reduction }) => Value

/** What a preference objective is, for validity rules and selectors. */
export interface PreferenceProperties {
  /** Whether it needs the frozen reference policy's log-probabilities (DPO, IPO, KTO). */
  readonly needsReference: boolean
  /** Whether it reads length-normalised (per-token mean) log-probabilities rather than whole-response sums. */
  readonly lengthNormalised: boolean
  /** Whether it trains an explicit scalar reward (a reward model) rather than the policy directly. */
  readonly rewardModel: boolean
  /** Whether it needs paired comparisons (KTO needs only good and bad answers, which pairs also provide). */
  readonly paired: boolean
  /** Whether the optimal margin stays finite when every preference is deterministic (IPO). */
  readonly boundedMargin: boolean
}

const preference = (
  properties: PreferenceProperties,
  description: string,
  space: Option<PreferenceLoss, PreferenceProperties>['space'],
  make: Option<PreferenceLoss, PreferenceProperties>['make'],
): Option<PreferenceLoss, PreferenceProperties> => ({ description, space, properties, make })

/**
 * The preference objectives as one slot, on the log-probabilities of chosen and rejected responses: DPO (default),
 * IPO, KTO (a pair read as one good and one bad answer), ORPO and SimPO (on per-token mean log-probabilities, no
 * reference) and the Bradley-Terry loss of a reward model (the "log-probabilities" are then rewards).
 */
export const preferenceLosses: OptionTable<PreferenceLoss, PreferenceProperties> = optionTable(
  {
    dpo: preference(
      { needsReference: true, lengthNormalised: false, rewardModel: false, paired: true, boundedMargin: false },
      'direct preference optimisation: the Bradley-Terry loss of beta times the policy-to-reference log-ratios',
      dict({
        beta: real(0.01, 1, { log: true, default: 0.1, label: 'β' }),
        labelSmoothing: real(0, 0.5, { default: 0, label: 'label noise ε' }),
      }),
      (k) => (b, o) => dpo(b, { ...o, beta: num(k['beta']), labelSmoothing: num(k['labelSmoothing']) }),
    ),
    ipo: preference(
      { needsReference: true, lengthNormalised: false, rewardModel: false, paired: true, boundedMargin: true },
      'identity preference optimisation: a squared loss to the margin 1 / (2τ)',
      dict({ tau: real(0.01, 2, { log: true, default: 0.1, label: 'τ' }) }),
      (k) => (b, o) => ipo(b, { ...o, tau: num(k['tau']) }),
    ),
    kto: preference(
      { needsReference: true, lengthNormalised: false, rewardModel: false, paired: false, boundedMargin: true },
      'Kahneman-Tversky optimisation: a prospect-theoretic loss on good and bad answers',
      dict({
        beta: real(0.01, 1, { log: true, default: 0.1, label: 'β' }),
        lambdaU: real(0.1, 10, { log: true, default: 1, label: 'weight of bad answers' }),
      }),
      (k) => (b, o) => {
        const [c, r] = [asTensor(b.policyChosen), asTensor(b.policyRejected)]
        const [rc, rr] = [asTensor(b.referenceChosen ?? 0), asTensor(b.referenceRejected ?? 0)]
        const n = c.shape[0] ?? 0
        const desirable = [...new Array<number>(n).fill(1), ...new Array<number>(r.shape[0] ?? 0).fill(0)]
        return kto(concat([c, r]), concat([broadcastTo(rc, c.shape), broadcastTo(rr, r.shape)]), desirable, {
          ...o,
          beta: num(k['beta']),
          lambdaU: num(k['lambdaU']),
        })
      },
    ),
    orpo: preference(
      { needsReference: false, lengthNormalised: true, rewardModel: false, paired: true, boundedMargin: false },
      'odds-ratio preference optimisation: the likelihood of the chosen answer plus an odds-ratio term, no reference',
      dict({ lambda: real(0.01, 2, { log: true, default: 0.1, label: 'λ' }) }),
      (k) => (b, o) => orpo(b.policyChosen, b.policyRejected, { ...o, lambda: num(k['lambda']) }),
    ),
    simpo: preference(
      { needsReference: false, lengthNormalised: true, rewardModel: false, paired: true, boundedMargin: false },
      'simple preference optimisation: reference-free DPO on length-normalised likelihoods with a target margin',
      dict({
        beta: real(0.1, 10, { log: true, default: 2, label: 'β' }),
        gamma: real(0, 3, { default: 0.5, label: 'target margin γ' }),
      }),
      (k) => (b, o) => simpo(b.policyChosen, b.policyRejected, { ...o, beta: num(k['beta']), gamma: num(k['gamma']) }),
    ),
    bradleyTerry: preference(
      { needsReference: false, lengthNormalised: false, rewardModel: true, paired: true, boundedMargin: false },
      'the pairwise loss of a reward model: the chosen reward beats the rejected by a logistic',
      dict({ margin: real(0, 3, { default: 0, label: 'margin' }) }),
      (k) => (b, o) => bradleyTerry(b.policyChosen, b.policyRejected, { ...o, margin: num(k['margin']) }),
    ),
  },
  { label: 'preference objective', description: 'the objective that aligns a model to preferences' },
)
