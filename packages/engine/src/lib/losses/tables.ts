/**
 * The objectives as option sets: the losses of regression and of binary classification, each the library's loss from
 * `losses` with its knobs bound and its properties (the task, convexity, smoothness, whether it is a likelihood), and
 * the penalties (none, ridge, lasso, elastic net) with their strength, their value on parameters, their proximal
 * operator where they have one, and their properties (smooth, has a prox, sparse). A method's objective slot is a
 * loss and a penalty from these tables; its validity rules read their properties (a proximal solver needs a prox,
 * IRLS a likelihood).
 */

import { proxElasticNet, proxL1, proxSquaredL2 } from '../optim/proximal.ts'
import { dict, type Option, type OptionTable, optionTable, real } from '../space/index.ts'
import { asTensor, type TensorLike, type Value } from '../tensor/index.ts'
import { exponential, hinge, logistic, modifiedHuber, perceptron, squaredHinge } from './margin.ts'
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
