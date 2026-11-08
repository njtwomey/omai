/**
 * The optimisers as an option set: each the library's gradient transformation from `optim` (SGD with momentum and
 * Nesterov, RMSprop, Adagrad, Adadelta, Adam and its relatives, Lion) as a shape that takes the learning rate (a
 * number or a schedule) and builds the transformation, with the knobs of its own and properties the validity rules and
 * selectors read (whether it adapts per coordinate, keeps a momentum, decays weights apart from the gradient). The
 * `gradient` mode of the smooth solvers takes an optimiser from this table beside a schedule from `schedule-table.ts`.
 */

import { bool, defaults, dict, type Option, type OptionTable, optionTable, real } from '../space/index.ts'
import { adam, adamax, adamw, amsgrad, lion, nadam, radam } from './adaptive.ts'
import { adadelta, adagrad, rmsprop, sgd } from './first-order.ts'
import type { GradientTransformation, LearningRate } from './transform.ts'

/** An optimiser shape: given the learning rate (a number or a schedule over steps), the gradient transformation. */
export type OptimiserShape = (learningRate: LearningRate) => GradientTransformation

/** What an optimiser does, for validity rules and selectors. */
export interface OptimiserProperties {
  /** Scales each coordinate by its own history of gradients. */
  readonly adaptive: boolean
  /** Keeps a running average of the gradient (a momentum). */
  readonly momentum: boolean
  /** Applies weight decay to the parameters apart from the gradient (AdamW, Lion), not as an L2 penalty. */
  readonly decoupledDecay: boolean
}

const shape = (
  properties: OptimiserProperties,
  description: string,
  space: Option<OptimiserShape, OptimiserProperties>['space'],
  make: Option<OptimiserShape, OptimiserProperties>['make'],
): Option<OptimiserShape, OptimiserProperties> => ({
  description,
  space,
  properties,
  // A configuration that names only some knobs gets the others' defaults.
  make: (k) => make({ ...(defaults(space) as Readonly<Record<string, never>>), ...k }),
})

const weightDecay = (fallback = 0): ReturnType<typeof real> => real(0, 1, { default: fallback, label: 'weight decay' })
const b1 = real(0, 0.999, { default: 0.9, label: 'first-moment decay β₁' })
const b2 = real(0.5, 0.99999, { default: 0.999, label: 'second-moment decay β₂' })
const eps = real(1e-12, 1e-2, { log: true, default: 1e-8, label: 'ε' })

const num = (k: Readonly<Record<string, unknown>>, name: string): number => k[name] as number

/**
 * The optimisers as one slot: SGD (default), RMSprop, Adagrad, Adadelta, Adam, AdamW, AMSGrad, Adamax, NAdam, RAdam
 * and Lion, each with its own knobs, all taking the learning rate (or a schedule) when made.
 *
 * @index table SGD, RMSprop, Adagrad, Adam and relatives as one slot.
 * @example
 * ```ts
 * import { optimisers } from 'omai-engine/optim'
 * optimisers.make({ kind: 'adam', b1: 0.8 })(0.01).name // => 'chain(addDecayedWeights, scaleByAdam, scaleByLearningRate)'
 * ```
 */
export const optimisers: OptionTable<OptimiserShape, OptimiserProperties> = optionTable(
  {
    sgd: shape(
      { adaptive: false, momentum: true, decoupledDecay: false },
      'stochastic gradient descent, with optional (Nesterov) momentum',
      dict({
        momentum: real(0, 0.999, { default: 0, label: 'momentum' }),
        nesterov: bool({ default: false, label: 'Nesterov momentum' }),
        weightDecay: weightDecay(),
      }),
      (k) => (learningRate) =>
        sgd({
          learningRate,
          momentum: num(k, 'momentum'),
          nesterov: k['nesterov'] as boolean,
          weightDecay: num(k, 'weightDecay'),
        }),
    ),
    rmsprop: shape(
      { adaptive: true, momentum: false, decoupledDecay: false },
      'a running mean of squared gradients scales each coordinate',
      dict({
        decay: real(0.5, 0.9999, { default: 0.99, label: 'decay' }),
        eps,
        momentum: real(0, 0.999, { default: 0, label: 'momentum' }),
        centered: bool({ default: false, label: 'centred' }),
        weightDecay: weightDecay(),
      }),
      (k) => (learningRate) =>
        rmsprop({
          learningRate,
          decay: num(k, 'decay'),
          eps: num(k, 'eps'),
          momentum: num(k, 'momentum'),
          centered: k['centered'] as boolean,
          weightDecay: num(k, 'weightDecay'),
        }),
    ),
    adagrad: shape(
      { adaptive: true, momentum: false, decoupledDecay: false },
      'the sum of all past squared gradients scales each coordinate',
      dict({
        eps: real(1e-12, 1e-2, { log: true, default: 1e-10, label: 'ε' }),
        initialAccumulator: real(0, 1, { default: 0, label: 'initial accumulator' }),
        weightDecay: weightDecay(),
      }),
      (k) => (learningRate) =>
        adagrad({
          learningRate,
          eps: num(k, 'eps'),
          initialAccumulator: num(k, 'initialAccumulator'),
          weightDecay: num(k, 'weightDecay'),
        }),
    ),
    adadelta: shape(
      { adaptive: true, momentum: false, decoupledDecay: false },
      'steps whose units match the parameters, from running averages of squared gradients and updates',
      dict({
        rho: real(0.5, 0.9999, { default: 0.9, label: 'ρ' }),
        eps: real(1e-10, 1e-2, { log: true, default: 1e-6, label: 'ε' }),
        weightDecay: weightDecay(),
      }),
      (k) => (learningRate) =>
        adadelta({ learningRate, rho: num(k, 'rho'), eps: num(k, 'eps'), weightDecay: num(k, 'weightDecay') }),
    ),
    adam: shape(
      { adaptive: true, momentum: true, decoupledDecay: false },
      'bias-corrected first and second moment estimates',
      dict({ b1, b2, eps, weightDecay: weightDecay() }),
      (k) => (learningRate) =>
        adam({
          learningRate,
          b1: num(k, 'b1'),
          b2: num(k, 'b2'),
          eps: num(k, 'eps'),
          weightDecay: num(k, 'weightDecay'),
        }),
    ),
    adamw: shape(
      { adaptive: true, momentum: true, decoupledDecay: true },
      'Adam with weight decay applied to the parameters apart from the gradient',
      dict({ b1, b2, eps, weightDecay: weightDecay(0.01) }),
      (k) => (learningRate) =>
        adamw({
          learningRate,
          b1: num(k, 'b1'),
          b2: num(k, 'b2'),
          eps: num(k, 'eps'),
          weightDecay: num(k, 'weightDecay'),
        }),
    ),
    amsgrad: shape(
      { adaptive: true, momentum: true, decoupledDecay: false },
      'Adam normalised by the running maximum of the second moment, so steps never grow',
      dict({ b1, b2, eps, weightDecay: weightDecay() }),
      (k) => (learningRate) =>
        amsgrad({
          learningRate,
          b1: num(k, 'b1'),
          b2: num(k, 'b2'),
          eps: num(k, 'eps'),
          weightDecay: num(k, 'weightDecay'),
        }),
    ),
    adamax: shape(
      { adaptive: true, momentum: true, decoupledDecay: false },
      'Adam with the infinity norm in place of the second moment',
      dict({ b1, b2, eps, weightDecay: weightDecay() }),
      (k) => (learningRate) =>
        adamax({
          learningRate,
          b1: num(k, 'b1'),
          b2: num(k, 'b2'),
          eps: num(k, 'eps'),
          weightDecay: num(k, 'weightDecay'),
        }),
    ),
    nadam: shape(
      { adaptive: true, momentum: true, decoupledDecay: false },
      'Adam with Nesterov momentum',
      dict({
        b1,
        b2,
        eps,
        momentumDecay: real(1e-4, 0.1, { log: true, default: 4e-3, label: 'momentum decay ψ' }),
        weightDecay: weightDecay(),
      }),
      (k) => (learningRate) =>
        nadam({
          learningRate,
          b1: num(k, 'b1'),
          b2: num(k, 'b2'),
          eps: num(k, 'eps'),
          momentumDecay: num(k, 'momentumDecay'),
          weightDecay: num(k, 'weightDecay'),
        }),
    ),
    radam: shape(
      { adaptive: true, momentum: true, decoupledDecay: false },
      'Adam with a rectified variance, so no warm-up is needed',
      dict({ b1, b2, eps, weightDecay: weightDecay() }),
      (k) => (learningRate) =>
        radam({
          learningRate,
          b1: num(k, 'b1'),
          b2: num(k, 'b2'),
          eps: num(k, 'eps'),
          weightDecay: num(k, 'weightDecay'),
        }),
    ),
    lion: shape(
      { adaptive: false, momentum: true, decoupledDecay: true },
      'the sign of an interpolated momentum, with decoupled weight decay',
      dict({
        b1,
        b2: real(0.5, 0.99999, { default: 0.99, label: 'momentum decay β₂' }),
        weightDecay: weightDecay(1e-3),
      }),
      (k) => (learningRate) =>
        lion({ learningRate, b1: num(k, 'b1'), b2: num(k, 'b2'), weightDecay: num(k, 'weightDecay') }),
    ),
  },
  { label: 'optimiser', description: 'how a gradient becomes a step' },
)
