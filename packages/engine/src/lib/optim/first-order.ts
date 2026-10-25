/**
 * Classic first-order optimisers as gradient transformations, matching torch.optim and optax: SGD with momentum and
 * Nesterov momentum, RMSprop, Adagrad, Adadelta, Rprop (sign-based per-coordinate steps), averaged SGD, signSGD and
 * signum, and the exponential moving average of updates.
 */

import { MdmlError } from '../base/index.ts'
import {
  add,
  clip,
  div,
  equal,
  greater,
  less,
  mul,
  sign,
  sqrt,
  square,
  sub,
  treeMap,
  treeZeros,
  where,
  zerosLike,
} from '../tensor/index.ts'
import {
  addDecayedWeights,
  chain,
  type GradientTransformation,
  type LearningRate,
  scaleByAdagrad,
  scaleByLearningRate,
  scaleByRms,
  type Tree,
  traceMomentum,
} from './transform.ts'

const rate = (lr: LearningRate, step: number): number => (typeof lr === 'number' ? lr : lr(step))

/** Options of {@link sgd}. */
export interface SgdOptions {
  readonly learningRate: LearningRate
  readonly momentum?: number
  /** Dampening of the gradient in the momentum buffer (torch's dampening). */
  readonly dampening?: number
  readonly nesterov?: boolean
  /** L2 penalty added to the gradient (torch's weight_decay). */
  readonly weightDecay?: number
}

/**
 * Stochastic gradient descent with optional momentum (with dampening), Nesterov momentum and L2 weight decay
 * (torch.optim.SGD).
 *
 * @param options - Learning rate, momentum, dampening, Nesterov and weight decay.
 * @returns The optimiser.
 * @concept optimisers
 * @example
 * ```ts
 * import { sgd, applyUpdates } from 'mdml-engine/optim'
 * import { tensor, toArray } from 'mdml-engine/tensor'
 * const opt = sgd({ learningRate: 0.1 })
 * const { updates } = opt.update(tensor([1, -2]), opt.init(tensor([0, 0])), tensor([0, 0]))
 * toArray(updates as never) // => [-0.1, 0.2]
 * ```
 */
export function sgd(options: SgdOptions): GradientTransformation<unknown[]> {
  const momentum = options.momentum ?? 0
  return chain(
    addDecayedWeights(options.weightDecay ?? 0),
    ...(momentum > 0 ? [traceMomentum(momentum, options.nesterov ?? false, options.dampening ?? 0)] : []),
    scaleByLearningRate(options.learningRate),
  )
}

/** Options of {@link rmsprop}. */
export interface RmspropOptions {
  readonly learningRate: LearningRate
  readonly decay?: number
  readonly eps?: number
  readonly momentum?: number
  /** Normalise by the estimated gradient variance (torch's centered). */
  readonly centered?: boolean
  readonly weightDecay?: number
}

/**
 * RMSprop (torch.optim.RMSprop): g / (sqrt(v) + eps), optionally centred, with optional momentum and L2 weight decay.
 *
 * @param options - Learning rate, decay (alpha), eps, momentum, centring and weight decay.
 * @returns The optimiser.
 * @example
 * ```ts
 * import { rmsprop } from 'mdml-engine/optim'
 * rmsprop({ learningRate: 0.01 }).name // => 'chain(addDecayedWeights, scaleByRms, scaleByLearningRate)'
 * ```
 */
export function rmsprop(options: RmspropOptions): GradientTransformation<unknown[]> {
  return chain(
    addDecayedWeights(options.weightDecay ?? 0),
    scaleByRms(options.decay ?? 0.99, options.eps ?? 1e-8, options.centered ?? false),
    ...((options.momentum ?? 0) > 0 ? [traceMomentum(options.momentum ?? 0)] : []),
    scaleByLearningRate(options.learningRate),
  )
}

/** Options of {@link adagrad}. */
export interface AdagradOptions {
  readonly learningRate: LearningRate
  readonly eps?: number
  readonly initialAccumulator?: number
  /** The rate at step t is learningRate / (1 + t lrDecay) (torch's lr_decay). */
  readonly lrDecay?: number
  readonly weightDecay?: number
}

/**
 * Adagrad (torch.optim.Adagrad), with torch's learning-rate decay lr / (1 + t lrDecay).
 *
 * @param options - Learning rate, eps, initial accumulator, learning-rate decay and weight decay.
 * @returns The optimiser.
 * @example
 * ```ts
 * import { adagrad } from 'mdml-engine/optim'
 * adagrad({ learningRate: 0.1 }).name // => 'chain(addDecayedWeights, scaleByAdagrad, scaleByLearningRate)'
 * ```
 */
export function adagrad(options: AdagradOptions): GradientTransformation<unknown[]> {
  const decay = options.lrDecay ?? 0
  const lr = options.learningRate
  return chain(
    addDecayedWeights(options.weightDecay ?? 0),
    scaleByAdagrad(options.eps ?? 1e-10, options.initialAccumulator ?? 0),
    scaleByLearningRate(decay === 0 ? lr : (t: number) => rate(lr, t) / (1 + t * decay)),
  )
}

/** State of {@link scaleByAdadelta}. */
export interface AdadeltaState {
  readonly squareAvg: Tree
  readonly accDelta: Tree
}

/**
 * Adadelta's scaling (Zeiler 2012, torch.optim.Adadelta): v = rho v + (1 - rho) g^2, delta = sqrt(u + eps) /
 * sqrt(v + eps) g, u = rho u + (1 - rho) delta^2.
 *
 * @param rho - Decay of both running averages.
 * @param eps - Added inside both square roots.
 * @returns The transformation.
 * @example
 * ```ts
 * import { scaleByAdadelta } from 'mdml-engine/optim'
 * import { tensor, toArray } from 'mdml-engine/tensor'
 * const t = scaleByAdadelta(0, 0)
 * toArray(t.update(tensor([2]), t.init(tensor([0]))).updates as never) // => [0]
 * ```
 */
export function scaleByAdadelta(rho = 0.9, eps = 1e-6): GradientTransformation<AdadeltaState> {
  return {
    name: 'scaleByAdadelta',
    init: (params) => ({ squareAvg: treeZeros(params), accDelta: treeZeros(params) }),
    update: (grads, state) => {
      const squareAvg = treeMap((g, v) => add(mul(v, rho), mul(square(g), 1 - rho)), grads, state.squareAvg)
      const updates = treeMap(
        (g, v, u) => mul(div(sqrt(add(u, eps)), sqrt(add(v, eps))), g),
        grads,
        squareAvg,
        state.accDelta,
      )
      const accDelta = treeMap((d, u) => add(mul(u, rho), mul(square(d), 1 - rho)), updates, state.accDelta)
      return { updates, state: { squareAvg, accDelta } }
    },
  }
}

/** Options of {@link adadelta}. */
export interface AdadeltaOptions {
  /** Default 1, as torch. */
  readonly learningRate?: LearningRate
  readonly rho?: number
  readonly eps?: number
  readonly weightDecay?: number
}

/**
 * Adadelta (torch.optim.Adadelta): per-coordinate steps whose units match the parameters, from running averages of
 * squared gradients and squared updates.
 *
 * @param options - Learning rate (default 1), rho, eps and L2 weight decay.
 * @returns The optimiser.
 * @example
 * ```ts
 * import { adadelta } from 'mdml-engine/optim'
 * adadelta().name // => 'chain(addDecayedWeights, scaleByAdadelta, scaleByLearningRate)'
 * ```
 */
export function adadelta(options: AdadeltaOptions = {}): GradientTransformation<unknown[]> {
  return chain(
    addDecayedWeights(options.weightDecay ?? 0),
    scaleByAdadelta(options.rho ?? 0.9, options.eps ?? 1e-6),
    scaleByLearningRate(options.learningRate ?? 1),
  )
}

/** Options of {@link rprop}. */
export interface RpropOptions {
  /** The initial step size of every coordinate (torch's lr, default 0.01). */
  readonly learningRate?: number
  /** Factors for a sign change and for an agreeing sign (default [0.5, 1.2]). */
  readonly etas?: readonly [number, number]
  /** Bounds on the step sizes (default [1e-6, 50]). */
  readonly stepSizes?: readonly [number, number]
}

/** State of {@link rprop}. */
export interface RpropState {
  readonly prev: Tree
  readonly stepSize: Tree
}

/**
 * Resilient backpropagation (Riedmiller and Braun 1993, torch.optim.Rprop): each coordinate keeps its own step size,
 * grown when the gradient keeps its sign and shrunk (skipping that step) when it flips; only the sign of the
 * gradient is used.
 *
 * @param options - Initial step size, the two factors and the step-size bounds.
 * @returns The optimiser (its updates already include the step sizes).
 * @example
 * ```ts
 * import { rprop } from 'mdml-engine/optim'
 * import { tensor, toArray } from 'mdml-engine/tensor'
 * const opt = rprop({ learningRate: 0.1 })
 * const s = opt.update(tensor([3, -1]), opt.init(tensor([0, 0])))
 * toArray(s.updates as never) // => [-0.1, 0.1]
 * ```
 */
export function rprop(options: RpropOptions = {}): GradientTransformation<RpropState> {
  const lr = options.learningRate ?? 0.01
  const [etaMinus, etaPlus] = options.etas ?? [0.5, 1.2]
  const [stepMin, stepMax] = options.stepSizes ?? [1e-6, 50]
  return {
    name: 'rprop',
    init: (params) => ({ prev: treeZeros(params), stepSize: treeMap((p) => add(zerosLike(p), lr), params) }),
    update: (grads, state) => {
      const factor = treeMap(
        (g, p) => {
          const s = sign(mul(g, p))
          return where(greater(s, 0), etaPlus, where(less(s, 0), etaMinus, 1))
        },
        grads,
        state.prev,
      )
      const stepSize = treeMap((st, f) => clip(mul(st, f), stepMin, stepMax), state.stepSize, factor)
      const g2 = treeMap((g, f) => where(equal(f, etaMinus), 0, g), grads, factor)
      const updates = treeMap((g, st) => mul(sign(g), mul(st, -1)), g2, stepSize)
      return { updates, state: { prev: g2, stepSize } }
    },
  }
}

/** Options of {@link asgd}. */
export interface AsgdOptions {
  readonly learningRate?: number
  /** Decay term (default 1e-4). */
  readonly lambd?: number
  /** Power of the step-size decay (default 0.75). */
  readonly alpha?: number
  /** The step at which averaging starts (default 1e6). */
  readonly t0?: number
  readonly weightDecay?: number
}

/** State of {@link asgd}: the step size, the averaging weight and the averaged parameters. */
export interface AsgdState {
  readonly count: number
  readonly eta: number
  readonly mu: number
  /** The averaged iterate (Polyak-Ruppert average once averaging has started). */
  readonly ax: Tree
}

/**
 * Averaged stochastic gradient descent (Polyak and Juditsky 1992; torch.optim.ASGD): p <- p (1 - lambd eta) - eta g,
 * with eta = lr / (1 + lambd lr t)^alpha and the running average of the iterates kept in the state from step t0.
 *
 * @param options - Learning rate, lambd, alpha, t0 and weight decay.
 * @returns The optimiser (needs `params` in `update`); `state.ax` is the averaged iterate.
 * @example
 * ```ts
 * import { asgd } from 'mdml-engine/optim'
 * import { tensor, toArray } from 'mdml-engine/tensor'
 * const opt = asgd({ learningRate: 0.1, lambd: 0 })
 * toArray(opt.update(tensor([2]), opt.init(tensor([0])), tensor([0])).updates as never) // => [-0.2]
 * ```
 */
export function asgd(options: AsgdOptions = {}): GradientTransformation<AsgdState> {
  const lr = options.learningRate ?? 0.01
  const lambd = options.lambd ?? 1e-4
  const alpha = options.alpha ?? 0.75
  const t0 = options.t0 ?? 1e6
  const wd = options.weightDecay ?? 0
  return {
    name: 'asgd',
    init: (params) => ({ count: 0, eta: lr, mu: 1, ax: treeZeros(params) }),
    update: (grads, state, params) => {
      if (params === undefined) throw new MdmlError('value', 'asgd needs the parameters')
      const { eta, mu } = state
      const g = wd === 0 ? grads : treeMap((gi, p) => add(gi, mul(p, wd)), grads, params)
      const next = treeMap((p, gi) => sub(mul(p, 1 - lambd * eta), mul(gi, eta)), params, g)
      const updates = treeMap((n, p) => sub(n, p), next, params)
      const ax = mu !== 1 ? treeMap((a, n) => add(a, mul(sub(n, a), mu)), state.ax, next) : next
      const count = state.count + 1
      return {
        updates,
        state: { count, eta: lr / (1 + lambd * lr * count) ** alpha, mu: 1 / Math.max(1, count - t0), ax },
      }
    },
  }
}

/** State of {@link ema}. */
export interface EmaState {
  readonly count: number
  readonly ema: Tree
}

/**
 * Exponential moving average of the updates (optax.ema): m = decay m + (1 - decay) g, optionally debiased by
 * 1 - decay^t.
 *
 * @param decay - The decay rate.
 * @param debias - Divide by 1 - decay^t.
 * @returns The transformation.
 * @example
 * ```ts
 * import { ema } from 'mdml-engine/optim'
 * import { tensor, toArray } from 'mdml-engine/tensor'
 * const t = ema(0.5, false)
 * toArray(t.update(tensor([2]), t.init(tensor([0]))).updates as never) // => [1]
 * ```
 */
export function ema(decay: number, debias = true): GradientTransformation<EmaState> {
  return {
    name: 'ema',
    init: (params) => ({ count: 0, ema: treeZeros(params) }),
    update: (grads, state) => {
      const count = state.count + 1
      const m = treeMap((g, e) => add(mul(e, decay), mul(g, 1 - decay)), grads, state.ema)
      const c = 1 - decay ** count
      return { updates: debias ? treeMap((e) => div(e, c), m) : m, state: { count, ema: m } }
    },
  }
}

/**
 * Replace each update by its sign (optax.scale_by_sign).
 *
 * @returns The transformation.
 * @example
 * ```ts
 * import { scaleBySign } from 'mdml-engine/optim'
 * import { tensor, toArray } from 'mdml-engine/tensor'
 * toArray(scaleBySign().update(tensor([3, -0.5, 0]), undefined).updates as never) // => [1, -1, 0]
 * ```
 */
export function scaleBySign(): GradientTransformation<undefined> {
  return {
    name: 'scaleBySign',
    init: () => undefined,
    update: (g, state) => ({ updates: treeMap((u) => sign(u), g), state }),
  }
}

/** Options of {@link signSgd}. */
export interface SignSgdOptions {
  readonly learningRate: LearningRate
  /** With momentum, the sign of an (undebiased) moving average of the gradient: signum (default 0, plain signSGD). */
  readonly momentum?: number
}

/**
 * signSGD and signum (Bernstein et al. 2018; optax.sign_sgd and optax.signum): steps of fixed length along the sign of
 * the gradient, or of its moving average.
 *
 * @param options - Learning rate and the moving-average decay.
 * @returns The optimiser.
 * @example
 * ```ts
 * import { signSgd } from 'mdml-engine/optim'
 * import { tensor, toArray } from 'mdml-engine/tensor'
 * const opt = signSgd({ learningRate: 0.1 })
 * toArray(opt.update(tensor([3, -0.5]), opt.init(tensor([0, 0]))).updates as never) // => [-0.1, 0.1]
 * ```
 */
export function signSgd(options: SignSgdOptions): GradientTransformation<unknown[]> {
  const m = options.momentum ?? 0
  return chain(...(m > 0 ? [ema(m, false)] : []), scaleBySign(), scaleByLearningRate(options.learningRate))
}
