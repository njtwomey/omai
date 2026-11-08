/**
 * The functional training step, Flax style: a `TrainState` holding parameters, optimiser state, buffers, the step and
 * its key, and the pieces of one step: loss and gradients on a minibatch, applying gradients through the optimiser with
 * frozen parameters masked, gradient accumulation and `trainStep`, which combines them.
 */

import { valueAndGrad } from '../../autodiff/index.ts'
import { applyUpdates, type GradientTransformation, globalNorm } from '../../optim/index.ts'
import { foldIn, type Key, key as makeKey } from '../../random/index.ts'
import { add, item, mul, treeLeaves, treeMap, type Value } from '../../tensor/index.ts'
import type { TapContext } from '../../trace/index.ts'
import { concreteTree, type Module, type Params } from '../module/module.ts'

/**
 * The state of a training run (Flax's TrainState): parameters, optimiser state, buffers (running statistics), the
 * step, the key the stochastic layers draw from, and what the last steps measured.
 */
export interface TrainState {
  readonly params: Params
  readonly optState: unknown
  /** Non-trainable state the forward pass updates (batch-norm running statistics). */
  readonly buffers: Params
  /** Minibatches processed so far. */
  readonly step: number
  /** Optimiser updates applied so far (fewer than steps with gradient accumulation). */
  readonly updates: number
  readonly epoch: number
  /** The run's key; step t's dropout key is foldIn(key, t). */
  readonly key: Key
  /** The loss and gradient norm of the last minibatch. */
  readonly loss: number
  readonly gradNorm: number
  /** The sum of this epoch's minibatch losses so far (for the epoch mean). */
  readonly epochLossSum: number
  /** The factor applied to every update (lowered by learning-rate reduction on plateau). */
  readonly lrScale: number
  /** Gradients summed over the minibatches of an accumulation window, and how many. */
  readonly accumulated?: { readonly grads: Params; readonly count: number }
  /** Set when a callback asked to stop. */
  readonly stopped?: string
}

/**
 * A fresh training state for a module and an optimiser.
 *
 * @param model - The module (its parameters and buffers are the starting point).
 * @param optimizer - The gradient transformation.
 * @param key - The run's key (default key(0)).
 * @returns The state at step 0.
 * @concept training
 * @example
 * ```ts
 * import { trainState, Linear } from 'omai-engine/nn'
 * import { sgd } from 'omai-engine/optim'
 * trainState(new Linear(2, 1), sgd({ learningRate: 0.1 })).step // => 0
 * ```
 */
export function trainState(model: Module<never, unknown>, optimizer: GradientTransformation, key?: Key): TrainState {
  return {
    params: model.params,
    optState: optimizer.init(model.params),
    buffers: model.buffers,
    step: 0,
    updates: 0,
    epoch: 0,
    key: key ?? makeKey(0),
    loss: Number.NaN,
    gradNorm: Number.NaN,
    epochLossSum: 0,
    lrScale: 1,
  }
}

/** Options of {@link applyGradients}. */
export interface ApplyGradientOptions {
  /** A tree like the parameters with 1 for trainable and 0 for frozen leaves (`Module.trainableMask()`). */
  readonly mask?: Params
  /** Multiply the updates (default: the state's lrScale). */
  readonly scale?: number
}

/**
 * Apply gradients through the optimiser (Flax's apply_gradients): transform, scale, mask frozen parameters, add.
 *
 * @param state - The training state.
 * @param grads - Gradients with the structure of the parameters.
 * @param optimizer - The gradient transformation the state was built with.
 * @param options - A trainable mask and an update scale.
 * @returns The state with new parameters, optimiser state and update count, and the updates applied.
 * @example
 * ```ts
 * import { applyGradients, trainState, Linear } from 'omai-engine/nn'
 * import { sgd } from 'omai-engine/optim'
 * import { treeMap, onesLike } from 'omai-engine/tensor'
 * const opt = sgd({ learningRate: 0.1 })
 * const s = trainState(new Linear(2, 1), opt)
 * applyGradients(s, treeMap((p) => onesLike(p), s.params), opt).state.updates // => 1
 * ```
 */
export function applyGradients(
  state: TrainState,
  grads: Params,
  optimizer: GradientTransformation,
  options: ApplyGradientOptions = {},
): { readonly state: TrainState; readonly updates: Params } {
  const r = optimizer.update(grads, state.optState, state.params)
  let updates = r.updates as Params
  const scale = options.scale ?? state.lrScale
  if (scale !== 1) updates = treeMap((u) => mul(u, scale), updates)
  if (options.mask !== undefined) updates = treeMap((u, m) => mul(u, m), updates, options.mask)
  return {
    state: {
      ...state,
      params: applyUpdates(state.params, updates) as Params,
      optState: r.state,
      updates: state.updates + 1,
    },
    updates,
  }
}

/** Result of {@link lossAndGrads}. */
export interface LossAndGrads {
  readonly loss: number
  readonly grads: Params
  /** The buffers after the forward pass (unchanged in evaluation). */
  readonly buffers: Params
}

/**
 * The loss of a minibatch, its gradient with respect to the parameters, and the updated buffers (one forward and
 * backward pass in training mode).
 *
 * @param model - The module.
 * @param params - The parameters to differentiate at.
 * @param buffers - The buffers to read.
 * @param x - The minibatch input.
 * @param loss - The objective of the output.
 * @param options - The dropout key, the training flag (default true), a tap context for activations and a penalty on
 *   the parameters added to the loss.
 * @returns The loss, gradients and buffers.
 * @example
 * ```ts
 * import { lossAndGrads, Linear } from 'omai-engine/nn'
 * import { sum } from 'omai-engine/tensor'
 * const m = new Linear(2, 1)
 * Object.keys(lossAndGrads(m, m.params, m.buffers, [[1, 2]], (y) => sum(y)).grads) // => ['bias', 'weight']
 * ```
 */
export function lossAndGrads<I>(
  model: Module<I, Value>,
  params: Params,
  buffers: Params,
  x: I,
  loss: (output: Value) => Value,
  options: {
    readonly key?: Key
    readonly train?: boolean
    readonly taps?: TapContext
    /** A regulariser of the parameters, added to the loss (weight decay, a smoothness penalty). */
    readonly penalty?: (params: Params) => Value
  } = {},
): LossAndGrads {
  let next = buffers
  const objective = (p: Params): Value => {
    const data = loss(
      model.apply(p, x, {
        train: options.train ?? true,
        buffers,
        onBuffers: (b) => {
          next = b
        },
        ...(options.key === undefined ? {} : { key: options.key }),
        ...(options.taps === undefined ? {} : { taps: options.taps }),
      }),
    )
    return options.penalty === undefined ? data : add(data, options.penalty(p))
  }
  const { value, grad } = valueAndGrad(objective)(params)
  return { loss: item(value), grads: grad, buffers: concreteTree(next) }
}

/**
 * One functional training step (Flax's train_step): loss and gradients on a minibatch with the step's dropout key,
 * then the optimiser update; buffers are carried along.
 *
 * @param model - The module.
 * @param state - The training state.
 * @param x - The minibatch input.
 * @param loss - The objective of the output.
 * @param optimizer - The gradient transformation.
 * @returns The next state.
 * @concept training
 * @example
 * ```ts
 * import { trainStep, trainState, Linear } from 'omai-engine/nn'
 * import { sgd } from 'omai-engine/optim'
 * import { sum, square } from 'omai-engine/tensor'
 * const m = new Linear(1, 1)
 * const opt = sgd({ learningRate: 0.1 })
 * trainStep(m, trainState(m, opt), [[1]], (y) => sum(square(y)), opt).step // => 1
 * ```
 */
export function trainStep<I>(
  model: Module<I, Value>,
  state: TrainState,
  x: I,
  loss: (output: Value) => Value,
  optimizer: GradientTransformation,
): TrainState {
  const r = lossAndGrads(model, state.params, state.buffers, x, loss, { key: foldIn(state.key, state.step) })
  const next = applyGradients(state, r.grads, optimizer).state
  return { ...next, buffers: r.buffers, step: state.step + 1, loss: r.loss, gradNorm: item(globalNorm(r.grads)) }
}

/**
 * Add gradients into an accumulation window (gradient accumulation over minibatches).
 *
 * @param window - The window so far (or none).
 * @param grads - This minibatch's gradients.
 * @returns The summed gradients and count.
 * @example
 * ```ts
 * import { accumulate } from 'omai-engine/nn'
 * import { tensor } from 'omai-engine/tensor'
 * accumulate(accumulate(undefined, { w: tensor(1) }), { w: tensor(2) }).count // => 2
 * ```
 */
export function accumulate(
  window: { readonly grads: Params; readonly count: number } | undefined,
  grads: Params,
): { readonly grads: Params; readonly count: number } {
  if (window === undefined) return { grads, count: 1 }
  return { grads: treeMap((a, b) => add(a, b), window.grads, grads), count: window.count + 1 }
}

/**
 * Whether any leaf of a trainable mask is zero (some parameters are frozen).
 *
 * @param mask - From `Module.trainableMask()`.
 * @returns True when at least one parameter is frozen.
 * @example
 * ```ts
 * import { hasFrozen, Linear } from 'omai-engine/nn'
 * hasFrozen(new Linear(2, 1).requiresGrad(false).trainableMask()) // => true
 * ```
 */
export function hasFrozen(mask: Params): boolean {
  return treeLeaves(mask).some((m) => item(m) === 0)
}
