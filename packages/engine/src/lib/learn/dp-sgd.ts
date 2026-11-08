/**
 * Differentially private training by DP-SGD (Abadi et al. 2016, opacus style): per-example gradients by `vmap` of
 * `grad`, clipped to a norm bound, summed, hidden in Gaussian noise scaled to the bound and divided by the expected
 * batch size; a gradient transformation for the noise, a step over `TrainState`, a stepped algorithm with Poisson
 * subsampling that accounts the privacy spent by the Renyi accountant of `privacy.ts` as it trains (epsilon at delta on
 * the tape), and the accountant's inverse, the noise multiplier a budget needs.
 */

import { valueAndGrad, vmap } from '../autodiff/index.ts'
import { OmaiError } from '../base/index.ts'
import { applyGradients, concreteTree, type Module, type Params, type TrainState, trainState } from '../nn/index.ts'
import { type GradientTransformation, globalNorm } from '../optim/index.ts'
import { bernoulli, foldIn, type Key, normal } from '../random/index.ts'
import {
  add,
  asTensor,
  div,
  expandDims,
  fromStorage,
  item,
  minimum,
  mul,
  primalOf,
  reshape,
  sqrt,
  sum,
  type Tensor,
  type TensorLike,
  toFlat,
  treeFlatten,
  treeMap,
  treeUnflatten,
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
import { DEFAULT_RDP_ORDERS, rdpToDp, subsampledGaussianRdp } from './privacy.ts'

/** A per-example objective of the model's output for one example (a batch of one) and its target. */
export type ExampleLoss = (output: Value, target: Value) => Value

/** The result of {@link perExampleGradients}. */
export interface PerExampleGradients {
  /** Gradients with the structure of the parameters and a leading axis of one entry per example. */
  readonly grads: Params
  /** Each example's loss. */
  readonly losses: readonly number[]
}

/**
 * Per-example gradients of a loss (torch.func's `vmap(grad(...))`, opacus's grad samples): the gradient of each example's
 * own loss with respect to the parameters, as one tree with a leading example axis.
 *
 * @param model - The module.
 * @param params - Parameters to differentiate at.
 * @param x - Inputs `[N, ...]`.
 * @param y - Targets `[N, ...]`.
 * @param loss - The loss of one example's output `[1, ...]` and target `[1, ...]`.
 * @param options - Buffers to read, and whether the model runs in training mode (default evaluation).
 * @returns The gradients and the losses.
 * @concept differential-privacy
 * @example
 * ```ts
 * import { perExampleGradients } from 'omai-engine/learn'
 * import { Linear } from 'omai-engine/nn'
 * import { sum } from 'omai-engine/tensor'
 * const m = new Linear(2, 1)
 * perExampleGradients(m, m.params, [[1, 2], [3, 4]], [[0], [0]], (o) => sum(o)).grads.weight.shape // => [2, 1, 2]
 * ```
 */
export function perExampleGradients(
  model: Module<Value, Value>,
  params: Params,
  x: TensorLike,
  y: TensorLike,
  loss: ExampleLoss,
  options: { readonly buffers?: Params; readonly train?: boolean } = {},
): PerExampleGradients {
  const buffers = options.buffers ?? model.buffers
  const one = (p: Params, xi: Value, yi: Value): Value =>
    reshape(loss(model.apply(p, expandDims(xi, 0), { train: options.train ?? false, buffers }), expandDims(yi, 0)), [])
  const perExample = vmap(
    (xi: Value, yi: Value) => {
      const r = valueAndGrad((p: Params) => one(p, xi, yi))(params)
      return [r.value, r.grad] as const
    },
    [0, 0],
    [0, 0],
  )
  const [values, grads] = perExample(asTensor(x), asTensor(y)) as unknown as readonly [Value, Params]
  return { grads: concreteTree(grads), losses: toFlat(primalOf(values)) }
}

/** The result of {@link clipPerExample}. */
export interface Clipped {
  /** The clipped gradients (same structure, leading example axis). */
  readonly grads: Params
  /** Each example's gradient norm before clipping. */
  readonly norms: readonly number[]
  /** The share of examples whose gradient was clipped. */
  readonly clippedShare: number
}

/** Each example's gradient norm over all leaves, `[N]`. */
function exampleNorms(grads: Params): Value {
  const leaves = treeFlatten(grads).leaves
  const n = leaves[0]?.shape[0] ?? 0
  let total: Value = fromStorage(new Float64Array(n), [n], 'float64')
  for (const leaf of leaves) total = add(total, sum(mul(reshape(leaf, [n, -1]), reshape(leaf, [n, -1])), 1))
  return sqrt(total)
}

/**
 * Clip each example's gradient to a norm bound (opacus's flat clipping): the whole gradient of an example, over every
 * parameter, is multiplied by min(1, C / (norm + 1e-6)), so no example moves the parameters by more than C however
 * unusual it is: the bounded sensitivity the Gaussian noise is calibrated to.
 *
 * @param grads - Per-example gradients (leading example axis), from {@link perExampleGradients}.
 * @param clipNorm - The bound C.
 * @returns The clipped gradients, the original norms and the clipped share.
 * @concept differential-privacy
 * @example
 * ```ts
 * import { clipPerExample } from 'omai-engine/learn'
 * import { tensor } from 'omai-engine/tensor'
 * clipPerExample({ w: tensor([[3, 4], [0.3, 0.4]]) }, 1).clippedShare // => 0.5
 * ```
 */
export function clipPerExample(grads: Params, clipNorm: number): Clipped {
  if (!(clipNorm > 0)) throw new OmaiError('value', 'clipPerExample: the clip norm must be positive')
  const norms = exampleNorms(grads)
  const factor = minimum(div(clipNorm, add(norms, 1e-6)), 1)
  const n = norms.shape[0] ?? 0
  const clipped = treeMap((leaf) => mul(leaf, reshape(factor, [n, ...leaf.shape.slice(1).map(() => 1)])), grads)
  const values = toFlat(norms)
  return { grads: clipped, norms: values, clippedShare: values.filter((v) => v > clipNorm).length / Math.max(1, n) }
}

/** Options of the private aggregation. */
export interface DpOptions {
  /** The per-example norm bound C. */
  readonly clipNorm: number
  /** The noise standard deviation over C (sigma). */
  readonly noiseMultiplier: number
  /** The divisor of the noisy sum: the expected batch size (default the batch's size). */
  readonly expectedBatchSize?: number
}

/**
 * The privatised mean gradient: sum the clipped per-example gradients, add N(0, (sigma C)^2) to every coordinate, and
 * divide by the expected batch size.
 *
 * @param clipped - Clipped per-example gradients (leading example axis).
 * @param options - Clip norm, noise multiplier and expected batch size.
 * @param key - The noise key.
 * @returns The private gradient (no example axis).
 * @concept differential-privacy
 * @example
 * ```ts
 * import { privateGradient } from 'omai-engine/learn'
 * import { key } from 'omai-engine/random'
 * import { tensor } from 'omai-engine/tensor'
 * privateGradient({ w: tensor([[1, 0], [1, 0]]) }, { clipNorm: 1, noiseMultiplier: 0 }, key(0)).w.data // => Float64Array.of(1, 0)
 * ```
 */
export function privateGradient(clipped: Params, options: DpOptions, key: Key): Params {
  const n = treeFlatten(clipped).leaves[0]?.shape[0] ?? 0
  return noisyMean(
    treeMap((leaf) => sum(leaf, 0), clipped) as Params,
    options,
    key,
    options.expectedBatchSize ?? Math.max(1, n),
  ).gradient
}

/** Add N(0, (sigma C)^2) to a summed gradient and divide by the batch size; also returns the noise. */
function noisyMean(total: Params, options: DpOptions, key: Key, divisor: number): { gradient: Params; noise: Params } {
  const { leaves, def } = treeFlatten(total)
  const noise = leaves.map((leaf, i) =>
    mul(normal(foldIn(key, i), [...leaf.shape]), options.noiseMultiplier * options.clipNorm),
  )
  const gradient = leaves.map((leaf, i) => div(add(leaf, noise[i] as Value), divisor))
  return { gradient: treeUnflatten(def, gradient) as Params, noise: treeUnflatten(def, noise) as Params }
}

/** The state of {@link dpGradientTransformation}. */
export interface DpNoiseState {
  readonly count: number
}

/**
 * The noise stage of DP-SGD as a gradient transformation: its input is the SUM of the clipped per-example gradients; it
 * adds N(0, (sigma C)^2) noise to every coordinate and divides by the expected batch size, drawing a fresh key from the
 * step count. Chain it before any optimiser (`chain(dpGradientTransformation(...), adam(...))`); the clipping is
 * {@link clipPerExample}, which needs the per-example gradients.
 *
 * @param options - Clip norm, noise multiplier, the expected batch size (default 1) and the key.
 * @returns The transformation.
 * @concept differential-privacy
 * @example
 * ```ts
 * import { dpGradientTransformation } from 'omai-engine/learn'
 * import { key } from 'omai-engine/random'
 * import { tensor } from 'omai-engine/tensor'
 * const t = dpGradientTransformation({ clipNorm: 1, noiseMultiplier: 0, expectedBatchSize: 2, key: key(0) })
 * t.update(tensor([2, 4]), t.init(tensor([0, 0]))).updates // => tensor([1, 2])
 * ```
 */
export function dpGradientTransformation(
  options: DpOptions & { readonly key: Key },
): GradientTransformation<DpNoiseState> {
  return {
    name: 'dpNoise',
    init: () => ({ count: 0 }),
    update: (g, state) => {
      const { leaves, def } = treeFlatten(g)
      const divisor = options.expectedBatchSize ?? 1
      const k = foldIn(options.key, state.count)
      const out = leaves.map((leaf, i) =>
        div(add(leaf, mul(normal(foldIn(k, i), [...leaf.shape]), options.noiseMultiplier * options.clipNorm)), divisor),
      )
      return { updates: treeUnflatten(def, out), state: { count: state.count + 1 } }
    },
  }
}

/** What one DP-SGD step reports besides the next state. */
export interface DpStepReport {
  readonly state: TrainState
  /** Mean loss of the batch. */
  readonly loss: number
  /** The share of examples clipped. */
  readonly clippedShare: number
  /** The norm of the noise added (before dividing by the batch size). */
  readonly noiseNorm: number
  readonly batchSize: number
}

/**
 * One DP-SGD step on a batch: per-example gradients, clipping, noise, then the optimiser, over a `TrainState` (the key
 * of the noise is the step's). The batch should be a Poisson sample of the data at the accountant's sampling rate for
 * the accounted guarantee to hold; `dpSgdAlgorithm` draws it so.
 *
 * @param model - The module.
 * @param state - The training state.
 * @param x - The batch inputs.
 * @param y - The batch targets.
 * @param loss - The per-example loss.
 * @param optimizer - The optimiser the state was built with.
 * @param options - Clip norm, noise multiplier and expected batch size.
 * @returns The next state and the step's report.
 * @concept differential-privacy
 * @example
 * ```ts
 * import { dpSgdStep } from 'omai-engine/learn'
 * import { Linear, trainState } from 'omai-engine/nn'
 * import { sgd } from 'omai-engine/optim'
 * import { sum } from 'omai-engine/tensor'
 * const m = new Linear(2, 1)
 * const opt = sgd({ learningRate: 0.1 })
 * dpSgdStep(m, trainState(m, opt), [[1, 2], [3, 4]], [[0], [0]], (o) => sum(o), opt, { clipNorm: 1, noiseMultiplier: 1 }).state.step // => 1
 * ```
 */
export function dpSgdStep(
  model: Module<Value, Value>,
  state: TrainState,
  x: TensorLike,
  y: TensorLike,
  loss: ExampleLoss,
  optimizer: GradientTransformation,
  options: DpOptions,
): DpStepReport {
  const xs = asTensor(x)
  const n = xs.shape[0] ?? 0
  let total: Params = treeMap((p) => mul(p, 0), state.params)
  let losses: readonly number[] = []
  let clippedShare = 0
  if (n > 0) {
    const per = perExampleGradients(model, state.params, xs, y, loss, { buffers: state.buffers })
    const clipped = clipPerExample(per.grads, options.clipNorm)
    total = treeMap((leaf) => sum(leaf, 0), clipped.grads) as Params
    losses = per.losses
    clippedShare = clipped.clippedShare
  }
  const { gradient, noise } = noisyMean(
    total,
    options,
    foldIn(state.key, state.step),
    options.expectedBatchSize ?? Math.max(1, n),
  )
  const applied = applyGradients(state, gradient, optimizer, { mask: model.trainableMask() })
  const meanLoss = losses.length === 0 ? Number.NaN : losses.reduce((a, b) => a + b, 0) / losses.length
  return {
    state: { ...applied.state, step: state.step + 1, loss: meanLoss, gradNorm: item(globalNorm(gradient)) },
    loss: meanLoss,
    clippedShare,
    noiseNorm: item(globalNorm(noise)),
    batchSize: n,
  }
}

/**
 * The epsilon that DP-SGD spends at delta: the Renyi accountant (`subsampledGaussianRdp`, converted by `rdpToDp`) for a
 * sampling rate, noise multiplier and number of steps.
 *
 * @param options - Sampling rate, noise multiplier, steps, delta and orders.
 * @returns epsilon and the best order.
 * @concept differential-privacy
 * @example
 * ```ts
 * import { dpEpsilon } from 'omai-engine/learn'
 * // Abadi et al.'s MNIST setting, q = 0.01, sigma = 4, 10000 steps.
 * dpEpsilon({ sampleRate: 0.01, noiseMultiplier: 4, steps: 10000, delta: 1e-5 }).epsilon < 2 // => true
 * ```
 */
export function dpEpsilon(options: {
  readonly sampleRate: number
  readonly noiseMultiplier: number
  readonly steps: number
  readonly delta: number
  readonly orders?: readonly number[]
}): { readonly epsilon: number; readonly order: number } {
  const orders = options.orders ?? DEFAULT_RDP_ORDERS
  return rdpToDp(
    orders,
    subsampledGaussianRdp(options.sampleRate, options.noiseMultiplier, options.steps, orders),
    options.delta,
  )
}

/**
 * The smallest noise multiplier whose DP-SGD run stays within a target epsilon (opacus's get_noise_multiplier):
 * bisection on sigma over the accountant, which is decreasing in sigma.
 *
 * @param options - Target epsilon, delta, sampling rate, steps; the tolerance on epsilon (default 0.01).
 * @returns The noise multiplier.
 * @throws OmaiError with code `convergence` when no sigma up to 1000 reaches the target.
 * @concept differential-privacy
 * @example
 * ```ts
 * import { noiseMultiplierFor, dpEpsilon } from 'omai-engine/learn'
 * const sigma = noiseMultiplierFor({ targetEpsilon: 3, delta: 1e-5, sampleRate: 0.01, steps: 1000 })
 * dpEpsilon({ sampleRate: 0.01, noiseMultiplier: sigma, steps: 1000, delta: 1e-5 }).epsilon <= 3 // => true
 * ```
 */
export function noiseMultiplierFor(options: {
  readonly targetEpsilon: number
  readonly delta: number
  readonly sampleRate: number
  readonly steps: number
  readonly epsilonTolerance?: number
  readonly orders?: readonly number[]
}): number {
  const eps = (sigma: number): number => dpEpsilon({ ...options, noiseMultiplier: sigma }).epsilon
  let hi = 1
  while (eps(hi) > options.targetEpsilon) {
    hi *= 2
    if (hi > 1000)
      throw new OmaiError('convergence', 'noiseMultiplierFor: no noise multiplier up to 1000 reaches the target')
  }
  let lo = hi / 2
  const tol = options.epsilonTolerance ?? 0.01
  for (let i = 0; i < 60; i++) {
    const mid = (lo + hi) / 2
    const e = eps(mid)
    if (e > options.targetEpsilon) lo = mid
    else {
      hi = mid
      if (options.targetEpsilon - e < tol) break
    }
  }
  return hi
}

/** The paths of {@link dpSgdAlgorithm}. */
export type DpSgdPath =
  | 'training/loss'
  | 'training/clippedShare'
  | 'training/noiseNorm'
  | 'training/batchSize'
  | 'budget/epsilon'
  | 'budget/rdp'

/** What a DP-SGD run logs. */
export const dpSgdContract: Contract<DpSgdState, DpSgdPath> = defineContract({
  'training/loss': scalar({ description: 'mean per-example loss of the Poisson batch' }),
  'training/clippedShare': scalar({ description: 'share of the batch whose gradient was clipped' }),
  'training/noiseNorm': scalar({ description: 'norm of the Gaussian noise added to the summed gradient' }),
  'training/batchSize': scalar({ description: 'size of the Poisson batch' }),
  'budget/epsilon': scalar({ description: 'epsilon spent so far at the target delta' }),
  'budget/rdp': tensorSource({ role: 'state', description: 'the accumulated Renyi DP at each order' }),
})

/** The state of a DP-SGD run: the training state and what has been spent. */
export interface DpSgdState extends TrainState {
  readonly epsilon: number
  readonly rdp: readonly number[]
  readonly clippedShare: number
  readonly noiseNorm: number
  readonly batchSize: number
}

/** Options of {@link dpSgdAlgorithm}. */
export interface DpSgdOptions extends DpOptions {
  /** The per-example loss. */
  readonly loss: ExampleLoss
  readonly optimizer: GradientTransformation
  /** The Poisson sampling rate q: each row joins a batch with this probability. */
  readonly sampleRate: number
  /** Steps to run. */
  readonly steps: number
  /** The delta epsilon is reported at. */
  readonly delta: number
  readonly orders?: readonly number[]
}

/**
 * DP-SGD as a stepped algorithm: each step draws a Poisson batch of the rows (each included with probability q),
 * clips and noises the gradients, takes the optimiser step, and adds the step's Renyi DP to the account, so the
 * epsilon spent at delta is on the tape beside the loss, the clipped share and the noise. The expected batch size
 * q n divides the noisy sum, as the accountant assumes.
 *
 * @param model - The module.
 * @param data - Inputs and targets.
 * @param options - Loss, optimiser, q, clip norm, noise multiplier, steps and delta.
 * @returns The algorithm.
 * @concept differential-privacy
 * @example
 * ```ts
 * import { dpSgdAlgorithm } from 'omai-engine/learn'
 * import { Linear } from 'omai-engine/nn'
 * import { sgd } from 'omai-engine/optim'
 * import { runAlgorithm } from 'omai-engine/trace'
 * import { sum, square, sub } from 'omai-engine/tensor'
 * const alg = dpSgdAlgorithm(new Linear(1, 1), { x: [[1], [2], [3], [4]], y: [[1], [2], [3], [4]] }, {
 *   loss: (o, t) => sum(square(sub(o, t))), optimizer: sgd({ learningRate: 0.01 }), sampleRate: 0.5, clipNorm: 1, noiseMultiplier: 1, steps: 5, delta: 1e-5,
 * })
 * runAlgorithm(alg, { maxSteps: 5 }).state.epsilon > 0 // => true
 * ```
 */
export function dpSgdAlgorithm(
  model: Module<Value, Value>,
  data: { readonly x: TensorLike; readonly y: TensorLike },
  options: DpSgdOptions,
): Algorithm<DpSgdState, DpSgdPath> {
  const x = asTensor(data.x) as Tensor
  const y = asTensor(data.y) as Tensor
  const n = x.shape[0] ?? 0
  const orders = options.orders ?? DEFAULT_RDP_ORDERS
  const perStep = subsampledGaussianRdp(options.sampleRate, options.noiseMultiplier, 1, orders)
  const expected = options.expectedBatchSize ?? Math.max(1, options.sampleRate * n)
  const rows = (t: Tensor, picked: readonly number[]): Tensor => {
    const d = t.data.length / n
    const out = new Float64Array(picked.length * d)
    for (const [i, r] of picked.entries()) out.set(t.data.subarray(r * d, (r + 1) * d), i * d)
    return fromStorage(out, [picked.length, ...t.shape.slice(1)], 'float64')
  }
  const optimizer = options.optimizer
  return {
    name: 'dpSgd',
    config: {
      sampleRate: options.sampleRate,
      clipNorm: options.clipNorm,
      noiseMultiplier: options.noiseMultiplier,
      steps: options.steps,
      delta: options.delta,
      optimizer: optimizer.name,
    },
    init: (key) => ({
      ...trainState(model, optimizer, key),
      epsilon: 0,
      rdp: orders.map(() => 0),
      clippedShare: 0,
      noiseNorm: 0,
      batchSize: 0,
    }),
    step: (s, ctx) => {
      const draw = toFlat(bernoulli(foldIn(ctx.key, 0), options.sampleRate, [n]))
      const picked = draw.flatMap((v, i) => (v ? [i] : []))
      const r = dpSgdStep(
        model,
        { ...s, key: foldIn(ctx.key, 1), step: s.step },
        rows(x, picked),
        rows(y, picked),
        options.loss,
        optimizer,
        { clipNorm: options.clipNorm, noiseMultiplier: options.noiseMultiplier, expectedBatchSize: expected },
      )
      const rdp = s.rdp.map((v, k) => v + (perStep[k] ?? 0))
      const epsilon = rdpToDp(orders, rdp, options.delta).epsilon
      ctx.log('training/loss', r.loss)
      ctx.log('training/clippedShare', r.clippedShare)
      ctx.log('training/noiseNorm', r.noiseNorm)
      ctx.log('training/batchSize', r.batchSize)
      ctx.log('budget/epsilon', epsilon)
      ctx.log('budget/rdp', rdp)
      return {
        ...r.state,
        key: s.key,
        epsilon,
        rdp,
        clippedShare: r.clippedShare,
        noiseNorm: r.noiseNorm,
        batchSize: r.batchSize,
      }
    },
    done: (s) => (s.step >= options.steps ? 'completed' : undefined),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as DpSgdState,
    observe: (s) => ({ metrics: { epsilon: s.epsilon, clippedShare: s.clippedShare, noiseNorm: s.noiseNorm } }),
  }
}
