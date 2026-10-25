/**
 * Mixture-of-experts layers: a router sends each token to a few of many feed-forward experts, so capacity grows without
 * the cost of running every expert. The routing rules (dense softmax, top-k, noisy top-k, Switch and expert choice),
 * the auxiliary load-balancing, router z- and importance losses, and the `MixtureOfExperts` layer that returns its
 * output with the routing and losses.
 */

import { MdmlError } from '../../base/index.ts'
import { foldIn, type Key, normal } from '../../random/index.ts'
import {
  add,
  argsort,
  div,
  fromStorage,
  logsumexp,
  mean,
  mul,
  primalOf,
  promoteFloat,
  reshape,
  scatterAdd,
  softmax,
  square,
  sum,
  type Tensor,
  type TensorLike,
  take,
  toFlat,
  type Value,
  variance,
} from '../../tensor/index.ts'
import { softplus } from '../functional/activations.ts'
import { Composite, ModuleList } from '../module/containers.ts'
import { Linear } from '../module/layers.ts'
import { type ApplyContext, initKey, isTraining, type Module, type Params } from '../module/module.ts'
import { FeedForward, GatedFeedForward } from './transformer.ts'

const fl = (x: TensorLike): Value => promoteFloat(x)[0] as Value

/** A routing decision for T tokens and E experts. */
export interface Routing {
  /** Combine weights `[T, E]`, zero where a token is not sent to an expert. */
  readonly weights: Value
  /** The router's probabilities `[T, E]` (softmax of the logits). */
  readonly probs: Value
  /** 1 where a token is dispatched to an expert, `[T, E]` (concrete). */
  readonly dispatch: Tensor
}

const dispatchOf = (rows: readonly (readonly number[])[], t: number, e: number): Tensor => {
  const d = new Float64Array(t * e)
  for (const [i, experts] of rows.entries()) for (const j of experts) d[i * e + j] = 1
  return fromStorage(d, [t, e], 'float64')
}

/** The k largest entries of each row of a concrete matrix. */
const topRows = (scores: Tensor, k: number): number[][] => {
  const [t = 0, e = 0] = scores.shape
  const order = toFlat(argsort(scores, -1, true))
  return Array.from({ length: t }, (_, i) => order.slice(i * e, i * e + k))
}

/**
 * Dense softmax routing (soft mixtures): every token goes to every expert with its router probability.
 *
 * @param logits - Router logits `[T, E]`.
 * @returns The routing.
 * @concept mixture-of-experts
 * @example
 * ```ts
 * import { blocks } from 'mdml-engine/nn'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(blocks.softmaxRouting([[0, 0]]).weights) // => [[0.5, 0.5]]
 * ```
 */
export function softmaxRouting(logits: TensorLike): Routing {
  const z = fl(logits)
  const probs = softmax(z, -1)
  const [t = 0, e = 0] = z.shape
  return { weights: probs, probs, dispatch: fromStorage(new Float64Array(t * e).fill(1), [t, e], 'float64') }
}

/**
 * Top-k routing (GShard, Mixtral): each token goes to its k most probable experts, with the probabilities renormalised
 * over those k (the softmax of the kept logits).
 *
 * @param logits - Router logits `[T, E]`.
 * @param k - Experts per token.
 * @param options - `normalise` (default true) the kept weights to sum to one.
 * @returns The routing.
 * @concept mixture-of-experts
 * @example
 * ```ts
 * import { blocks } from 'mdml-engine/nn'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(blocks.topKRouting([[2, 0, 1]], 1).weights) // => [[1, 0, 0]]
 * ```
 */
export function topKRouting(logits: TensorLike, k: number, options: { readonly normalise?: boolean } = {}): Routing {
  const z = fl(logits)
  const [t = 0, e = 0] = z.shape
  if (k < 1 || k > e) throw new MdmlError('value', `topKRouting: k must be in [1, ${e}]`)
  const probs = softmax(z, -1)
  const dispatch = dispatchOf(topRows(primalOf(z), k), t, e)
  let weights = mul(probs, dispatch)
  // The kept probabilities are positive, so each row sum is too.
  if (options.normalise ?? true) weights = div(weights, sum(weights, -1, true))
  return { weights, probs, dispatch }
}

/**
 * Noisy top-k gating (Shazeer et al. 2017): in training, add Gaussian noise with a learnt, input-dependent scale
 * softplus(noise logits) to the logits before choosing the top k, which spreads the load.
 *
 * @param logits - Router logits `[T, E]`.
 * @param noiseLogits - `[T, E]`, the noise scale before the softplus.
 * @param k - Experts per token.
 * @param options - The key (needed in training) and the training flag.
 * @returns The routing (the noisy logits decide and weight).
 * @reference Shazeer et al. (2017), Outrageously large neural networks: the sparsely-gated mixture-of-experts layer.
 * @example
 * ```ts
 * import { blocks } from 'mdml-engine/nn'
 * blocks.noisyTopKRouting([[2, 0, 1]], [[0, 0, 0]], 2, { training: false }).dispatch.shape // => [1, 3]
 * ```
 */
export function noisyTopKRouting(
  logits: TensorLike,
  noiseLogits: TensorLike,
  k: number,
  options: { readonly key?: Key; readonly training?: boolean } = {},
): Routing {
  const z = fl(logits)
  if (options.training === false) return topKRouting(z, k)
  if (options.key === undefined) throw new MdmlError('value', 'noisyTopKRouting: training needs a key')
  const noisy = add(z, mul(normal(options.key, z.shape), softplus(fl(noiseLogits))))
  return topKRouting(noisy, k)
}

/**
 * Switch routing (Fedus et al. 2022): each token goes to its most probable expert, weighted by that probability; an
 * expert takes at most capacity = ceil(capacityFactor T / E) tokens, the rest are dropped (they pass through the
 * residual connection of the surrounding block).
 *
 * @param logits - Router logits `[T, E]`.
 * @param capacityFactor - Capacity slack (default 1.25).
 * @returns The routing.
 * @reference Fedus, Zoph and Shazeer (2022), Switch transformers.
 * @example
 * ```ts
 * import { blocks } from 'mdml-engine/nn'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(blocks.switchRouting([[1, 0], [1, 0], [1, 0]], 0.5).dispatch) // => [[1, 0], [0, 0], [0, 0]]
 * ```
 */
export function switchRouting(logits: TensorLike, capacityFactor = 1.25): Routing {
  const z = fl(logits)
  const [t = 0, e = 0] = z.shape
  const probs = softmax(z, -1)
  const capacity = Math.ceil((capacityFactor * t) / e)
  const used = new Array<number>(e).fill(0)
  const rows = topRows(primalOf(z), 1).map(([j = 0]) => {
    if ((used[j] ?? 0) >= capacity) return []
    used[j] = (used[j] ?? 0) + 1
    return [j]
  })
  const dispatch = dispatchOf(rows, t, e)
  return { weights: mul(probs, dispatch), probs, dispatch }
}

/**
 * Expert-choice routing (Zhou et al. 2022): each expert picks the `capacity` tokens with the highest affinity to it,
 * so every expert is fully loaded and a token may go to any number of experts.
 *
 * @param logits - Router logits `[T, E]`.
 * @param capacity - Tokens per expert (default ceil(T / E)).
 * @returns The routing (weights are the token-to-expert softmax at the chosen pairs).
 * @reference Zhou et al. (2022), Mixture-of-experts with expert choice routing.
 * @example
 * ```ts
 * import { blocks } from 'mdml-engine/nn'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(blocks.expertChoiceRouting([[3, 0], [2, 1], [0, 3]], 1).dispatch) // => [[1, 0], [0, 0], [0, 1]]
 * ```
 */
export function expertChoiceRouting(logits: TensorLike, capacity?: number): Routing {
  const z = fl(logits)
  const [t = 0, e = 0] = z.shape
  const c = Math.min(t, capacity ?? Math.ceil(t / e))
  const probs = softmax(z, -1)
  const p = toFlat(primalOf(probs))
  const d = new Float64Array(t * e)
  for (let j = 0; j < e; j++) {
    const order = Array.from({ length: t }, (_, i) => i).sort(
      (a, b) => (p[b * e + j] ?? 0) - (p[a * e + j] ?? 0) || a - b,
    )
    for (const i of order.slice(0, c)) d[i * e + j] = 1
  }
  const dispatch = fromStorage(d, [t, e], 'float64')
  return { weights: mul(probs, dispatch), probs, dispatch }
}

/**
 * The Switch Transformer's load-balancing loss: E sum_e f_e P_e, with f_e the fraction of tokens dispatched to expert
 * e and P_e its mean router probability; 1 when the load is uniform, larger when it is not.
 *
 * @param routing - A routing decision.
 * @returns The loss (differentiable in the router probabilities).
 * @example
 * ```ts
 * import { blocks } from 'mdml-engine/nn'
 * import { item } from 'mdml-engine/tensor'
 * item(blocks.loadBalancingLoss(blocks.topKRouting([[1, 0], [0, 1]], 1))) // => 1
 * ```
 */
export function loadBalancingLoss(routing: Routing): Value {
  const e = routing.probs.shape[1] ?? 1
  const f = mean(routing.dispatch, 0)
  const total = sum(f)
  const fraction = mul(f, div(1, total))
  return mul(sum(mul(fraction, mean(routing.probs, 0))), e)
}

/**
 * The router z-loss of ST-MoE (Zoph et al. 2022): the mean squared log-partition of the router logits, which keeps
 * them small and the routing numerically stable.
 *
 * @param logits - Router logits `[T, E]`.
 * @returns The loss.
 * @reference Zoph et al. (2022), ST-MoE: designing stable and transferable sparse expert models.
 * @example
 * ```ts
 * import { blocks } from 'mdml-engine/nn'
 * import { item } from 'mdml-engine/tensor'
 * item(blocks.routerZLoss([[0, 0]])) // => 0.4804530139182014
 * ```
 */
export function routerZLoss(logits: TensorLike): Value {
  return mean(square(logsumexp(fl(logits), -1)))
}

/**
 * The importance loss of Shazeer et al. (2017): the squared coefficient of variation of each expert's total gate
 * weight over the batch.
 *
 * @param routing - A routing decision.
 * @returns The loss.
 * @example
 * ```ts
 * import { blocks } from 'mdml-engine/nn'
 * import { item } from 'mdml-engine/tensor'
 * item(blocks.importanceLoss(blocks.softmaxRouting([[0, 0]]))) // => 0
 * ```
 */
export function importanceLoss(routing: Routing): Value {
  const importance = sum(routing.weights, 0)
  return div(variance(importance), square(mean(importance)))
}

/** How a {@link MixtureOfExperts} routes. */
export type RoutingKind = 'softmax' | 'topk' | 'noisy-topk' | 'switch' | 'expert-choice'

/** Options of {@link MixtureOfExperts}. */
export interface MixtureOfExpertsOptions {
  /** Experts per token for top-k routing (default 2). */
  readonly k?: number
  readonly routing?: RoutingKind
  /** Switch routing's capacity factor (default 1.25), or expert-choice's tokens per expert. */
  readonly capacityFactor?: number
  /** Expert hidden width (default 4 dim). */
  readonly hidden?: number
  /** Gated (SwiGLU) experts (default true). */
  readonly gated?: boolean
  /** Custom experts instead of feed-forward blocks. */
  readonly expert?: (i: number) => Module
  readonly key?: Key
}

/** The output of a {@link MixtureOfExperts} layer: the output, the routing and its auxiliary losses. */
export interface MixtureOfExpertsOutput {
  readonly output: Value
  readonly routing: Routing
  /** The load-balancing loss (add a small multiple to the training loss). */
  readonly balanceLoss: Value
  readonly zLoss: Value
  /** Tokens dispatched to each expert. */
  readonly load: readonly number[]
}

/**
 * A sparse mixture-of-experts layer (Shazeer et al. 2017; Switch, GShard, Mixtral): a linear router (`gate`) scores
 * the experts for each token, the routing chooses some, and each expert runs only on the tokens dispatched to it
 * (gathered, processed, and scattered back with their weights), so cost grows with k, not with the number of experts.
 * Returns the load-balancing and z losses for training.
 *
 * @concept mixture-of-experts
 * @example
 * ```ts
 * import { blocks } from 'mdml-engine/nn'
 * import { zeros } from 'mdml-engine/tensor'
 * new blocks.MixtureOfExperts(8, 4, { k: 2 }).forward(zeros([2, 3, 8])).output.shape // => [2, 3, 8]
 * ```
 */
export class MixtureOfExperts extends Composite<Value, MixtureOfExpertsOutput> {
  readonly kind = 'MixtureOfExperts'
  readonly dim: number
  readonly numExperts: number
  readonly k: number
  readonly routing: RoutingKind
  readonly capacityFactor: number

  constructor(dim: number, numExperts: number, options: MixtureOfExpertsOptions = {}) {
    const key = initKey('MixtureOfExperts', options.key)
    const routing = options.routing ?? 'topk'
    const hidden = options.hidden ?? 4 * dim
    const make =
      options.expert ??
      ((i: number): Module =>
        (options.gated ?? true)
          ? new GatedFeedForward(dim, hidden, { key: foldIn(foldIn(key, 2), i) })
          : new FeedForward(dim, hidden, { key: foldIn(foldIn(key, 2), i) }))
    super({
      gate: new Linear(dim, numExperts, { bias: false, key: foldIn(key, 0) }),
      ...(routing === 'noisy-topk' ? { noise: new Linear(dim, numExperts, { bias: false, key: foldIn(key, 1) }) } : {}),
      experts: new ModuleList(Array.from({ length: numExperts }, (_, i) => make(i))),
    })
    this.dim = dim
    this.numExperts = numExperts
    this.k = options.k ?? 2
    this.routing = routing
    this.capacityFactor = options.capacityFactor ?? 1.25
  }

  /** The routing of a token matrix `[T, D]`. */
  route(params: Params, tokens: Value, ctx: ApplyContext = {}): Routing {
    const p = this.parts(params, ctx)
    const logits = p.run<Value, Value>('gate', tokens)
    switch (this.routing) {
      case 'softmax':
        return softmaxRouting(logits)
      case 'switch':
        return switchRouting(logits, this.capacityFactor)
      case 'expert-choice':
        return expertChoiceRouting(logits, Math.ceil(((tokens.shape[0] ?? 0) / this.numExperts) * this.capacityFactor))
      case 'noisy-topk':
        return noisyTopKRouting(logits, p.run<Value, Value>('noise', tokens), this.k, {
          training: isTraining(ctx),
          ...(ctx.key === undefined ? {} : { key: foldIn(ctx.key, 7) }),
        })
      default:
        return topKRouting(logits, this.k)
    }
  }

  apply(params: Params, x: Value, ctx: ApplyContext = {}): MixtureOfExpertsOutput {
    const shape = x.shape
    const tokens = reshape(x, [-1, this.dim])
    const t = tokens.shape[0] ?? 0
    const routing = this.route(params, tokens, ctx)
    const p = this.parts(params, ctx)
    const experts = this.part<ModuleList<Module>>('experts')
    const ep = (params['experts'] ?? {}) as Params
    const listCtx = p.context('experts')
    const d = toFlat(routing.dispatch)
    const load: number[] = []
    let out: Value | undefined
    for (const [j, expert] of experts.modules.entries()) {
      const rows: number[] = []
      for (let i = 0; i < t; i++) if ((d[i * this.numExperts + j] ?? 0) > 0) rows.push(i)
      load.push(rows.length)
      if (rows.length === 0) continue
      const idx = fromStorage(Int32Array.from(rows), [rows.length], 'int32')
      const y = expert.apply((ep[String(j)] ?? {}) as Params, take(tokens, idx, 0), {
        ...listCtx,
        ...(listCtx.key === undefined ? {} : { key: foldIn(listCtx.key, j) }),
      })
      const w = reshape(take(take(routing.weights, [j], 1), idx, 0), [rows.length, 1])
      const scatterIdx = fromStorage(
        Int32Array.from(rows.flatMap((r) => new Array<number>(this.dim).fill(r))),
        [rows.length, this.dim],
        'int32',
      )
      const placed = scatterAdd([t, this.dim], scatterIdx, mul(y, w), 0)
      out = out === undefined ? placed : add(out, placed)
    }
    const logits = p.run<Value, Value>('gate', tokens)
    p.done()
    return {
      output: reshape(out ?? mul(tokens, 0), shape),
      routing,
      balanceLoss: loadBalancingLoss(routing),
      zLoss: routerZLoss(logits),
      load,
    }
  }

  outputShape(s: readonly number[]): number[] {
    return [...s]
  }

  override extraRepr(): string {
    return `dim=${this.dim}, experts=${this.numExperts}, routing=${this.routing}${this.routing.includes('topk') ? `, k=${this.k}` : ''}`
  }
}
