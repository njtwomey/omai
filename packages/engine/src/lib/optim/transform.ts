import { WhiteboxError } from '../base/index.ts'
import {
  add,
  clip,
  div,
  isValue,
  item,
  maximum,
  mul,
  sqrt,
  square,
  sub,
  sum,
  tensor,
  treeFlatten,
  treeLeaves,
  treeMap,
  treeUnflatten,
  treeZeros,
  type Value,
  zerosLike,
} from '../tensor/index.ts'

/** A tree of values (parameters, gradients or updates): nested records and arrays with tensor leaves. */
export type Tree = unknown

/** A learning rate: a constant or a function of the step count (from 0). */
export type LearningRate = number | ((step: number) => number)

/**
 * A gradient transformation (optax's design): `init` builds a state from the parameters, and `update` maps gradients
 * to updates, threading the state. Optimisers are chains of transformations; parameters move by
 * {@link applyUpdates}.
 *
 * @concept optimisers
 */
export interface GradientTransformation<S = unknown> {
  readonly name: string
  init(params: Tree): S
  update(grads: Tree, state: S, params?: Tree): { readonly updates: Tree; readonly state: S }
}

const rate = (lr: LearningRate, step: number): number => (typeof lr === 'number' ? lr : lr(step))

/**
 * Add updates to parameters.
 *
 * @param params - The parameters.
 * @param updates - The updates (already scaled by the learning rate).
 * @returns The new parameters.
 * @example
 * ```ts
 * import { applyUpdates } from 'whitebox-engine/optim'
 * import { tensor, toArray } from 'whitebox-engine/tensor'
 * toArray((applyUpdates({ w: tensor([1, 2]) }, { w: tensor([0.5, -1]) }) as { w: never }).w) // => [1.5, 1]
 * ```
 */
export function applyUpdates(params: Tree, updates: Tree): Tree {
  return treeMap((p, u) => add(p, u), params, updates)
}

/**
 * Compose transformations: each one's updates are the next one's gradients.
 *
 * @param transformations - The transformations, applied in order.
 * @returns The chained transformation.
 * @example
 * ```ts
 * import { chain, clipByGlobalNorm, scaleByLearningRate } from 'whitebox-engine/optim'
 * chain(clipByGlobalNorm(1), scaleByLearningRate(0.1)).name // => 'chain(clipByGlobalNorm, scaleByLearningRate)'
 * ```
 */
export function chain(...transformations: readonly GradientTransformation[]): GradientTransformation<unknown[]> {
  return {
    name: `chain(${transformations.map((t) => t.name).join(', ')})`,
    init: (params) => transformations.map((t) => t.init(params)),
    update: (grads, state, params) => {
      let updates = grads
      const next = transformations.map((t, i) => {
        const r = t.update(updates, state[i], params)
        updates = r.updates
        return r.state
      })
      return { updates, state: next }
    },
  }
}

/**
 * Multiply by -learningRate(step): the last link of a descent chain.
 *
 * @param learningRate - A constant or a schedule.
 * @returns The transformation.
 * @example
 * ```ts
 * import { scaleByLearningRate } from 'whitebox-engine/optim'
 * import { tensor, toArray } from 'whitebox-engine/tensor'
 * const t = scaleByLearningRate(0.1)
 * toArray(t.update(tensor([2]), t.init(tensor([0]))).updates as never) // => [-0.2]
 * ```
 */
export function scaleByLearningRate(learningRate: LearningRate): GradientTransformation<{ readonly count: number }> {
  return {
    name: 'scaleByLearningRate',
    init: () => ({ count: 0 }),
    update: (grads, state) => {
      const lr = rate(learningRate, state.count)
      return { updates: treeMap((g) => mul(g, -lr), grads), state: { count: state.count + 1 } }
    },
  }
}

/**
 * The global L2 norm of a tree: sqrt of the sum of squares of every leaf.
 *
 * @param tree - A tree of values.
 * @returns The norm (a 0-d value).
 * @example
 * ```ts
 * import { globalNorm } from 'whitebox-engine/optim'
 * import { item, tensor } from 'whitebox-engine/tensor'
 * item(globalNorm({ a: tensor([3]), b: tensor([4]) })) // => 5
 * ```
 */
export function globalNorm(tree: Tree): Value {
  const squares = treeLeaves(tree).map((leaf) => sum(square(leaf)))
  return sqrt(squares.reduce((acc, x) => add(acc, x), squares.length === 0 ? tensor(0) : (squares.shift() as Value)))
}

/**
 * Rescale gradients whose global norm exceeds `maxNorm` to have norm `maxNorm` (torch.nn.utils.clip_grad_norm_).
 *
 * @param maxNorm - The largest allowed norm.
 * @returns The transformation.
 * @example
 * ```ts
 * import { clipByGlobalNorm } from 'whitebox-engine/optim'
 * import { tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(clipByGlobalNorm(1).update(tensor([3, 4]), undefined).updates as never).map((v) => Math.round(v * 1e12) / 1e12) // => [0.6, 0.8]
 * ```
 */
export function clipByGlobalNorm(maxNorm: number): GradientTransformation<undefined> {
  return {
    name: 'clipByGlobalNorm',
    init: () => undefined,
    update: (grads, state) => {
      const norm = item(globalNorm(grads))
      const factor = norm > maxNorm ? maxNorm / norm : 1
      return { updates: factor === 1 ? grads : treeMap((g) => mul(g, factor), grads), state }
    },
  }
}

/**
 * Clip each gradient entry to [-maxValue, maxValue] (torch.nn.utils.clip_grad_value_).
 *
 * @param maxValue - The bound.
 * @returns The transformation.
 * @example
 * ```ts
 * import { clipByValue } from 'whitebox-engine/optim'
 * import { tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(clipByValue(1).update(tensor([3, -0.5]), undefined).updates as never) // => [1, -0.5]
 * ```
 */
export function clipByValue(maxValue: number): GradientTransformation<undefined> {
  return {
    name: 'clipByValue',
    init: () => undefined,
    update: (grads, state) => ({ updates: treeMap((g) => clip(g, -maxValue, maxValue), grads), state }),
  }
}

/**
 * Add weightDecay times the parameters to the gradients (L2 regularisation when placed before the optimiser's
 * scaling, decoupled weight decay as in AdamW when placed after it).
 *
 * @param weightDecay - The coefficient.
 * @returns The transformation (needs `params` in `update`).
 * @example
 * ```ts
 * import { addDecayedWeights } from 'whitebox-engine/optim'
 * import { tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(addDecayedWeights(0.5).update(tensor([1]), undefined, tensor([2])).updates as never) // => [2]
 * ```
 */
export function addDecayedWeights(weightDecay: number): GradientTransformation<undefined> {
  return {
    name: 'addDecayedWeights',
    init: () => undefined,
    update: (grads, state, params) => {
      if (weightDecay === 0 || params === undefined) return { updates: grads, state }
      return { updates: treeMap((g, p) => add(g, mul(p, weightDecay)), grads, params), state }
    },
  }
}

/** State of {@link traceMomentum}. */
export interface TraceState {
  readonly count: number
  readonly trace: Tree
}

/**
 * Momentum: trace = decay trace + (1 - dampening) g (the first step takes trace = g), and updates = trace, or
 * g + decay trace with Nesterov (torch.optim.SGD's momentum buffer).
 *
 * @param decay - The momentum coefficient.
 * @param nesterov - Use Nesterov momentum.
 * @param dampening - Dampening of the gradient added to the trace (torch's dampening, 0 by default).
 * @returns The transformation.
 * @example
 * ```ts
 * import { traceMomentum } from 'whitebox-engine/optim'
 * import { tensor, toArray } from 'whitebox-engine/tensor'
 * const t = traceMomentum(0.5)
 * const first = t.update(tensor([2]), t.init(tensor([0])))
 * toArray(t.update(tensor([2]), first.state).updates as never) // => [3]
 * ```
 */
export function traceMomentum(decay: number, nesterov = false, dampening = 0): GradientTransformation<TraceState> {
  return {
    name: 'traceMomentum',
    init: (params) => ({ count: 0, trace: treeZeros(params) }),
    update: (grads, state) => {
      const keep = state.count === 0 ? 1 : 1 - dampening
      const trace = treeMap((g, t) => add(keep === 1 ? g : mul(g, keep), mul(t, decay)), grads, state.trace)
      const updates = nesterov ? treeMap((g, t) => add(g, mul(t, decay)), grads, trace) : trace
      return { updates, state: { count: state.count + 1, trace } }
    },
  }
}

/** State of {@link scaleByAdam}. */
export interface AdamState {
  readonly count: number
  readonly mu: Tree
  readonly nu: Tree
  readonly nuMax?: Tree
}

/**
 * Adam's scaling: bias-corrected first and second moment estimates, updates = m_hat / (sqrt(v_hat) + eps), with the
 * AMSGrad variant keeping the running maximum of the second moment (torch.optim.Adam).
 *
 * @param b1 - First-moment decay.
 * @param b2 - Second-moment decay.
 * @param eps - Added to the denominator.
 * @param amsgrad - Use the AMSGrad maximum.
 * @returns The transformation.
 * @example
 * ```ts
 * import { scaleByAdam } from 'whitebox-engine/optim'
 * import { tensor, toArray } from 'whitebox-engine/tensor'
 * const t = scaleByAdam()
 * toArray(t.update(tensor([2]), t.init(tensor([0]))).updates as never)[0] // => 0.999999995
 * ```
 */
export function scaleByAdam(b1 = 0.9, b2 = 0.999, eps = 1e-8, amsgrad = false): GradientTransformation<AdamState> {
  return {
    name: 'scaleByAdam',
    init: (params) => ({
      count: 0,
      mu: treeZeros(params),
      nu: treeZeros(params),
      ...(amsgrad ? { nuMax: treeZeros(params) } : {}),
    }),
    update: (grads, state) => {
      const count = state.count + 1
      const mu = treeMap((g, m) => add(mul(m, b1), mul(g, 1 - b1)), grads, state.mu)
      const nu = treeMap((g, v) => add(mul(v, b2), mul(square(g), 1 - b2)), grads, state.nu)
      const nuMax = amsgrad ? treeMap((v, vm) => maximum(v, vm), nu, state.nuMax) : undefined
      const c1 = 1 - b1 ** count
      const c2 = Math.sqrt(1 - b2 ** count)
      // torch's form: (m / c1) / (sqrt(v) / c2 + eps).
      const updates = treeMap((m, v) => div(div(m, c1), add(div(sqrt(v), c2), eps)), mu, nuMax ?? nu)
      return { updates, state: { count, mu, nu, ...(nuMax === undefined ? {} : { nuMax }) } }
    },
  }
}

/** State of {@link scaleByRms}. */
export interface RmsState {
  readonly nu: Tree
  /** The running mean of the gradient (centred RMSprop only). */
  readonly mu?: Tree
}

/**
 * RMSprop's scaling: updates = g / (sqrt(v) + eps) with v = decay v + (1 - decay) g^2, or, centred, g / (sqrt(v -
 * m^2) + eps) with m the running mean of the gradient (torch.optim.RMSprop).
 *
 * @param decay - Second-moment decay (torch's alpha).
 * @param eps - Added to the denominator.
 * @param centered - Normalise by the estimated variance instead of the second moment.
 * @returns The transformation.
 * @example
 * ```ts
 * import { scaleByRms } from 'whitebox-engine/optim'
 * import { tensor, toArray } from 'whitebox-engine/tensor'
 * const t = scaleByRms(0.75, 0)
 * toArray(t.update(tensor([4]), t.init(tensor([0]))).updates as never) // => [2]
 * ```
 */
export function scaleByRms(decay = 0.99, eps = 1e-8, centered = false): GradientTransformation<RmsState> {
  return {
    name: 'scaleByRms',
    init: (params) => ({ nu: treeZeros(params), ...(centered ? { mu: treeZeros(params) } : {}) }),
    update: (grads, state) => {
      const nu = treeMap((g, v) => add(mul(v, decay), mul(square(g), 1 - decay)), grads, state.nu)
      if (!centered) return { updates: treeMap((g, v) => div(g, add(sqrt(v), eps)), grads, nu), state: { nu } }
      const mu = treeMap((g, m) => add(m, mul(sub(g, m), 1 - decay)), grads, state.mu)
      const updates = treeMap((g, v, m) => div(g, add(sqrt(sub(v, square(m))), eps)), grads, nu, mu)
      return { updates, state: { nu, mu } }
    },
  }
}

/**
 * Adagrad's scaling: updates = g / (sqrt(sum of g^2) + eps) (torch.optim.Adagrad).
 *
 * @param eps - Added to the denominator.
 * @param initialAccumulator - Starting value of the sum.
 * @returns The transformation.
 * @example
 * ```ts
 * import { scaleByAdagrad } from 'whitebox-engine/optim'
 * scaleByAdagrad().name // => 'scaleByAdagrad'
 * ```
 */
export function scaleByAdagrad(eps = 1e-10, initialAccumulator = 0): GradientTransformation<{ readonly sum: Tree }> {
  return {
    name: 'scaleByAdagrad',
    init: (params) => ({ sum: treeMap((p) => add(mul(p, 0), initialAccumulator), params) }),
    update: (grads, state) => {
      const total = treeMap((g, s) => add(s, square(g)), grads, state.sum)
      return { updates: treeMap((g, s) => div(g, add(sqrt(s), eps)), grads, total), state: { sum: total } }
    },
  }
}

/**
 * The identity transformation: updates are the gradients (optax.identity).
 *
 * @returns The transformation.
 * @example
 * ```ts
 * import { identity } from 'whitebox-engine/optim'
 * import { tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(identity().update(tensor([1, 2]), undefined).updates as never) // => [1, 2]
 * ```
 */
export function identity(): GradientTransformation<undefined> {
  return { name: 'identity', init: () => undefined, update: (grads, state) => ({ updates: grads, state }) }
}

/**
 * Zero updates: freezes the parameters it is applied to (optax.set_to_zero), typically inside {@link multiTransform}.
 *
 * @returns The transformation.
 * @example
 * ```ts
 * import { setToZero } from 'whitebox-engine/optim'
 * import { tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(setToZero().update(tensor([1, 2]), undefined).updates as never) // => [0, 0]
 * ```
 */
export function setToZero(): GradientTransformation<undefined> {
  return {
    name: 'setToZero',
    init: () => undefined,
    update: (grads, state) => ({ updates: treeMap((g) => zerosLike(g), grads), state }),
  }
}

/**
 * Multiply updates by a constant (optax.scale).
 *
 * @param factor - The factor.
 * @returns The transformation.
 * @example
 * ```ts
 * import { scale } from 'whitebox-engine/optim'
 * import { tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(scale(-2).update(tensor([1, 2]), undefined).updates as never) // => [-2, -4]
 * ```
 */
export function scale(factor: number): GradientTransformation<undefined> {
  return {
    name: 'scale',
    init: () => undefined,
    update: (g, state) => ({ updates: treeMap((u) => mul(u, factor), g), state }),
  }
}

/**
 * Layer-wise trust ratio (LARS and LAMB, optax.scale_by_trust_ratio): each leaf's update is multiplied by
 * |param| / |update|, or by 1 when either norm is zero.
 *
 * @returns The transformation (needs `params` in `update`).
 * @example
 * ```ts
 * import { scaleByTrustRatio } from 'whitebox-engine/optim'
 * import { tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(scaleByTrustRatio().update(tensor([3, 4]), undefined, tensor([0, 10])).updates as never) // => [6, 8]
 * ```
 */
export function scaleByTrustRatio(): GradientTransformation<undefined> {
  return {
    name: 'scaleByTrustRatio',
    init: () => undefined,
    update: (grads, state, params) => {
      if (params === undefined) throw new WhiteboxError('value', 'scaleByTrustRatio needs the parameters')
      const updates = treeMap(
        (u, p) => {
          const pn = item(sqrt(sum(square(p))))
          const un = item(sqrt(sum(square(u))))
          return pn === 0 || un === 0 ? u : mul(u, pn / un)
        },
        grads,
        params,
      )
      return { updates, state }
    },
  }
}

/** A tree of labels (strings or booleans) with the structure of the parameters, or a function computing it. */
export type Labels = unknown

// The leaves of a label tree in the order treeFlatten visits a parameter tree (records by sorted key, arrays by index).
function labelLeaves(labels: unknown): unknown[] {
  const out: unknown[] = []
  const visit = (node: unknown): void => {
    if (Array.isArray(node)) for (const child of node) visit(child)
    else if (typeof node === 'object' && node !== null && !isValue(node))
      for (const key of Object.keys(node).sort()) visit((node as Record<string, unknown>)[key])
    else out.push(node)
  }
  visit(labels)
  return out
}

function resolveLabels(labels: unknown, tree: Tree, count: number): unknown[] {
  const leaves = labelLeaves(typeof labels === 'function' ? (labels as (p: Tree) => unknown)(tree) : labels)
  if (leaves.length !== count)
    throw new WhiteboxError('value', `labels have ${leaves.length} leaves but the parameters have ${count}`)
  return leaves
}

/** State of {@link masked}. */
export interface MaskedState {
  readonly inner: unknown
}

/**
 * Apply a transformation only to the leaves selected by a mask, passing the other updates through unchanged
 * (optax.masked). The inner transformation sees the selected leaves as an array, so global operations such as
 * {@link clipByGlobalNorm} act on the selection only.
 *
 * @param inner - The transformation for the selected leaves.
 * @param mask - A tree of booleans with the structure of the parameters, or a function of the parameters returning one.
 * @returns The transformation.
 * @example
 * ```ts
 * import { masked, scale } from 'whitebox-engine/optim'
 * import { tensor, toArray } from 'whitebox-engine/tensor'
 * const t = masked(scale(10), { w: true, b: false })
 * const g = { w: tensor([1]), b: tensor([1]) }
 * const u = t.update(g, t.init(g), g).updates as { w: never; b: never }
 * [toArray(u.w), toArray(u.b)] // => [[10], [1]]
 * ```
 */
export function masked(inner: GradientTransformation, mask: Labels): GradientTransformation<MaskedState> {
  const pick = (tree: Tree, leaves: readonly Value[]): boolean[] =>
    resolveLabels(mask, tree, leaves.length).map((m) => m === true)
  return {
    name: `masked(${inner.name})`,
    init: (params) => {
      const leaves = treeLeaves(params)
      const m = pick(params, leaves)
      return { inner: inner.init(leaves.filter((_, i) => m[i])) }
    },
    update: (grads, state, params) => {
      const { leaves, def } = treeFlatten(grads)
      const m = pick(params ?? grads, leaves)
      const pLeaves = params === undefined ? undefined : treeLeaves(params).filter((_, i) => m[i])
      const r = inner.update(
        leaves.filter((_, i) => m[i]),
        state.inner,
        pLeaves,
      )
      const out = r.updates as Value[]
      let k = 0
      const merged = leaves.map((g, i) => (m[i] ? (out[k++] as Value) : g))
      return { updates: treeUnflatten(def, merged), state: { inner: r.state } }
    },
  }
}

/**
 * Different transformations for different parameter groups (optax.multi_transform, torch's parameter groups): each
 * leaf's label picks the transformation that updates it; each transformation sees its group's leaves as an array.
 *
 * @param transforms - The transformation for each label.
 * @param labels - A tree of labels with the structure of the parameters, or a function of the parameters returning one.
 * @returns The transformation; its state holds one state per label.
 * @throws WhiteboxError with code `value` when a label has no transformation.
 * @example
 * ```ts
 * import { multiTransform, scaleByLearningRate, setToZero } from 'whitebox-engine/optim'
 * import { tensor, toArray } from 'whitebox-engine/tensor'
 * const t = multiTransform({ train: scaleByLearningRate(0.5), frozen: setToZero() }, { w: 'train', b: 'frozen' })
 * const g = { w: tensor([2]), b: tensor([2]) }
 * const u = t.update(g, t.init(g), g).updates as { w: never; b: never }
 * [toArray(u.w), toArray(u.b)] // => [[-1], [0]]
 * ```
 */
export function multiTransform(
  transforms: Readonly<Record<string, GradientTransformation>>,
  labels: Labels,
): GradientTransformation<Readonly<Record<string, unknown>>> {
  const groups = (tree: Tree, leaves: readonly Value[]): string[] => {
    const ls = resolveLabels(labels, tree, leaves.length).map(String)
    for (const l of ls)
      if (transforms[l] === undefined) throw new WhiteboxError('value', `no transformation for label '${l}'`)
    return ls
  }
  const names = Object.keys(transforms)
  return {
    name: `multiTransform(${names.map((n) => `${n}: ${transforms[n]?.name}`).join(', ')})`,
    init: (params) => {
      const leaves = treeLeaves(params)
      const ls = groups(params, leaves)
      const state: Record<string, unknown> = {}
      for (const n of names)
        state[n] = (transforms[n] as GradientTransformation).init(leaves.filter((_, i) => ls[i] === n))
      return state
    },
    update: (grads, state, params) => {
      const { leaves, def } = treeFlatten(grads)
      const ls = groups(params ?? grads, leaves)
      const pLeaves = params === undefined ? undefined : treeLeaves(params)
      const merged: Value[] = [...leaves]
      const next: Record<string, unknown> = {}
      for (const n of names) {
        const idx = ls.flatMap((l, i) => (l === n ? [i] : []))
        const r = (transforms[n] as GradientTransformation).update(
          idx.map((i) => leaves[i] as Value),
          state[n],
          pLeaves === undefined ? undefined : idx.map((i) => pLeaves[i] as Value),
        )
        const out = r.updates as Value[]
        for (const [k, i] of idx.entries()) merged[i] = out[k] as Value
        next[n] = r.state
      }
      return { updates: treeUnflatten(def, merged), state: next }
    },
  }
}
