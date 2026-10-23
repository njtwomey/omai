import {
  add,
  clip,
  div,
  item,
  maximum,
  mul,
  sqrt,
  square,
  sum,
  tensor,
  treeLeaves,
  treeMap,
  treeZeros,
  type Value,
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
 * toArray(clipByGlobalNorm(1).update(tensor([3, 4]), undefined).updates as never) // => [0.6, 0.8]
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

/**
 * Momentum: trace = decay trace + g, and updates = trace (or g + decay trace with Nesterov), as torch.optim.SGD with no
 * dampening.
 *
 * @param decay - The momentum coefficient.
 * @param nesterov - Use Nesterov momentum.
 * @returns The transformation.
 * @example
 * ```ts
 * import { traceMomentum } from 'whitebox-engine/optim'
 * traceMomentum(0.9).name // => 'traceMomentum'
 * ```
 */
export function traceMomentum(decay: number, nesterov = false): GradientTransformation<{ readonly trace: Tree }> {
  return {
    name: 'traceMomentum',
    init: (params) => ({ trace: treeZeros(params) }),
    update: (grads, state) => {
      const trace = treeMap((g, t) => add(g, mul(t, decay)), grads, state.trace)
      const updates = nesterov ? treeMap((g, t) => add(g, mul(t, decay)), grads, trace) : trace
      return { updates, state: { trace } }
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

/**
 * RMSprop's scaling: updates = g / (sqrt(v) + eps) with v = decay v + (1 - decay) g^2 (torch.optim.RMSprop).
 *
 * @param decay - Second-moment decay (torch's alpha).
 * @param eps - Added to the denominator.
 * @returns The transformation.
 * @example
 * ```ts
 * import { scaleByRms } from 'whitebox-engine/optim'
 * scaleByRms().name // => 'scaleByRms'
 * ```
 */
export function scaleByRms(decay = 0.99, eps = 1e-8): GradientTransformation<{ readonly nu: Tree }> {
  return {
    name: 'scaleByRms',
    init: (params) => ({ nu: treeZeros(params) }),
    update: (grads, state) => {
      const nu = treeMap((g, v) => add(mul(v, decay), mul(square(g), 1 - decay)), grads, state.nu)
      return { updates: treeMap((g, v) => div(g, add(sqrt(v), eps)), grads, nu), state: { nu } }
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
 * Polyak averaging of parameter trees (optax.incremental_update): `stepSize * new + (1 - stepSize) * old`, the soft
 * target-network update of DDPG, TD3 and SAC and the exponential moving average of weights.
 *
 * @param newTree - The new (online) parameters.
 * @param oldTree - The old (target) parameters, with the same structure.
 * @param stepSize - The interpolation weight `tau` in [0, 1].
 * @returns The averaged parameters.
 * @example
 * ```ts
 * import { incrementalUpdate } from 'whitebox-engine/optim'
 * import { tensor, toArray } from 'whitebox-engine/tensor'
 * toArray((incrementalUpdate({ w: tensor([4]) }, { w: tensor([0]) }, 0.25) as { w: never }).w) // => [1]
 * ```
 */
export function incrementalUpdate(newTree: Tree, oldTree: Tree, stepSize: number): Tree {
  return treeMap((n, o) => add(mul(n, stepSize), mul(o, 1 - stepSize)), newTree, oldTree)
}
