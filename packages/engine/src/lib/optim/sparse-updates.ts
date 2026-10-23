import { WhiteboxError } from '../base/index.ts'
import {
  abs,
  add,
  any,
  cast,
  div,
  greater,
  less,
  lessEqual,
  maximum,
  minimum,
  mul,
  neg,
  notEqual,
  reshape,
  sign,
  sqrt,
  square,
  sub,
  treeMap,
  treeZeros,
  type Value,
  where,
  zerosLike,
} from '../tensor/index.ts'
import { chain, type GradientTransformation, type LearningRate, scaleByLearningRate, type Tree } from './transform.ts'

const rate = (lr: LearningRate, step: number): number => (typeof lr === 'number' ? lr : lr(step))

/**
 * The active set of a sparse gradient: entries that are non-zero, or (row-wise, for embedding tables) whole rows with
 * any non-zero entry, as a float mask of 0s and 1s.
 *
 * @param grad - The gradient.
 * @param rowwise - Treat rows (the first axis) as the unit of sparsity.
 * @returns The mask, with the gradient's shape.
 * @example
 * ```ts
 * import { activeMask } from 'whitebox-engine/optim'
 * import { tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(activeMask(tensor([[0, 2], [0, 0]]), true)) // => [[1, 1], [0, 0]]
 * toArray(activeMask(tensor([[0, 2], [0, 0]]))) // => [[0, 1], [0, 0]]
 * ```
 */
export function activeMask(grad: Value, rowwise = false): Value {
  const nz = notEqual(grad, 0)
  if (!rowwise || grad.shape.length < 2) return cast(nz, 'float64')
  const axes = grad.shape.slice(1).map((_, i) => i + 1)
  const rows = reshape(cast(any(nz, axes), 'float64'), [grad.shape[0] ?? 0, ...axes.map(() => 1)])
  return add(zerosLike(grad), rows)
}

/** State of {@link scaleBySparseAdam}. */
export interface SparseAdamState {
  readonly count: number
  readonly mu: Tree
  readonly nu: Tree
}

/**
 * SparseAdam's scaling (torch.optim.SparseAdam): the moments are updated only on the active entries of the gradient
 * (so rows of an embedding table that were not looked up keep their statistics), and the step there is
 * sqrt(1 - b2^t) / (1 - b1^t) m / (sqrt(v) + eps); inactive entries do not move.
 *
 * @param b1 - First-moment decay.
 * @param b2 - Second-moment decay.
 * @param eps - Added to the denominator.
 * @param rowwise - Rows are the unit of sparsity (embedding tables).
 * @returns The transformation.
 * @example
 * ```ts
 * import { scaleBySparseAdam } from 'whitebox-engine/optim'
 * import { tensor, toArray } from 'whitebox-engine/tensor'
 * const t = scaleBySparseAdam(0.9, 0.999, 0)
 * toArray(t.update(tensor([2, 0]), t.init(tensor([0, 0]))).updates as never).map((v) => Math.round(v * 1e12) / 1e12) // => [1, 0]
 * ```
 */
export function scaleBySparseAdam(
  b1 = 0.9,
  b2 = 0.999,
  eps = 1e-8,
  rowwise = false,
): GradientTransformation<SparseAdamState> {
  return {
    name: 'scaleBySparseAdam',
    init: (params) => ({ count: 0, mu: treeZeros(params), nu: treeZeros(params) }),
    update: (grads, state) => {
      const count = state.count + 1
      const masks = treeMap((g) => activeMask(g, rowwise), grads)
      const mu = treeMap((g, m, a) => add(m, mul(mul(sub(g, m), 1 - b1), a)), grads, state.mu, masks)
      const nu = treeMap((g, v, a) => add(v, mul(mul(sub(square(g), v), 1 - b2), a)), grads, state.nu, masks)
      const size = Math.sqrt(1 - b2 ** count) / (1 - b1 ** count)
      const updates = treeMap((m, v, a) => where(a, mul(div(m, add(sqrt(v), eps)), size), 0), mu, nu, masks)
      return { updates, state: { count, mu, nu } }
    },
  }
}

/** Options of {@link sparseAdam}. */
export interface SparseAdamOptions {
  readonly learningRate: LearningRate
  readonly b1?: number
  readonly b2?: number
  readonly eps?: number
  /** Rows are the unit of sparsity (embedding tables). */
  readonly rowwise?: boolean
}

/**
 * SparseAdam (torch.optim.SparseAdam): lazy Adam for sparse gradients, such as embedding lookups.
 *
 * @param options - Learning rate, betas, eps and row-wise sparsity.
 * @returns The optimiser.
 * @concept sparse-optimisers
 * @example
 * ```ts
 * import { sparseAdam } from 'whitebox-engine/optim'
 * sparseAdam({ learningRate: 1e-3 }).name // => 'chain(scaleBySparseAdam, scaleByLearningRate)'
 * ```
 */
export function sparseAdam(options: SparseAdamOptions): GradientTransformation<unknown[]> {
  return chain(
    scaleBySparseAdam(options.b1 ?? 0.9, options.b2 ?? 0.999, options.eps ?? 1e-8, options.rowwise ?? false),
    scaleByLearningRate(options.learningRate),
  )
}

/** Options of {@link lazyAdagrad}. */
export interface LazyAdagradOptions {
  readonly learningRate: LearningRate
  readonly eps?: number
  readonly initialAccumulator?: number
  /** The rate at step t is learningRate / (1 + t lrDecay). */
  readonly lrDecay?: number
  /** L2 penalty added to the gradient on the active entries only (lazy regularisation). */
  readonly weightDecay?: number
  readonly rowwise?: boolean
}

/** State of {@link lazyAdagrad}. */
export interface LazyAdagradState {
  readonly count: number
  readonly sum: Tree
}

/**
 * Lazy Adagrad (torch.optim.Adagrad on sparse gradients): the squared-gradient sums and the parameters change only on
 * the active entries; weight decay, if any, is applied lazily to those entries too, so untouched rows cost nothing.
 *
 * @param options - Learning rate, eps, initial accumulator, learning-rate decay, lazy weight decay and row-wise
 *   sparsity.
 * @returns The optimiser (needs `params` in `update` when weight decay is used).
 * @example
 * ```ts
 * import { lazyAdagrad } from 'whitebox-engine/optim'
 * import { tensor, toArray } from 'whitebox-engine/tensor'
 * const opt = lazyAdagrad({ learningRate: 0.5, eps: 0 })
 * toArray(opt.update(tensor([4, 0]), opt.init(tensor([1, 1])), tensor([1, 1])).updates as never) // => [-0.5, 0]
 * ```
 */
export function lazyAdagrad(options: LazyAdagradOptions): GradientTransformation<LazyAdagradState> {
  const eps = options.eps ?? 1e-10
  const init = options.initialAccumulator ?? 0
  const decay = options.lrDecay ?? 0
  const wd = options.weightDecay ?? 0
  const rowwise = options.rowwise ?? false
  return {
    name: 'lazyAdagrad',
    init: (params) => ({ count: 0, sum: treeMap((p) => add(zerosLike(p), init), params) }),
    update: (grads, state, params) => {
      if (wd !== 0 && params === undefined)
        throw new WhiteboxError('value', 'lazyAdagrad weight decay needs the parameters')
      const clr = rate(options.learningRate, state.count) / (1 + state.count * decay)
      const masks = treeMap((g) => activeMask(g, rowwise), grads)
      const g = wd === 0 ? grads : treeMap((gi, p, a) => add(gi, mul(mul(p, wd), a)), grads, params, masks)
      const total = treeMap((gi, s, a) => add(s, mul(square(gi), a)), g, state.sum, masks)
      const updates = treeMap((gi, s, a) => where(a, mul(div(gi, add(sqrt(s), eps)), -clr), 0), g, total, masks)
      return { updates, state: { count: state.count + 1, sum: total } }
    },
  }
}

/** Options of {@link sgdCumulativeL1}. */
export interface CumulativeL1Options {
  readonly learningRate: LearningRate
  /** The L1 strength per step (C / N in Tsuruoka et al.). */
  readonly l1: number
  /** Apply the penalty only to active coordinates (the paper's lazy form, default true). */
  readonly lazy?: boolean
}

/** State of {@link sgdCumulativeL1}. */
export interface CumulativeL1State {
  readonly count: number
  /** The total L1 penalty every weight could have received so far. */
  readonly u: number
  /** The penalty each weight has actually received. */
  readonly q: Tree
}

/**
 * SGD with L1 regularisation by cumulative penalty (Tsuruoka, Tsujii and Ananiadou 2009): after each gradient step,
 * every (active) weight is pulled towards zero by the penalty it has not yet received, u - q_i, clipped at zero, so
 * weights become exactly zero and the penalty applied matches what full-gradient L1 would apply.
 *
 * @param options - Learning rate, L1 strength and laziness.
 * @returns The optimiser (needs `params` in `update`).
 * @concept sparse-optimisers
 * @example
 * ```ts
 * import { sgdCumulativeL1 } from 'whitebox-engine/optim'
 * import { tensor, toArray } from 'whitebox-engine/tensor'
 * const opt = sgdCumulativeL1({ learningRate: 1, l1: 0.5 })
 * toArray(opt.update(tensor([0.2, -1]), opt.init(tensor([1, 0])), tensor([1, 0])).updates as never) // => [-0.7, 0.5]
 * ```
 */
export function sgdCumulativeL1(options: CumulativeL1Options): GradientTransformation<CumulativeL1State> {
  const lazy = options.lazy ?? true
  return {
    name: 'sgdCumulativeL1',
    init: (params) => ({ count: 0, u: 0, q: treeZeros(params) }),
    update: (grads, state, params) => {
      if (params === undefined) throw new WhiteboxError('value', 'sgdCumulativeL1 needs the parameters')
      const eta = rate(options.learningRate, state.count)
      const u = state.u + eta * options.l1
      const stepped = treeMap((p, g) => sub(p, mul(g, eta)), params, grads)
      const penalised = treeMap(
        (z, q, g) => {
          const pos = maximum(sub(z, add(q, u)), 0)
          const negv = minimum(add(z, sub(u, q)), 0)
          const w = where(greater(z, 0), pos, where(less(z, 0), negv, z))
          return lazy ? where(notEqual(g, 0), w, z) : w
        },
        stepped,
        state.q,
        grads,
      )
      const q = treeMap((qi, w, z) => add(qi, sub(w, z)), state.q, penalised, stepped)
      const updates = treeMap((w, p) => sub(w, p), penalised, params)
      return { updates, state: { count: state.count + 1, u, q } }
    },
  }
}

/**
 * The truncation operator T1(v, alpha, theta) of Langford, Li and Zhang (2009): entries in [0, theta] shrink towards
 * zero by alpha (stopping at zero), entries in [-theta, 0] likewise, larger entries are left alone. With theta =
 * infinity it is soft thresholding.
 *
 * @param v - The values.
 * @param alpha - The shrinkage (the gravity times the step).
 * @param theta - The threshold beyond which nothing is shrunk.
 * @returns T1(v).
 * @example
 * ```ts
 * import { truncate } from 'whitebox-engine/optim'
 * import { tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(truncate(tensor([0.3, -0.05, 2]), 0.1, 1)) // => [0.19999999999999998, 0, 2]
 * ```
 */
export function truncate(v: Value, alpha: number, theta: number): Value {
  const inside = lessEqual(abs(v), theta)
  const shrunk = mul(sign(v), maximum(sub(abs(v), alpha), 0))
  return where(inside, shrunk, v)
}

/** Options of {@link truncatedGradient}. */
export interface TruncatedGradientOptions {
  readonly learningRate: LearningRate
  /** The gravity g: the L1 pull per unit step. */
  readonly gravity: number
  /** Only weights with |w| <= theta are truncated (default infinity). */
  readonly theta?: number
  /** Truncate every K steps with K times the gravity (default 1). */
  readonly every?: number
}

/**
 * Truncated gradient (Langford, Li and Zhang 2009): SGD whose weights are truncated towards zero by
 * {@link truncate} every K steps with gravity K eta g, producing sparse models online.
 *
 * @param options - Learning rate, gravity, threshold and period.
 * @returns The optimiser (needs `params` in `update`).
 * @concept sparse-optimisers
 * @example
 * ```ts
 * import { truncatedGradient } from 'whitebox-engine/optim'
 * import { tensor, toArray } from 'whitebox-engine/tensor'
 * const opt = truncatedGradient({ learningRate: 0.5, gravity: 0.2 })
 * const u = opt.update(tensor([0.4, 0]), opt.init(tensor([1, 0.05])), tensor([1, 0.05])).updates
 * toArray(u as never).map((v) => Math.round(v * 1e12) / 1e12) // => [-0.3, -0.05]
 * ```
 */
export function truncatedGradient(
  options: TruncatedGradientOptions,
): GradientTransformation<{ readonly count: number }> {
  const k = options.every ?? 1
  const theta = options.theta ?? Number.POSITIVE_INFINITY
  return {
    name: 'truncatedGradient',
    init: () => ({ count: 0 }),
    update: (grads, state, params) => {
      if (params === undefined) throw new WhiteboxError('value', 'truncatedGradient needs the parameters')
      const eta = rate(options.learningRate, state.count)
      const count = state.count + 1
      const alpha = count % k === 0 ? k * eta * options.gravity : 0
      const updates = treeMap(
        (p, g) => {
          const stepped = sub(p, mul(g, eta))
          return sub(alpha === 0 ? stepped : truncate(stepped, alpha, theta), p)
        },
        params,
        grads,
      )
      return { updates, state: { count } }
    },
  }
}

/** Options of {@link ftrlProximal}. */
export interface FtrlProximalOptions {
  /** The per-coordinate learning-rate scale alpha (default 0.1). */
  readonly alpha?: number
  /** The learning-rate offset beta (default 1). */
  readonly beta?: number
  /** L1 strength lambda1 (default 0). */
  readonly l1?: number
  /** L2 strength lambda2 (default 0). */
  readonly l2?: number
  readonly rowwise?: boolean
}

/** State of {@link ftrlProximal}: the per-coordinate z (adjusted gradient sums) and n (squared-gradient sums). */
export interface FtrlProximalState {
  readonly z: Tree
  readonly n: Tree
}

/**
 * FTRL-Proximal (McMahan et al. 2013, "Ad click prediction: a view from the trenches"): per-coordinate
 * follow-the-regularised-leader with adaptive learning rates eta_i = alpha / (beta + sqrt(n_i)), L1 and L2. Weights
 * are computed in closed form from z and n, and are exactly zero while |z_i| <= l1; only active coordinates change.
 *
 * @param options - alpha, beta, L1, L2 and row-wise sparsity.
 * @returns The optimiser (needs `params` in `update`; its updates move the parameters to the FTRL solution).
 * @concept sparse-optimisers
 * @example
 * ```ts
 * import { ftrlProximal } from 'whitebox-engine/optim'
 * import { tensor, toArray } from 'whitebox-engine/tensor'
 * const opt = ftrlProximal({ alpha: 1, beta: 0, l1: 1 })
 * const p = tensor([0, 0])
 * toArray(opt.update(tensor([3, 0.5]), opt.init(p), p).updates as never) // => [-0.6666666666666666, 0]
 * ```
 */
export function ftrlProximal(options: FtrlProximalOptions = {}): GradientTransformation<FtrlProximalState> {
  const alpha = options.alpha ?? 0.1
  const beta = options.beta ?? 1
  const l1 = options.l1 ?? 0
  const l2 = options.l2 ?? 0
  const rowwise = options.rowwise ?? false
  return {
    name: 'ftrlProximal',
    init: (params) => ({ z: treeZeros(params), n: treeZeros(params) }),
    update: (grads, state, params) => {
      if (params === undefined) throw new WhiteboxError('value', 'ftrlProximal needs the parameters')
      const masks = treeMap((g) => activeMask(g, rowwise), grads)
      const sigma = treeMap((g, n) => div(sub(sqrt(add(n, square(g))), sqrt(n)), alpha), grads, state.n)
      const z = treeMap((zi, g, s, w, a) => add(zi, mul(sub(g, mul(s, w)), a)), state.z, grads, sigma, params, masks)
      const n = treeMap((ni, g, a) => add(ni, mul(square(g), a)), state.n, grads, masks)
      const updates = treeMap(
        (zi, ni, w, a) => {
          const shrunk = div(neg(sub(zi, mul(sign(zi), l1))), add(div(add(sqrt(ni), beta), alpha), l2))
          const target = where(lessEqual(abs(zi), l1), 0, shrunk)
          return mul(sub(target, w), a)
        },
        z,
        n,
        params,
        masks,
      )
      return { updates, state: { z, n } }
    },
  }
}

/**
 * Whether a FTRL weight would be non-zero: |z_i| > l1 (the sparsity pattern FTRL-Proximal maintains).
 *
 * @param state - A {@link ftrlProximal} state.
 * @param l1 - The L1 strength used.
 * @returns A tree of 0/1 masks.
 * @example
 * ```ts
 * import { ftrlProximal, ftrlSupport } from 'whitebox-engine/optim'
 * import { tensor, toArray } from 'whitebox-engine/tensor'
 * const opt = ftrlProximal({ l1: 1 })
 * const p = tensor([0, 0])
 * toArray(ftrlSupport(opt.update(tensor([3, 0.5]), opt.init(p), p).state, 1) as never) // => [1, 0]
 * ```
 */
export function ftrlSupport(state: FtrlProximalState, l1: number): Tree {
  return treeMap((z) => cast(greater(abs(z), l1), 'float64'), state.z)
}
