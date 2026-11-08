/**
 * Curvature-aware preconditioned optimisers: natural gradient descent with Fisher or empirical Fisher matrices, K-FAC
 * (Kronecker-factored curvature for dense layers, Martens and Grosse), Shampoo (per-mode preconditioners, Gupta et
 * al.), and building blocks such as matrix inverse p-th roots, fixed preconditioners and diagonal preconditioning.
 */

import { OmaiError } from '../base/index.ts'
import { eigh, solve } from '../numerics/linalg/index.ts'
import {
  add,
  asTensor,
  concat,
  div,
  eye,
  fromStorage,
  item,
  matmul,
  maximum,
  moveaxis,
  mul,
  ones,
  pow,
  ravel,
  reshape,
  sliceAxis,
  sqrt,
  type Tensor,
  type TensorLike,
  tensordot,
  toFlat,
  toVector,
  trace,
  transpose,
  treeFlatten,
  treeLeaves,
  treeMap,
  treePaths,
  treeUnflatten,
  treeZeros,
  type Value,
} from '../tensor/index.ts'
import type { GradientTransformation, LearningRate, ParamTree } from './transform.ts'

const rate = (lr: LearningRate, step: number): number => (typeof lr === 'number' ? lr : lr(step))

/**
 * The inverse p-th root (A + ridge I)^(-1/p) of a symmetric positive semi-definite matrix, by eigendecomposition with
 * eigenvalues floored at zero (optax.matrix_inverse_pth_root; the preconditioner of Shampoo).
 *
 * @param a - A symmetric matrix.
 * @param p - The root.
 * @param ridge - Added to the diagonal (default 0).
 * @returns The matrix power.
 * @example
 * ```ts
 * import { matrixInversePthRoot } from 'omai-engine/optim'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(matrixInversePthRoot([[16, 0], [0, 1]], 4)) // => [[0.5, 0], [0, 1]]
 * ```
 */
export function matrixInversePthRoot(a: TensorLike, p: number, ridge = 0): Value {
  const { eigenvalues, eigenvectors } = eigh(a)
  const scaled = pow(add(maximum(eigenvalues, 0), ridge), -1 / p)
  return matmul(mul(eigenvectors, scaled), transpose(eigenvectors))
}

/**
 * The empirical Fisher matrix G^T G / N from per-example gradients (rows of G).
 *
 * @param perExampleGrads - An N x d matrix of per-example gradients.
 * @returns The d x d matrix.
 * @example
 * ```ts
 * import { empiricalFisher } from 'omai-engine/optim'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(empiricalFisher([[1, 0], [1, 2]])) // => [[1, 1], [1, 2]]
 * ```
 */
export function empiricalFisher(perExampleGrads: TensorLike): Value {
  const g = asTensor(perExampleGrads)
  return div(matmul(transpose(g), g), g.shape[0] ?? 1)
}

/**
 * The natural-gradient direction (F + damping I)^-1 g (Amari 1998).
 *
 * @param fisher - The Fisher (or Gauss-Newton) matrix, d x d.
 * @param grad - The gradient, a d-vector.
 * @param damping - Tikhonov damping.
 * @returns The direction.
 * @example
 * ```ts
 * import { naturalGradientDirection } from 'omai-engine/optim'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(naturalGradientDirection([[2, 0], [0, 4]], [2, 2], 0)) // => [1, 0.5]
 * ```
 */
export function naturalGradientDirection(fisher: TensorLike, grad: TensorLike, damping = 0): Value {
  const f = asTensor(fisher)
  const n = f.shape[0] ?? 0
  return solve(add(f, mul(eye(n), damping)), asTensor(grad))
}

/** A transformation whose curvature statistics are fed by the caller (Fisher matrices, layer activations). */
export interface PreconditionedTransformation<S, O> extends GradientTransformation<S> {
  /** Fold new curvature statistics into the state. */
  observe(state: S, statistics: O): S
}

/** Curvature statistics for {@link naturalGradient}: a Fisher matrix or per-example gradients (rows). */
export interface FisherStatistics {
  readonly fisher?: TensorLike
  readonly perExampleGrads?: TensorLike
}

/** State of {@link naturalGradient}. */
export interface NaturalGradientState {
  readonly count: number
  readonly fisher?: Value
}

/** Options of {@link naturalGradient}. */
export interface NaturalGradientOptions {
  readonly learningRate: LearningRate
  /** Tikhonov damping added to the Fisher (default 1e-3). */
  readonly damping?: number
  /** Exponential averaging of observed Fisher matrices (default 0: use the latest). */
  readonly decay?: number
}

/**
 * Natural gradient descent: updates = -lr (F + damping I)^-1 g over the flattened parameters, with F an exponential
 * average of the Fisher matrices (or empirical Fishers of per-example gradients) passed to `observe`. Before any
 * observation it is plain gradient descent.
 *
 * @param options - Learning rate, damping and averaging.
 * @returns The transformation, with `observe(state, { fisher } | { perExampleGrads })`.
 * @concept natural-gradient
 * @example
 * ```ts
 * import { naturalGradient } from 'omai-engine/optim'
 * import { tensor, toArray } from 'omai-engine/tensor'
 * const opt = naturalGradient({ learningRate: 1, damping: 0 })
 * const s = opt.observe(opt.init(tensor([0, 0])), { fisher: [[2, 0], [0, 4]] })
 * toArray(opt.update(tensor([2, 2]), s).updates as never) // => [-1, -0.5]
 * ```
 */
export function naturalGradient(
  options: NaturalGradientOptions,
): PreconditionedTransformation<NaturalGradientState, FisherStatistics> {
  const damping = options.damping ?? 1e-3
  const decay = options.decay ?? 0
  return {
    name: 'naturalGradient',
    init: () => ({ count: 0 }),
    observe: (state, stats) => {
      const f =
        stats.fisher !== undefined
          ? asTensor(stats.fisher)
          : stats.perExampleGrads !== undefined
            ? empiricalFisher(stats.perExampleGrads)
            : undefined
      if (f === undefined)
        throw new OmaiError('value', 'naturalGradient.observe needs a fisher or per-example gradients')
      const fisher = state.fisher === undefined || decay === 0 ? f : add(mul(state.fisher, decay), mul(f, 1 - decay))
      return { ...state, fisher }
    },
    update: (grads, state) => {
      const lr = rate(options.learningRate, state.count)
      const { flat, unravel } = ravel(grads)
      const direction =
        state.fisher === undefined
          ? flat
          : Float64Array.from(toFlat(naturalGradientDirection(state.fisher, flat, damping)))
      return { updates: unravel(direction.map((v) => -lr * v)), state: { ...state, count: state.count + 1 } }
    },
  }
}

/** Kronecker factors of one dense layer's Fisher block: A from its inputs, G from the gradients of its outputs. */
export interface KfacFactors {
  readonly a: Value
  readonly g: Value
}

/**
 * K-FAC's Kronecker factors for a dense layer y = x W^T (+ b): A = E[x x^T] (with a column of ones appended when the
 * layer has a bias) and G = E[dy dy^T], estimated from a batch (Martens and Grosse 2015).
 *
 * @param activations - The layer inputs, N x in.
 * @param outputGrads - Gradients of the loss with respect to the layer outputs, N x out (per example, unscaled).
 * @param bias - Whether the layer has a bias.
 * @returns The factors.
 * @example
 * ```ts
 * import { kfacFactors } from 'omai-engine/optim'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(kfacFactors([[1, 0], [0, 1]], [[2], [0]]).g) // => [[2]]
 * ```
 */
export function kfacFactors(activations: TensorLike, outputGrads: TensorLike, bias = false): KfacFactors {
  const x = asTensor(activations)
  const n = x.shape[0] ?? 1
  const xa = bias ? concat([x, ones([n, 1])], 1) : x
  return { a: div(matmul(transpose(xa), xa), n), g: empiricalFisher(outputGrads) }
}

/**
 * Precondition a dense layer's gradient by its Kronecker-factored Fisher: (G + damping_g I)^-1 dW (A + damping_a I)^-1,
 * with factored Tikhonov damping split by pi = sqrt((tr A / dim A) / (tr G / dim G)) (Martens and Grosse 2015,
 * section 6.3). Equals solving (A kron G + damping) vec(dW) up to the damping split.
 *
 * @param gradW - The weight gradient, out x in (or out x (in + 1) with the bias as the last column).
 * @param factors - The Kronecker factors.
 * @param damping - The damping lambda (each factor gets sqrt(lambda) scaled by pi).
 * @returns The preconditioned gradient.
 * @example
 * ```ts
 * import { kfacPrecondition } from 'omai-engine/optim'
 * import { tensor, toArray } from 'omai-engine/tensor'
 * toArray(kfacPrecondition([[4, 8]], { a: tensor([[2, 0], [0, 4]]), g: tensor([[2]]) }, 0)) // => [[1, 1]]
 * ```
 */
export function kfacPrecondition(gradW: TensorLike, factors: KfacFactors, damping = 1e-3): Value {
  const dimA = factors.a.shape[0] ?? 1
  const dimG = factors.g.shape[0] ?? 1
  const trA = item(trace(factors.a)) / dimA
  const trG = item(trace(factors.g)) / dimG
  const pi = trA > 0 && trG > 0 ? Math.sqrt(trA / trG) : 1
  const root = Math.sqrt(damping)
  const aD = add(factors.a, mul(eye(dimA), pi * root))
  const gD = add(factors.g, mul(eye(dimG), root / pi))
  const left = solve(gD, asTensor(gradW))
  return transpose(solve(aD, transpose(left)))
}

/** One dense layer for {@link kfac}: the paths of its weight (and bias) in the parameter tree and a batch's statistics. */
export interface KfacLayer {
  /** The path of the weight (out x in) in the parameter tree, as `treePaths` names it. */
  readonly weight: string
  /** The path of the bias, if the layer has one. */
  readonly bias?: string
  readonly activations: TensorLike
  readonly outputGrads: TensorLike
}

/** State of {@link kfac}: running factors per weight path, with the path of the layer's bias. */
export interface KfacState {
  readonly count: number
  readonly factors: Readonly<Record<string, KfacFactors & { readonly bias?: string }>>
  readonly momentum?: ParamTree
}

/** Options of {@link kfac}. */
export interface KfacOptions {
  readonly learningRate: LearningRate
  /** Tikhonov damping lambda (default 1e-3). */
  readonly damping?: number
  /** Exponential averaging of the factors (default 0.95). */
  readonly decay?: number
  /** Heavy-ball momentum on the preconditioned updates (default 0). */
  readonly momentum?: number
}

/**
 * K-FAC for dense layers (Martens and Grosse 2015): each observed layer's weight (and bias) gradient is preconditioned
 * by the inverse of its Kronecker-factored Fisher block, with running averages of the factors fed through `observe`
 * (the layer inputs and the per-example gradients of its outputs); other parameters take plain gradient steps.
 *
 * @param options - Learning rate, damping, factor decay and momentum.
 * @returns The transformation, with `observe(state, layers)`.
 * @concept natural-gradient
 * @example
 * ```ts
 * import { kfac } from 'omai-engine/optim'
 * import { tensor, toArray } from 'omai-engine/tensor'
 * const opt = kfac({ learningRate: 1, damping: 0, decay: 0 })
 * const p = { w: tensor([[0, 0]]) }
 * const s = opt.observe(opt.init(p), [{ weight: 'w', activations: [[1, 0], [0, 2]], outputGrads: [[1], [1]] }])
 * toArray((opt.update({ w: tensor([[1, 4]]) }, s).updates as { w: never }).w) // => [[-2, -2]]
 * ```
 */
export function kfac(options: KfacOptions): PreconditionedTransformation<KfacState, readonly KfacLayer[]> {
  const damping = options.damping ?? 1e-3
  const decay = options.decay ?? 0.95
  const beta = options.momentum ?? 0
  return {
    name: 'kfac',
    init: (params) => ({ count: 0, factors: {}, ...(beta > 0 ? { momentum: treeZeros(params) } : {}) }),
    observe: (state, layers) => {
      const factors: Record<string, KfacFactors & { readonly bias?: string }> = { ...state.factors }
      for (const layer of layers) {
        const f = kfacFactors(layer.activations, layer.outputGrads, layer.bias !== undefined)
        const old = factors[layer.weight]
        const avg =
          old === undefined || decay === 0
            ? f
            : { a: add(mul(old.a, decay), mul(f.a, 1 - decay)), g: add(mul(old.g, decay), mul(f.g, 1 - decay)) }
        factors[layer.weight] = layer.bias === undefined ? avg : { ...avg, bias: layer.bias }
      }
      return { ...state, factors }
    },
    update: (grads, state) => {
      const lr = rate(options.learningRate, state.count)
      const paths = Object.keys(treePaths(grads))
      const { leaves, def } = treeFlatten(grads)
      const index = new Map(paths.map((p, i) => [p, i]))
      const out: Value[] = [...leaves]
      for (const [weight, f] of Object.entries(state.factors)) {
        const wi = index.get(weight)
        if (wi === undefined) continue
        const w = leaves[wi] as Value
        const bi = f.bias === undefined ? undefined : index.get(f.bias)
        if (f.bias !== undefined && bi === undefined) throw new OmaiError('value', `kfac: no bias at '${f.bias}'`)
        const inputs = (w.shape[1] ?? 0) + (bi === undefined ? 0 : 1)
        if (f.a.shape[0] !== inputs || f.g.shape[0] !== w.shape[0])
          throw new OmaiError('shape', `kfac: factors for '${weight}' do not match its gradient`)
        if (bi === undefined) {
          out[wi] = kfacPrecondition(w, f, damping)
        } else {
          const joint = concat([w, reshape(leaves[bi] as Value, [-1, 1])], 1)
          const pre = kfacPrecondition(joint, f, damping)
          const cols = w.shape[1] ?? 0
          out[wi] = sliceAxis(pre, 1, 0, cols)
          out[bi] = reshape(sliceAxis(pre, 1, cols, cols + 1), (leaves[bi] as Value).shape)
        }
      }
      const pre = treeUnflatten(def, out)
      if (beta === 0) return { updates: treeMap((u) => mul(u, -lr), pre), state: { ...state, count: state.count + 1 } }
      const momentum = treeMap((u, m) => add(mul(m, beta), u), pre, state.momentum)
      return { updates: treeMap((m) => mul(m, -lr), momentum), state: { ...state, count: state.count + 1, momentum } }
    },
  }
}

/** State of {@link shampoo}. */
export interface ShampooState {
  readonly count: number
  /** Per leaf, the per-mode statistics H_i (initialised to eps I). */
  readonly stats: readonly (readonly Value[])[]
  /** Per leaf, the per-mode preconditioners H_i^(-1/(2k)) (identity for modes above maxDim). */
  readonly roots: readonly (readonly (Value | undefined)[])[]
  readonly momentum?: ParamTree
}

/** Options of {@link shampoo}. */
export interface ShampooOptions {
  readonly learningRate: LearningRate
  /** The statistics start at eps I (default 1e-4). */
  readonly eps?: number
  /** Recompute the inverse roots every this many steps (default 1). */
  readonly preconditionEvery?: number
  /** Modes longer than this are not preconditioned (default 1024). */
  readonly maxDim?: number
  /** Heavy-ball momentum on the preconditioned gradient (default 0). */
  readonly momentum?: number
}

/**
 * Shampoo (Gupta, Koren and Singer 2018): for a k-dimensional gradient G, each mode i keeps H_i += G_(i) G_(i)^T
 * (the mode-i unfolding) and the update preconditions G along every mode by H_i^(-1/(2k)): L^(-1/4) G R^(-1/4) for a
 * matrix, H^(-1/2) g for a vector.
 *
 * @param options - Learning rate, eps, refresh period, the largest preconditioned mode and momentum.
 * @returns The optimiser.
 * @concept preconditioning
 * @example
 * ```ts
 * import { shampoo } from 'omai-engine/optim'
 * import { tensor, toArray } from 'omai-engine/tensor'
 * const opt = shampoo({ learningRate: 1 })
 * toArray(opt.update(tensor([3, 4]), opt.init(tensor([0, 0]))).updates as never).map((v) => Math.round(v * 1e4) / 1e4) // => [-0.6, -0.8]
 * ```
 */
export function shampoo(options: ShampooOptions): GradientTransformation<ShampooState> {
  const eps = options.eps ?? 1e-4
  const every = options.preconditionEvery ?? 1
  const maxDim = options.maxDim ?? 1024
  const beta = options.momentum ?? 0
  const unfold = (g: Value, i: number): Value => reshape(moveaxis(g, i, 0), [g.shape[i] ?? 0, -1])
  return {
    name: 'shampoo',
    init: (params) => ({
      count: 0,
      stats: treeLeaves(params).map((p) => p.shape.map((d) => (d <= maxDim ? mul(eye(d), eps) : eye(0)))),
      roots: treeLeaves(params).map((p) => p.shape.map(() => undefined)),
      ...(beta > 0 ? { momentum: treeZeros(params) } : {}),
    }),
    update: (grads, state) => {
      const lr = rate(options.learningRate, state.count)
      const refresh = state.count % every === 0
      const { leaves, def } = treeFlatten(grads)
      const stats: Value[][] = []
      const roots: (Value | undefined)[][] = []
      const pre = leaves.map((g, li) => {
        const k = g.shape.length
        const ls: Value[] = []
        const rs: (Value | undefined)[] = []
        let out = g
        for (let i = 0; i < k; i++) {
          const d = g.shape[i] ?? 0
          const old = state.stats[li]?.[i] as Value
          if (d > maxDim) {
            ls.push(old)
            rs.push(undefined)
            continue
          }
          const gi = unfold(g, i)
          const h = add(old, matmul(gi, transpose(gi)))
          ls.push(h)
          const r = refresh ? matrixInversePthRoot(h, 2 * k) : (state.roots[li]?.[i] ?? matrixInversePthRoot(h, 2 * k))
          rs.push(r)
          out = moveaxis(tensordot(r, moveaxis(out, i, 0), 1), 0, i)
        }
        stats.push(ls)
        roots.push(rs)
        return out
      })
      const tree = treeUnflatten(def, pre)
      const next = { count: state.count + 1, stats, roots }
      if (beta === 0) return { updates: treeMap((u) => mul(u, -lr), tree), state: next }
      const momentum = treeMap((u, m) => add(mul(m, beta), u), tree, state.momentum)
      return { updates: treeMap((m) => mul(m, -lr), momentum), state: { ...next, momentum } }
    },
  }
}

/**
 * A fixed preconditioner: updates = P g over the flattened parameters, with P a matrix, a vector (diagonal) or a
 * function of the flat gradient (for example a solve with a known Hessian approximation). The function takes the flat
 * gradient as a tensor and returns a tensor-like (migration, D-009: it took a `Float64Array` and returned an
 * `ArrayLike<number>`).
 *
 * @param preconditioner - The matrix, diagonal or function.
 * @throws OmaiError with code `shape` when the function returns something that is not a vector.
 * @returns The transformation.
 * @concept preconditioning
 * @example
 * ```ts
 * import { preconditionBy } from 'omai-engine/optim'
 * import { tensor, toArray } from 'omai-engine/tensor'
 * toArray(preconditionBy([0.5, 2]).update(tensor([2, 2]), undefined).updates as never) // => [1, 4]
 * ```
 */
export function preconditionBy(
  preconditioner: TensorLike | ((grad: Tensor<'float64', [number]>) => TensorLike),
): GradientTransformation<undefined> {
  const apply =
    typeof preconditioner === 'function'
      ? (g: Float64Array): ArrayLike<number> =>
          toVector(preconditioner(fromStorage(g, [g.length], 'float64')), 'preconditionBy')
      : (g: Float64Array): ArrayLike<number> => {
          const p = asTensor(preconditioner)
          const gv = asTensor(g)
          return toFlat(p.shape.length === 1 ? mul(p, gv) : matmul(p, gv))
        }
  return {
    name: 'preconditionBy',
    init: () => undefined,
    update: (grads, state) => {
      const { flat, unravel } = ravel(grads)
      return { updates: unravel(apply(flat)), state }
    },
  }
}

/**
 * Normalise a gradient by the square root of a diagonal curvature estimate: g / (sqrt(d) + eps), the diagonal
 * preconditioning shared by Adagrad, RMSprop and Adam, exposed for custom estimates d (for example a diagonal Fisher).
 *
 * @param grads - The gradients.
 * @param diagonal - The diagonal estimate, with the structure of the gradients.
 * @param eps - Added to the square root.
 * @returns The preconditioned gradients.
 * @example
 * ```ts
 * import { diagonalPrecondition } from 'omai-engine/optim'
 * import { tensor, toArray } from 'omai-engine/tensor'
 * toArray(diagonalPrecondition(tensor([2, 2]), tensor([4, 16]), 0) as never) // => [1, 0.5]
 * ```
 */
export function diagonalPrecondition(grads: ParamTree, diagonal: ParamTree, eps = 1e-8): ParamTree {
  return treeMap((g, d) => div(g, add(sqrt(d), eps)), grads, diagonal)
}
