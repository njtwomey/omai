/**
 * Loss landscapes (Li, Xu, Taylor, Studer and Goldstein 2018): one-dimensional slices along a line between two
 * parameter vectors (the barrier between two minima, and along a bent Bezier path that avoids it) and slices along
 * random directions, one- or two-dimensional, with filter normalisation so that the pictures of two networks, or of one
 * network at two scales, can be compared; and the sharpness, the largest Hessian eigenvalue, by Lanczos or power
 * iteration on Hessian-vector products. The loss is any scalar function of a parameter tree.
 */

import { MdmlError } from '../base/index.ts'
import { dnrm2 } from '../numerics/linalg/blas.ts'
import { eigsh } from '../numerics/sparse/index.ts'
import { foldIn, type Key, normal } from '../random/index.ts'
import {
  add,
  asTensor,
  div,
  fromStorage,
  item,
  maximum,
  mul,
  norm,
  primalOf,
  reshape,
  type Tensor,
  type TensorLike,
  toFlat,
  toVector,
  transpose,
  treeFlatten,
  treeMap,
  treeUnflatten,
  type Value,
} from '../tensor/index.ts'
import { flatObjective, type Objective } from './minimize/objective.ts'
import type { ParamTree } from './transform.ts'

/** A loss as a function of a parameter tree: a scalar. */
export type LossFunction = Objective

/** How a direction is normalised against the parameters it is applied to. */
export interface FilterNormOptions {
  /**
   * `filter` (default): each filter (a slice along `filterAxis`) of the direction gets the norm of the matching filter
   * of the parameters; `layer`: each tensor gets the norm of the parameter tensor; `none`: no normalisation.
   */
  readonly mode?: 'filter' | 'layer' | 'none'
  /** The axis that indexes filters (0 for PyTorch's `[out, in, ...]` weights; default 0). */
  readonly filterAxis?: number
  /** What to do with one-dimensional tensors (biases, normalisation scales): zero them as Li et al. do (default), or normalise them as layers. */
  readonly biases?: 'zero' | 'normalise'
}

const lossAt = (loss: LossFunction, params: ParamTree): number => item(primalOf(loss(params)))

const leafNorm = (v: Value): Value => norm(reshape(v, [-1]), 2)

/** Rescale one leaf of a direction against the matching parameter leaf. */
function normaliseLeaf(d: Value, theta: Value, options: FilterNormOptions): Value {
  const mode = options.mode ?? 'filter'
  if (mode === 'none') return d
  if (theta.shape.length <= 1 && (options.biases ?? 'zero') === 'zero') return mul(d, 0)
  if (mode === 'layer' || theta.shape.length <= 1) return mul(d, div(leafNorm(theta), maximum(leafNorm(d), 1e-30)))
  const axis = options.filterAxis ?? 0
  const rank = theta.shape.length
  if (axis < 0 || axis >= rank)
    throw new MdmlError('value', `filterNormalise: filterAxis ${axis} is outside a rank-${rank} tensor`)
  // View each tensor as [filters, rest] by moving the filter axis first.
  const perm = [axis, ...Array.from({ length: rank }, (_, i) => i).filter((i) => i !== axis)]
  const inverse = perm.map((_, i) => perm.indexOf(i))
  const moved = (v: Value): Value => transpose(v, perm)
  const rows = (v: Value): Value => reshape(moved(v), [theta.shape[axis] ?? 1, -1])
  const scale = div(norm(rows(theta), 2, 1, true), maximum(norm(rows(d), 2, 1, true), 1e-30))
  return transpose(reshape(mul(rows(d), scale), [...moved(d).shape]), inverse)
}

/**
 * Filter-normalise a direction against parameters (Li et al. 2018): every filter of the direction is rescaled to the
 * norm of the corresponding filter of the parameters, so that a step of a given size means the same relative change in
 * every filter, and the slice does not depend on the scale of the weights.
 *
 * @param direction - A tree shaped like the parameters.
 * @param params - The point the direction is applied at.
 * @param options - Mode, filter axis and the treatment of biases.
 * @returns The normalised direction.
 * @concept loss-landscape
 * @example
 * ```ts
 * import { filterNormalise } from 'mdml-engine/optim'
 * import { tensor, toArray } from 'mdml-engine/tensor'
 * toArray(filterNormalise(tensor([[1, 1], [2, 0]]), tensor([[3, 4], [0, 1]]))) // => [[3.5355339059327378, 3.5355339059327378], [1, 0]]
 * ```
 */
export function filterNormalise<P extends ParamTree>(direction: P, params: P, options: FilterNormOptions = {}): P {
  const d = treeFlatten(asTree(direction))
  const t = treeFlatten(asTree(params))
  const out = d.leaves.map((leaf, i) => normaliseLeaf(leaf, t.leaves[i] as Value, options))
  return treeUnflatten(d.def, out) as P
}

/** Numeric leaves of a user-supplied tree as tensors. */
function asTree(x: unknown): ParamTree {
  const walk = (v: unknown): unknown => {
    if (Array.isArray(v) && v.length > 0 && v.every((e) => Array.isArray(e) || typeof e === 'number'))
      return asTensor(v as never)
    if (typeof v === 'number') return asTensor(v)
    if (Array.isArray(v)) return v.map(walk)
    if (typeof v === 'object' && v !== null && !('shape' in v))
      return Object.fromEntries(Object.entries(v).map(([k, e]) => [k, walk(e)]))
    return v
  }
  return walk(x) as ParamTree
}

/**
 * A random Gaussian direction in parameter space, filter-normalised against the parameters.
 *
 * @param params - The point (a tree of tensors).
 * @param key - The key.
 * @param options - Normalisation options.
 * @returns A direction shaped like the parameters.
 * @concept loss-landscape
 * @example
 * ```ts
 * import { randomDirection } from 'mdml-engine/optim'
 * import { key } from 'mdml-engine/random'
 * import { tensor } from 'mdml-engine/tensor'
 * randomDirection(tensor([[3, 4], [0, 2]]), key(0)).shape // => [2, 2]
 * ```
 */
export function randomDirection<P extends ParamTree>(params: P, key: Key, options: FilterNormOptions = {}): P {
  const t = treeFlatten(asTree(params))
  const noise = t.leaves.map((leaf, i) => normal(foldIn(key, i), [...leaf.shape]) as Value)
  const raw = treeUnflatten(t.def, noise) as P
  return filterNormalise(raw, asTree(params) as P, options)
}

const along = (params: ParamTree, direction: ParamTree, alpha: number): ParamTree =>
  treeMap((p, d) => add(p, mul(d, alpha)), asTree(params), asTree(direction))

/** The loss along a line through the parameters. */
export interface Slice1d {
  readonly alphas: readonly number[]
  readonly losses: readonly number[]
}

/**
 * The loss along one direction: L(theta + alpha d) for each alpha. With alpha = 0 it is the loss at the point. Takes
 * the grid as a tensor-like (migration, D-009: `alphas` was `readonly number[]`).
 *
 * @param loss - The loss.
 * @param params - The centre.
 * @param direction - The direction (for example from {@link randomDirection}).
 * @param alphas - Where to evaluate: a vector (a tensor, a typed array or a list).
 * @returns The alphas and losses.
 * @throws MdmlError with code `shape` unless `alphas` has rank 1, and `not-differentiable` for a traced one.
 * @concept loss-landscape
 * @example
 * ```ts
 * import { slice1d } from 'mdml-engine/optim'
 * import { square, sum, tensor, type Value } from 'mdml-engine/tensor'
 * slice1d((x: Value) => sum(square(x)), tensor([1, 0]), tensor([1, 0]), [-1, 0, 1]).losses // => [0, 1, 4]
 * ```
 */
export function slice1d(loss: LossFunction, params: ParamTree, direction: ParamTree, alphas: TensorLike): Slice1d {
  const grid = toVector(alphas, 'slice1d')
  return { alphas: grid, losses: grid.map((a) => lossAt(loss, along(params, direction, a))) }
}

/** A grid of losses over two directions. */
export interface Slice2d {
  readonly alphas: readonly number[]
  readonly betas: readonly number[]
  /** Losses `[alphas.length, betas.length]`: entry `[i, j]` is L(theta + alphas[i] d1 + betas[j] d2). */
  readonly losses: Tensor
}

/**
 * The loss over a plane through the parameters, spanned by two directions (Li et al.'s contour plots): L(theta + alpha
 * d1 + beta d2) on a grid. Takes the grids as tensor-likes (migration, D-009: `alphas` and `betas` were
 * `readonly number[]`).
 *
 * @param loss - The loss.
 * @param params - The centre.
 * @param d1 - First direction.
 * @param d2 - Second direction.
 * @param alphas - Grid along `d1`: a vector (a tensor, a typed array or a list).
 * @param betas - Grid along `d2`: the same.
 * @returns The grid of losses.
 * @throws MdmlError with code `shape` unless a grid has rank 1, and `not-differentiable` for a traced one.
 * @concept loss-landscape
 * @example
 * ```ts
 * import { slice2d } from 'mdml-engine/optim'
 * import { square, sum, tensor, type Value } from 'mdml-engine/tensor'
 * slice2d((x: Value) => sum(square(x)), tensor([0, 0]), tensor([1, 0]), tensor([0, 1]), [0, 1], [0, 2]).losses.shape // => [2, 2]
 * ```
 */
export function slice2d(
  loss: LossFunction,
  params: ParamTree,
  d1: ParamTree,
  d2: ParamTree,
  alphas: TensorLike,
  betas: TensorLike,
): Slice2d {
  const as = toVector(alphas, 'slice2d')
  const bs = toVector(betas, 'slice2d')
  const data = new Float64Array(as.length * bs.length)
  as.forEach((a, i) => {
    const row = along(params, d1, a)
    bs.forEach((b, j) => {
      data[i * bs.length + j] = lossAt(loss, along(row, d2, b))
    })
  })
  return { alphas: as, betas: bs, losses: fromStorage(data, [as.length, bs.length], 'float64') }
}

/**
 * A filter-normalised slice, 1-D or 2-D, around a point: draws one or two random directions from the key, normalises
 * them, and evaluates the loss on the grid. The directions are returned so the slice can be redrawn after training, or
 * so a trajectory can be projected onto them. Takes the grids as tensor-likes (migration, D-009: they were
 * `readonly number[]`).
 *
 * @param loss - The loss.
 * @param params - The centre.
 * @param key - Key for the directions.
 * @param grid - Alphas (and betas for a plane).
 * @param options - Normalisation.
 * @returns The directions and the slice.
 * @concept loss-landscape
 * @example
 * ```ts
 * import { randomSlice } from 'mdml-engine/optim'
 * import { key } from 'mdml-engine/random'
 * import { square, sum, tensor, type Value } from 'mdml-engine/tensor'
 * randomSlice((x: Value) => sum(square(x)), tensor([[1, 2], [3, 4]]), key(0), { alphas: [0, 0.5] }).slice.losses[0] // => 30
 * ```
 */
export function randomSlice(
  loss: LossFunction,
  params: ParamTree,
  key: Key,
  grid: { readonly alphas: TensorLike; readonly betas?: TensorLike },
  options: FilterNormOptions = {},
): { readonly directions: readonly ParamTree[]; readonly slice: Slice1d | Slice2d } {
  const d1 = randomDirection(params, foldIn(key, 0), options)
  if (grid.betas === undefined) return { directions: [d1], slice: slice1d(loss, params, d1, grid.alphas) }
  const d2 = randomDirection(params, foldIn(key, 1), options)
  return { directions: [d1, d2], slice: slice2d(loss, params, d1, d2, grid.alphas, grid.betas) }
}

/**
 * The loss along the straight line between two parameter vectors: L((1 - t) a + t b), with t = 0 at `a` and t = 1 at
 * `b` (values outside [0, 1] extrapolate). Takes the positions as a tensor-like (migration, D-009: `ts` was
 * `readonly number[]`).
 *
 * @param loss - The loss.
 * @param a - One endpoint.
 * @param b - The other.
 * @param ts - Where to evaluate: a vector (a tensor, a typed array or a list).
 * @returns The positions and losses.
 * @throws MdmlError with code `shape` unless `ts` has rank 1, and `not-differentiable` for a traced one.
 * @concept mode-connectivity
 * @example
 * ```ts
 * import { interpolationSlice } from 'mdml-engine/optim'
 * import { square, sum, tensor, type Value } from 'mdml-engine/tensor'
 * interpolationSlice((x: Value) => sum(square(x)), tensor([-1]), tensor([1]), [0, 0.5, 1]).losses // => [1, 0, 1]
 * ```
 */
export function interpolationSlice(loss: LossFunction, a: ParamTree, b: ParamTree, ts: TensorLike): Slice1d {
  const grid = toVector(ts, 'interpolationSlice')
  const diff = treeMap((x, y) => add(y, mul(x, -1)), asTree(a), asTree(b))
  return { alphas: grid, losses: grid.map((t) => lossAt(loss, along(a, diff, t))) }
}

/**
 * The loss barrier of a path between two minima: the largest amount by which the loss along the path exceeds the
 * straight line between its end losses, max_t [L(t) - ((1 - t) L(0) + t L(1))] (zero when the path is no higher than
 * the chord). Evaluate it on {@link interpolationSlice} for the linear path, or on {@link bezierSlice} for a bent one.
 *
 * @param slice - A path slice with positions in [0, 1].
 * @returns The barrier height (at least 0).
 * @concept mode-connectivity
 * @example
 * ```ts
 * import { barrier } from 'mdml-engine/optim'
 * barrier({ alphas: [0, 0.5, 1], losses: [1, 3, 1] }) // => 2
 * ```
 */
export function barrier(slice: Slice1d): number {
  const n = slice.losses.length
  if (n < 2) return 0
  const [t0, t1] = [slice.alphas[0] as number, slice.alphas[n - 1] as number]
  const [l0, l1] = [slice.losses[0] as number, slice.losses[n - 1] as number]
  let worst = 0
  slice.losses.forEach((l, i) => {
    const t = ((slice.alphas[i] as number) - t0) / (t1 - t0)
    worst = Math.max(worst, l - ((1 - t) * l0 + t * l1))
  })
  return worst
}

/**
 * The loss along a quadratic Bezier curve (Garipov et al. 2018): from `a` to `b` through a control point `c`,
 * (1 - t)^2 a + 2 (1 - t) t c + t^2 b. With the control point chosen to lower the loss the curve connects two minima
 * with no barrier where the straight line has one; choosing it is a training problem, this evaluates a given one.
 * Takes the positions as a tensor-like (migration, D-009: `ts` was `readonly number[]`).
 *
 * @param loss - The loss.
 * @param a - Start (a minimum).
 * @param c - Control point.
 * @param b - End.
 * @param ts - Positions in [0, 1]: a vector (a tensor, a typed array or a list).
 * @returns The positions and losses.
 * @throws MdmlError with code `shape` unless `ts` has rank 1, and `not-differentiable` for a traced one.
 * @concept mode-connectivity
 * @example
 * ```ts
 * import { bezierSlice } from 'mdml-engine/optim'
 * import { sum, tensor, type Value } from 'mdml-engine/tensor'
 * bezierSlice((x: Value) => sum(x), tensor([0]), tensor([2]), tensor([0]), [0, 0.5, 1]).losses // => [0, 1, 0]
 * ```
 */
export function bezierSlice(loss: LossFunction, a: ParamTree, c: ParamTree, b: ParamTree, ts: TensorLike): Slice1d {
  const grid = toVector(ts, 'bezierSlice')
  const point = (t: number): ParamTree =>
    treeMap(
      (x, y, z) => add(add(mul(x, (1 - t) ** 2), mul(y, 2 * (1 - t) * t)), mul(z, t * t)),
      asTree(a),
      asTree(c),
      asTree(b),
    )
  return { alphas: grid, losses: grid.map((t) => lossAt(loss, point(t))) }
}

/** The top Hessian eigenvalue and its eigenvector. */
export interface Sharpness {
  /** The largest eigenvalue of the Hessian (the sharpness). */
  readonly value: number
  /** The eigenvector, as a tree shaped like the parameters. */
  readonly vector: ParamTree
  /** Hessian-vector products used. */
  readonly products: number
}

/**
 * The sharpness of the loss at a point: the largest eigenvalue of the Hessian, found without forming it, from
 * Hessian-vector products (forward over reverse differentiation) by Lanczos (`eigsh`, the largest algebraic eigenvalue,
 * the default) or by power iteration (the eigenvalue of largest magnitude, the one that decides stability at the edge of
 * stability, where gradient descent with step eta is stable only while it is below 2 / eta).
 *
 * @param loss - The loss.
 * @param params - The point.
 * @param options - `method`, power iteration `iterations` and `tol`, and a start vector `v0` (takes a tensor-like;
 *   migration, D-009: was `ArrayLike<number>`).
 * @returns The eigenvalue, its eigenvector as a tree, and the number of products.
 * @concept sharpness
 * @example
 * ```ts
 * import { sharpness } from 'mdml-engine/optim'
 * import { mul, sum, tensor, type Value } from 'mdml-engine/tensor'
 * // L = x0^2 + 3 x1^2 has Hessian diag(2, 6).
 * sharpness((x: Value) => sum(mul(mul(x, x), tensor([1, 3]))), tensor([0.5, 0.5])).value.toFixed(6) // => '6.000000'
 * ```
 */
export function sharpness(
  loss: LossFunction,
  params: ParamTree,
  options: {
    readonly method?: 'lanczos' | 'power'
    readonly iterations?: number
    readonly tol?: number
    readonly v0?: TensorLike
  } = {},
): Sharpness {
  const v0 = options.v0 === undefined ? undefined : toVector(options.v0, 'sharpness')
  const f = flatObjective(loss, asTree(params))
  const x = f.x0
  let products = 0
  const matvec = (v: ArrayLike<number>): Float64Array => {
    products++
    return f.hvp(x, v)
  }
  if (f.n === 1) {
    const h = matvec([1])
    return { value: h[0] as number, vector: f.unravel([1]), products }
  }
  if ((options.method ?? 'lanczos') === 'lanczos') {
    const op = {
      shape: [f.n, f.n] as const,
      matvec: (v: Value): Value => fromStorage(matvec(toVec(v)), [f.n], 'float64'),
    }
    const r = eigsh(op, 1, {
      which: 'LA',
      ...(v0 === undefined ? {} : { v0 }),
      tol: options.tol ?? 1e-10,
    })
    const top = r.values.data[0] as number
    // The eigenvectors are the columns of an [n, 1] matrix, so the single column is the data in order.
    return { value: top, vector: f.unravel(r.vectors.data), products }
  }
  let v = Float64Array.from(v0 ?? Array.from({ length: f.n }, (_, i) => Math.cos(1.3 * i + 0.4)))
  const s = dnrm2(v)
  v = v.map((q) => q / s)
  let value = Number.NaN
  for (let it = 0; it < (options.iterations ?? 200); it++) {
    const hv = matvec(v)
    const next = v.reduce((acc, q, i) => acc + q * (hv[i] as number), 0)
    const n = dnrm2(hv)
    v = hv.map((q) => q / (n === 0 ? 1 : n))
    if (Math.abs(next - value) <= (options.tol ?? 1e-10) * Math.max(1, Math.abs(next))) {
      value = next
      break
    }
    value = next
  }
  return { value, vector: f.unravel(v), products }
}

const toVec = (v: Value): Float64Array => Float64Array.from(toFlat(primalOf(v)))

/**
 * The edge-of-stability threshold 2 / eta of gradient descent with step size eta: the sharpness above which the
 * iteration diverges along the top eigendirection.
 *
 * @param learningRate - The step size eta.
 * @returns 2 / eta.
 * @concept sharpness
 * @example
 * ```ts
 * import { stabilityThreshold } from 'mdml-engine/optim'
 * stabilityThreshold(0.1) // => 20
 * ```
 */
export function stabilityThreshold(learningRate: number): number {
  return 2 / learningRate
}
