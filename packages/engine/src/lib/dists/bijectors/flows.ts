/**
 * The layers of normalising flows beyond affine coupling: planar and radial flows (Rezende and Mohamed), activation
 * normalisation (ActNorm) and Glow's invertible linear layers (a free matrix, and the LU parameterisation with a cheap
 * log-determinant), inverse autoregressive flow and a MADE conditioner for autoregressive layers, the monotone
 * rational-quadratic spline of neural spline flows (as an elementwise bijector and as a coupling layer), and residual
 * flows, whose log-determinant is a power series of the Jacobian's traces, exact or by Hutchinson's estimator. Each is a
 * `Bijector` with an inverse and a log-determinant, and differentiable in its parameters.
 */

import { jacobian, jvp } from '../../autodiff/index.ts'
import { MdmlError } from '../../base/index.ts'
import { inv, lu, norm, slogdet } from '../../numerics/linalg/index.ts'
import { foldIn, type Key, normal } from '../../random/index.ts'
import {
  add,
  broadcastTo,
  cast,
  concat,
  cumsum,
  div,
  exp,
  expandDims,
  eye,
  greaterEqual,
  lessEqual,
  log,
  logicalAnd,
  matmul,
  maximum,
  mean,
  minimum,
  mul,
  neg,
  promoteFloat,
  reshape,
  sliceAxis,
  softmax,
  softplus,
  sqrt,
  square,
  squeeze,
  stack,
  stopGradient,
  sub,
  sum,
  type TensorLike,
  take,
  takeAlongAxis,
  tanh,
  toFlat,
  transpose,
  tril,
  triu,
  type Value,
  where,
} from '../../tensor/index.ts'
import { type Bijector, defineBijector } from './bijector.ts'
import { madeMasks, type ShiftLogScale } from './structured.ts'

const lastSize = (x: Value): number => x.shape[x.shape.length - 1] ?? 0
/** Zeros shaped like the batch of a vector `x` (its last axis reduced). */
const batchZeros = (x: Value): Value => sum(mul(x, 0), -1)
/** A diagonal matrix with the entries of a vector. */
const diagonalMatrix = (v: Value): Value => mul(eye(lastSize(v)), v)

// --- Planar and radial flows -----------------------------------------------------------------------------------------------

/** Bisection of a function increasing in a tensor argument: the root of f(a) = target, elementwise. */
function bisect(f: (a: Value) => Value, target: Value, low: Value, high: Value): Value {
  let [lo, hi] = [low, high]
  for (let i = 0; i < 80; i++) {
    const mid = mul(add(lo, hi), 0.5)
    const below = lessEqual(f(mid), target)
    lo = where(below, mid, lo)
    hi = where(below, hi, mid)
  }
  return mul(add(lo, hi), 0.5)
}

/**
 * A planar flow (Rezende and Mohamed, 2015): y = x + u_hat tanh(w . x + b), a contraction and expansion of the space along
 * a hyperplane, with u_hat adjusted so that w . u_hat >= -1 (which makes the map invertible). The log-determinant is
 * log |1 + u_hat . w (1 - tanh^2(w . x + b))|; the inverse solves a monotone scalar equation for w . x by bisection
 * (with one Newton step carrying the gradients to the parameters).
 *
 * @param u - The direction `[d]`.
 * @param w - The normal `[d]`.
 * @param b - The offset (scalar).
 * @returns The bijector (vector-valued).
 * @concept normalizing-flows
 * @example
 * ```ts
 * import { bijectors } from 'mdml-engine/dists'
 * import { toArray } from 'mdml-engine/tensor'
 * const layer = bijectors.planar([1, 0], [1, 0], 0)
 * Math.abs(toArray(layer.inverse(layer.forward([1, 2])))[0] - 1) < 1e-10 // => true
 * ```
 */
export function planar(u: TensorLike, w: TensorLike, b: TensorLike): Bijector {
  const [uu, ww, bb] = promoteFloat(u, w, b) as [Value, Value, Value]
  const wTu = sum(mul(ww, uu))
  const normSq = sum(square(ww))
  // u_hat = u + (softplus(w.u) - 1 - w.u) w / |w|^2, so that w . u_hat = softplus(w.u) - 1 >= -1.
  const uHat = add(uu, mul(div(sub(sub(softplus(wTu), 1), wTu), normSq), ww))
  const c = sum(mul(ww, uHat))
  return defineBijector({
    name: 'planar',
    inputRank: 1,
    forward: (x) => add(x, mul(expandDims(tanh(add(sum(mul(x, ww), -1), bb)), -1), uHat)),
    inverse: (y) => {
      const wy = sum(mul(y, ww), -1)
      const g = (a: Value): Value => add(a, mul(c, tanh(add(a, bb))))
      const bound = add(neg(minimum(c, 0)), 1)
      const root = bisect(g, stopGradient(wy), sub(stopGradient(wy), bound), add(stopGradient(wy), bound))
      // One Newton step from the constant root: the value is unchanged, the gradients are those of the implicit root.
      const slope = add(mul(c, sub(1, square(tanh(add(root, bb))))), 1)
      const alpha = sub(root, div(sub(g(root), wy), slope))
      return sub(y, mul(expandDims(tanh(add(alpha, bb)), -1), uHat))
    },
    forwardLogDetJacobian: (x) => log(add(mul(sub(1, square(tanh(add(sum(mul(x, ww), -1), bb)))), c), 1)),
  })
}

/**
 * A radial flow (Rezende and Mohamed, 2015): y = x + beta / (alpha + r) (x - x0) with r = |x - x0|, which contracts or
 * expands the space around a centre. alpha > 0 and beta_hat = softplus(beta) - alpha keeps the map invertible; the
 * log-determinant is (d - 1) log(1 + beta h) + log(1 + beta h + beta h' r) with h = 1 / (alpha + r), and the inverse
 * solves for the radius by bisection.
 *
 * @param center - The centre x0 `[d]`.
 * @param alpha - The softening, a positive scalar.
 * @param beta - An unconstrained scalar (mapped to beta_hat >= -alpha).
 * @returns The bijector (vector-valued).
 * @example
 * ```ts
 * import { bijectors } from 'mdml-engine/dists'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(bijectors.radial([0, 0], 1, -50).forward([0, 0])) // => [0, 0]
 * ```
 */
export function radial(center: TensorLike, alpha: TensorLike, beta: TensorLike): Bijector {
  const [x0, a, rawBeta] = promoteFloat(center, alpha, beta) as [Value, Value, Value]
  const betaHat = sub(softplus(rawBeta), a)
  const dim = lastSize(x0)
  const radius = (v: Value): Value => sqrt(sum(square(sub(v, x0)), -1))
  return defineBijector({
    name: 'radial',
    inputRank: 1,
    forward: (x) => {
      const r = radius(x)
      return add(x, mul(expandDims(div(betaHat, add(a, r)), -1), sub(x, x0)))
    },
    inverse: (y) => {
      const ry = radius(y)
      const m = (r: Value): Value => add(r, div(mul(betaHat, r), add(a, r)))
      const high = mul(add(stopGradient(ry), 1e-12), maximum(div(a, maximum(add(a, stopGradient(betaHat)), 1e-12)), 1))
      const root = bisect(m, stopGradient(ry), mul(stopGradient(ry), 0), add(high, 1e-9))
      const slope = add(1, div(mul(betaHat, a), square(add(a, root))))
      const r = sub(root, div(sub(m(root), ry), slope))
      return add(x0, div(sub(y, x0), expandDims(add(1, div(betaHat, add(a, r))), -1)))
    },
    forwardLogDetJacobian: (x) => {
      const r = radius(x)
      const h = div(1, add(a, r))
      const bh = mul(betaHat, h)
      const first = mul(log(add(bh, 1)), dim - 1)
      return add(first, log(add(add(bh, 1), mul(mul(betaHat, neg(square(h))), r))))
    },
  })
}

// --- ActNorm and invertible linear layers ----------------------------------------------------------------------------------

/**
 * Activation normalisation (ActNorm, Kingma and Dhariwal, 2018): a per-feature affine map y = (x + bias) exp(logScale)
 * whose parameters are initialised from a batch so that its output has zero mean and unit variance
 * ({@link actNormInitialise}) and then trained.
 *
 * @param bias - The shift added first `[d]`.
 * @param logScale - The log of the per-feature scale `[d]`.
 * @param options - `positions`, how many positions share the parameters (the pixels of an image: the log-determinant is
 *   multiplied by it; default 1).
 * @returns The bijector acting on the last axis.
 * @concept normalizing-flows
 * @example
 * ```ts
 * import { bijectors } from 'mdml-engine/dists'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(bijectors.actNorm([1, 0], [0, Math.LN2]).forward([1, 3])) // => [2, 6]
 * ```
 */
export function actNorm(
  bias: TensorLike,
  logScale: TensorLike,
  options: { readonly positions?: number } = {},
): Bijector {
  const [b, s] = promoteFloat(bias, logScale) as [Value, Value]
  const positions = options.positions ?? 1
  return defineBijector({
    name: 'actNorm',
    inputRank: 1,
    forward: (x) => mul(add(x, b), exp(s)),
    inverse: (y) => sub(mul(y, exp(neg(s))), b),
    forwardLogDetJacobian: (x) => mul(add(batchZeros(x), sum(s)), positions),
  })
}

/**
 * The data-dependent initialisation of {@link actNorm} (Glow): bias = -mean and logScale = -log(std + eps) over the
 * batch, so the first forward pass outputs zero mean and unit variance per feature.
 *
 * @param data - A batch `[N, d]`.
 * @param options - eps (default 1e-6).
 * @returns The bias and log-scale.
 * @example
 * ```ts
 * import { actNormInitialise } from 'mdml-engine/dists/bijectors'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(actNormInitialise([[1], [3]]).bias) // => [-2]
 * ```
 */
export function actNormInitialise(
  data: TensorLike,
  options: { readonly eps?: number } = {},
): { readonly bias: Value; readonly logScale: Value } {
  const [x] = promoteFloat(data) as [Value]
  const centre = mean(x, 0)
  const spread = sqrt(mean(square(sub(x, centre)), 0))
  return { bias: neg(centre), logScale: neg(log(add(spread, options.eps ?? 1e-6))) }
}

/**
 * Glow's invertible linear layer (Kingma and Dhariwal, 2018), the 1 x 1 convolution of an image when applied along the
 * channel axis: y = W x for a square matrix W, with log |det W| (times `positions`) as the log-determinant and W^-1 as the
 * inverse.
 *
 * @param weight - The matrix `[d, d]` (invertible).
 * @param options - `positions`, how many positions share the matrix (default 1).
 * @returns The bijector acting on the last axis.
 * @example
 * ```ts
 * import { bijectors } from 'mdml-engine/dists'
 * import { item } from 'mdml-engine/tensor'
 * item(bijectors.invertibleLinear([[2, 0], [0, 3]]).forwardLogDetJacobian([1, 1])) === Math.log(6) // => true
 * ```
 */
export function invertibleLinear(weight: TensorLike, options: { readonly positions?: number } = {}): Bijector {
  const [w] = promoteFloat(weight) as [Value]
  return linearLayer('invertibleLinear', w, () => slogdet(w).logabsdet, options.positions ?? 1)
}

/** A bijector y = W x on the last axis, for a matrix W and its log |det|. */
function linearLayer(name: string, w: Value, logAbsDet: () => Value, positions: number): Bijector {
  return defineBijector({
    name,
    inputRank: 1,
    forward: (x) => matmul(x, transpose(w)),
    inverse: (y) => matmul(y, transpose(inv(w))),
    forwardLogDetJacobian: (x) => mul(add(batchZeros(x), logAbsDet()), positions),
  })
}

/** The parameters of an LU-parameterised linear layer W = P L U. */
export interface LuParameters {
  /** The strictly lower triangle of L (unit diagonal) `[d, d]`. */
  readonly lower: Value
  /** The strictly upper triangle of U `[d, d]`. */
  readonly upper: Value
  /** log |diag U| `[d]`. */
  readonly logDiagonal: Value
  /** sign(diag U), constant `[d]`. */
  readonly signs: Value
  /** The permutation P, W = P L U with P[i][permutation[i]] = 1. */
  readonly permutation: readonly number[]
}

/**
 * The LU parameters of a matrix, to initialise {@link luLinear}.
 *
 * @param weight - An invertible matrix `[d, d]`.
 * @returns W = P L U in the layer's parameters.
 * @example
 * ```ts
 * import { luParameters } from 'mdml-engine/dists/bijectors'
 * luParameters([[0, 1], [1, 0]]).permutation // => [1, 0]
 * ```
 */
export function luParameters(weight: TensorLike): LuParameters {
  const { p, l, u } = lu(weight)
  const d = lastSize(l)
  const diagU = sum(mul(u, eye(d)), -1)
  const perm = (Array.from(toFlat(p)) as number[]).reduce<number[]>((acc, v, i) => {
    if (v === 1) acc[Math.floor(i / d)] = i % d
    return acc
  }, new Array<number>(d).fill(0))
  return {
    lower: tril(l, -1),
    upper: triu(u, 1),
    logDiagonal: log(maximum(diagU, neg(diagU))),
    signs: where(greaterEqual(diagU, 0), 1, -1),
    permutation: perm,
  }
}

/**
 * Glow's LU-parameterised invertible linear layer: W = P L U with L unit lower triangular, U upper triangular with
 * diagonal sign exp(logDiagonal), and P a fixed permutation, so the log-determinant is the sum of `logDiagonal` (O(d),
 * not O(d^3)) and every parameter is unconstrained.
 *
 * @param parameters - The LU parameters (see {@link luParameters}).
 * @param options - `positions`, how many positions share the matrix (default 1).
 * @returns The bijector acting on the last axis.
 * @example
 * ```ts
 * import { luLinear, luParameters } from 'mdml-engine/dists/bijectors'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(luLinear(luParameters([[2, 0], [0, 3]])).forward([1, 1])) // => [2, 3]
 * ```
 */
export function luLinear(parameters: LuParameters, options: { readonly positions?: number } = {}): Bijector {
  const d = lastSize(parameters.logDiagonal)
  const l = add(tril(parameters.lower, -1), eye(d))
  const u = add(triu(parameters.upper, 1), diagonalMatrix(mul(parameters.signs, exp(parameters.logDiagonal))))
  // W = P L U, with P[i][permutation[i]] = 1: row i of P M is row permutation[i] of M.
  const w = take(matmul(l, u), [...parameters.permutation], 0)
  return linearLayer('luLinear', w, () => sum(parameters.logDiagonal), options.positions ?? 1)
}

// --- Autoregressive layers -------------------------------------------------------------------------------------------------------

/**
 * An inverse autoregressive flow layer (IAF, Kingma et al., 2016): y_i = x_i exp(s_i(x_{<i})) + m_i(x_{<i}), the
 * conditioner reading the layer's input. The forward map is one pass (fast sampling, the use in variational inference);
 * the inverse takes d sequential passes. It is the mirror image of {@link autoregressive} (MAF), whose density
 * direction is the single pass.
 *
 * @param conditioner - Maps `[..., d]` to `{ shift, logScale }`, with output i depending only on inputs before i.
 * @returns The bijector (vector-valued; log-determinant sum of s at x).
 * @concept normalizing-flows
 * @example
 * ```ts
 * import { bijectors } from 'mdml-engine/dists'
 * import { concat, mul, sliceAxis, toArray, zerosLike } from 'mdml-engine/tensor'
 * const shiftByPrevious = (x) => ({ shift: concat([zerosLike(sliceAxis(x, -1, 0, 1)), sliceAxis(x, -1, 0, 1)], -1), logScale: mul(x, 0) })
 * toArray(bijectors.inverseAutoregressive(shiftByPrevious).forward([1, 1])) // => [1, 2]
 * ```
 */
export function inverseAutoregressive(conditioner: (x: Value) => ShiftLogScale): Bijector {
  return defineBijector({
    name: 'inverseAutoregressive',
    inputRank: 1,
    forward: (x) => {
      const { shift, logScale } = conditioner(x)
      return add(mul(x, exp(logScale)), shift)
    },
    inverse: (y) => {
      let x: Value = mul(y, 0)
      for (let i = 0; i < lastSize(y); i++) {
        const { shift, logScale } = conditioner(x)
        x = mul(sub(y, shift), exp(neg(logScale)))
      }
      return x
    },
    forwardLogDetJacobian: (x) => sum(conditioner(x).logScale, -1),
  })
}

/** The parameters of a MADE network: a weight and a bias for every hidden layer and the output layer. */
export interface MadeParameters {
  /** Layer weights `[in, out]`: `[d, h1]`, ..., `[h_last, 2 d]` (the output holds the shift, then the log-scale). */
  readonly weights: readonly Value[]
  readonly biases: readonly Value[]
}

/**
 * Initial parameters of a MADE network (Germain et al., 2015) for a conditioner of dimension `d`: normal weights scaled by
 * 1 / sqrt(fan-in), the output layer small so the flow starts near the identity.
 *
 * @param inputDim - d (at least 2).
 * @param hiddenSizes - The hidden widths.
 * @param k - The random key.
 * @returns The weights and biases.
 * @example
 * ```ts
 * import { madeParameters } from 'mdml-engine/dists/bijectors'
 * import { key } from 'mdml-engine/random'
 * madeParameters(3, [8], key(0)).weights.map((w) => w.shape) // => [[3, 8], [8, 6]]
 * ```
 */
export function madeParameters(inputDim: number, hiddenSizes: readonly number[], k: Key): MadeParameters {
  const widths = [inputDim, ...hiddenSizes, 2 * inputDim]
  const weights: Value[] = []
  const biases: Value[] = []
  for (let l = 0; l + 1 < widths.length; l++) {
    const [fanIn, fanOut] = [widths[l] ?? 1, widths[l + 1] ?? 1]
    const scale = (l + 2 === widths.length ? 0.01 : 1) / Math.sqrt(fanIn)
    weights.push(mul(normal(foldIn(k, l), [fanIn, fanOut]), scale))
    biases.push(mul(normal(foldIn(k, 100 + l), [fanOut]), 0))
  }
  return { weights, biases }
}

/**
 * The MADE conditioner (a masked tanh network) for {@link autoregressive} and {@link inverseAutoregressive}: the
 * weights are multiplied by the connectivity masks of {@link madeMasks}, so output i of the shift and of the log-scale
 * depends only on inputs before i. The parameters can be trained through the conditioner.
 *
 * @param parameters - The network's weights and biases (see {@link madeParameters}).
 * @returns The conditioner `[..., d]` to `{ shift, logScale }`.
 * @example
 * ```ts
 * import { madeConditioner, madeParameters } from 'mdml-engine/dists/bijectors'
 * import { key } from 'mdml-engine/random'
 * const conditioner = madeConditioner(madeParameters(3, [8], key(1)))
 * conditioner([[1, 2, 3]]).shift.shape // => [1, 3]
 * ```
 */
export function madeConditioner(parameters: MadeParameters): (y: Value) => ShiftLogScale {
  const d = parameters.weights[0]?.shape[0] ?? 0
  const hidden = parameters.weights.slice(0, -1).map((w) => w.shape[1] ?? 0)
  const masks = madeMasks(d, hidden)
  const last = masks.length - 1
  return (y) => {
    let h: Value = y
    for (const [l, w] of parameters.weights.entries()) {
      const mask = l === last ? concat([masks[l] as Value, masks[l] as Value], 1) : (masks[l] as Value)
      h = add(matmul(h, mul(w, mask)), parameters.biases[l] as Value)
      if (l < last) h = tanh(h)
    }
    return { shift: sliceAxis(h, -1, 0, d), logScale: sliceAxis(h, -1, d, 2 * d) }
  }
}

// --- Rational-quadratic splines --------------------------------------------------------------------------------------------------

/**
 * The unconstrained parameters of a monotone rational-quadratic spline with `K` bins on every coordinate: bin widths and
 * heights `[..., K]` (softmax-normalised) and the `K - 1` interior knot derivatives `[..., K - 1]` (softplus-positive).
 */
export interface SplineParameters {
  readonly widths: Value
  readonly heights: Value
  readonly derivatives: Value
}

/** Options of the rational-quadratic spline. */
export interface SplineOptions {
  /** The spline acts on [-bound, bound] and is the identity outside (default 3). */
  readonly bound?: number
  /** The least share of the interval a bin takes in width and in height (default 1e-3). */
  readonly minBinSize?: number
  /** The least derivative at an interior knot (default 1e-3). */
  readonly minDerivative?: number
}

/** The K + 1 knots from -bound to bound for unnormalised bin sizes `[..., K]`. */
function splineKnots(unnormalised: Value, minSize: number, bound: number): Value {
  const k = lastSize(unnormalised)
  const sizes = add(mul(softmax(unnormalised, -1), 1 - minSize * k), minSize)
  const cumulative = cumsum(sizes, -1)
  const interior = sub(mul(sliceAxis(cumulative, -1, 0, k - 1), 2 * bound), bound)
  const edge = (value: number): Value => add(mul(sliceAxis(cumulative, -1, 0, 1), 0), value)
  return concat([edge(-bound), interior, edge(bound)], -1)
}

/** One side of a spline: the transformed values and the log-derivative of the forward map there. */
interface SplineResult {
  readonly value: Value
  readonly logDerivative: Value
}

/**
 * The monotone rational-quadratic spline of neural spline flows (Durkan et al., 2019) on [-bound, bound], with
 * linear (identity) tails beyond, and its log-derivative, forward or inverse. The input and the parameters have the same
 * leading shape.
 *
 * @param x - The inputs `[...]`.
 * @param parameters - Widths and heights `[..., K]` and derivatives `[..., K - 1]` (leading shape broadcast to x's).
 * @param inverse - Whether to invert the spline (the log-derivative returned is still the forward map's, at the preimage).
 * @param options - bound, minBinSize and minDerivative.
 * @returns The mapped values and the forward log-derivative.
 * @concept normalizing-flows
 * @example
 * ```ts
 * import { rationalQuadraticSplineMap } from 'mdml-engine/dists/bijectors'
 * import { item, zeros } from 'mdml-engine/tensor'
 * // Outside [-3, 3] the spline is the identity.
 * const p = { widths: zeros([4]), heights: zeros([4]), derivatives: zeros([3]) }
 * item(rationalQuadraticSplineMap(5, p, false).value) // => 5
 * ```
 */
export function rationalQuadraticSplineMap(
  x: TensorLike,
  parameters: SplineParameters,
  inverse: boolean,
  options: SplineOptions = {},
): SplineResult {
  const [input] = promoteFloat(x) as [Value]
  const bound = options.bound ?? 3
  const minSize = options.minBinSize ?? 1e-3
  const minDerivative = options.minDerivative ?? 1e-3
  const k = lastSize(parameters.widths)
  const shape = [...input.shape, k + 1]
  const xKnots = broadcastTo(splineKnots(parameters.widths, minSize, bound), shape)
  const yKnots = broadcastTo(splineKnots(parameters.heights, minSize, bound), shape)
  const interiorDerivatives = add(softplus(parameters.derivatives), minDerivative)
  const ones = add(mul(sliceAxis(interiorDerivatives, -1, 0, 1), 0), 1)
  const derivatives = broadcastTo(concat([ones, interiorDerivatives, ones], -1), shape)
  const inside = logicalAnd(greaterEqual(input, -bound), lessEqual(input, bound))
  // Outside the bound the spline is evaluated at the edge; the gradient of an input at or inside the edge is the identity's.
  const clamped = where(inside, input, maximum(minimum(stopGradient(input), bound), -bound))
  // The bin of each input: the number of interior knots at or below it.
  const search = inverse ? yKnots : xKnots
  const interior = sliceAxis(search, -1, 1, k)
  const bin = stopGradient(
    cast(
      sum(cast(greaterEqual(expandDims(stopGradient(clamped), -1), stopGradient(interior)), 'float64'), -1),
      'int32',
    ),
  )
  const gather = (table: Value, offset: number): Value =>
    squeeze(takeAlongAxis(table, expandDims(add(bin, offset), -1), -1), -1)
  const [x0, x1, y0, y1] = [gather(xKnots, 0), gather(xKnots, 1), gather(yKnots, 0), gather(yKnots, 1)]
  const [d0, d1] = [gather(derivatives, 0), gather(derivatives, 1)]
  const [width, height] = [sub(x1, x0), sub(y1, y0)]
  const slope = div(height, width)
  let xi: Value
  if (inverse) {
    const delta = sub(clamped, y0)
    const q = sub(add(d1, d0), mul(slope, 2))
    const a = add(mul(delta, q), mul(height, sub(slope, d0)))
    const b = sub(mul(height, d0), mul(delta, q))
    const c = neg(mul(slope, delta))
    const discriminant = maximum(sub(square(b), mul(mul(a, c), 4)), 0)
    xi = div(mul(c, 2), sub(neg(b), sqrt(discriminant)))
  } else {
    xi = div(sub(clamped, x0), width)
  }
  const xi1 = mul(xi, sub(1, xi))
  const denominator = add(slope, mul(sub(add(d1, d0), mul(slope, 2)), xi1))
  const logDerivative = sub(
    add(mul(log(slope), 2), log(add(add(mul(d1, square(xi)), mul(mul(slope, xi1), 2)), mul(d0, square(sub(1, xi)))))),
    mul(log(denominator), 2),
  )
  const mapped = inverse
    ? add(x0, mul(xi, width))
    : add(y0, div(mul(height, add(mul(slope, square(xi)), mul(d0, xi1))), denominator))
  return {
    value: where(inside, mapped, input),
    logDerivative: where(inside, logDerivative, mul(input, 0)),
  }
}

/**
 * The unconstrained spline parameters from the flat output of a conditioner: for each of `dim` coordinates, K widths, K
 * heights and K - 1 derivatives in a row of 3 K - 1 numbers.
 *
 * @param raw - The conditioner output `[..., dim (3 K - 1)]`.
 * @param bins - K.
 * @returns The parameters with a trailing axis of K (or K - 1) per coordinate `[..., dim, ...]`.
 * @example
 * ```ts
 * import { splineParametersFromVector } from 'mdml-engine/dists/bijectors'
 * import { zeros } from 'mdml-engine/tensor'
 * splineParametersFromVector(zeros([1, 2 * 11]), 4).derivatives.shape // => [1, 2, 3]
 * ```
 */
export function splineParametersFromVector(raw: TensorLike, bins: number): SplineParameters {
  const [r] = promoteFloat(raw) as [Value]
  const per = 3 * bins - 1
  const dim = lastSize(r) / per
  if (!Number.isInteger(dim))
    throw new MdmlError('shape', `splineParametersFromVector: ${lastSize(r)} is not a multiple of ${per}`)
  const rows = reshape(r, [...r.shape.slice(0, -1), dim, per])
  return {
    widths: sliceAxis(rows, -1, 0, bins),
    heights: sliceAxis(rows, -1, bins, 2 * bins),
    derivatives: sliceAxis(rows, -1, 2 * bins, per),
  }
}

/**
 * An elementwise rational-quadratic spline bijector with fixed parameters: each of the last axis's coordinates has its
 * own monotone spline on [-bound, bound] (the identity outside).
 *
 * @param parameters - Widths and heights `[d, K]` and derivatives `[d, K - 1]`.
 * @param options - bound, minBinSize and minDerivative.
 * @returns The bijector (log-determinant per coordinate; use `inputRank` 0 semantics).
 * @example
 * ```ts
 * import { bijectors } from 'mdml-engine/dists'
 * import { zeros } from 'mdml-engine/tensor'
 * bijectors.rationalQuadraticSpline({ widths: zeros([2, 4]), heights: zeros([2, 4]), derivatives: zeros([2, 3]) }).name // => 'rationalQuadraticSpline'
 * ```
 */
export function rationalQuadraticSpline(parameters: SplineParameters, options: SplineOptions = {}): Bijector {
  return defineBijector({
    name: 'rationalQuadraticSpline',
    sign: 1,
    forward: (x) => rationalQuadraticSplineMap(x, parameters, false, options).value,
    inverse: (y) => rationalQuadraticSplineMap(y, parameters, true, options).value,
    forwardLogDetJacobian: (x) => rationalQuadraticSplineMap(x, parameters, false, options).logDerivative,
    inverseLogDetJacobian: (y) => neg(rationalQuadraticSplineMap(y, parameters, true, options).logDerivative),
  })
}

/**
 * A rational-quadratic spline coupling layer (neural spline flow, Durkan et al., 2019): the first `split` coordinates
 * pass through and condition a monotone spline on each of the rest. More expressive than an affine coupling at the same
 * cost, with an exact inverse (a quadratic per bin).
 *
 * @param split - How many leading coordinates pass through.
 * @param conditioner - Maps x_a `[..., split]` to the spline parameters of the remaining coordinates
 *   (`[..., d - split, K]`, `[..., d - split, K]`, `[..., d - split, K - 1]`; see {@link splineParametersFromVector}).
 * @param options - bound, minBinSize and minDerivative.
 * @returns The bijector (vector-valued).
 * @concept normalizing-flows
 * @example
 * ```ts
 * import { bijectors } from 'mdml-engine/dists'
 * import { zeros } from 'mdml-engine/tensor'
 * const layer = bijectors.rationalQuadraticCoupling(1, () => ({ widths: zeros([1, 4]), heights: zeros([1, 4]), derivatives: zeros([1, 3]) }))
 * layer.forward([0.3, 5]).shape // => [2]
 * ```
 */
export function rationalQuadraticCoupling(
  split: number,
  conditioner: (xa: Value) => SplineParameters,
  options: SplineOptions = {},
): Bijector {
  const apply = (v: Value, inverse: boolean): { out: Value; logDerivative: Value } => {
    const a = sliceAxis(v, -1, 0, split)
    const b = sliceAxis(v, -1, split, lastSize(v))
    const mapped = rationalQuadraticSplineMap(b, conditioner(a), inverse, options)
    return { out: concat([a, mapped.value], -1), logDerivative: sum(mapped.logDerivative, -1) }
  }
  return defineBijector({
    name: 'rationalQuadraticCoupling',
    inputRank: 1,
    forward: (x) => apply(x, false).out,
    inverse: (y) => apply(y, true).out,
    forwardLogDetJacobian: (x) => apply(x, false).logDerivative,
    inverseLogDetJacobian: (y) => neg(apply(y, true).logDerivative),
  })
}

// --- Residual flows --------------------------------------------------------------------------------------------------------------

/** Options of {@link residualFlow}. */
export interface ResidualFlowOptions {
  /** The number of terms of the power series for the log-determinant (default 20). */
  readonly terms?: number
  /** 'exact' (default) takes the exact traces of the Jacobian's powers; 'hutchinson' estimates them with random vectors. */
  readonly estimator?: 'exact' | 'hutchinson'
  /** The random key of the Hutchinson estimator. */
  readonly key?: Key
  /** Random vectors per call of the Hutchinson estimator (default 1). */
  readonly samples?: number
  /** The stopping tolerance of the fixed-point inverse (default 1e-12). */
  readonly tolerance?: number
  /** The most fixed-point iterations of the inverse (default 500). */
  readonly maxIterations?: number
}

/**
 * Scale a weight matrix so its spectral norm is at most `coefficient` < 1, the Lipschitz constraint that makes a residual
 * block a contraction (Behrmann et al., 2019; Chen et al., 2019): W min(1, coefficient / sigma_max(W)).
 *
 * @param weight - A matrix.
 * @param coefficient - The bound (default 0.9).
 * @returns The scaled matrix, differentiable.
 * @example
 * ```ts
 * import { lipschitzNormalise } from 'mdml-engine/dists/bijectors'
 * import { item, mul } from 'mdml-engine/tensor'
 * import { norm } from 'mdml-engine/numerics/linalg'
 * Math.abs(item(norm(lipschitzNormalise([[3, 0], [0, 1]], 0.5), 2)) - 0.5) < 1e-12 // => true
 * ```
 */
export function lipschitzNormalise(weight: TensorLike, coefficient = 0.9): Value {
  const [w] = promoteFloat(weight) as [Value]
  return mul(w, minimum(div(coefficient, norm(w, 2)), 1))
}

/**
 * A residual flow layer (iResNet, Behrmann et al., 2019; Residual Flows, Chen et al., 2019): y = x + g(x) for a
 * contraction g (Lipschitz constant below one, for example a network with {@link lipschitzNormalise}d weights). The
 * inverse is the fixed-point iteration x <- y - g(x); the log-determinant is the power series log det(I + J) = sum_k
 * (-1)^(k+1) tr(J^k) / k, truncated at `terms` (exact traces of the powers of the Jacobian, or Hutchinson's estimator
 * with Gaussian vectors and forward-mode products, unbiased for the truncated series).
 *
 * @param g - The residual map on `[..., d]`, acting on each row independently and able to take a single vector.
 * @param options - Terms, estimator and its key and samples, and the inverse's tolerance.
 * @returns The bijector (vector-valued).
 * @throws MdmlError with code `value` for the Hutchinson estimator without a key.
 * @concept normalizing-flows
 * @example
 * ```ts
 * import { bijectors } from 'mdml-engine/dists'
 * import { item, mul } from 'mdml-engine/tensor'
 * // g(x) = x / 2: y = 1.5 x, log det = 2 log 1.5.
 * const layer = bijectors.residualFlow((x) => mul(x, 0.5), { terms: 60 })
 * Math.abs(item(layer.forwardLogDetJacobian([1, 2])) - 2 * Math.log(1.5)) < 1e-12 // => true
 * ```
 */
export function residualFlow(g: (x: Value) => Value, options: ResidualFlowOptions = {}): Bijector {
  const terms = options.terms ?? 20
  const estimator = options.estimator ?? 'exact'
  if (estimator === 'hutchinson' && options.key === undefined)
    throw new MdmlError('value', 'residualFlow: the Hutchinson estimator needs a key')
  const exactSeries = (x: Value): Value => {
    const d = lastSize(x)
    const rows = reshape(x, [-1, d])
    const n = rows.shape[0] ?? 1
    const logDets = Array.from({ length: n }, (_, i) => {
      const row = squeeze(sliceAxis(rows, 0, i, i + 1), 0)
      const jacobianOfRow = jacobian((v: Value) => g(v))(row)
      let power: Value = jacobianOfRow
      let total: Value = mul(sum(mul(jacobianOfRow, eye(d))), 0)
      for (let k = 1; k <= terms; k++) {
        const trace = sum(mul(power, eye(d)))
        total = add(total, mul(trace, (k % 2 === 1 ? 1 : -1) / k))
        if (k < terms) power = matmul(power, jacobianOfRow)
      }
      return total
    })
    return reshape(stack(logDets, 0), x.shape.slice(0, -1))
  }
  const hutchinson = (x: Value): Value => {
    const samples = options.samples ?? 1
    const parts = Array.from({ length: samples }, (_, s) => {
      const v = normal(foldIn(options.key as Key, s), [...x.shape])
      let w: Value = v
      let total: Value = batchZeros(x)
      for (let k = 1; k <= terms; k++) {
        w = jvp(g, x, w).tangent
        total = add(total, mul(sum(mul(v, w), -1), (k % 2 === 1 ? 1 : -1) / k))
      }
      return total
    })
    return div(
      parts.reduce((a, b) => add(a, b)),
      samples,
    )
  }
  return defineBijector({
    name: 'residualFlow',
    inputRank: 1,
    forward: (x) => add(x, g(x)),
    inverse: (y) => {
      let x: Value = y
      for (let i = 0; i < (options.maxIterations ?? 500); i++) {
        const next = sub(y, g(x))
        const change = Math.max(...Array.from(toFlat(stopGradient(sub(next, x))), Math.abs))
        x = next
        if (change < (options.tolerance ?? 1e-12)) break
      }
      return x
    },
    forwardLogDetJacobian: (x) => (estimator === 'exact' ? exactSeries(x) : hutchinson(x)),
  })
}
