/**
 * Penalties added to a loss to constrain the parameters: L1 (lasso), L2 (ridge, weight decay), elastic net, the group
 * and sparse group lasso, the nuclear norm (a convex surrogate for low rank), soft orthogonality of weight matrices and
 * an entropy (confidence) penalty on predicted distributions. Each takes one value or any tree of parameters.
 */

import { WhiteboxError } from '../base/index.ts'
import { svdvals } from '../numerics/linalg/index.ts'
import { entr } from '../numerics/special/index.ts'
import {
  abs,
  add,
  asTensor,
  equal,
  eye,
  isValue,
  matmul,
  mean,
  mul,
  promoteFloat,
  segmentSum,
  softmax,
  sqrt,
  square,
  sub,
  sum,
  type TensorLike,
  tensor,
  transpose,
  treeLeaves,
  type Value,
  where,
} from '../tensor/index.ts'

/** Parameters to penalise: one value or any pytree of values (every leaf counts). */
export type Penalised = TensorLike | { readonly [key: string]: unknown } | readonly unknown[]

/** Whether a node is array data (numbers, nested number arrays or a typed array) rather than a branch. */
const isArrayData = (node: unknown): boolean =>
  typeof node === 'number' || ArrayBuffer.isView(node) || (Array.isArray(node) && node.every((c) => isArrayData(c)))

/** The leaves of a parameter tree whose leaves may be values or plain (nested) number arrays. */
const leaves = (params: Penalised): Value[] => {
  if (isValue(params) || isArrayData(params)) return [asTensor(params as TensorLike)]
  if (Array.isArray(params)) return params.flatMap((p) => leaves(p as Penalised))
  if (typeof params === 'object' && params !== null) {
    return Object.keys(params)
      .sort()
      .flatMap((k) => leaves((params as Record<string, unknown>)[k] as Penalised))
  }
  return treeLeaves(params)
}

const total = (params: Penalised, f: (v: Value) => Value): Value => {
  let acc: Value | undefined
  for (const leaf of leaves(params)) {
    const [v] = promoteFloat(leaf) as [Value]
    const s = sum(f(v))
    acc = acc === undefined ? s : add(acc, s)
  }
  return acc ?? tensor(0)
}

/**
 * L1 (lasso) penalty strength * sum |w| over every parameter, which drives coefficients to exactly zero.
 *
 * @param params - A value or a pytree of values.
 * @param options - strength (default 1).
 * @returns The penalty, differentiable away from zero.
 * @concept regularisation
 * @example
 * ```ts
 * import { l1Penalty } from 'whitebox-engine/losses'
 * import { item } from 'whitebox-engine/tensor'
 * item(l1Penalty({ w: [1, -2], b: [0.5] }, { strength: 2 })) // => 7
 * ```
 */
export function l1Penalty(params: Penalised, options: { readonly strength?: number } = {}): Value {
  return mul(total(params, abs), options.strength ?? 1)
}

/**
 * L2 (ridge, weight-decay) penalty (strength / 2) sum w^2, whose gradient strength * w is the weight decay of SGD.
 *
 * @param params - A value or a pytree of values.
 * @param options - strength (default 1).
 * @returns The penalty, differentiable.
 * @example
 * ```ts
 * import { l2Penalty } from 'whitebox-engine/losses'
 * import { item } from 'whitebox-engine/tensor'
 * item(l2Penalty([3, 4])) // => 12.5
 * ```
 */
export function l2Penalty(params: Penalised, options: { readonly strength?: number } = {}): Value {
  return mul(total(params, square), (options.strength ?? 1) / 2)
}

/**
 * Elastic-net penalty in scikit-learn's parameterisation: alpha (l1Ratio ||w||_1 + (1 - l1Ratio) / 2 ||w||_2^2).
 *
 * @param params - A value or a pytree of values.
 * @param options - alpha (default 1) and l1Ratio (default 0.5).
 * @returns The penalty, differentiable away from zero.
 * @example
 * ```ts
 * import { elasticNet } from 'whitebox-engine/losses'
 * import { item } from 'whitebox-engine/tensor'
 * item(elasticNet([2], { alpha: 1, l1Ratio: 0.5 })) // => 2
 * ```
 */
export function elasticNet(
  params: Penalised,
  options: { readonly alpha?: number; readonly l1Ratio?: number } = {},
): Value {
  const alpha = options.alpha ?? 1
  const rho = options.l1Ratio ?? 0.5
  return mul(add(mul(total(params, abs), rho), mul(total(params, square), (1 - rho) / 2)), alpha)
}

/**
 * Group-lasso penalty (Yuan and Lin, 2006): strength * sum_g sqrt(p_g) ||w_g||_2 over groups of coefficients, which
 * zeroes whole groups; with `l1Ratio` it becomes the sparse group lasso (Simon et al., 2013),
 * (1 - l1Ratio) group term + l1Ratio ||w||_1.
 *
 * @param weights - Coefficients `[P]`.
 * @param groups - A group id in 0..G-1 for each coefficient.
 * @param options - strength (default 1) and l1Ratio (default 0).
 * @returns The penalty, differentiable away from zero groups (whose subgradient is taken as 0).
 * @throws WhiteboxError with code `shape` when the groups do not match the coefficients.
 * @example
 * ```ts
 * import { groupLasso } from 'whitebox-engine/losses'
 * import { item } from 'whitebox-engine/tensor'
 * item(groupLasso([3, 4, 0], [0, 0, 1])) === 5 * Math.SQRT2 // => true
 * ```
 */
export function groupLasso(
  weights: TensorLike,
  groups: readonly number[],
  options: { readonly strength?: number; readonly l1Ratio?: number } = {},
): Value {
  const [w] = promoteFloat(weights) as [Value]
  if (w.shape.reduce((a, b) => a * b, 1) !== groups.length)
    throw new WhiteboxError('shape', 'groupLasso: one group id per coefficient')
  let g = 0
  for (const id of groups) g = Math.max(g, id + 1)
  const sizes = Array.from({ length: g }, () => 0)
  for (const id of groups) sizes[id] = (sizes[id] ?? 0) + 1
  const squares = segmentSum(square(w), tensor(groups, { dtype: 'int32' }), g)
  // sqrt at zero has an infinite derivative; empty groups contribute zero with a zero gradient.
  const zero = equal(squares, 0)
  const norms = where(zero, 0, sqrt(where(zero, 1, squares)))
  const group = sum(mul(norms, tensor(sizes.map(Math.sqrt))))
  const rho = options.l1Ratio ?? 0
  const penalty = rho === 0 ? group : add(mul(group, 1 - rho), mul(sum(abs(w)), rho))
  return mul(penalty, options.strength ?? 1)
}

/**
 * Nuclear-norm (trace-norm) penalty strength * sum of singular values, the convex surrogate of rank for low-rank
 * matrix learning.
 *
 * @param matrix - A matrix `[m, n]`.
 * @param options - strength (default 1).
 * @returns The penalty, differentiable where the singular values are distinct and non-zero.
 * @example
 * ```ts
 * import { nuclearNorm } from 'whitebox-engine/losses'
 * import { item } from 'whitebox-engine/tensor'
 * item(nuclearNorm([[3, 0], [0, -2]])) // => 5
 * ```
 */
export function nuclearNorm(matrix: TensorLike, options: { readonly strength?: number } = {}): Value {
  return mul(sum(svdvals(matrix)), options.strength ?? 1)
}

/**
 * Soft orthogonality penalty (Bansal, Chen and Wang, 2018): strength ||W^T W - I||_F^2, which keeps the columns of a
 * weight matrix orthonormal; `double` adds ||W W^T - I||_F^2 (the double soft orthogonality for non-square W).
 *
 * @param matrix - A matrix `[m, n]`.
 * @param options - strength (default 1) and double (default false).
 * @returns The penalty, differentiable.
 * @example
 * ```ts
 * import { orthogonality } from 'whitebox-engine/losses'
 * import { item } from 'whitebox-engine/tensor'
 * item(orthogonality([[0, 1], [1, 0]])) // => 0
 * ```
 */
export function orthogonality(
  matrix: TensorLike,
  options: { readonly strength?: number; readonly double?: boolean } = {},
): Value {
  const [w] = promoteFloat(matrix) as [Value]
  const m = w.shape[0] ?? 0
  const n = w.shape[1] ?? 0
  let penalty = sum(square(sub(matmul(transpose(w), w), eye(n))))
  if (options.double === true) penalty = add(penalty, sum(square(sub(matmul(w, transpose(w)), eye(m)))))
  return mul(penalty, options.strength ?? 1)
}

/**
 * Entropy penalty on predicted distributions: -strength * mean H(p), the confidence penalty of Pereyra et al. (2017)
 * that discourages over-confident outputs when added to a loss (a negative strength minimises entropy instead, as in
 * entropy minimisation for semi-supervised learning).
 *
 * @param input - Logits (default) or probabilities `[..., K]`.
 * @param options - strength (default 1) and logits (default true).
 * @returns The penalty, differentiable.
 * @example
 * ```ts
 * import { entropyPenalty } from 'whitebox-engine/losses'
 * import { item } from 'whitebox-engine/tensor'
 * item(entropyPenalty([[0, 0]])) === -Math.LN2 // => true
 * ```
 */
export function entropyPenalty(
  input: TensorLike,
  options: { readonly strength?: number; readonly logits?: boolean } = {},
): Value {
  const [x] = promoteFloat(input) as [Value]
  const p = (options.logits ?? true) ? softmax(x, -1) : x
  return mul(mean(sum(entr(p), -1)), -(options.strength ?? 1))
}
