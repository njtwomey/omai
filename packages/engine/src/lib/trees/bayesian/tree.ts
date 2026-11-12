/**
 * The structure of a Bayesian tree and the operations on it that need only the data: the tree as nested nodes (a leaf, or
 * a split on a feature at a threshold with the rows at or below it going left), the routing of rows to leaves as one
 * tensor expression, the leaf boxes, the statistics each leaf collects, and the summaries a prediction averages. The
 * statistics are sums of per-row statistics over a leaf's rows (`segmentSum`), so they are additive and exact.
 */

import { OmaiError } from '../../base/index.ts'
import {
  asTensor,
  cast,
  full,
  lessEqual,
  reshape,
  segmentSum,
  sliceAxis,
  type TensorLike,
  take,
  toFlat,
  type Value,
  where,
} from '../../tensor/index.ts'
import type { LeafModel, LeafPosterior } from './leaf.ts'

/** A leaf of a Bayesian tree. */
export interface BayesLeaf {
  readonly kind: 'leaf'
}

/** A split: rows with `x[feature] <= threshold` go left. `rule` is the index in the candidate set, when there is one. */
export interface BayesSplit {
  readonly kind: 'split'
  readonly feature: number
  readonly threshold: number
  readonly rule?: number
  readonly left: BayesTree
  readonly right: BayesTree
}

/**
 * A decision tree as nested nodes. It carries no parameters: the leaf parameters are integrated out, and what a leaf
 * says is its posterior given the rows that reach it.
 *
 * @index structure A Bayesian decision tree: nested leaves and splits, no parameters.
 */
export type BayesTree = BayesLeaf | BayesSplit

/** The tree of one leaf, no splits. */
export const bayesLeaf: BayesLeaf = { kind: 'leaf' }

/**
 * A split node.
 *
 * @param feature - The feature index.
 * @param threshold - The threshold; rows with `x[feature] <= threshold` go left.
 * @param left - The subtree for the rows at or below the threshold.
 * @param right - The subtree for the rows above it.
 * @param rule - The index of the split in a candidate set, when it came from one.
 * @returns The tree.
 * @concept bayesian-trees
 * @example
 * ```ts
 * import { bayesLeaf, bayesSplit, bayesLeafCount } from 'omai-engine/trees'
 * bayesLeafCount(bayesSplit(0, 0.5, bayesLeaf, bayesSplit(0, 0.8, bayesLeaf, bayesLeaf))) // => 3
 * ```
 */
export function bayesSplit(
  feature: number,
  threshold: number,
  left: BayesTree,
  right: BayesTree,
  rule?: number,
): BayesSplit {
  return rule === undefined
    ? { kind: 'split', feature, threshold, left, right }
    : { kind: 'split', feature, threshold, rule, left, right }
}

/**
 * The number of leaves of a tree.
 *
 * @param tree - The tree.
 * @returns The leaves.
 * @concept bayesian-trees
 * @example
 * ```ts
 * import { bayesLeaf, bayesLeafCount } from 'omai-engine/trees'
 * bayesLeafCount(bayesLeaf) // => 1
 * ```
 */
export function bayesLeafCount(tree: BayesTree): number {
  return tree.kind === 'leaf' ? 1 : bayesLeafCount(tree.left) + bayesLeafCount(tree.right)
}

/**
 * The depth of a tree (a leaf is 0).
 *
 * @param tree - The tree.
 * @returns The depth.
 * @concept bayesian-trees
 * @example
 * ```ts
 * import { bayesLeaf, bayesSplit, bayesDepth } from 'omai-engine/trees'
 * bayesDepth(bayesSplit(0, 0.5, bayesLeaf, bayesLeaf)) // => 1
 * ```
 */
export function bayesDepth(tree: BayesTree): number {
  return tree.kind === 'leaf' ? 0 : 1 + Math.max(bayesDepth(tree.left), bayesDepth(tree.right))
}

/** The data a Bayesian tree is fitted to: the inputs, the per-row leaf statistics of the target, and the leaf model. */
export interface BayesData {
  /** The inputs, `[n, d]`. */
  readonly x: Value
  /** The per-row statistics of the target under the leaf model, `[n, size]`. */
  readonly rows: Value
  readonly model: LeafModel
  readonly n: number
  readonly d: number
}

/** Inputs as a float matrix `[n, d]` (a vector is one feature). */
export function inputMatrix(x: TensorLike): Value {
  const xs = cast(asTensor(x), 'float64')
  return xs.shape.length === 1 ? reshape(xs, [-1, 1]) : xs
}

/**
 * Bundle inputs and target with a leaf model.
 *
 * @param x - The inputs, `[n, d]` (a vector is one feature).
 * @param y - The target: reals (Normal-Gamma leaf) or labels `0..K-1` (Dirichlet leaf).
 * @param model - The leaf model.
 * @returns The data.
 * @throws OmaiError with code `shape` unless `x` and `y` have the same number of rows.
 * @concept bayesian-trees
 * @example
 * ```ts
 * import { normalGammaLeaf, bayesData } from 'omai-engine/trees'
 * const model = normalGammaLeaf({ m0: 0, kappa0: 1, alpha0: 2, beta0: 1 })
 * bayesData([[0], [1], [2]], [0, 1, 2], model).d // => 1
 * ```
 */
export function bayesData(x: TensorLike, y: TensorLike, model: LeafModel): BayesData {
  const xm = inputMatrix(x)
  const n = xm.shape[0] ?? 0
  const rows = model.statistics(y)
  if ((rows.shape[0] ?? 0) !== n) {
    throw new OmaiError('shape', `bayesData: ${n} rows of inputs but ${rows.shape[0] ?? 0} targets`)
  }
  return { x: xm, rows, model, n, d: xm.shape[1] ?? 1 }
}

/** One feature of the inputs as a vector. */
export function column(x: Value, feature: number): Value {
  return reshape(sliceAxis(x, 1, feature, feature + 1), [-1])
}

/** The leaf of each row (leaves numbered in preorder, left before right) and the number of leaves. */
export interface LeafAssignment {
  readonly ids: Value
  readonly leaves: number
}

/**
 * Route rows to leaves: one tensor expression per split, `where(x <= threshold, left, right)`, so the whole tree is a
 * handful of tensor operations. Leaves are numbered in preorder, left before right.
 *
 * @param tree - The tree.
 * @param x - The inputs, `[n, d]`.
 * @returns The leaf id of each row (int32) and the number of leaves.
 * @concept bayesian-trees
 * @example
 * ```ts
 * import { bayesLeaf, bayesSplit, leafIds } from 'omai-engine/trees'
 * import { toFlat } from 'omai-engine/tensor'
 * toFlat(leafIds(bayesSplit(0, 1.5, bayesLeaf, bayesLeaf), [[1], [2], [0]]).ids) // => [0, 1, 0]
 * ```
 */
export function leafIds(tree: BayesTree, x: TensorLike): LeafAssignment {
  const xm = inputMatrix(x)
  const n = xm.shape[0] ?? 0
  let next = 0
  const route = (node: BayesTree): Value => {
    if (node.kind === 'leaf') return full([n], next++, { dtype: 'int32' })
    const left = route(node.left)
    const right = route(node.right)
    return where(lessEqual(column(xm, node.feature), node.threshold), left, right)
  }
  const ids = route(tree)
  return { ids, leaves: next }
}

/** The statistics of every leaf of a tree on the data, `[leaves, size]`: the sums of the per-row statistics. */
export function leafStatistics(data: BayesData, tree: BayesTree): Value {
  const { ids, leaves } = leafIds(tree, data.x)
  return segmentSum(data.rows, ids, leaves)
}

/**
 * The exact log marginal likelihood of the target given a tree, with every leaf parameter integrated out: the sum over
 * leaves of the leaf evidence.
 *
 * @param data - The data.
 * @param tree - The tree.
 * @returns The log evidence of the tree.
 * @concept bayesian-trees
 * @index function Exact log marginal likelihood of a tree: the sum of its leaves' evidences.
 * @example
 * ```ts
 * import { bayesLeaf, bayesSplit, normalGammaLeaf, bayesData, treeLogEvidence } from 'omai-engine/trees'
 * const model = normalGammaLeaf({ m0: 0, kappa0: 1, alpha0: 2, beta0: 1 })
 * const data = bayesData([[0], [1], [2], [3]], [0, 0, 5, 5], model)
 * const split = treeLogEvidence(data, bayesSplit(0, 1.5, bayesLeaf, bayesLeaf))
 * split > treeLogEvidence(data, bayesLeaf) // => true
 * ```
 */
export function treeLogEvidence(data: BayesData, tree: BayesTree): number {
  const evidence = data.model.logEvidence(leafStatistics(data, tree))
  return toFlat(evidence).reduce((a, b) => a + b, 0)
}

/** The box of a leaf: the lower (exclusive) and upper (inclusive) bound per feature. */
export interface LeafBox {
  readonly id: number
  readonly depth: number
  readonly lower: readonly number[]
  readonly upper: readonly number[]
}

/**
 * The region of every leaf as a box, leaves in preorder (the order of {@link leafIds}).
 *
 * @param tree - The tree.
 * @param d - The number of features.
 * @returns One box per leaf.
 * @concept bayesian-trees
 * @example
 * ```ts
 * import { bayesLeaf, bayesSplit, leafBoxes } from 'omai-engine/trees'
 * leafBoxes(bayesSplit(0, 1.5, bayesLeaf, bayesLeaf), 1).map((b) => b.upper[0]) // => [1.5, Infinity]
 * ```
 */
export function leafBoxes(tree: BayesTree, d: number): LeafBox[] {
  const boxes: LeafBox[] = []
  const walk = (node: BayesTree, depth: number, lower: readonly number[], upper: readonly number[]): void => {
    if (node.kind === 'leaf') {
      boxes.push({ id: boxes.length, depth, lower, upper })
      return
    }
    const cut = (bounds: readonly number[], value: number): number[] =>
      bounds.map((b, j) => (j === node.feature ? value : b))
    walk(node.left, depth + 1, lower, cut(upper, node.threshold))
    walk(node.right, depth + 1, cut(lower, node.threshold), upper)
  }
  const inf = Number.POSITIVE_INFINITY
  walk(
    tree,
    0,
    Array.from({ length: d }, () => -inf),
    Array.from({ length: d }, () => inf),
  )
  return boxes
}

/** A leaf of a fitted tree: its box and its posterior. */
export interface FittedLeaf extends LeafBox, LeafPosterior {}

/**
 * The leaves of a tree with their posteriors on the data: box, rows, posterior mean, spread, precision and step size.
 *
 * @param data - The data.
 * @param tree - The tree.
 * @returns One entry per leaf, in preorder.
 * @concept bayesian-trees
 * @example
 * ```ts
 * import { bayesLeaf, bayesSplit, fittedLeaves, normalGammaLeaf, bayesData } from 'omai-engine/trees'
 * const model = normalGammaLeaf({ m0: 0, kappa0: 1, alpha0: 2, beta0: 1 })
 * const data = bayesData([[0], [1], [2], [3]], [0, 0, 5, 5], model)
 * fittedLeaves(data, bayesSplit(0, 1.5, bayesLeaf, bayesLeaf)).map((l) => l.n) // => [2, 2]
 * ```
 */
export function fittedLeaves(data: BayesData, tree: BayesTree): FittedLeaf[] {
  const size = data.model.size
  const flat = toFlat(leafStatistics(data, tree))
  return leafBoxes(tree, data.d).map((box, i) => ({
    ...box,
    ...data.model.describe(flat.slice(i * size, (i + 1) * size)),
  }))
}

/**
 * The posterior-predictive summary of a tree at new inputs, `[g, outputs]`: each row's leaf looked up in the summary of
 * the leaf's statistics. A prediction averages these summaries over trees and reads the result with the model's `read`.
 *
 * @param data - The data the tree is fitted to.
 * @param tree - The tree.
 * @param xq - The query inputs, `[g, d]`.
 * @returns The summaries.
 * @concept bayesian-trees
 * @example
 * ```ts
 * import { bayesLeaf, bayesSplit, normalGammaLeaf, bayesData, treeSummary } from 'omai-engine/trees'
 * const model = normalGammaLeaf({ m0: 0, kappa0: 1, alpha0: 2, beta0: 1 })
 * const data = bayesData([[0], [1], [2], [3]], [0, 0, 4, 4], model)
 * treeSummary(data, bayesSplit(0, 1.5, bayesLeaf, bayesLeaf), [[0.5]]).shape // => [1, 3]
 * ```
 */
export function treeSummary(data: BayesData, tree: BayesTree, xq: TensorLike): Value {
  const summaries = data.model.summary(leafStatistics(data, tree))
  return take(summaries, leafIds(tree, xq).ids, 0)
}
