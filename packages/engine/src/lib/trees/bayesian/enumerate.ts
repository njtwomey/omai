/**
 * The exact posterior over trees for a small problem: every tree of depth at most two over a candidate rule set,
 * weighted by prior times exact evidence, as one `[C, 1 + C, 1 + C]` tensor of log weights (a root rule, then for each of its
 * two children a leaf or a rule). Given the root rule the two subtrees are independent, so the weights are sums of three
 * vectors, the normaliser is one `logsumexp`, and the posterior-predictive average over all trees is a weighted sum of
 * region summaries, not a loop over trees. It is the reference the sampler is checked against.
 */

import { OmaiError } from '../../base/index.ts'
import {
  add,
  argsort,
  cast,
  concat,
  exp,
  expandDims,
  full,
  isfinite,
  lessEqual,
  logsumexp,
  mul,
  oneHot,
  reshape,
  sliceAxis,
  stack,
  sub,
  sum,
  type TensorLike,
  take,
  toFlat,
  type Value,
  where,
} from '../../tensor/index.ts'
import type { LeafPrediction } from './leaf.ts'
import { logLeaf, logSplit } from './prior.ts'
import { childRegion, type Region, splitTable, type TreeSpace } from './space.ts'
import { type BayesTree, bayesLeaf, bayesSplit, inputMatrix } from './tree.ts'

/** The exact posterior over all trees of depth at most two over a rule set. */
export interface TreeEnumeration {
  readonly space: TreeSpace
  /** The log weight (prior times evidence) of the tree with no split. */
  readonly logStump: number
  /** The log weights of the trees with a root rule: `[C, 1 + C, 1 + C]`, indexed by the root rule, then the left and the right child as 0 for a leaf and `1 + c` for rule `c`; `-Infinity` for a non-tree. */
  readonly logWeights: Value
  /** The log of the sum of all weights: the log marginal likelihood of the data under the prior over trees. */
  readonly logEvidence: number
  /** The number of trees with positive prior, the stump included. */
  readonly trees: number
  /** The posterior probability that the tree is a stump. */
  readonly stump: number
  /** The posterior probability of each root rule, `[C]`. */
  readonly root: readonly number[]
  /** The posterior expected number of times each rule is used anywhere in the tree, `[C]`. */
  readonly usage: readonly number[]
  /** The posterior probability of 0, 1, 2 and 3 splits. */
  readonly splits: readonly number[]
}

/** The log weights of the options of one child region: a leaf, or a split by each rule (`-Infinity` for unavailable rules), `[1 + C]`. */
function optionWeights(space: TreeSpace, region: Region): Value {
  const { prior } = space
  const c = space.rules.features.length
  const available = region.valid.length
  const table = splitTable(space, region)
  const leaf = logLeaf(prior, region.depth, available) + region.logEvidence
  const mask = sum(oneHot(region.valid, c), 0)
  // The grandchildren are leaves for certain (the depth limit is two), which has probability one.
  const split = add(add(table.left, table.right), logSplit(prior, region.depth, available))
  const options = where(cast(mask, 'bool'), split, Number.NEGATIVE_INFINITY)
  return concat([reshape(full([1], leaf), [1]), reshape(options, [-1])], 0)
}

/**
 * Enumerate every tree of depth at most two over the candidate rules and weight each by its prior times its exact
 * evidence. It is exact, and it costs a matrix product per region and one tensor of `C (1 + C)^2` weights, so it is for
 * a handful of features and a few dozen candidate thresholds.
 *
 * @param space - The space; its prior's `maxDepth` must be at most 2.
 * @returns The enumeration.
 * @throws OmaiError with code `value` when `maxDepth` is above 2.
 * @concept bayesian-trees
 * @index function Exact posterior over all depth-two trees over a rule set.
 * @example
 * ```ts
 * import { enumerateTrees, normalGammaLeaf, quantileRules, bayesData, treePrior, treeSpace } from 'omai-engine/trees'
 * const xs = [[0], [1], [2], [3], [4], [5]]
 * const model = normalGammaLeaf({ m0: 2.5, kappa0: 1, alpha0: 2, beta0: 2 })
 * const data = bayesData(xs, [0, 0.1, 0, 5, 5.1, 5], model)
 * const posterior = enumerateTrees(treeSpace(data, treePrior({ maxDepth: 2 }), quantileRules(xs, 5)))
 * posterior.root.indexOf(Math.max(...posterior.root)) // => 2
 * ```
 */
export function enumerateTrees(space: TreeSpace): TreeEnumeration {
  const { prior, root } = space
  if (prior.maxDepth > 2) throw new OmaiError('value', 'enumerateTrees: the prior must have maxDepth <= 2')
  const c = space.rules.features.length
  const a = c + 1
  const never = full([a], Number.NEGATIVE_INFINITY)
  const available = root.valid.length
  const rootMask = sum(oneHot(root.valid, c), 0)
  const rootTerm = where(cast(rootMask, 'bool'), logSplit(prior, 0, available), Number.NEGATIVE_INFINITY)
  const valid = new Set(root.valid)
  const optionsLeft: Value[] = []
  const optionsRight: Value[] = []
  for (let r = 0; r < c; r++) {
    optionsLeft.push(valid.has(r) ? optionWeights(space, childRegion(space, root, r, 'L')) : never)
    optionsRight.push(valid.has(r) ? optionWeights(space, childRegion(space, root, r, 'R')) : never)
  }
  const left = stack(optionsLeft, 0)
  const right = stack(optionsRight, 0)
  const weights = add(add(reshape(rootTerm, [c, 1, 1]), reshape(left, [c, a, 1])), reshape(right, [c, 1, a]))
  const logStump = logLeaf(prior, 0, available) + root.logEvidence
  const flat = reshape(weights, [-1])
  const logEvidence = toFlat(logsumexp(concat([reshape(full([1], logStump), [1]), flat], 0)))[0] ?? 0
  const p = exp(sub(weights, logEvidence))
  const finite = toFlat(sum(cast(isfinite(weights), 'float64')))[0] ?? 0
  const rootProbability = toFlat(sum(p, [1, 2]))
  const usageLeft = toFlat(sum(p, [0, 2]))
  const usageRight = toFlat(sum(p, [0, 1]))
  const block = (rows: [number, number], cols: [number, number]): number =>
    toFlat(sum(sliceAxis(sliceAxis(p, 1, rows[0], rows[1]), 2, cols[0], cols[1])))[0] ?? 0
  const none = block([0, 1], [0, 1])
  const onlyLeft = block([1, a], [0, 1])
  const onlyRight = block([0, 1], [1, a])
  const both = block([1, a], [1, a])
  return {
    space,
    logStump,
    logWeights: weights,
    logEvidence,
    trees: 1 + finite,
    stump: Math.exp(logStump - logEvidence),
    root: rootProbability,
    usage: rootProbability.map((v, i) => v + (usageLeft[i + 1] ?? 0) + (usageRight[i + 1] ?? 0)),
    splits: [Math.exp(logStump - logEvidence), none, onlyLeft + onlyRight, both],
  }
}

/** A tree and its posterior probability. */
export interface WeightedTree {
  readonly tree: BayesTree
  readonly probability: number
}

/**
 * The most probable trees of an enumeration, with their posterior probabilities.
 *
 * @param e - The enumeration.
 * @param k - How many trees.
 * @returns The trees, most probable first.
 * @concept bayesian-trees
 * @example
 * ```ts
 * import { enumerateTrees, normalGammaLeaf, quantileRules, topTrees, bayesData, treePrior, treeSpace } from 'omai-engine/trees'
 * const xs = [[0], [1], [2], [3], [4], [5]]
 * const model = normalGammaLeaf({ m0: 2.5, kappa0: 1, alpha0: 2, beta0: 2 })
 * const data = bayesData(xs, [0, 0.1, 0, 5, 5.1, 5], model)
 * const posterior = enumerateTrees(treeSpace(data, treePrior({ maxDepth: 2 }), quantileRules(xs, 5)))
 * topTrees(posterior, 1)[0]?.tree.kind // => 'split'
 * ```
 */
export function topTrees(e: TreeEnumeration, k: number): WeightedTree[] {
  const { rules } = e.space
  const c = rules.features.length
  const a = c + 1
  const flat = reshape(e.logWeights, [-1])
  const order = toFlat(argsort(flat, 0, true)).slice(0, k)
  const logs = toFlat(flat)
  const node = (option: number): BayesTree =>
    option === 0
      ? bayesLeaf
      : bayesSplit(rules.features[option - 1] ?? 0, rules.thresholds[option - 1] ?? 0, bayesLeaf, bayesLeaf, option - 1)
  const trees = order.map((i) => {
    const r = Math.floor(i / (a * a))
    const left = Math.floor(i / a) % a
    const right = i % a
    return {
      tree: bayesSplit(rules.features[r] ?? 0, rules.thresholds[r] ?? 0, node(left), node(right), r),
      probability: Math.exp((logs[i] ?? Number.NEGATIVE_INFINITY) - e.logEvidence),
    }
  })
  const stump = { tree: bayesLeaf as BayesTree, probability: e.stump }
  return [...trees, stump].sort((x, y) => y.probability - x.probability).slice(0, k)
}

/**
 * The posterior-predictive of the average over all trees, exactly: at each query point, the probability-weighted mean of
 * the leaf summaries over every tree. Given a root rule the two subtrees are independent, so each root rule's share is
 * computed from its two children's option weights and the sums over trees are tensor sums.
 *
 * @param e - The enumeration.
 * @param xq - The query inputs, `[g, d]`.
 * @returns The predictive mean and spreads (regression) or class probabilities (classification) at each point.
 * @concept bayesian-trees
 * @index function Exact posterior-predictive average over all trees of an enumeration.
 * @example
 * ```ts
 * import { enumerateTrees, normalGammaLeaf, predictEnumerated, quantileRules, bayesData, treePrior, treeSpace } from 'omai-engine/trees'
 * const xs = [[0], [1], [2], [3], [4], [5]]
 * const model = normalGammaLeaf({ m0: 2.5, kappa0: 1, alpha0: 2, beta0: 2 })
 * const data = bayesData(xs, [0, 0.1, 0, 5, 5.1, 5], model)
 * const posterior = enumerateTrees(treeSpace(data, treePrior({ maxDepth: 2 }), quantileRules(xs, 5)))
 * const [low = 0, high = 0] = predictEnumerated(posterior, [[0.5], [4.5]]).mean
 * high - low > 3 // => true
 * ```
 */
export function predictEnumerated(e: TreeEnumeration, xq: TensorLike): LeafPrediction {
  const { space } = e
  const { model } = space.data
  const c = space.rules.features.length
  const xm = inputMatrix(xq)
  const g = xm.shape[0] ?? 0
  // Which way each query point goes under each rule, `[g, c]`, as 0/1 floats.
  const goes = cast(
    lessEqual(take(xm, space.rules.features as number[], 1), reshape(space.rules.thresholds as number[], [1, c])),
    'float64',
  )
  const summarise = (stats: Value): Value => model.summary(stats)
  const rootSummary = reshape(summarise(space.root.stats), [1, model.outputs])
  let total: Value = mul(full([g, model.outputs], 1), e.stump)
  total = mul(total, rootSummary)
  const available = new Set(space.root.valid)
  const options = (region: Region): Value => {
    // Option probabilities among the child's options, and the summary each option predicts at every query point.
    const weights = optionWeights(space, region)
    const share = exp(sub(weights, logsumexp(weights)))
    const leaf = reshape(summarise(region.stats), [1, model.outputs])
    const below = summarise(region.left)
    const above = summarise(sub(region.stats, region.left))
    const under = expandDims(goes, 2)
    const predicted = add(
      mul(under, reshape(below, [1, c, model.outputs])),
      mul(sub(1, under), reshape(above, [1, c, model.outputs])),
    )
    const viaSplit = sum(mul(predicted, reshape(sliceAxis(share, 0, 1, c + 1), [1, c, 1])), 1)
    const viaLeaf = mul(leaf, sliceAxis(share, 0, 0, 1))
    return add(viaLeaf, viaSplit)
  }
  const logZ = e.logEvidence
  for (let r = 0; r < c; r++) {
    if (!available.has(r)) continue
    const left = childRegion(space, space.root, r, 'L')
    const right = childRegion(space, space.root, r, 'R')
    const wl = optionWeights(space, left)
    const wr = optionWeights(space, right)
    const rootTerm = logSplit(space.prior, 0, space.root.valid.length)
    const share = Math.exp(rootTerm + (toFlat(logsumexp(wl))[0] ?? 0) + (toFlat(logsumexp(wr))[0] ?? 0) - logZ)
    const side = sliceAxis(goes, 1, r, r + 1)
    const mixed = add(mul(side, options(left)), mul(sub(1, side), options(right)))
    total = add(total, mul(mixed, share))
  }
  return model.read(total)
}
