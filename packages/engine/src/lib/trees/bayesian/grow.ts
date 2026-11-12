/**
 * The myopic Bayesian tree: grow one tree one split at a time, always taking the split with the largest posterior odds
 * over the whole tree so far, and stop when no leaf has a split that pays for itself under the prior. The odds of a split
 * at a leaf are the prior odds that the leaf splits, times the exact Bayes factor of the two-leaf tree against the one-leaf
 * tree (the leaf parameters are integrated out), over the number of rules the prior chose between. Every decision is
 * recorded, so the growth can be replayed split by split. The search at a leaf covers every feature and every threshold
 * with a handful of tensor operations ({@link searchBayesSplits}).
 */

import { arange, lessEqual, logicalNot, maskedSelect, sum, take, toFlat, type Value } from '../../tensor/index.ts'
import type { Algorithm } from '../../trace/index.ts'
import { runAlgorithm } from '../../trace/index.ts'
import { logLeaf, splitProbability, type TreePrior, treePrior } from './prior.ts'
import { type BayesSplitSearch, searchBayesSplits } from './search.ts'
import { type BayesData, type BayesTree, bayesLeaf, bayesSplit, column, type FittedLeaf, fittedLeaves } from './tree.ts'

/**
 * The log posterior odds that a leaf splits by the best rule, against staying a leaf: the prior odds of splitting, one
 * over the number of rules, the two new leaves' probability of not splitting further, and the log Bayes factor of the split.
 *
 * @param prior - The tree prior.
 * @param depth - The leaf's depth.
 * @param available - The rules available at the leaf.
 * @param gain - The log Bayes factor of the split against no split.
 * @returns The log odds (`-Infinity` when the leaf cannot split); positive means the split pays for itself.
 * @concept bayesian-trees
 * @example
 * ```ts
 * import { splitLogOdds, treePrior } from 'omai-engine/trees'
 * splitLogOdds(treePrior(), 0, 10, 5) > 0 // => true
 * splitLogOdds(treePrior(), 0, 10, 0) > 0 // => false
 * ```
 */
export function splitLogOdds(prior: TreePrior, depth: number, available: number, gain: number): number {
  const p = splitProbability(prior, depth, available)
  if (p === 0) return Number.NEGATIVE_INFINITY
  const children = logLeaf(prior, depth + 1, 1)
  return Math.log(p) - Math.log(1 - p) - Math.log(available) + 2 * children + gain
}

/** A node of a tree in growth. */
export interface BayesGrownNode {
  readonly id: number
  readonly parent: number
  readonly depth: number
  /** The rows that reach the node (int32 indices). */
  readonly rows: Value
  readonly n: number
  readonly stats: readonly number[]
  readonly logEvidence: number
  /** The rules available here. */
  readonly available: number
  /** The best split, with its gain and odds, when the node can split at all. */
  readonly best?: {
    readonly feature: number
    readonly threshold: number
    readonly gain: number
    readonly logOdds: number
  }
  /** The step at which the node was split, once it has been. */
  readonly splitAt?: number
  readonly left?: number
  readonly right?: number
}

/** One split decision of the grower. */
export interface SplitDecision {
  readonly step: number
  readonly node: number
  readonly depth: number
  readonly feature: number
  readonly threshold: number
  /** The rows at the node. */
  readonly n: number
  readonly nLeft: number
  readonly nRight: number
  /** The log Bayes factor of the split against the node as a leaf. */
  readonly gain: number
  /** The log posterior odds of the split, the prior included. */
  readonly logOdds: number
  /** The log evidence of the tree after the split. */
  readonly logEvidence: number
  readonly leaves: number
}

/** The state of the grower. */
export interface BayesGrowState {
  readonly nodes: readonly BayesGrownNode[]
  /** The decisions so far, in order. */
  readonly steps: readonly SplitDecision[]
}

/** The paths the grower logs. */
export type BayesGrowPath = 'bayesTree/split' | 'bayesTree/leaves' | 'bayesTree/logEvidence'

function born(
  data: BayesData,
  prior: TreePrior,
  id: number,
  parent: number,
  depth: number,
  rows: Value | undefined,
): BayesGrownNode {
  const found: BayesSplitSearch = searchBayesSplits(data, rows, prior.minLeaf)
  const all = rows ?? arange(0, data.n, 1, { dtype: 'int32' })
  const best =
    found.best === undefined
      ? undefined
      : {
          feature: found.best.feature,
          threshold: found.best.threshold,
          gain: found.best.gain,
          logOdds: splitLogOdds(prior, depth, found.available, found.best.gain),
        }
  return {
    id,
    parent,
    depth,
    rows: all,
    n: all.shape[0] ?? 0,
    stats: toFlat(sum(take(data.rows, all, 0), 0)),
    logEvidence: found.logEvidence,
    available: found.available,
    ...(best === undefined ? {} : { best }),
  }
}

const openLeaves = (nodes: readonly BayesGrownNode[]): BayesGrownNode[] =>
  nodes.filter((n) => n.splitAt === undefined && (n.best?.logOdds ?? Number.NEGATIVE_INFINITY) > 0)

/**
 * The myopic grower as a stepped algorithm: each step splits the leaf whose best split has the largest posterior odds,
 * and the run ends when no leaf has odds above one (or the tree has `maxSplits` splits).
 *
 * Logs, when subscribed: `bayesTree/split` (the decision), `bayesTree/leaves` and `bayesTree/logEvidence` (the exact
 * log marginal likelihood of the tree so far).
 *
 * @param data - The data.
 * @param options - The prior and the most splits to make (default no limit beyond the prior's depth).
 * @returns The algorithm.
 * @concept bayesian-trees
 * @index algorithm Myopic Bayesian tree: grow one split at a time by posterior odds, stop when none pays.
 * @example
 * ```ts
 * import { bayesGrowAlgorithm, normalGammaLeaf, bayesData } from 'omai-engine/trees'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const model = normalGammaLeaf({ m0: 2.5, kappa0: 1, alpha0: 2, beta0: 2 })
 * const data = bayesData([[0], [1], [2], [3], [4], [5]], [0, 0.1, 0, 5, 5.1, 5], model)
 * runAlgorithm(bayesGrowAlgorithm(data), { maxSteps: 10 }).state.steps.length // => 1
 * ```
 */
export function bayesGrowAlgorithm(
  data: BayesData,
  options: { readonly prior?: TreePrior; readonly maxSplits?: number } = {},
): Algorithm<BayesGrowState, BayesGrowPath> {
  const prior = options.prior ?? treePrior()
  const maxSplits = options.maxSplits ?? Number.POSITIVE_INFINITY
  return {
    name: 'bayesGrow',
    config: { ...prior, maxSplits: Number.isFinite(maxSplits) ? maxSplits : -1 },
    init: () => ({ nodes: [born(data, prior, 0, -1, 0, undefined)], steps: [] }),
    step: (state, ctx) => {
      const open = openLeaves(state.nodes)
      const top = open.reduce<BayesGrownNode | undefined>(
        (a, b) => (a === undefined || (b.best?.logOdds ?? 0) > (a.best?.logOdds ?? 0) ? b : a),
        undefined,
      )
      if (top?.best === undefined) return state
      const { feature, threshold, gain, logOdds } = top.best
      const goesLeft = lessEqual(take(column(data.x, feature), top.rows, 0), threshold)
      const leftRows = maskedSelect(top.rows, goesLeft)
      const rightRows = maskedSelect(top.rows, logicalNot(goesLeft))
      const left = born(data, prior, state.nodes.length, top.id, top.depth + 1, leftRows)
      const right = born(data, prior, state.nodes.length + 1, top.id, top.depth + 1, rightRows)
      const split: BayesGrownNode = { ...top, splitAt: state.steps.length, left: left.id, right: right.id }
      const nodes = [...state.nodes.map((n) => (n.id === top.id ? split : n)), left, right]
      const leaves = nodes.filter((n) => n.splitAt === undefined)
      const logEvidence = leaves.reduce((a, n) => a + n.logEvidence, 0)
      const decision: SplitDecision = {
        step: state.steps.length,
        node: top.id,
        depth: top.depth,
        feature,
        threshold,
        n: top.n,
        nLeft: left.n,
        nRight: right.n,
        gain,
        logOdds,
        logEvidence,
        leaves: leaves.length,
      }
      if (ctx.wants('bayesTree/split')) ctx.log('bayesTree/split', { ...decision })
      if (ctx.wants('bayesTree/leaves')) ctx.log('bayesTree/leaves', leaves.length)
      if (ctx.wants('bayesTree/logEvidence')) ctx.log('bayesTree/logEvidence', logEvidence)
      return { nodes, steps: [...state.steps, decision] }
    },
    done: (state) =>
      state.steps.length >= maxSplits || openLeaves(state.nodes).length === 0 ? 'converged' : undefined,
  }
}

/**
 * The tree after the first `k` splits of a growth (all of them by default).
 *
 * @param state - The state of {@link bayesGrowAlgorithm}.
 * @param k - How many splits to keep.
 * @returns The tree.
 * @concept bayesian-trees
 * @example
 * ```ts
 * import { bayesGrowAlgorithm, normalGammaLeaf, treeAfter, bayesData } from 'omai-engine/trees'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const model = normalGammaLeaf({ m0: 2.5, kappa0: 1, alpha0: 2, beta0: 2 })
 * const data = bayesData([[0], [1], [2], [3], [4], [5]], [0, 0.1, 0, 5, 5.1, 5], model)
 * const run = runAlgorithm(bayesGrowAlgorithm(data), { maxSteps: 10 })
 * treeAfter(run.state, 0).kind // => 'leaf'
 * treeAfter(run.state).kind // => 'split'
 * ```
 */
export function treeAfter(state: BayesGrowState, k: number = state.steps.length): BayesTree {
  const build = (id: number): BayesTree => {
    const node = state.nodes[id]
    const at = node?.splitAt
    if (node === undefined || at === undefined || at >= k || node.best === undefined) return bayesLeaf
    return bayesSplit(node.best.feature, node.best.threshold, build(node.left ?? 0), build(node.right ?? 0))
  }
  return build(0)
}

/** A tree grown by the myopic method. */
export interface GrownBayesTree {
  readonly tree: BayesTree
  readonly state: BayesGrowState
  /** The leaves with their posteriors. */
  readonly leaves: readonly FittedLeaf[]
  /** The decisions, in order. */
  readonly steps: readonly SplitDecision[]
  /** The exact log marginal likelihood of the tree. */
  readonly logEvidence: number
}

/**
 * Grow a Bayesian tree greedily: one split at a time by posterior odds, until none pays for itself.
 *
 * @param data - The data.
 * @param options - The prior (default {@link treePrior}) and a limit on the number of splits.
 * @returns The tree, its leaves with their posteriors, and every decision.
 * @concept bayesian-trees
 * @index function Grow a Bayesian tree greedily; returns the tree, leaf posteriors and decisions.
 * @example
 * ```ts
 * import { growBayesTree, normalGammaLeaf, bayesData } from 'omai-engine/trees'
 * const model = normalGammaLeaf({ m0: 2.5, kappa0: 1, alpha0: 2, beta0: 2 })
 * const data = bayesData([[0], [1], [2], [3], [4], [5]], [0, 0.1, 0, 5, 5.1, 5], model)
 * const grown = growBayesTree(data)
 * grown.steps.map((s) => s.threshold) // => [2.5]
 * grown.leaves.map((l) => l.n) // => [3, 3]
 * ```
 */
export function growBayesTree(
  data: BayesData,
  options: { readonly prior?: TreePrior; readonly maxSplits?: number } = {},
): GrownBayesTree {
  const run = runAlgorithm(bayesGrowAlgorithm(data, options), { maxSteps: 10_000 })
  const tree = treeAfter(run.state)
  return {
    tree,
    state: run.state,
    leaves: fittedLeaves(data, tree),
    steps: run.state.steps,
    logEvidence: run.state.nodes.filter((n) => n.splitAt === undefined).reduce((a, n) => a + n.logEvidence, 0),
  }
}
