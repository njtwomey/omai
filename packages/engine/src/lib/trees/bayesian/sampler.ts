/**
 * A Metropolis-Hastings sampler over tree structures (Chipman, George and McCulloch 1998): grow (split a leaf by a random
 * available rule), prune (turn a split whose children are both leaves back into a leaf) and change (draw a new rule for a
 * split). The leaf parameters are integrated out, so the chain moves on structures only and a move's acceptance uses the
 * exact prior and evidence of the two trees; the regions are cached, so a move costs a lookup once the regions are known.
 * The draws are trees; a prediction averages the leaf summaries over them.
 */

import { OmaiError } from '../../base/index.ts'
import { foldIn, type Key, key as makeKey, uniform } from '../../random/index.ts'
import { ess, rhat } from '../../sampling/diagnostics/index.ts'
import { metropolisAcceptance } from '../../sampling/index.ts'
import { div, stack, sum, type TensorLike, tensor, toFlat, type Value } from '../../tensor/index.ts'
import type { Algorithm } from '../../trace/index.ts'
import type { LeafPrediction } from './leaf.ts'
import { type PlacedNode, placeNodes, scoreTree, type TreeSpace } from './space.ts'
import { type BayesTree, bayesLeaf, bayesLeafCount, bayesSplit, treeSummary } from './tree.ts'

/** The probabilities of the three moves from a tree that is not a single leaf (a single leaf can only grow). */
export interface TreeMoves {
  readonly grow: number
  readonly prune: number
  readonly change: number
}

/** The default move probabilities: grow 0.25, prune 0.25, change 0.5. */
export const defaultMoves: TreeMoves = { grow: 0.25, prune: 0.25, change: 0.5 }

/** The state of one chain: the tree, its log posterior (up to the constant), and how the moves have fared. */
export interface TreeChainState {
  readonly tree: BayesTree
  readonly logPosterior: number
  readonly moves: Readonly<Record<MoveKind, MoveCount>>
}

/** The three kinds of move. */
export type MoveKind = 'grow' | 'prune' | 'change'

/** How often a kind of move was proposed and accepted. */
export interface MoveCount {
  readonly proposed: number
  readonly accepted: number
}

const noMoves: Readonly<Record<MoveKind, MoveCount>> = {
  grow: { proposed: 0, accepted: 0 },
  prune: { proposed: 0, accepted: 0 },
  change: { proposed: 0, accepted: 0 },
}

/** The path of sides from the root to a node, e.g. `'LR'`. */
function replaceAt(tree: BayesTree, path: string, node: BayesTree): BayesTree {
  if (path === '') return node
  if (tree.kind === 'leaf') throw new OmaiError('value', 'replaceAt: the path leaves the tree')
  return path[0] === 'L'
    ? bayesSplit(tree.feature, tree.threshold, replaceAt(tree.left, path.slice(1), node), tree.right, tree.rule)
    : bayesSplit(tree.feature, tree.threshold, tree.left, replaceAt(tree.right, path.slice(1), node), tree.rule)
}

const growable = (space: TreeSpace, placed: readonly PlacedNode[]): PlacedNode[] =>
  placed.filter((p) => p.node.kind === 'leaf' && p.region.depth < space.prior.maxDepth && p.region.valid.length > 0)

const prunable = (placed: readonly PlacedNode[]): PlacedNode[] =>
  placed.filter((p) => p.node.kind === 'split' && p.node.left.kind === 'leaf' && p.node.right.kind === 'leaf')

const pickFrom = <T>(items: readonly T[], u: number): T =>
  items[Math.min(items.length - 1, Math.floor(u * items.length))] as T

/** One Metropolis-Hastings step: the new state, and the move proposed (when one could be made) and whether it was accepted. */
export function treeStep(
  space: TreeSpace,
  state: TreeChainState,
  k: Key,
  moves: TreeMoves = defaultMoves,
): TreeChainState {
  const u = toFlat(uniform(k, [5]))
  const { tree } = state
  const stump = tree.kind === 'leaf'
  const pGrow = (t: BayesTree): number => (t.kind === 'leaf' ? 1 : moves.grow)
  const placed = placeNodes(space, tree)
  const choose = stump
    ? 'grow'
    : (u[0] ?? 0) < moves.grow
      ? 'grow'
      : (u[0] ?? 0) < moves.grow + moves.prune
        ? 'prune'
        : 'change'
  let proposal: BayesTree | undefined
  let logForward = 0
  let logBackward = 0
  if (choose === 'grow') {
    const options = growable(space, placed)
    if (options.length > 0) {
      const at = pickFrom(options, u[1] ?? 0)
      const rule = pickFrom(at.region.valid, u[2] ?? 0)
      proposal = replaceAt(
        tree,
        at.path,
        bayesSplit(space.rules.features[rule] ?? 0, space.rules.thresholds[rule] ?? 0, bayesLeaf, bayesLeaf, rule),
      )
      const after = placeNodes(space, proposal)
      logForward = Math.log(pGrow(tree)) - Math.log(options.length) - Math.log(at.region.valid.length)
      logBackward = Math.log(moves.prune) - Math.log(prunable(after).length)
    }
  } else if (choose === 'prune') {
    const options = prunable(placed)
    const at = pickFrom(options, u[1] ?? 0)
    proposal = replaceAt(tree, at.path, bayesLeaf)
    const after = placeNodes(space, proposal)
    logForward = Math.log(moves.prune) - Math.log(options.length)
    logBackward = Math.log(pGrow(proposal)) - Math.log(growable(space, after).length) - Math.log(at.region.valid.length)
  } else {
    const options = placed.filter((p) => p.node.kind === 'split')
    const at = pickFrom(options, u[1] ?? 0)
    const rule = pickFrom(at.region.valid, u[2] ?? 0)
    const node = at.node
    if (node.kind === 'split') {
      proposal = replaceAt(
        tree,
        at.path,
        bayesSplit(space.rules.features[rule] ?? 0, space.rules.thresholds[rule] ?? 0, node.left, node.right, rule),
      )
    }
  }
  const counts = state.moves[choose]
  if (proposal === undefined) {
    return {
      ...state,
      moves: { ...state.moves, [choose]: { proposed: counts.proposed + 1, accepted: counts.accepted } },
    }
  }
  const score = scoreTree(space, proposal).logPosterior
  const alpha = metropolisAcceptance(state.logPosterior, score, logForward, logBackward)
  const accepted = (u[3] ?? 0) < alpha
  const moved = { proposed: counts.proposed + 1, accepted: counts.accepted + (accepted ? 1 : 0) }
  return {
    tree: accepted ? proposal : tree,
    logPosterior: accepted ? score : state.logPosterior,
    moves: { ...state.moves, [choose]: moved },
  }
}

/** The path of the sampler's log: the log posterior and the leaf count after each move. */
export type TreeSamplerPath = 'sampler/logPosterior' | 'sampler/leaves'

/**
 * The Metropolis-Hastings chain over trees as a stepped algorithm: one move per step from the single-leaf tree, with the
 * log posterior and the number of leaves logged.
 *
 * @param space - The space of trees.
 * @param moves - The move probabilities (default {@link defaultMoves}).
 * @returns The algorithm.
 * @concept bayesian-trees
 * @index algorithm Metropolis-Hastings over tree structures: grow, prune and change, leaf parameters integrated out.
 * @example
 * ```ts
 * import { normalGammaLeaf, quantileRules, bayesData, treePrior, treeSampler, treeSpace } from 'omai-engine/trees'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const xs = [[0], [1], [2], [3], [4], [5]]
 * const data = bayesData(xs, [0, 0.1, 0, 5, 5.1, 5], normalGammaLeaf({ m0: 2.5, kappa0: 1, alpha0: 2, beta0: 2 }))
 * const space = treeSpace(data, treePrior({ maxDepth: 2 }), quantileRules(xs, 5))
 * runAlgorithm(treeSampler(space), { maxSteps: 50, seed: 1 }).state.logPosterior < 0 // => true
 * ```
 */
export function treeSampler(
  space: TreeSpace,
  moves: TreeMoves = defaultMoves,
): Algorithm<TreeChainState, TreeSamplerPath> {
  return {
    name: 'treeSampler',
    config: { ...moves, maxDepth: space.prior.maxDepth, rules: space.rules.features.length },
    init: () => ({ tree: bayesLeaf, logPosterior: scoreTree(space, bayesLeaf).logPosterior, moves: noMoves }),
    step: (state, ctx) => {
      const next = treeStep(space, state, ctx.key, moves)
      if (ctx.wants('sampler/logPosterior')) ctx.log('sampler/logPosterior', next.logPosterior)
      if (ctx.wants('sampler/leaves')) ctx.log('sampler/leaves', bayesLeafCount(next.tree))
      return next
    },
  }
}

/** Options of {@link sampleTrees}. */
export interface SampleTreesOptions {
  /** The key the chains' keys derive from (default `key(0)`). */
  readonly key?: Key
  /** Independent chains, each started from the single-leaf tree (default 4). */
  readonly chains?: number
  /** Moves per chain after the warm-up (default 2000). */
  readonly iterations?: number
  /** Moves discarded at the start of each chain (default 500). */
  readonly warmup?: number
  /** Keep every `thin`-th move (default 5). */
  readonly thin?: number
  readonly moves?: TreeMoves
}

/** The draws of {@link sampleTrees}. */
export interface TreeSamples {
  readonly space: TreeSpace
  /** The kept trees, chain by chain. */
  readonly draws: readonly BayesTree[]
  /** The chain of each draw. */
  readonly chainOf: readonly number[]
  /** The log posterior and the number of leaves of each kept draw, `[chains, kept]`. */
  readonly logPosterior: Value
  readonly leaves: Value
  /** How the three kinds of move fared over all chains and all moves (warm-up included). */
  readonly moves: Readonly<Record<MoveKind, MoveCount & { readonly rate: number }>>
  /** The overall acceptance rate of the moves. */
  readonly acceptance: number
  /** Convergence of the log posterior and of the leaf count across chains. */
  readonly diagnostics: {
    readonly logPosterior: { readonly rhat: number; readonly ess: number }
    readonly leaves: { readonly rhat: number; readonly ess: number }
  }
}

/**
 * Draw trees from the posterior by running several Metropolis-Hastings chains over structures from the single-leaf tree
 * and keeping every `thin`-th move after the warm-up, with the acceptance rate of each kind of move and the split-R-hat
 * and effective sample size of the log posterior and of the leaf count.
 *
 * @param space - The space of trees.
 * @param options - Chains, moves, warm-up, thinning and the key.
 * @returns The draws and how the chains behaved.
 * @concept bayesian-trees
 * @index function Posterior draws of whole trees by Metropolis-Hastings, with acceptance rates, R-hat and ESS.
 * @example
 * ```ts
 * import { normalGammaLeaf, quantileRules, sampleTrees, bayesData, treePrior, treeSpace } from 'omai-engine/trees'
 * const xs = [[0], [1], [2], [3], [4], [5]]
 * const data = bayesData(xs, [0, 0.1, 0, 5, 5.1, 5], normalGammaLeaf({ m0: 2.5, kappa0: 1, alpha0: 2, beta0: 2 }))
 * const space = treeSpace(data, treePrior({ maxDepth: 2 }), quantileRules(xs, 5))
 * sampleTrees(space, { chains: 2, iterations: 100, warmup: 20, thin: 10 }).draws.length // => 20
 * ```
 */
export function sampleTrees(space: TreeSpace, options: SampleTreesOptions = {}): TreeSamples {
  const chains = options.chains ?? 4
  const iterations = options.iterations ?? 2000
  const warmup = options.warmup ?? 500
  const thin = options.thin ?? 5
  const moves = options.moves ?? defaultMoves
  const base = options.key ?? makeKey(0)
  const draws: BayesTree[] = []
  const chainOf: number[] = []
  const logs: number[][] = []
  const sizes: number[][] = []
  const totals: Record<MoveKind, { proposed: number; accepted: number }> = {
    grow: { proposed: 0, accepted: 0 },
    prune: { proposed: 0, accepted: 0 },
    change: { proposed: 0, accepted: 0 },
  }
  for (let c = 0; c < chains; c++) {
    const chainKey = foldIn(base, c)
    let state: TreeChainState = {
      tree: bayesLeaf,
      logPosterior: scoreTree(space, bayesLeaf).logPosterior,
      moves: noMoves,
    }
    const lp: number[] = []
    const lv: number[] = []
    for (let t = 0; t < warmup + iterations; t++) {
      state = treeStep(space, state, foldIn(chainKey, t), moves)
      if (t >= warmup && (t - warmup) % thin === 0) {
        draws.push(state.tree)
        chainOf.push(c)
        lp.push(state.logPosterior)
        lv.push(bayesLeafCount(state.tree))
      }
    }
    logs.push(lp)
    sizes.push(lv)
    for (const kind of ['grow', 'prune', 'change'] as const) {
      totals[kind].proposed += state.moves[kind].proposed
      totals[kind].accepted += state.moves[kind].accepted
    }
  }
  const logPosterior = tensor(logs)
  const leaves = tensor(sizes)
  const rate = (m: MoveCount): number => (m.proposed === 0 ? 0 : m.accepted / m.proposed)
  const all = Object.values(totals).reduce((a, m) => ({
    proposed: a.proposed + m.proposed,
    accepted: a.accepted + m.accepted,
  }))
  return {
    space,
    draws,
    chainOf,
    logPosterior,
    leaves,
    moves: {
      grow: { ...totals.grow, rate: rate(totals.grow) },
      prune: { ...totals.prune, rate: rate(totals.prune) },
      change: { ...totals.change, rate: rate(totals.change) },
    },
    acceptance: rate(all),
    diagnostics: {
      logPosterior: { rhat: rhat(logPosterior), ess: ess(logPosterior) },
      leaves: { rhat: rhat(leaves), ess: ess(leaves) },
    },
  }
}

/**
 * The posterior-predictive of the average over the drawn trees: the mean of the leaf summaries over the draws, read with
 * the leaf model.
 *
 * @param samples - The draws.
 * @param xq - The query inputs, `[g, d]`.
 * @returns The predictive mean and spreads (regression) or class probabilities (classification) at each point.
 * @concept bayesian-trees
 * @index function Posterior-predictive average over drawn trees.
 * @example
 * ```ts
 * import { normalGammaLeaf, predictSampled, quantileRules, sampleTrees, bayesData, treePrior, treeSpace } from 'omai-engine/trees'
 * const xs = [[0], [1], [2], [3], [4], [5]]
 * const data = bayesData(xs, [0, 0.1, 0, 5, 5.1, 5], normalGammaLeaf({ m0: 2.5, kappa0: 1, alpha0: 2, beta0: 2 }))
 * const space = treeSpace(data, treePrior({ maxDepth: 2 }), quantileRules(xs, 5))
 * const draws = sampleTrees(space, { chains: 2, iterations: 200, warmup: 50, thin: 5 })
 * const [low = 0, high = 0] = predictSampled(draws, [[0.5], [4.5]]).mean
 * high - low > 3 // => true
 * ```
 */
export function predictSampled(samples: TreeSamples, xq: TensorLike): LeafPrediction {
  const { data } = samples.space
  const each = samples.draws.map((tree) => treeSummary(data, tree, xq))
  return data.model.read(div(sum(stack(each, 0), 0), samples.draws.length))
}

/**
 * The mean curve of `k` of the drawn trees, evenly spaced through the draws: what each tree says at the query points.
 *
 * @param samples - The draws.
 * @param xq - The query inputs, `[g, d]`.
 * @param k - How many trees.
 * @returns One row of predictive means per tree.
 * @concept bayesian-trees
 * @example
 * ```ts
 * import { drawnCurves, normalGammaLeaf, quantileRules, sampleTrees, bayesData, treePrior, treeSpace } from 'omai-engine/trees'
 * const xs = [[0], [1], [2], [3], [4], [5]]
 * const data = bayesData(xs, [0, 0.1, 0, 5, 5.1, 5], normalGammaLeaf({ m0: 2.5, kappa0: 1, alpha0: 2, beta0: 2 }))
 * const space = treeSpace(data, treePrior({ maxDepth: 2 }), quantileRules(xs, 5))
 * const draws = sampleTrees(space, { chains: 1, iterations: 100, warmup: 20, thin: 10 })
 * drawnCurves(draws, [[0.5], [4.5]], 3).length // => 3
 * ```
 */
export function drawnCurves(samples: TreeSamples, xq: TensorLike, k: number): number[][] {
  const { data } = samples.space
  const n = samples.draws.length
  const stride = Math.max(1, Math.floor(n / k))
  return samples.draws
    .filter((_, i) => i % stride === 0)
    .slice(0, k)
    .map((tree) => data.model.read(treeSummary(data, tree, xq)).mean as number[])
}

/** The posterior over rules from draws: how often each rule is the root and is used anywhere. */
export interface RulePosterior {
  /** The probability that the tree is a single leaf. */
  readonly stump: number
  /** The probability of each rule at the root, `[C]`. */
  readonly root: readonly number[]
  /** The expected number of times each rule is used anywhere in a tree, `[C]`. */
  readonly usage: readonly number[]
}

/**
 * The posterior over rules, estimated from draws, for comparison with the exact enumeration.
 *
 * @param samples - The draws.
 * @returns The probability of each root rule and the expected use of each rule.
 * @concept bayesian-trees
 * @example
 * ```ts
 * import { normalGammaLeaf, quantileRules, rulePosterior, sampleTrees, bayesData, treePrior, treeSpace } from 'omai-engine/trees'
 * const xs = [[0], [1], [2], [3], [4], [5]]
 * const data = bayesData(xs, [0, 0.1, 0, 5, 5.1, 5], normalGammaLeaf({ m0: 2.5, kappa0: 1, alpha0: 2, beta0: 2 }))
 * const space = treeSpace(data, treePrior({ maxDepth: 2 }), quantileRules(xs, 5))
 * const draws = sampleTrees(space, { chains: 2, iterations: 200, warmup: 50, thin: 5 })
 * rulePosterior(draws).root.length // => 5
 * ```
 */
export function rulePosterior(samples: TreeSamples): RulePosterior {
  const c = samples.space.rules.features.length
  const root = Array.from({ length: c }, () => 0)
  const usage = Array.from({ length: c }, () => 0)
  let stump = 0
  const used = (node: BayesTree): void => {
    if (node.kind === 'leaf') return
    const r = samples.space.ruleIndex.get(`${node.feature}:${node.threshold}`) ?? -1
    usage[r] = (usage[r] ?? 0) + 1
    used(node.left)
    used(node.right)
  }
  for (const tree of samples.draws) {
    if (tree.kind === 'leaf') stump += 1
    else {
      const r = samples.space.ruleIndex.get(`${tree.feature}:${tree.threshold}`) ?? -1
      root[r] = (root[r] ?? 0) + 1
    }
    used(tree)
  }
  const n = samples.draws.length
  return { stump: stump / n, root: root.map((v) => v / n), usage: usage.map((v) => v / n) }
}
