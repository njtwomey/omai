/**
 * Pruning decision trees. Minimal cost-complexity pruning as in scikit-learn: the weakest link (the internal node whose
 * subtree buys the least impurity per leaf) is cut, step by step, as an algorithm whose trace is the pruning path of
 * effective alphas; `pruneTree` stops at a given alpha and `costComplexityPath` runs to the root. Reduced-error pruning
 * (Quinlan 1987) replaces a subtree by a leaf bottom-up whenever that does not raise the error on held-out rows, again
 * one node per step.
 */

import { MdmlError } from '../base/index.ts'
import { type PruningPath, pruningPath } from '../exposition/index.ts'
import type { LearnData } from '../learn/estimator.ts'
import { codesFor, matrixView, sampleWeightsOf, targetVector } from '../learn/validation.ts'
import { type Algorithm, runAlgorithm, stateFromJson, stateToJson } from '../trace/index.ts'
import { isLeaf, type Tree } from './tree.ts'

/** The state of {@link weakestLinkAlgorithm}: the subtree so far and the path walked. */
export interface PruneState {
  /** Per node: 1 when it is a leaf of the current subtree (an original leaf or a pruned node). */
  readonly leaf: readonly number[]
  /** Per node: 1 while it is part of the current subtree. */
  readonly inSubtree: readonly number[]
  /** Per node: 1 while it is an internal node of the current subtree. */
  readonly candidate: readonly number[]
  /** The summed leaf risk below each node, and its leaf count. */
  readonly rBranch: readonly number[]
  readonly nLeaves: readonly number[]
  /** The path: effective alphas, total leaf impurity and leaf counts, from the full tree. */
  readonly alphas: readonly number[]
  readonly impurities: readonly number[]
  readonly leaves: readonly number[]
  /** The node pruned last (-1 before the first step). */
  readonly pruned: number
}

/** Parents of the nodes of a tree (-1 for the root). */
function parentsOf(tree: Tree): Int32Array {
  const parent = new Int32Array(tree.feature.length).fill(-1)
  for (let i = 0; i < tree.feature.length; i++)
    if (!isLeaf(tree, i)) {
      parent[tree.left[i] ?? 0] = i
      parent[tree.right[i] ?? 0] = i
    }
  return parent
}

/**
 * Minimal cost-complexity pruning (scikit-learn's `_cost_complexity_prune`) as an algorithm: each step cuts the weakest
 * link, the internal node with the smallest effective alpha (its risk minus its subtree's, per leaf removed), and
 * records the alpha, the total leaf impurity and the leaf count of the pruned tree. It converges at the root. Logs, when
 * subscribed: `prune/alpha`, `prune/leaves`, `prune/trainError` (the total weighted leaf impurity) and the event
 * `prune/pruned` with the node cut.
 *
 * @param tree - The fully grown tree.
 * @param stopAlpha - Stop before cutting a link whose alpha exceeds this (default none: prune to the root).
 * @returns The algorithm.
 * @concept decision-trees
 * @example
 * ```ts
 * import { weakestLinkAlgorithm } from 'mdml-engine/trees'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const tree = { task: 'regression' as const, classes: [], nFeatures: 1, feature: [0, -2, -2], threshold: [0.5, -2, -2], left: [1, -1, -1], right: [2, -1, -1], value: [[1], [0], [2]], impurity: [1, 0, 0], nSamples: [2, 1, 1], weightedNSamples: [2, 1, 1], missingLeft: [false, false, false], categories: [[], [], []] }
 * runAlgorithm(weakestLinkAlgorithm(tree), { maxSteps: 10 }).state.alphas // => [0, 1]
 * ```
 */
export function weakestLinkAlgorithm(tree: Tree, stopAlpha: number = Number.POSITIVE_INFINITY): Algorithm<PruneState> {
  const n = tree.feature.length
  const total = tree.weightedNSamples[0] ?? 1
  const rNode = Float64Array.from(
    { length: n },
    (_, i) => ((tree.weightedNSamples[i] ?? 0) * (tree.impurity[i] ?? 0)) / total,
  )
  const parent = parentsOf(tree)
  const weakest = (s: PruneState): { readonly alpha: number; readonly node: number } => {
    let alpha = Number.MAX_VALUE
    let node = 0
    for (let i = 0; i < n; i++) {
      if (!s.candidate[i]) continue
      const a = ((rNode[i] ?? 0) - (s.rBranch[i] ?? 0)) / ((s.nLeaves[i] ?? 0) - 1)
      if (a < alpha) {
        alpha = a
        node = i
      }
    }
    return { alpha, node }
  }
  return {
    name: 'costComplexityPruning',
    config: { stopAlpha: Number.isFinite(stopAlpha) ? stopAlpha : null },
    init: () => {
      const leaf = Array.from({ length: n }, (_, i) => (isLeaf(tree, i) ? 1 : 0))
      const rBranch = new Array<number>(n).fill(0)
      const nLeaves = new Array<number>(n).fill(0)
      for (let l = 0; l < n; l++) {
        if (!leaf[l]) continue
        rBranch[l] = rNode[l] ?? 0
        let node = l
        while (node !== 0) {
          const p = parent[node] ?? 0
          rBranch[p] = (rBranch[p] ?? 0) + (rNode[l] ?? 0)
          nLeaves[p] = (nLeaves[p] ?? 0) + 1
          node = p
        }
      }
      return {
        leaf,
        inSubtree: new Array<number>(n).fill(1),
        candidate: leaf.map((v) => (v ? 0 : 1)),
        rBranch,
        nLeaves,
        alphas: [0],
        impurities: [rBranch[0] ?? 0],
        leaves: [nLeaves[0] || 1],
        pruned: -1,
      }
    },
    step: (s, ctx) => {
      const { alpha, node: pruned } = weakest(s)
      const leaf = [...s.leaf]
      const inSubtree = [...s.inSubtree]
      const candidate = [...s.candidate]
      const rBranch = [...s.rBranch]
      const nLeaves = [...s.nLeaves]
      const stack = [pruned]
      while (stack.length > 0) {
        const node = stack.pop() ?? 0
        if (!inSubtree[node]) continue
        candidate[node] = 0
        leaf[node] = 0
        inSubtree[node] = 0
        if (!isLeaf(tree, node)) stack.push(tree.left[node] ?? 0, tree.right[node] ?? 0)
      }
      leaf[pruned] = 1
      inSubtree[pruned] = 1
      const removed = (nLeaves[pruned] ?? 0) - 1
      nLeaves[pruned] = 0
      const diff = (rNode[pruned] ?? 0) - (rBranch[pruned] ?? 0)
      rBranch[pruned] = rNode[pruned] ?? 0
      let node = parent[pruned] ?? -1
      while (node !== -1) {
        nLeaves[node] = (nLeaves[node] ?? 0) - removed
        rBranch[node] = (rBranch[node] ?? 0) + diff
        node = parent[node] ?? -1
      }
      const leaves = Math.max(nLeaves[0] ?? 0, 1)
      if (ctx.wants('prune/alpha')) ctx.log('prune/alpha', alpha)
      if (ctx.wants('prune/leaves')) ctx.log('prune/leaves', leaves)
      if (ctx.wants('prune/trainError')) ctx.log('prune/trainError', rBranch[0] ?? 0)
      ctx.emit('prune/pruned', { node: pruned, alpha })
      return {
        leaf,
        inSubtree,
        candidate,
        rBranch,
        nLeaves,
        alphas: [...s.alphas, alpha],
        impurities: [...s.impurities, rBranch[0] ?? 0],
        leaves: [...s.leaves, leaves],
        pruned,
      }
    },
    done: (s) => (!s.candidate[0] || weakest(s).alpha > stopAlpha ? 'converged' : undefined),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as PruneState,
  }
}

/**
 * The tree with some nodes turned into leaves, the nodes below them dropped and the rest renumbered in depth-first
 * order (the numbering scikit-learn gives a pruned tree).
 *
 * @param tree - The tree.
 * @param isLeafNow - Per node, whether it is a leaf now (truthy for original leaves and pruned nodes).
 * @returns The collapsed tree.
 * @example
 * ```ts
 * import { collapseTree, stumpTree } from 'mdml-engine/trees'
 * collapseTree(stumpTree({ feature: 0, threshold: 0.5, left: 0, right: 1 }), [1, 0, 0]).left // => [-1]
 * ```
 */
export function collapseTree(tree: Tree, isLeafNow: ArrayLike<number | boolean>): Tree {
  const keep: number[] = []
  const walk = (node: number): void => {
    keep.push(node)
    if (!isLeafNow[node] && !isLeaf(tree, node)) {
      walk(tree.left[node] ?? 0)
      walk(tree.right[node] ?? 0)
    }
  }
  walk(0)
  const index = new Map(keep.map((old, i) => [old, i]))
  const leafNow = (old: number): boolean => Boolean(isLeafNow[old]) || isLeaf(tree, old)
  return {
    ...tree,
    feature: keep.map((o) => (leafNow(o) ? -2 : (tree.feature[o] ?? -2))),
    threshold: keep.map((o) => (leafNow(o) ? -2 : (tree.threshold[o] ?? -2))),
    left: keep.map((o) => (leafNow(o) ? -1 : (index.get(tree.left[o] ?? -1) ?? -1))),
    right: keep.map((o) => (leafNow(o) ? -1 : (index.get(tree.right[o] ?? -1) ?? -1))),
    value: keep.map((o) => tree.value[o] ?? []),
    impurity: keep.map((o) => tree.impurity[o] ?? 0),
    nSamples: keep.map((o) => tree.nSamples[o] ?? 0),
    weightedNSamples: keep.map((o) => tree.weightedNSamples[o] ?? 0),
    missingLeft: keep.map((o) => (leafNow(o) ? false : tree.missingLeft[o] === true)),
    categories: keep.map((o) => (leafNow(o) ? [] : [...(tree.categories[o] ?? [])])),
  }
}

/**
 * The minimal cost-complexity pruning path (scikit-learn's `cost_complexity_pruning_path`): the effective alphas at
 * which weakest links are pruned, with the total leaf impurity (weighted by node weight over total weight) and the
 * number of leaves of each pruned tree, from the full tree to the root.
 *
 * @param tree - The fully grown tree.
 * @returns The path as an exposition record.
 * @example
 * ```ts
 * import { costComplexityPath, stumpTree } from 'mdml-engine/trees'
 * costComplexityPath({ ...stumpTree({ feature: 0, threshold: 0.5, left: 0, right: 2 }), impurity: [1, 0, 0] }).alphas // => [0, 1]
 * ```
 */
export function costComplexityPath(tree: Tree): PruningPath {
  const s = runAlgorithm(weakestLinkAlgorithm(tree), { maxSteps: tree.feature.length + 1 }).state
  return pruningPath({ alphas: [...s.alphas], impurities: [...s.impurities], leaves: [...s.leaves] })
}

/**
 * The subtree with the largest cost complexity no greater than alpha (scikit-learn's `ccp_alpha`): weakest links are
 * pruned while their effective alpha is at most alpha, and the remaining nodes are renumbered in depth-first order.
 *
 * @param tree - The tree.
 * @param alpha - The complexity parameter (0 returns the tree unchanged).
 * @returns The pruned tree.
 * @example
 * ```ts
 * import { pruneTree, stumpTree } from 'mdml-engine/trees'
 * const t = { ...stumpTree({ feature: 0, threshold: 0.5, left: 0, right: 2 }), impurity: [1, 0, 0] }
 * pruneTree(t, 0.5).left.length // => 3
 * pruneTree(t, 1).left.length // => 1
 * ```
 */
export function pruneTree(tree: Tree, alpha: number): Tree {
  if (!(alpha > 0)) return tree
  const s = runAlgorithm(weakestLinkAlgorithm(tree, alpha), { maxSteps: tree.feature.length + 1 }).state
  return collapseTree(tree, s.leaf)
}

/** The state of {@link reducedErrorAlgorithm}. */
export interface ReducedErrorState {
  /** The next position in the bottom-up order. */
  readonly at: number
  /** Per node: 1 once it has been turned into a leaf. */
  readonly pruned: readonly number[]
  /** Per node, once visited: the least held-out error of the subtree rooted there. */
  readonly error: readonly number[]
}

/**
 * Reduced-error pruning (Quinlan 1987) as an algorithm over a held-out set: nodes are visited bottom-up, one per step;
 * an internal node becomes a leaf (predicting its training majority or mean) when that makes the held-out error no
 * larger than its children's combined error. Logs `prune/leaves`, `prune/trainError` (here the held-out error of the
 * tree so far) and the event `prune/pruned`.
 *
 * @param tree - The tree.
 * @param held - Held-out data with targets (and optional weights).
 * @returns The algorithm.
 * @concept decision-trees
 * @example
 * ```ts
 * import { reducedErrorAlgorithm, stumpTree } from 'mdml-engine/trees'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const t = stumpTree({ feature: 0, threshold: 0.5, left: 0, right: 1 })
 * // On held-out rows where the split does not help, the stump is pruned to its root.
 * runAlgorithm(reducedErrorAlgorithm(t, { x: [[0], [1]], y: [0.5, 0.5] }), { maxSteps: 5 }).state.pruned // => [1, 0, 0]
 * ```
 */
export function reducedErrorAlgorithm(tree: Tree, held: LearnData): Algorithm<ReducedErrorState> {
  if (held.y === undefined) throw new MdmlError('value', 'reducedErrorAlgorithm: held-out targets are needed')
  const m = matrixView(held.x, 'reducedErrorAlgorithm', { nFeatures: tree.nFeatures, allowNaN: true })
  const w = sampleWeightsOf(held.weights, m.n, 'reducedErrorAlgorithm')
  const y =
    tree.task === 'classification'
      ? Float64Array.from(codesFor(held.y, tree.classes, 'reducedErrorAlgorithm'))
      : targetVector(held.y, 'reducedErrorAlgorithm', m.n)
  const n = tree.feature.length
  // The held-out rows reaching each node, and the error if the node were a leaf.
  const reach: number[][] = Array.from({ length: n }, () => [])
  for (let i = 0; i < m.n; i++) {
    let node = 0
    reach[0]?.push(i)
    while (!isLeaf(tree, node)) {
      const v = m.data[i * m.d + (tree.feature[node] ?? 0)] ?? 0
      const cats = tree.categories[node] ?? []
      const left = Number.isNaN(v)
        ? tree.missingLeft[node] === true
        : cats.length > 0
          ? cats.includes(v)
          : v <= (tree.threshold[node] ?? 0)
      node = left ? (tree.left[node] ?? 0) : (tree.right[node] ?? 0)
      reach[node]?.push(i)
    }
  }
  const leafError = Float64Array.from({ length: n }, (_, node) => {
    const v = tree.value[node] ?? []
    let e = 0
    for (const i of reach[node] ?? []) {
      if (tree.task === 'classification') {
        let best = 0
        for (let c = 1; c < v.length; c++) if ((v[c] ?? 0) > (v[best] ?? 0)) best = c
        e += (y[i] ?? 0) === best ? 0 : (w[i] ?? 0)
      } else e += (w[i] ?? 0) * ((y[i] ?? 0) - (v[0] ?? 0)) ** 2
    }
    return e
  })
  const order: number[] = []
  const walk = (node: number): void => {
    if (!isLeaf(tree, node)) {
      walk(tree.left[node] ?? 0)
      walk(tree.right[node] ?? 0)
    }
    order.push(node)
  }
  walk(0)
  return {
    name: 'reducedErrorPruning',
    config: { nodes: n },
    init: () => ({ at: 0, pruned: new Array<number>(n).fill(0), error: new Array<number>(n).fill(0) }),
    step: (s, ctx) => {
      const node = order[s.at] ?? 0
      const pruned = [...s.pruned]
      const error = [...s.error]
      if (isLeaf(tree, node)) error[node] = leafError[node] ?? 0
      else {
        const below = (s.error[tree.left[node] ?? 0] ?? 0) + (s.error[tree.right[node] ?? 0] ?? 0)
        if ((leafError[node] ?? 0) <= below) {
          pruned[node] = 1
          error[node] = leafError[node] ?? 0
          ctx.emit('prune/pruned', { node })
        } else error[node] = below
      }
      if (ctx.wants('prune/leaves') || ctx.wants('prune/trainError')) {
        const now = collapseTree(tree, pruned)
        if (ctx.wants('prune/leaves')) ctx.log('prune/leaves', now.left.filter((l) => l < 0).length)
      }
      if (ctx.wants('prune/trainError')) ctx.log('prune/trainError', error[0] ?? 0)
      return { at: s.at + 1, pruned, error }
    },
    done: (s) => (s.at >= order.length ? 'converged' : undefined),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as ReducedErrorState,
  }
}

/**
 * Reduced-error pruning: the tree with every subtree that does not help on held-out rows replaced by a leaf.
 *
 * @param tree - The tree.
 * @param held - Held-out data with targets.
 * @returns The pruned tree.
 * @example
 * ```ts
 * import { reducedErrorPrune, stumpTree } from 'mdml-engine/trees'
 * const t = stumpTree({ feature: 0, threshold: 0.5, left: 0, right: 1 })
 * reducedErrorPrune(t, { x: [[0], [1]], y: [0.5, 0.5] }).left // => [-1]
 * reducedErrorPrune(t, { x: [[0], [1]], y: [0, 1] }).left // => [1, -1, -1]
 * ```
 */
export function reducedErrorPrune(tree: Tree, held: LearnData): Tree {
  const r = runAlgorithm(reducedErrorAlgorithm(tree, held), { maxSteps: tree.feature.length + 1 })
  return collapseTree(tree, r.state.pruned)
}
