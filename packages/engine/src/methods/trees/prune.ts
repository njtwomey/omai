import { type PruningPath, pruningPath } from '../../lib/exposition/index.ts'
import { isLeaf, type TreeParams } from './tree.ts'

// TODO: reduced-error pruning.

interface Pruning {
  readonly alphas: number[]
  readonly impurities: number[]
  readonly leaves: number[]
  /** Whether each node stays in the final subtree. */
  readonly inSubtree: Uint8Array
  /** Whether each kept node became a leaf. */
  readonly leaf: Uint8Array
}

/** Weakest-link pruning (scikit-learn's `_cost_complexity_prune`), until the root or until alpha would exceed `stop`. */
function weakestLink(tree: TreeParams, stop: number): Pruning {
  const n = tree.feature.length
  const total = tree.weightedNSamples[0] ?? 1
  const rNode = Float64Array.from(
    { length: n },
    (_, i) => ((tree.weightedNSamples[i] ?? 0) * (tree.impurity[i] ?? 0)) / total,
  )
  const parent = new Int32Array(n).fill(-1)
  for (let i = 0; i < n; i++)
    if (!isLeaf(tree, i)) {
      parent[tree.left[i] ?? 0] = i
      parent[tree.right[i] ?? 0] = i
    }
  const leaf = Uint8Array.from({ length: n }, (_, i) => (isLeaf(tree, i) ? 1 : 0))
  const rBranch = new Float64Array(n)
  const nLeaves = new Int32Array(n)
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
  const candidate = Uint8Array.from(leaf, (v) => (v ? 0 : 1))
  const inSubtree = new Uint8Array(n).fill(1)
  const alphas = [0]
  const impurities = [rBranch[0] ?? 0]
  const leaves = [nLeaves[0] || 1]
  while (candidate[0]) {
    let alpha = Number.MAX_VALUE
    let pruned = 0
    for (let i = 0; i < n; i++) {
      if (!candidate[i]) continue
      const a = ((rNode[i] ?? 0) - (rBranch[i] ?? 0)) / ((nLeaves[i] ?? 0) - 1)
      if (a < alpha) {
        alpha = a
        pruned = i
      }
    }
    if (stop < alpha) break
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
    alphas.push(alpha)
    impurities.push(rBranch[0] ?? 0)
    leaves.push(Math.max(nLeaves[0] ?? 0, 1))
  }
  return { alphas, impurities, leaves, inSubtree, leaf }
}

/**
 * The minimal cost-complexity pruning path (scikit-learn's `cost_complexity_pruning_path`): the effective alphas at
 * which weakest links are pruned, with the total leaf impurity (weighted by node weight / total weight) and the
 * number of leaves of each pruned tree, from the full tree to the root.
 *
 * @param tree - The fully grown tree.
 * @returns The path as an exposition record.
 */
export function costComplexityPruningPath(tree: TreeParams): PruningPath {
  const p = weakestLink(tree, Number.POSITIVE_INFINITY)
  return pruningPath({ alphas: p.alphas, impurities: p.impurities, leaves: p.leaves })
}

/**
 * The subtree with the largest cost complexity no greater than alpha (scikit-learn's `ccp_alpha`): weakest links are
 * pruned while their effective alpha is at most alpha, and the remaining nodes are renumbered in depth-first order.
 *
 * @param tree - The tree.
 * @param alpha - The complexity parameter (0 returns the tree unchanged).
 * @returns The pruned tree.
 */
export function pruneTree(tree: TreeParams, alpha: number): TreeParams {
  if (!(alpha > 0)) return tree
  const p = weakestLink(tree, alpha)
  const keep: number[] = []
  const walk = (node: number): void => {
    keep.push(node)
    if (!p.leaf[node] && !isLeaf(tree, node)) {
      walk(tree.left[node] ?? 0)
      walk(tree.right[node] ?? 0)
    }
  }
  walk(0)
  const index = new Map(keep.map((old, i) => [old, i]))
  const leafNow = (old: number): boolean => Boolean(p.leaf[old]) || isLeaf(tree, old)
  return {
    ...tree,
    feature: keep.map((o) => (leafNow(o) ? -1 : (tree.feature[o] ?? -1))),
    threshold: keep.map((o) => (leafNow(o) ? Number.NaN : (tree.threshold[o] ?? Number.NaN))),
    left: keep.map((o) => (leafNow(o) ? -1 : (index.get(tree.left[o] ?? -1) ?? -1))),
    right: keep.map((o) => (leafNow(o) ? -1 : (index.get(tree.right[o] ?? -1) ?? -1))),
    value: keep.map((o) => tree.value[o] ?? []),
    impurity: keep.map((o) => tree.impurity[o] ?? 0),
    nSamples: keep.map((o) => tree.nSamples[o] ?? 0),
    weightedNSamples: keep.map((o) => tree.weightedNSamples[o] ?? 0),
  }
}

/**
 * Impurity-based feature importances (mean decrease in impurity), normalised to sum to one (scikit-learn's
 * `feature_importances_`).
 *
 * @param tree - The tree.
 * @returns One importance per feature.
 */
export function featureImportancesOf(tree: TreeParams): number[] {
  const out = new Array<number>(tree.nFeatures).fill(0)
  for (let i = 0; i < tree.feature.length; i++) {
    if (isLeaf(tree, i)) continue
    const l = tree.left[i] ?? 0
    const r = tree.right[i] ?? 0
    const w = (k: number): number => tree.weightedNSamples[k] ?? 0
    const imp = (k: number): number => tree.impurity[k] ?? 0
    const f = tree.feature[i] ?? 0
    out[f] = (out[f] ?? 0) + w(i) * imp(i) - w(l) * imp(l) - w(r) * imp(r)
  }
  const total = out.reduce((a, b) => a + b, 0)
  return total > 0 ? out.map((v) => v / total) : out
}
