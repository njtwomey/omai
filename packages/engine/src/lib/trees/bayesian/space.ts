/**
 * The space of trees over a finite set of candidate rules (a few quantile thresholds per feature), where a tree's prior and
 * exact evidence are cheap: a node's region is the set of rows that reach it, kept as a 0/1 indicator and remembered under
 * the path of rules and sides that leads to it, so the statistics, the evidence and the rules still available at a region
 * are computed once, with one matrix product over all candidate rules, and reused by every tree that has the region. The
 * enumeration and the sampler both work in this space.
 */

import { OmaiError } from '../../base/index.ts'
import {
  add,
  cast,
  expandDims,
  lessEqual,
  linspace,
  matmul,
  mul,
  ones,
  reshape,
  round,
  sliceAxis,
  sub,
  type TensorLike,
  take,
  toFlat,
  transpose,
  unique,
  type Value,
} from '../../tensor/index.ts'
import { logLeaf, logSplit, type TreePrior } from './prior.ts'
import { type BayesData, type BayesTree, column, inputMatrix } from './tree.ts'

/** A finite set of candidate rules: rule `c` sends a row left when `x[features[c]] <= thresholds[c]`. */
export interface Rules {
  readonly features: readonly number[]
  readonly thresholds: readonly number[]
}

/**
 * Candidate rules at evenly spaced ranks of the distinct values of each feature: thresholds at the midpoints of
 * consecutive distinct values, thinned to at most `perFeature` per feature.
 *
 * @param x - The inputs, `[n, d]`.
 * @param perFeature - The most candidate thresholds per feature.
 * @returns The rules, feature by feature in increasing threshold.
 * @concept bayesian-trees
 * @index function Candidate split rules: midpoints of the distinct values of each feature, thinned to quantile positions.
 * @example
 * ```ts
 * import { quantileRules } from 'omai-engine/trees'
 * quantileRules([[0], [1], [2], [3], [4]], 2).thresholds // => [0.5, 3.5]
 * ```
 */
export function quantileRules(x: TensorLike, perFeature: number): Rules {
  const xm = inputMatrix(x)
  const d = xm.shape[1] ?? 1
  const features: number[] = []
  const thresholds: number[] = []
  for (let f = 0; f < d; f++) {
    const values = unique(column(xm, f)).values
    const m = (values.shape[0] ?? 0) - 1
    if (m < 1) continue
    const mids = mul(add(sliceAxis(values, 0, 0, m), sliceAxis(values, 0, 1, m + 1)), 0.5)
    const picks = m <= perFeature ? mids : take(mids, unique(round(linspace(0, m - 1, perFeature))).values, 0)
    for (const t of toFlat(picks)) {
      features.push(f)
      thresholds.push(t)
    }
  }
  return { features, thresholds }
}

/** The rows that reach a node, with what is known about them: the node's statistics, evidence and available rules. */
export interface Region {
  /** The path of rules and sides from the root: `''` for the root, `'3L.7R.'` after rule 3 left and rule 7 right. */
  readonly key: string
  readonly depth: number
  /** 1 for the rows in the region, `[n]`. */
  readonly indicator: Value
  /** The sum of the rows' statistics, `[size]`. */
  readonly stats: Value
  /** The rows in the region. */
  readonly n: number
  /** The log evidence of the region as a leaf. */
  readonly logEvidence: number
  /** The statistics of the rows that go left under each rule, `[C, size]`. */
  readonly left: Value
  /** The rules that leave at least `minLeaf` rows on each side. */
  readonly valid: readonly number[]
  /** The evidence of the two sides of every rule, `[C]` each, once asked for. */
  table?: { readonly left: Value; readonly right: Value }
}

/** Trees over a finite set of rules on some data under a prior, with the regions seen so far. */
export interface TreeSpace {
  readonly data: BayesData
  readonly prior: TreePrior
  readonly rules: Rules
  /** The rules' decisions on every row, `[n, C]`: 1 where the row goes left. */
  readonly mask: Value
  readonly root: Region
  /** The regions seen so far, by path. */
  readonly regions: Map<string, Region>
  /** The index of a rule by feature and threshold. */
  readonly ruleIndex: ReadonlyMap<string, number>
}

const ruleKey = (feature: number, threshold: number): string => `${feature}:${threshold}`

/**
 * Make the space of trees over a set of candidate rules.
 *
 * @param data - The data.
 * @param prior - The tree prior.
 * @param rules - The candidate rules.
 * @returns The space.
 * @throws OmaiError with code `value` when there are no rules.
 * @concept bayesian-trees
 * @index function Space of trees over a finite rule set: exact prior and evidence.
 * @example
 * ```ts
 * import { normalGammaLeaf, quantileRules, bayesData, treePrior, treeSpace } from 'omai-engine/trees'
 * const model = normalGammaLeaf({ m0: 0, kappa0: 1, alpha0: 2, beta0: 1 })
 * const data = bayesData([[0], [1], [2], [3], [4], [5]], [0, 0, 0, 5, 5, 5], model)
 * const space = treeSpace(data, treePrior({ maxDepth: 2 }), quantileRules([[0], [1], [2], [3], [4], [5]], 5))
 * space.root.valid.length // => 3
 * ```
 */
export function treeSpace(data: BayesData, prior: TreePrior, rules: Rules): TreeSpace {
  if (rules.features.length === 0) throw new OmaiError('value', 'treeSpace: no candidate rules')
  const columns = take(data.x, rules.features as number[], 1)
  const mask = cast(lessEqual(columns, reshape(rules.thresholds as number[], [1, -1])), 'float64')
  const ruleIndex = new Map(rules.features.map((f, c) => [ruleKey(f, rules.thresholds[c] ?? 0), c] as const))
  const space = { data, prior, rules, mask, regions: new Map<string, Region>(), ruleIndex }
  const root = regionOf(space, '', 0, ones([data.n]))
  return { ...space, root }
}

type Building = Pick<TreeSpace, 'data' | 'prior' | 'rules' | 'mask' | 'regions' | 'ruleIndex'>

function regionOf(space: Building, key: string, depth: number, indicator: Value): Region {
  const { model } = space.data
  const weighted = mul(space.data.rows, expandDims(indicator, 1))
  const stats = matmul(indicator, space.data.rows)
  const left = matmul(transpose(space.mask), weighted)
  const n = toFlat(model.count(stats))[0] ?? 0
  const nLeft = toFlat(model.count(left))
  const valid = nLeft.flatMap((l, c) => (l >= space.prior.minLeaf && n - l >= space.prior.minLeaf ? [c] : []))
  const region: Region = {
    key,
    depth,
    indicator,
    stats,
    n,
    logEvidence: toFlat(model.logEvidence(stats))[0] ?? 0,
    left,
    valid,
  }
  space.regions.set(key, region)
  return region
}

/**
 * The region of the rows that go one way under a rule at a region (cached by path).
 *
 * @param space - The space.
 * @param parent - The region the rule splits.
 * @param rule - The rule's index.
 * @param side - `'L'` for the rows at or below the threshold, `'R'` for those above.
 * @returns The child region.
 * @concept bayesian-trees
 * @example
 * ```ts
 * import { childRegion, normalGammaLeaf, quantileRules, bayesData, treePrior, treeSpace } from 'omai-engine/trees'
 * const xs = [[0], [1], [2], [3], [4], [5]]
 * const model = normalGammaLeaf({ m0: 0, kappa0: 1, alpha0: 2, beta0: 1 })
 * const space = treeSpace(bayesData(xs, [0, 0, 0, 5, 5, 5], model), treePrior(), quantileRules(xs, 5))
 * childRegion(space, space.root, 2, 'L').n // => 3
 * ```
 */
export function childRegion(space: TreeSpace, parent: Region, rule: number, side: 'L' | 'R'): Region {
  const key = `${parent.key}${rule}${side}.`
  const cached = space.regions.get(key)
  if (cached !== undefined) return cached
  const decision = reshape(sliceAxis(space.mask, 1, rule, rule + 1), [-1])
  const goes = side === 'L' ? decision : sub(1, decision)
  return regionOf(space, key, parent.depth + 1, mul(parent.indicator, goes))
}

/**
 * The evidence of the two sides of every rule at a region, `[C]` each: the left and the right as leaves.
 *
 * @param space - The space.
 * @param region - The region.
 * @returns The two evidence vectors.
 * @concept bayesian-trees
 * @example
 * ```ts
 * import { normalGammaLeaf, quantileRules, splitTable, bayesData, treePrior, treeSpace } from 'omai-engine/trees'
 * const xs = [[0], [1], [2], [3], [4], [5]]
 * const model = normalGammaLeaf({ m0: 0, kappa0: 1, alpha0: 2, beta0: 1 })
 * const space = treeSpace(bayesData(xs, [0, 0, 0, 5, 5, 5], model), treePrior(), quantileRules(xs, 5))
 * splitTable(space, space.root).left.shape // => [5]
 * ```
 */
export function splitTable(space: TreeSpace, region: Region): { readonly left: Value; readonly right: Value } {
  if (region.table === undefined) {
    const { model } = space.data
    region.table = {
      left: model.logEvidence(region.left),
      right: model.logEvidence(sub(region.stats, region.left)),
    }
  }
  return region.table
}

/** The index of a split node's rule in the space. */
export function ruleOf(
  space: TreeSpace,
  node: { readonly feature: number; readonly threshold: number; readonly rule?: number },
): number {
  const c = node.rule ?? space.ruleIndex.get(ruleKey(node.feature, node.threshold))
  if (c === undefined)
    throw new OmaiError('value', `a split on feature ${node.feature} at ${node.threshold} is not a candidate rule`)
  return c
}

/** The log prior and the log evidence of a tree in a space, `-Infinity` for a tree the prior rules out. */
export interface TreeScore {
  readonly logPrior: number
  readonly logEvidence: number
  /** The log posterior up to the constant: prior times evidence. */
  readonly logPosterior: number
}

/**
 * The exact log prior, log evidence and unnormalised log posterior of a tree over the candidate rules. The prior is the
 * product over nodes: a leaf contributes its probability of not splitting, a split its probability of splitting times one
 * over the rules available at its region; the evidence is the sum of the leaf evidences.
 *
 * @param space - The space.
 * @param tree - A tree whose splits are candidate rules.
 * @returns The score; every field is `-Infinity` when a split is not available at its region.
 * @concept bayesian-trees
 * @index function Exact log prior and log evidence of a tree over a candidate rule set.
 * @example
 * ```ts
 * import { bayesLeaf, normalGammaLeaf, quantileRules, scoreTree, bayesData, treePrior, treeSpace } from 'omai-engine/trees'
 * const xs = [[0], [1], [2], [3], [4], [5]]
 * const model = normalGammaLeaf({ m0: 0, kappa0: 1, alpha0: 2, beta0: 1 })
 * const space = treeSpace(bayesData(xs, [0, 0, 0, 5, 5, 5], model), treePrior(), quantileRules(xs, 5))
 * Math.exp(scoreTree(space, bayesLeaf).logPrior) // => 0.5
 * ```
 */
export function scoreTree(space: TreeSpace, tree: BayesTree): TreeScore {
  let logPrior = 0
  let logEvidence = 0
  const walk = (node: BayesTree, region: Region): void => {
    if (node.kind === 'leaf') {
      logPrior += logLeaf(space.prior, region.depth, region.valid.length)
      logEvidence += region.logEvidence
      return
    }
    const rule = ruleOf(space, node)
    if (!region.valid.includes(rule)) {
      logPrior = Number.NEGATIVE_INFINITY
      return
    }
    logPrior += logSplit(space.prior, region.depth, region.valid.length)
    walk(node.left, childRegion(space, region, rule, 'L'))
    if (logPrior > Number.NEGATIVE_INFINITY) walk(node.right, childRegion(space, region, rule, 'R'))
  }
  walk(tree, space.root)
  if (logPrior === Number.NEGATIVE_INFINITY) {
    return { logPrior, logEvidence: Number.NEGATIVE_INFINITY, logPosterior: Number.NEGATIVE_INFINITY }
  }
  return { logPrior, logEvidence, logPosterior: logPrior + logEvidence }
}

/** A tree's node with the region it sits at and its place in the tree. */
export interface PlacedNode {
  readonly node: BayesTree
  readonly region: Region
  /** The sides taken from the root: `'LR'` is the right child of the left child. */
  readonly path: string
}

/**
 * Every node of a tree with its region and path, root first, left before right.
 *
 * @param space - The space.
 * @param tree - A tree whose splits are candidate rules available at their regions.
 * @returns The nodes.
 * @concept bayesian-trees
 * @example
 * ```ts
 * import { bayesLeaf, bayesSplit, normalGammaLeaf, placeNodes, quantileRules, bayesData, treePrior, treeSpace } from 'omai-engine/trees'
 * const xs = [[0], [1], [2], [3], [4], [5]]
 * const model = normalGammaLeaf({ m0: 0, kappa0: 1, alpha0: 2, beta0: 1 })
 * const space = treeSpace(bayesData(xs, [0, 0, 0, 5, 5, 5], model), treePrior(), quantileRules(xs, 5))
 * placeNodes(space, bayesSplit(0, 2.5, bayesLeaf, bayesLeaf)).map((p) => p.path) // => ['', 'L', 'R']
 * ```
 */
export function placeNodes(space: TreeSpace, tree: BayesTree): PlacedNode[] {
  const out: PlacedNode[] = []
  const walk = (node: BayesTree, region: Region, path: string): void => {
    out.push({ node, region, path })
    if (node.kind === 'leaf') return
    const rule = ruleOf(space, node)
    walk(node.left, childRegion(space, region, rule, 'L'), `${path}L`)
    walk(node.right, childRegion(space, region, rule, 'R'), `${path}R`)
  }
  walk(tree, space.root, '')
  return out
}
