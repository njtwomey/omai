/**
 * The prior over tree structures (Chipman, George and McCulloch 1998): a node at depth `d` splits with probability
 * `alpha (1 + d)^(-beta)`, and a split's feature and threshold are uniform among the rules that leave at least
 * `minLeaf` rows on each side of the node. A node with no such rule, or at the depth limit, is a leaf for certain.
 */

import { OmaiError } from '../../base/index.ts'

/** The tree prior. */
export interface TreePrior {
  /** The probability that the root splits (before the depth decay). Default 0.5; Chipman et al. use 0.95 for ensembles of small trees. */
  readonly alpha: number
  /** The decay with depth: a node at depth `d` splits with probability `alpha (1 + d)^(-beta)`. Default 2. */
  readonly beta: number
  /** The deepest a tree may grow: nodes at this depth are leaves. Default 3. */
  readonly maxDepth: number
  /** A split must leave at least this many rows on each side. Default 2. */
  readonly minLeaf: number
}

/**
 * The tree prior with defaults for what is not given.
 *
 * @param options - Any of `alpha`, `beta`, `maxDepth`, `minLeaf`.
 * @returns The prior.
 * @throws OmaiError with code `value` unless `0 < alpha < 1`, `beta >= 0`, `maxDepth >= 0` and `minLeaf >= 1`.
 * @concept bayesian-trees
 * @index function Tree-structure prior: split probability decaying with depth, minimum leaf size, depth limit.
 * @example
 * ```ts
 * import { treePrior } from 'omai-engine/trees'
 * treePrior({ maxDepth: 2 }) // => { alpha: 0.5, beta: 2, maxDepth: 2, minLeaf: 2 }
 * ```
 */
export function treePrior(options: Partial<TreePrior> = {}): TreePrior {
  const p = { alpha: 0.5, beta: 2, maxDepth: 3, minLeaf: 2, ...options }
  if (!(p.alpha > 0 && p.alpha < 1 && p.beta >= 0 && p.maxDepth >= 0 && p.minLeaf >= 1)) {
    throw new OmaiError(
      'value',
      `treePrior: need 0 < alpha < 1, beta >= 0, maxDepth >= 0, minLeaf >= 1; got ${JSON.stringify(p)}`,
    )
  }
  return p
}

/**
 * The probability that a node splits: `alpha (1 + depth)^(-beta)`, or 0 at the depth limit or when no rule is available.
 *
 * @param prior - The prior.
 * @param depth - The node's depth.
 * @param available - How many rules are available at the node.
 * @returns The probability.
 * @concept bayesian-trees
 * @example
 * ```ts
 * import { splitProbability, treePrior } from 'omai-engine/trees'
 * splitProbability(treePrior(), 1, 10) // => 0.125
 * ```
 */
export function splitProbability(prior: TreePrior, depth: number, available: number): number {
  return depth >= prior.maxDepth || available <= 0 ? 0 : prior.alpha * (1 + depth) ** -prior.beta
}

/** The log probability that a node is a leaf (0 when it cannot split). */
export function logLeaf(prior: TreePrior, depth: number, available: number): number {
  return Math.log(1 - splitProbability(prior, depth, available))
}

/** The log probability that a node splits by one particular rule: the split probability times `1 / available`. */
export function logSplit(prior: TreePrior, depth: number, available: number): number {
  const p = splitProbability(prior, depth, available)
  return p === 0 ? Number.NEGATIVE_INFINITY : Math.log(p) - Math.log(available)
}
