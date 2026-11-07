/**
 * Exposition records for decision trees (CART, ID3, C4.5, CHAID, and trees inside ensembles): every node with its
 * samples, impurity, prediction and the split candidates it considered, so a view can explain how the tree was grown,
 * plus the cost-complexity pruning path. `treeRecord` checks that the nodes form a tree.
 */

import { MdmlError } from '../base/index.ts'
import { seal } from './_check.ts'

/** A split a tree considered at a node. */
export interface SplitCandidate {
  readonly feature: number
  readonly threshold: number
  /** Impurity of the left and right children, and the weighted decrease from the parent. */
  readonly impurityLeft: number
  readonly impurityRight: number
  readonly decrease: number
  /** For a categorical split, the category codes sent left (the threshold is then meaningless). */
  readonly categories?: readonly number[]
  /** Where rows with a missing value go, when some did (or a rule for future ones). */
  readonly missingLeft?: boolean
  /** The score the split was ranked by, when it is not `decrease` (Friedman's improvement, C4.5's gain ratio). */
  readonly score?: number
}

/** One node of a grown tree, with everything needed to explain how it was grown. */
export interface TreeNode {
  readonly id: number
  readonly parent: number | null
  readonly depth: number
  /** Indices of the training rows that reach the node. */
  readonly rows: readonly number[]
  /** Class counts (classification) or the mean target (regression, one entry). */
  readonly value: readonly number[]
  readonly impurity: number
  /** Every candidate split evaluated (possibly truncated to the best few per feature). */
  readonly candidates: readonly SplitCandidate[]
  /** The chosen split, absent at leaves. */
  readonly split?: {
    readonly feature: number
    readonly threshold: number
    readonly left: number
    readonly right: number
    /** For a categorical split, the category codes sent left. */
    readonly categories?: readonly number[]
    /** Whether rows with a missing value go left. */
    readonly missingLeft?: boolean
    /** For a multiway split (ID3, C4.5, CHAID): every branch, with the category codes it takes. */
    readonly branches?: readonly { readonly child: number; readonly categories: readonly number[] }[]
  }
  /** When the node was grown (0 for the root). */
  readonly order: number
}

/** A tree with its growth (CART, ID3, C4.5, CHAID, trees inside ensembles). */
export interface TreeRecord {
  readonly kind: 'tree'
  readonly criterion: string
  readonly features: readonly string[]
  readonly classes?: readonly string[]
  readonly nodes: readonly TreeNode[]
}

/**
 * A tree record, checked to be a tree (one root, children referencing existing nodes).
 *
 * @param tree - The record without its `kind`.
 * @returns The record.
 * @throws MdmlError `value` when the nodes do not form a tree.
 * @example
 * ```ts
 * import { treeRecord } from 'mdml-engine/exposition'
 * treeRecord({ criterion: 'gini', features: ['x'], nodes: [{ id: 0, parent: null, depth: 0, rows: [0, 1], value: [1, 1], impurity: 0.5, candidates: [], order: 0 }] }).nodes.length // => 1
 * ```
 */
export function treeRecord(tree: Omit<TreeRecord, 'kind'>): TreeRecord {
  const ids = new Set(tree.nodes.map((n) => n.id))
  if (tree.nodes.filter((n) => n.parent === null).length !== 1)
    throw new MdmlError('value', 'treeRecord: a tree has exactly one root')
  for (const n of tree.nodes) {
    if (n.parent !== null && !ids.has(n.parent))
      throw new MdmlError('value', `treeRecord: node ${n.id} has an unknown parent ${n.parent}`)
    if (n.split !== undefined && !(ids.has(n.split.left) && ids.has(n.split.right)))
      throw new MdmlError('value', `treeRecord: node ${n.id} splits into unknown children`)
  }
  return seal({ kind: 'tree', ...tree })
}

/** The cost-complexity pruning path: the effective alphas and the impurities and sizes of the pruned trees. */
export interface PruningPath {
  readonly kind: 'pruningPath'
  readonly alphas: readonly number[]
  readonly impurities: readonly number[]
  readonly leaves: readonly number[]
}

/**
 * A pruning path, with its shapes checked.
 *
 * @param path - The path without its `kind`.
 * @returns The record.
 * @throws MdmlError `shape` when the columns differ in length.
 * @example
 * ```ts
 * import { pruningPath } from 'mdml-engine/exposition'
 * pruningPath({ alphas: [0, 0.1], impurities: [0, 0.2], leaves: [4, 1] }).leaves // => [4, 1]
 * ```
 */
export function pruningPath(path: Omit<PruningPath, 'kind'>): PruningPath {
  if (path.impurities.length !== path.alphas.length || path.leaves.length !== path.alphas.length)
    throw new MdmlError('shape', 'pruningPath: alphas, impurities and leaves must have one entry per step')
  return seal({ kind: 'pruningPath', ...path })
}
