/**
 * A fitted decision tree as plain arrays, in the layout of scikit-learn's `tree_` (split feature, threshold, children
 * and node values), and what can be done with one: the leaf and decision path of each row, predictions, depth and leaf
 * counts, a structure for drawing with mdml-render, and indented text as in `export_text`.
 */

import { argmaxOf } from '../../lib/base/index.ts'
import { matrixView } from '../../lib/learn/index.ts'
import { type Structure, structure } from '../../lib/scene/index.ts'
import { fromStorage, type Tensor, type TensorLike } from '../../lib/tensor/index.ts'

/**
 * A fitted decision tree as plain arrays (scikit-learn's `tree_`), one entry per node in growth order: the split
 * feature and threshold (-1 and NaN at leaves; rows with x[feature] <= threshold go left), the children (-1 at
 * leaves), the node's value (weighted class counts for classification, the mean target for regression), impurity,
 * sample count and weighted sample count.
 */
export type TreeParams = {
  readonly task: 'classification' | 'regression'
  /** Class labels for classification (empty for regression). */
  readonly classes: number[]
  readonly nFeatures: number
  readonly feature: number[]
  readonly threshold: number[]
  readonly left: number[]
  readonly right: number[]
  readonly value: number[][]
  readonly impurity: number[]
  readonly nSamples: number[]
  readonly weightedNSamples: number[]
}

/** Whether a node is a leaf. */
export function isLeaf(tree: TreeParams, node: number): boolean {
  return (tree.left[node] ?? -1) < 0
}

/**
 * The leaf each row reaches (scikit-learn's `apply`).
 *
 * @param tree - The tree.
 * @param x - Features `[m, d]`.
 * @returns Leaf ids, one per row.
 */
export function applyTree(tree: TreeParams, x: TensorLike): Int32Array {
  const m = matrixView(x, 'applyTree', { nFeatures: tree.nFeatures, allowNaN: true })
  const out = new Int32Array(m.n)
  for (let i = 0; i < m.n; i++) {
    let node = 0
    while (!isLeaf(tree, node)) {
      const v = m.data[i * m.d + (tree.feature[node] ?? 0)] ?? 0
      node = v <= (tree.threshold[node] ?? 0) ? (tree.left[node] ?? 0) : (tree.right[node] ?? 0)
    }
    out[i] = node
  }
  return out
}

/**
 * The nodes each row passes through from the root to its leaf (scikit-learn's `decision_path`, as lists).
 *
 * @param tree - The tree.
 * @param x - Features `[m, d]`.
 * @returns One path of node ids per row.
 */
export function decisionPath(tree: TreeParams, x: TensorLike): number[][] {
  const m = matrixView(x, 'decisionPath', { nFeatures: tree.nFeatures, allowNaN: true })
  return Array.from({ length: m.n }, (_, i) => {
    const path = [0]
    let node = 0
    while (!isLeaf(tree, node)) {
      const v = m.data[i * m.d + (tree.feature[node] ?? 0)] ?? 0
      node = v <= (tree.threshold[node] ?? 0) ? (tree.left[node] ?? 0) : (tree.right[node] ?? 0)
      path.push(node)
    }
    return path
  })
}

/**
 * Per-row predictions of a tree: class probabilities `[m, k]` (leaf class proportions) for classification, means `[m]`
 * for regression.
 *
 * @param tree - The tree.
 * @param x - Features `[m, d]`.
 * @returns The predictions.
 */
export function treeOutputs(tree: TreeParams, x: TensorLike): Tensor {
  const leaves = applyTree(tree, x)
  if (tree.task === 'regression')
    return fromStorage(
      Float64Array.from(leaves, (l) => tree.value[l]?.[0] ?? 0),
      [leaves.length],
      'float64',
    )
  const k = tree.classes.length
  const out = new Float64Array(leaves.length * k)
  leaves.forEach((l, i) => {
    const v = tree.value[l] ?? []
    const total = v.reduce((a, b) => a + b, 0) || 1
    for (let c = 0; c < k; c++) out[i * k + c] = (v[c] ?? 0) / total
  })
  return fromStorage(out, [leaves.length, k], 'float64')
}

/**
 * Predicted labels or values of a tree.
 *
 * @param tree - The tree.
 * @param x - Features `[m, d]`.
 * @returns Class labels (the leaf's majority) or means, `[m]`.
 */
export function treePredict(tree: TreeParams, x: TensorLike): Tensor {
  if (tree.task === 'regression') return treeOutputs(tree, x)
  const leaves = applyTree(tree, x)
  return fromStorage(
    Float64Array.from(leaves, (l) => tree.classes[argmaxOf(tree.value[l] ?? [])] ?? 0),
    [leaves.length],
    'float64',
  )
}

/** The depth of a tree (0 for a single leaf). */
export function treeDepth(tree: TreeParams): number {
  const depth = (node: number): number =>
    isLeaf(tree, node) ? 0 : 1 + Math.max(depth(tree.left[node] ?? 0), depth(tree.right[node] ?? 0))
  return depth(0)
}

/** The number of leaves of a tree. */
export function nLeaves(tree: TreeParams): number {
  return tree.left.filter((l) => l < 0).length
}

const fmt = (v: number): string => (Number.isInteger(v) ? String(v) : v.toFixed(3))

/**
 * A tree as a structure for mdml-render: internal nodes labelled with their test, leaves with their prediction,
 * edges labelled true and false.
 *
 * @param tree - The tree.
 * @param features - Feature names (default `x0, x1, ...`).
 * @param classNames - Class names (default the labels).
 * @returns The structure.
 */
export function treeToStructure(
  tree: TreeParams,
  features: readonly string[] = [],
  classNames: readonly string[] = [],
): Structure {
  const name = (j: number): string => features[j] ?? `x${j}`
  const nodes = tree.feature.map((f, id) => {
    const leaf = isLeaf(tree, id)
    const value = tree.value[id] ?? []
    const prediction =
      tree.task === 'regression'
        ? fmt(value[0] ?? 0)
        : (classNames[argmaxOf(value)] ?? String(tree.classes[argmaxOf(value)] ?? ''))
    return {
      id: String(id),
      label: leaf ? prediction : `${name(f)} <= ${fmt(tree.threshold[id] ?? 0)}`,
      role: leaf ? ('leaf' as const) : ('internal' as const),
      detail: `samples = ${tree.nSamples[id] ?? 0}, impurity = ${fmt(tree.impurity[id] ?? 0)}`,
      data: { value, impurity: tree.impurity[id] ?? 0, samples: tree.nSamples[id] ?? 0 },
    }
  })
  const edges = tree.left.flatMap((l, id) =>
    l < 0
      ? []
      : [
          { from: String(id), to: String(l), label: 'true', directed: true },
          { from: String(id), to: String(tree.right[id] ?? 0), label: 'false', directed: true },
        ],
  )
  return structure('decision-tree', nodes, edges, { direction: 'down' })
}

/**
 * A tree as indented text (scikit-learn's `export_text`).
 *
 * @param tree - The tree.
 * @param features - Feature names (default `x0, x1, ...`).
 * @returns The text, one line per test or leaf.
 */
export function exportText(tree: TreeParams, features: readonly string[] = []): string {
  const lines: string[] = []
  const name = (j: number): string => features[j] ?? `x${j}`
  const leafText = (id: number): string => {
    const v = tree.value[id] ?? []
    return tree.task === 'regression' ? `value: [${fmt(v[0] ?? 0)}]` : `class: ${tree.classes[argmaxOf(v)] ?? ''}`
  }
  const walk = (id: number, depth: number): void => {
    const pad = `${'|   '.repeat(depth)}|--- `
    if (isLeaf(tree, id)) {
      lines.push(`${pad}${leafText(id)}`)
      return
    }
    const f = name(tree.feature[id] ?? 0)
    const t = (tree.threshold[id] ?? 0).toFixed(2)
    lines.push(`${pad}${f} <= ${t}`)
    walk(tree.left[id] ?? 0, depth + 1)
    lines.push(`${pad}${f} >  ${t}`)
    walk(tree.right[id] ?? 0, depth + 1)
  }
  walk(0, 0)
  return lines.join('\n')
}
