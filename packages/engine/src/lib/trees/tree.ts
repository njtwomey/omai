/**
 * A fitted decision tree as plain arrays in the layout of scikit-learn's `tree_` (split feature, threshold, children,
 * node values, impurities and sample counts), and what can be done with one: the leaf and the decision path of each
 * row, predictions, depth and leaf counts, the region of the feature space each node covers, impurity importances, a
 * structure for drawing and indented text. Splits may be numeric (`x <= threshold`) or categorical (a set of codes
 * going left), and each split says where a missing value goes. The scalar-valued `TreeModel` that tree SHAP and rule
 * extraction read (moved here from `explain`) is a view of a tree, made by `treeModel`.
 */

import { argmaxOf, MdmlError } from '../base/index.ts'
import { type MatrixView, matrixView } from '../learn/validation.ts'
import { type Structure, structure } from '../scene/index.ts'
import { fromStorage, type Tensor, type TensorLike } from '../tensor/index.ts'

/**
 * A fitted decision tree as plain arrays (scikit-learn's `tree_`), one entry per node in growth order. Internal nodes
 * test `feature[i]`: a numeric split sends rows with `x <= threshold[i]` left, a categorical split (`categories[i]`
 * not empty) sends rows whose code is listed left, and a missing value goes left when `missingLeft[i]`. Leaves have
 * `left[i] = -1`, `feature[i] = -2` and `threshold[i] = -2`. `value` holds the weighted class counts of a classifier
 * and the (weighted) mean target, in a one-entry list, of a regressor.
 */
export type FittedTree = {
  readonly task: 'classification' | 'regression'
  /** Class labels of a classifier (empty for regression). */
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
  /** Whether a missing value at the node's split goes to the left child. */
  readonly missingLeft: boolean[]
  /** The category codes sent left by a categorical split (empty for numeric splits and leaves). */
  readonly categories: number[][]
}

/**
 * A binary decision tree as arrays with one scalar per node (sklearn's `tree_`, as tree SHAP and rule extraction read
 * it): node i splits on `feature[i]` at `threshold[i]` (going left when x <= threshold), leaves have
 * `childrenLeft[i] = -1`; `value[i]` is the node's prediction and `cover[i]` its (weighted) number of training samples.
 */
export interface TreeModel {
  readonly childrenLeft: readonly number[]
  readonly childrenRight: readonly number[]
  readonly feature: readonly number[]
  readonly threshold: readonly number[]
  readonly value: readonly number[]
  readonly cover: readonly number[]
}

/**
 * The prediction of a scalar-valued tree for each row.
 *
 * @param tree - The tree.
 * @param x - Rows `[n, d]`.
 * @returns Predictions `[n]`.
 * @example
 * ```ts
 * import { treePredict } from 'mdml-engine/trees'
 * import { toFlat } from 'mdml-engine/tensor'
 * const stump = { childrenLeft: [1, -1, -1], childrenRight: [2, -1, -1], feature: [0, -2, -2], threshold: [0.5, 0, 0], value: [0.5, 0, 1], cover: [2, 1, 1] }
 * toFlat(treePredict(stump, [[0], [1]])) // => [0, 1]
 * ```
 */
export function treePredict(tree: TreeModel, x: TensorLike): Tensor {
  const m = matrixView(x, 'treePredict')
  return fromStorage(
    Float64Array.from({ length: m.n }, (_, i) => tree.value[leafOfModel(tree, m, i)] ?? 0),
    [m.n],
    'float64',
  )
}

/**
 * The leaf of a scalar-valued tree that a row of a matrix reaches.
 *
 * @param tree - The tree.
 * @param m - The matrix view.
 * @param i - The row.
 * @returns The leaf's node id.
 * @example
 * ```ts
 * import { leafOfModel } from 'mdml-engine/trees'
 * import { matrixView } from 'mdml-engine/learn'
 * const stump = { childrenLeft: [1, -1, -1], childrenRight: [2, -1, -1], feature: [0, -2, -2], threshold: [0.5, 0, 0], value: [0.5, 0, 1], cover: [2, 1, 1] }
 * leafOfModel(stump, matrixView([[0], [1]], 'demo'), 1) // => 2
 * ```
 */
export function leafOfModel(tree: TreeModel, m: MatrixView, i: number): number {
  let node = 0
  for (let guard = 0; (tree.childrenLeft[node] ?? -1) >= 0 && guard < tree.value.length; guard++) {
    const f = tree.feature[node] ?? 0
    node =
      (m.data[i * m.d + f] ?? 0) <= (tree.threshold[node] ?? 0)
        ? (tree.childrenLeft[node] ?? 0)
        : (tree.childrenRight[node] ?? 0)
  }
  return node
}

/**
 * Whether a node is a leaf.
 *
 * @param tree - The tree.
 * @param node - The node id.
 * @returns `true` for leaves.
 * @example
 * ```ts
 * import { isLeaf, stumpTree } from 'mdml-engine/trees'
 * const t = stumpTree({ feature: 0, threshold: 0.5, left: 0, right: 1 })
 * isLeaf(t, 1) // => true
 * ```
 */
export function isLeaf(tree: FittedTree, node: number): boolean {
  return (tree.left[node] ?? -1) < 0
}

/**
 * A one-split regression tree, for tests and examples: the left leaf predicts `left`, the right leaf `right`.
 *
 * @param split - The feature, the threshold and the two leaf values.
 * @returns The tree over `feature + 1` features.
 * @example
 * ```ts
 * import { predictTree, stumpTree } from 'mdml-engine/trees'
 * import { toFlat } from 'mdml-engine/tensor'
 * toFlat(predictTree(stumpTree({ feature: 0, threshold: 0.5, left: 0, right: 1 }), [[0], [1]])) // => [0, 1]
 * ```
 */
export function stumpTree(split: {
  readonly feature: number
  readonly threshold: number
  readonly left: number
  readonly right: number
}): FittedTree {
  return {
    task: 'regression',
    classes: [],
    nFeatures: split.feature + 1,
    feature: [split.feature, -2, -2],
    threshold: [split.threshold, -2, -2],
    left: [1, -1, -1],
    right: [2, -1, -1],
    value: [[(split.left + split.right) / 2], [split.left], [split.right]],
    impurity: [0, 0, 0],
    nSamples: [2, 1, 1],
    weightedNSamples: [2, 1, 1],
    missingLeft: [false, false, false],
    categories: [[], [], []],
  }
}

function goesLeft(tree: FittedTree, node: number, v: number): boolean {
  if (Number.isNaN(v)) return tree.missingLeft[node] === true
  const cats = tree.categories[node] ?? []
  if (cats.length > 0) return cats.includes(v)
  return v <= (tree.threshold[node] ?? 0)
}

/**
 * The leaf each row reaches (scikit-learn's `apply`). Missing values (NaN) follow the split's `missingLeft`.
 *
 * @param tree - The tree.
 * @param x - Features `[m, d]`.
 * @returns Leaf ids, one per row.
 * @example
 * ```ts
 * import { applyTree, stumpTree } from 'mdml-engine/trees'
 * Array.from(applyTree(stumpTree({ feature: 0, threshold: 0.5, left: 0, right: 1 }), [[0], [1], [0.2]])) // => [1, 2, 1]
 * ```
 */
export function applyTree(tree: FittedTree, x: TensorLike): Int32Array {
  const m = matrixView(x, 'applyTree', { nFeatures: tree.nFeatures, allowNaN: true })
  const out = new Int32Array(m.n)
  for (let i = 0; i < m.n; i++) {
    let node = 0
    while (!isLeaf(tree, node)) {
      const v = m.data[i * m.d + (tree.feature[node] ?? 0)] ?? 0
      node = goesLeft(tree, node, v) ? (tree.left[node] ?? 0) : (tree.right[node] ?? 0)
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
 * @example
 * ```ts
 * import { decisionPath, stumpTree } from 'mdml-engine/trees'
 * decisionPath(stumpTree({ feature: 0, threshold: 0.5, left: 0, right: 1 }), [[0], [1]]) // => [[0, 1], [0, 2]]
 * ```
 */
export function decisionPath(tree: FittedTree, x: TensorLike): number[][] {
  const m = matrixView(x, 'decisionPath', { nFeatures: tree.nFeatures, allowNaN: true })
  return Array.from({ length: m.n }, (_, i) => {
    const path = [0]
    let node = 0
    while (!isLeaf(tree, node)) {
      const v = m.data[i * m.d + (tree.feature[node] ?? 0)] ?? 0
      node = goesLeft(tree, node, v) ? (tree.left[node] ?? 0) : (tree.right[node] ?? 0)
      path.push(node)
    }
    return path
  })
}

/**
 * Per-row outputs of a tree: class probabilities `[m, k]` (the leaf's class proportions) for a classifier, the leaf
 * means `[m]` for a regressor.
 *
 * @param tree - The tree.
 * @param x - Features `[m, d]`.
 * @returns The outputs.
 * @example
 * ```ts
 * import { stumpTree, treeOutputs } from 'mdml-engine/trees'
 * import { toFlat } from 'mdml-engine/tensor'
 * toFlat(treeOutputs(stumpTree({ feature: 0, threshold: 0.5, left: 3, right: 5 }), [[0], [1]])) // => [3, 5]
 * ```
 */
export function treeOutputs(tree: FittedTree, x: TensorLike): Tensor {
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
 * Predicted labels (the leaf's majority class) or values (the leaf's mean) of a tree.
 *
 * @param tree - The tree.
 * @param x - Features `[m, d]`.
 * @returns Class labels or means, `[m]`.
 * @example
 * ```ts
 * import { predictTree, stumpTree } from 'mdml-engine/trees'
 * import { toFlat } from 'mdml-engine/tensor'
 * toFlat(predictTree(stumpTree({ feature: 0, threshold: 0.5, left: 1, right: 2 }), [[0.9]])) // => [2]
 * ```
 */
export function predictTree(tree: FittedTree, x: TensorLike): Tensor {
  if (tree.task === 'regression') return treeOutputs(tree, x)
  const leaves = applyTree(tree, x)
  return fromStorage(
    Float64Array.from(leaves, (l) => tree.classes[argmaxOf(tree.value[l] ?? [])] ?? 0),
    [leaves.length],
    'float64',
  )
}

/**
 * The depth of a tree (0 for a single leaf).
 *
 * @param tree - The tree.
 * @returns The longest root-to-leaf path in edges.
 * @example
 * ```ts
 * import { stumpTree, treeDepth } from 'mdml-engine/trees'
 * treeDepth(stumpTree({ feature: 0, threshold: 0, left: 0, right: 1 })) // => 1
 * ```
 */
export function treeDepth(tree: FittedTree): number {
  const depth = (node: number): number =>
    isLeaf(tree, node) ? 0 : 1 + Math.max(depth(tree.left[node] ?? 0), depth(tree.right[node] ?? 0))
  return depth(0)
}

/**
 * The number of leaves of a tree.
 *
 * @param tree - The tree.
 * @returns The leaf count.
 * @example
 * ```ts
 * import { leafCount, stumpTree } from 'mdml-engine/trees'
 * leafCount(stumpTree({ feature: 0, threshold: 0, left: 0, right: 1 })) // => 2
 * ```
 */
export function leafCount(tree: FittedTree): number {
  return tree.left.filter((l) => l < 0).length
}

/** The region of feature space a node covers: per feature, the open lower bound and the closed upper bound. */
export interface NodeRegion {
  readonly node: number
  /** `lower[j] < x[j] <= upper[j]` (infinite where unbounded). Categorical splits do not bound a region. */
  readonly lower: readonly number[]
  readonly upper: readonly number[]
}

/**
 * The box of feature space each node covers: the root is everything, and each numeric split cuts its parent's box at
 * the threshold. Leaves' boxes tile the space, which is how a tree's decision regions are drawn.
 *
 * @param tree - The tree.
 * @returns One region per node, by node id.
 * @example
 * ```ts
 * import { nodeRegions, stumpTree } from 'mdml-engine/trees'
 * const r = nodeRegions(stumpTree({ feature: 0, threshold: 0.5, left: 0, right: 1 }))
 * r[1]?.upper[0] // => 0.5
 * ```
 */
export function nodeRegions(tree: FittedTree): NodeRegion[] {
  const d = tree.nFeatures
  const out: NodeRegion[] = new Array<NodeRegion>(tree.feature.length)
  out[0] = {
    node: 0,
    lower: new Array<number>(d).fill(Number.NEGATIVE_INFINITY),
    upper: new Array<number>(d).fill(Number.POSITIVE_INFINITY),
  }
  const stack = [0]
  while (stack.length > 0) {
    const node = stack.pop() ?? 0
    const region = out[node] as NodeRegion
    if (isLeaf(tree, node)) continue
    const f = tree.feature[node] ?? 0
    const numeric = (tree.categories[node] ?? []).length === 0
    const l = tree.left[node] ?? 0
    const r = tree.right[node] ?? 0
    const t = tree.threshold[node] ?? 0
    out[l] = {
      node: l,
      lower: region.lower,
      upper: numeric ? region.upper.map((u, j) => (j === f ? Math.min(u, t) : u)) : region.upper,
    }
    out[r] = {
      node: r,
      lower: numeric ? region.lower.map((u, j) => (j === f ? Math.max(u, t) : u)) : region.lower,
      upper: region.upper,
    }
    stack.push(l, r)
  }
  return out
}

/**
 * Impurity-based feature importances (mean decrease in impurity), normalised to sum to one (scikit-learn's
 * `feature_importances_`).
 *
 * @param tree - The tree.
 * @returns One importance per feature (all zero for a single leaf).
 * @example
 * ```ts
 * import { stumpTree, treeFeatureImportances } from 'mdml-engine/trees'
 * treeFeatureImportances({ ...stumpTree({ feature: 1, threshold: 0, left: 0, right: 1 }), impurity: [0.5, 0, 0] }) // => [0, 1]
 * ```
 */
export function treeFeatureImportances(tree: FittedTree): number[] {
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

/**
 * A tree as scalar-valued arrays for tree SHAP and rule extraction: a regressor's leaf means, a classifier's
 * probability of one class (or, with `output: 'label'`, the majority class label), with the node cover.
 *
 * @param tree - The tree.
 * @param output - A class index (default the last class, the positive class of a binary problem), or `'label'`.
 * @returns The scalar-valued tree. Categorical splits are not representable and are refused.
 * @throws MdmlError with code `value` when the tree has categorical splits.
 * @example
 * ```ts
 * import { stumpTree, treeModel } from 'mdml-engine/trees'
 * treeModel(stumpTree({ feature: 0, threshold: 0.5, left: 0, right: 1 })).value // => [0.5, 0, 1]
 * ```
 */
export function treeModel(tree: FittedTree, output?: number | 'label'): TreeModel {
  if (tree.categories.some((c) => c.length > 0))
    throw new MdmlError('value', 'treeModel: categorical splits have no scalar-valued form')
  const k = tree.classes.length
  const value = tree.value.map((v) => {
    if (tree.task === 'regression') return v[0] ?? 0
    if (output === 'label') return tree.classes[argmaxOf(v)] ?? 0
    const total = v.reduce((a, b) => a + b, 0) || 1
    return (v[output ?? k - 1] ?? 0) / total
  })
  return {
    childrenLeft: tree.left,
    childrenRight: tree.right,
    feature: tree.feature,
    threshold: tree.threshold,
    value,
    cover: tree.weightedNSamples,
  }
}

const fmt = (v: number): string => (Number.isInteger(v) ? String(v) : v.toFixed(3))

function testText(tree: FittedTree, node: number, name: (j: number) => string): string {
  const f = name(tree.feature[node] ?? 0)
  const cats = tree.categories[node] ?? []
  return cats.length > 0 ? `${f} in {${cats.join(', ')}}` : `${f} <= ${fmt(tree.threshold[node] ?? 0)}`
}

/**
 * A tree as a structure for drawing: internal nodes labelled with their test, leaves with their prediction, edges
 * labelled true and false.
 *
 * @param tree - The tree.
 * @param features - Feature names (default `x0, x1, ...`).
 * @param classNames - Class names (default the labels).
 * @returns The structure.
 * @example
 * ```ts
 * import { stumpTree, treeToStructure } from 'mdml-engine/trees'
 * treeToStructure(stumpTree({ feature: 0, threshold: 0.5, left: 0, right: 1 })).nodes.map((n) => n.label) // => ['x0 <= 0.500', '0', '1']
 * ```
 */
export function treeToStructure(
  tree: FittedTree,
  features: readonly string[] = [],
  classNames: readonly string[] = [],
): Structure {
  const name = (j: number): string => features[j] ?? `x${j}`
  const nodes = tree.feature.map((_, id) => {
    const leaf = isLeaf(tree, id)
    const value = tree.value[id] ?? []
    const prediction =
      tree.task === 'regression'
        ? fmt(value[0] ?? 0)
        : (classNames[argmaxOf(value)] ?? String(tree.classes[argmaxOf(value)] ?? ''))
    return {
      id: String(id),
      label: leaf ? prediction : testText(tree, id, name),
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
 * @example
 * ```ts
 * import { exportText, stumpTree } from 'mdml-engine/trees'
 * exportText(stumpTree({ feature: 0, threshold: 0.5, left: 0, right: 1 })).split('\n')[0] // => '|--- x0 <= 0.50'
 * ```
 */
export function exportText(tree: FittedTree, features: readonly string[] = []): string {
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
    const cats = tree.categories[id] ?? []
    const f = name(tree.feature[id] ?? 0)
    const t = (tree.threshold[id] ?? 0).toFixed(2)
    lines.push(cats.length > 0 ? `${pad}${f} in {${cats.join(', ')}}` : `${pad}${f} <= ${t}`)
    walk(tree.left[id] ?? 0, depth + 1)
    lines.push(cats.length > 0 ? `${pad}${f} not in {${cats.join(', ')}}` : `${pad}${f} >  ${t}`)
    walk(tree.right[id] ?? 0, depth + 1)
  }
  walk(0, 0)
  return lines.join('\n')
}
