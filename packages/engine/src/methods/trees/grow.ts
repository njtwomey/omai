// TODO: ID3, C4.5 (gain ratio, multiway splits) and CHAID for comparison with CART (classic.ts).
import { WhiteboxError } from '../../lib/base/index.ts'
import { categorical, type Distribution, normal } from '../../lib/dists/index.ts'
import { type Exposition, type SplitCandidate, type TreeNode, treeRecord } from '../../lib/exposition/index.ts'
import {
  classNamesOf,
  decisionSurfaceOf,
  defineExplainedEstimator,
  type ExplainedEstimator,
  ExplainedEstimatorClass,
  encodeLabels,
  type FitOptions,
  type Fitted,
  featureNamesOf,
  type LearnData,
  type MatrixView,
  matrixView,
  sampleWeightsOf,
  targetVector,
} from '../../lib/learn/index.ts'
import { foldIn, type Key, permutation } from '../../lib/random/index.ts'
import type { Structure } from '../../lib/scene/index.ts'
import { choice, dict, int, real, type Space } from '../../lib/space/index.ts'
import { fromStorage, type Tensor, type TensorLike, toFlat } from '../../lib/tensor/index.ts'
import { type Algorithm, channels, runAlgorithm } from '../../lib/trace/index.ts'
import {
  type ClassificationCriterion,
  type Criterion,
  impurityOf,
  nodeValue,
  type RegressionCriterion,
  type SplitProblem,
  type SplitSearch,
  searchSplits,
} from './criteria.ts'
import { costComplexityPruningPath, featureImportancesOf, pruneTree } from './prune.ts'
import {
  applyTree,
  decisionPath,
  exportText,
  type TreeParams,
  treeOutputs,
  treePredict,
  treeToStructure,
} from './tree.ts'

/** How many features a split may look at: all, a count, a fraction, or sqrt / log2 of the number of features. */
export type MaxFeatures = number | 'sqrt' | 'log2' | 'all'

/** The growth options shared by every tree (scikit-learn's DecisionTree parameters). */
export interface GrowOptions {
  readonly criterion: Criterion
  readonly splitter: 'best' | 'random'
  readonly maxDepth?: number
  readonly minSamplesSplit: number
  readonly minSamplesLeaf: number
  readonly minWeightFractionLeaf: number
  readonly maxFeatures: MaxFeatures
  /** Grow best-first up to this many leaves (depth-first when absent). */
  readonly maxLeafNodes?: number
  readonly minImpurityDecrease: number
  /** Candidates kept per feature in the tree record (default all). */
  readonly record?: number
}

/** A node waiting to be grown, or grown. */
interface Grown extends TreeNode {
  readonly isLeft: boolean
  readonly weight: number
  readonly search?: SplitSearch
}

/** The state of {@link growTreeAlgorithm}: the nodes so far and the frontier. */
export interface GrowState {
  readonly nodes: readonly Grown[]
  /** Depth-first: rows waiting to become nodes. Best-first: ids of leaves that may still be split. */
  readonly stack: readonly {
    readonly rows: Int32Array
    readonly parent: number
    readonly isLeft: boolean
    readonly depth: number
  }[]
  readonly frontier: readonly number[]
}

const EPSILON = 2.220446049250313e-16

function featureCount(m: MaxFeatures, d: number): number {
  if (m === 'all') return d
  if (m === 'sqrt') return Math.max(1, Math.floor(Math.sqrt(d)))
  if (m === 'log2') return Math.max(1, Math.floor(Math.log2(d)))
  if (m > 0 && m < 1) return Math.max(1, Math.floor(m * d))
  return Math.min(d, Math.max(1, Math.floor(m)))
}

/**
 * Growing a decision tree as a traced algorithm, one node per step: depth-first (scikit-learn's DepthFirstTreeBuilder,
 * node ids in preorder) or best-first when `maxLeafNodes` is set (the leaf with the largest weighted impurity decrease
 * is split next). Each node records its rows, value, impurity, every candidate split evaluated and the chosen one.
 * Records nodes, leaves and the impurity decrease per step.
 *
 * @param problem - Data, criterion and leaf limits.
 * @param options - Growth options.
 * @returns The algorithm.
 */
export function growTreeAlgorithm(problem: SplitProblem, options: GrowOptions): Algorithm<GrowState> {
  const { x } = problem
  const nf = featureCount(options.maxFeatures, x.d)
  const maxDepth = options.maxDepth ?? Number.POSITIVE_INFINITY
  const all = Array.from({ length: x.d }, (_, j) => j)
  const featuresFor = (key: Key): number[] =>
    nf >= x.d && options.splitter === 'best' ? all : Array.from(toFlat(permutation(key, x.d)))
  const leafReasons = (rows: Int32Array, weight: number, depth: number, impurity: number): boolean =>
    depth >= maxDepth ||
    rows.length < options.minSamplesSplit ||
    rows.length < 2 * problem.minSamplesLeaf ||
    weight < 2 * problem.minWeightLeaf ||
    impurity <= EPSILON
  const search = (rows: Int32Array, key: Key): SplitSearch => {
    const order = featuresFor(foldIn(key, 0))
    // Look at nf features, continuing past them until a valid split is found (scikit-learn's rule).
    let result: SplitSearch = { candidates: [] }
    const seen: number[] = []
    for (const j of order) {
      seen.push(j)
      if (seen.length < nf && seen.length < order.length) continue
      result = searchSplits(problem, rows, seen, {
        splitter: options.splitter,
        key: foldIn(key, 1),
        ...(options.record === undefined ? {} : { record: options.record }),
      })
      if (result.best !== undefined || seen.length >= order.length) break
    }
    return result
  }
  const makeNode = (
    id: number,
    rows: Int32Array,
    parent: number,
    isLeft: boolean,
    depth: number,
    order: number,
    key: Key,
  ): Grown => {
    let weight = 0
    for (const r of rows) weight += problem.w[r] ?? 0
    const impurity = impurityOf(problem, rows)
    const value = nodeValue(problem, rows)
    const leaf = leafReasons(rows, weight, depth, impurity)
    const s = leaf ? undefined : search(rows, key)
    const valid = s?.best !== undefined && s.best.decrease + EPSILON >= options.minImpurityDecrease ? s : undefined
    return {
      id,
      parent: parent < 0 ? null : parent,
      depth,
      rows: Array.from(rows),
      value,
      impurity,
      candidates: s?.candidates ?? [],
      order,
      isLeft,
      weight,
      ...(valid === undefined ? {} : { search: valid }),
    }
  }
  const link = (nodes: Grown[], child: Grown): void => {
    if (child.parent === null) return
    const p = nodes[child.parent]
    if (p === undefined || p.search?.best === undefined) return
    const b = p.search.best
    const split = p.split ?? { feature: b.feature, threshold: b.threshold, left: -1, right: -1 }
    nodes[child.parent] = { ...p, split: child.isLeft ? { ...split, left: child.id } : { ...split, right: child.id } }
  }
  const bestFirst = options.maxLeafNodes !== undefined
  const allRows = Int32Array.from({ length: x.n }, (_, i) => i)
  return {
    name: 'growTree',
    config: { criterion: problem.criterion, splitter: options.splitter, bestFirst },
    schema: channels({
      metrics: {
        nodes: { description: 'nodes grown' },
        leaves: { description: 'leaves so far' },
        decrease: { description: 'weighted impurity decrease of the last split' },
      },
    }),
    init: (key) => {
      if (!bestFirst)
        return { nodes: [], stack: [{ rows: allRows, parent: -1, isLeft: false, depth: 0 }], frontier: [] }
      const root = makeNode(0, allRows, -1, false, 0, 0, key)
      return { nodes: [root], stack: [], frontier: root.search === undefined ? [] : [0] }
    },
    step: (s, ctx) => {
      const nodes = [...s.nodes]
      if (!bestFirst) {
        const stack = [...s.stack]
        const next = stack.pop()
        if (next === undefined) return s
        const node = makeNode(nodes.length, next.rows, next.parent, next.isLeft, next.depth, ctx.t, ctx.key)
        nodes.push(node)
        link(nodes, node)
        const b = node.search?.best
        if (b !== undefined) {
          stack.push({ rows: b.rightRows, parent: node.id, isLeft: false, depth: node.depth + 1 })
          stack.push({ rows: b.leftRows, parent: node.id, isLeft: true, depth: node.depth + 1 })
        }
        return { nodes, stack, frontier: [] }
      }
      const frontier = [...s.frontier]
      let pick = 0
      for (let i = 1; i < frontier.length; i++) {
        const a = nodes[frontier[i] ?? 0]?.search?.best?.decrease ?? -1
        const c = nodes[frontier[pick] ?? 0]?.search?.best?.decrease ?? -1
        if (a > c) pick = i
      }
      const id = frontier.splice(pick, 1)[0] ?? 0
      const node = nodes[id] as Grown
      const b = node.search?.best
      if (b === undefined) return { ...s, frontier }
      const left = makeNode(nodes.length, b.leftRows, id, true, node.depth + 1, ctx.t + 1, foldIn(ctx.key, 0))
      nodes.push(left)
      link(nodes, left)
      const right = makeNode(nodes.length, b.rightRows, id, false, node.depth + 1, ctx.t + 1, foldIn(ctx.key, 1))
      nodes.push(right)
      link(nodes, right)
      for (const c of [left, right]) if (c.search !== undefined) frontier.push(c.id)
      return { nodes, stack: [], frontier }
    },
    done: (s) => {
      if (!bestFirst) return s.stack.length === 0 && s.nodes.length > 0 ? 'converged' : undefined
      const leaves = s.nodes.filter((n) => n.split === undefined).length
      return s.frontier.length === 0 || leaves >= (options.maxLeafNodes ?? Number.POSITIVE_INFINITY)
        ? 'converged'
        : undefined
    },
    observe: (s) => {
      const last = s.nodes[s.nodes.length - 1]
      return {
        metrics: {
          nodes: s.nodes.length,
          leaves: s.nodes.filter((n) => n.split === undefined).length,
          decrease: last?.search?.best?.decrease ?? 0,
        },
      }
    },
  }
}

/** A grown tree: its arrays and the nodes of its record. */
export interface GrownTree {
  readonly tree: TreeParams
  readonly nodes: readonly TreeNode[]
}

/** The tree's data: features, targets (class codes or values), weights and classes. */
export interface TreeData {
  readonly x: MatrixView
  readonly y: Float64Array
  readonly w: Float64Array
  /** Class labels (empty for regression). */
  readonly classes: readonly number[]
}

/**
 * Grow a tree on prepared data (the entry point ensembles use), running {@link growTreeAlgorithm}.
 *
 * @param data - Features, targets, weights and classes.
 * @param options - Growth options.
 * @param run - Key and tracing.
 * @returns The tree's arrays and record nodes (leaves of a best-first tree that were never split have no split).
 */
export function growTree(
  data: TreeData,
  options: GrowOptions,
  run: FitOptions & { readonly key?: Key } = {},
): GrownTree {
  const total = data.w.reduce((a, b) => a + b, 0)
  const problem: SplitProblem = {
    x: data.x,
    y: data.y,
    w: data.w,
    k: data.classes.length,
    criterion: options.criterion,
    minSamplesLeaf: options.minSamplesLeaf,
    minWeightLeaf: options.minWeightFractionLeaf * total,
    totalWeight: total,
  }
  const r = runAlgorithm(growTreeAlgorithm(problem, options), {
    maxSteps: 100000,
    ...(run.key === undefined ? {} : { key: run.key }),
    ...(run.parent !== undefined
      ? { parent: run.parent }
      : {
          ...(run.tracer === undefined ? {} : { tracer: run.tracer }),
          ...(run.policy === undefined ? {} : { policy: run.policy }),
        }),
  })
  const nodes: TreeNode[] = r.state.nodes.map(({ isLeft: _l, weight: _w, search: _s, ...node }) => node)
  const tree: TreeParams = {
    task: data.classes.length > 0 ? 'classification' : 'regression',
    classes: [...data.classes],
    nFeatures: data.x.d,
    feature: nodes.map((n) => n.split?.feature ?? -1),
    threshold: nodes.map((n) => n.split?.threshold ?? Number.NaN),
    left: nodes.map((n) => n.split?.left ?? -1),
    right: nodes.map((n) => n.split?.right ?? -1),
    value: nodes.map((n) => [...n.value]),
    impurity: nodes.map((n) => n.impurity),
    nSamples: nodes.map((n) => n.rows.length),
    weightedNSamples: r.state.nodes.map((n) => n.weight),
  }
  return { tree, nodes }
}

/** Hyperparameters of the decision trees. */
export interface DecisionTreeHyper extends GrowOptions {
  /** Minimal cost-complexity pruning (default 0, none). */
  readonly ccpAlpha: number
}

/** Verbs of a fitted tree. */
export interface TreeVerbs {
  readonly predict: (x: TensorLike) => Tensor
  readonly predictProba?: (x: TensorLike) => Tensor
  /** Categorical leaf proportions (classifiers), or normal with the leaf mean and standard deviation (regressors). */
  readonly predictDistribution: (x: TensorLike) => Distribution
  /** The leaf each row reaches. */
  readonly apply: (x: TensorLike) => Int32Array
  readonly decisionPath: (x: TensorLike) => number[][]
  /** The tree as a structure for drawing. */
  readonly describe: (features?: readonly string[], classNames?: readonly string[]) => Structure
  readonly exportText: (features?: readonly string[]) => string
  readonly classes?: number[]
  readonly nFeaturesIn: number
  readonly featureImportances: number[]
}

/** A fitted decision tree. */
export type FittedTree = Fitted<TreeParams> & TreeVerbs

/** The verbs of a tree, from its arrays. */
export function treeVerbs(tree: TreeParams): TreeVerbs {
  const classifier = tree.task === 'classification'
  return {
    predict: (x) => treePredict(tree, x),
    ...(classifier ? { predictProba: (x: TensorLike) => treeOutputs(tree, x), classes: tree.classes } : {}),
    predictDistribution: (x) => {
      if (classifier) return categorical({ probs: treeOutputs(tree, x) })
      const leaves = applyTree(tree, x)
      const loc = Float64Array.from(leaves, (l) => tree.value[l]?.[0] ?? 0)
      const scale = Float64Array.from(leaves, (l) => Math.sqrt(Math.max(tree.impurity[l] ?? 0, 1e-12)))
      return normal(fromStorage(loc, [loc.length], 'float64'), fromStorage(scale, [scale.length], 'float64'))
    },
    apply: (x) => applyTree(tree, x),
    decisionPath: (x) => decisionPath(tree, x),
    describe: (features, classNames) => treeToStructure(tree, features, classNames),
    exportText: (features) => exportText(tree, features),
    nFeaturesIn: tree.nFeatures,
    featureImportances: featureImportancesOf(tree),
  }
}

/** Read a tree's data for a classifier or regressor. */
export function treeData(data: LearnData, classifier: boolean, who: string): TreeData {
  const x = matrixView(data.x, who)
  const w = sampleWeightsOf(data.weights, x.n, who)
  if (!classifier) return { x, y: targetVector(data.y ?? [], who, x.n), w, classes: [] }
  const { classes, codes } = encodeLabels(data.y ?? [], who, x.n)
  return { x, y: Float64Array.from(codes), w, classes }
}

function treeEstimator(
  name: string,
  classifier: boolean,
  hyper: DecisionTreeHyper,
): ExplainedEstimator<DecisionTreeHyper, FittedTree> {
  const est: ExplainedEstimator<DecisionTreeHyper, FittedTree> = defineExplainedEstimator(
    {
      name,
      kind: classifier ? 'classifier' : 'regressor',
      hyperSpace: (h: DecisionTreeHyper): Space =>
        dict({
          criterion: choice(
            classifier ? ['gini', 'entropy'] : ['squared_error', 'absolute_error', 'poisson', 'friedman_mse'],
            { label: 'criterion', description: 'The impurity a split minimises.', default: h.criterion },
          ),
          maxDepth: int(1, 20, { default: h.maxDepth ?? 5, label: 'maximum depth' }),
          minSamplesLeaf: int(1, 50, { default: h.minSamplesLeaf, label: 'minimum leaf size' }),
          ccpAlpha: real(0, 0.1, { default: h.ccpAlpha, label: 'pruning alpha' }),
        }),
      fit: (data: LearnData, h: DecisionTreeHyper, options: FitOptions) => {
        const prepared = treeData(data, classifier, name)
        if (classifier !== ['gini', 'entropy', 'log_loss'].includes(h.criterion))
          throw new WhiteboxError('value', `${name}: criterion ${h.criterion} does not suit this task`)
        const grown = growTree(prepared, h, options)
        const tree = pruneTree(grown.tree, h.ccpAlpha)
        const features = featureNamesOf(data, prepared.x.d)
        const classNames = classifier ? classNamesOf(data, prepared.classes) : undefined
        const exposition: Exposition[] = [
          treeRecord({
            criterion: h.criterion,
            features,
            ...(classNames === undefined ? {} : { classes: classNames }),
            nodes: grown.nodes,
          }),
          costComplexityPruningPath(grown.tree),
        ]
        if (prepared.x.d === 2)
          exposition.push(
            decisionSurfaceOf(est.revive(tree), data.x, {
              featureNames: features,
              ...(classNames === undefined ? {} : { classNames }),
            }),
          )
        return { params: tree, exposition }
      },
      methods: (p: TreeParams) => treeVerbs(p),
    },
    hyper,
  )
  return est
}

function treeDefaults(hyper: Partial<DecisionTreeHyper>, criterion: Criterion): DecisionTreeHyper {
  return {
    criterion: hyper.criterion ?? criterion,
    splitter: hyper.splitter ?? 'best',
    ...(hyper.maxDepth === undefined ? {} : { maxDepth: hyper.maxDepth }),
    minSamplesSplit: hyper.minSamplesSplit ?? 2,
    minSamplesLeaf: hyper.minSamplesLeaf ?? 1,
    minWeightFractionLeaf: hyper.minWeightFractionLeaf ?? 0,
    maxFeatures: hyper.maxFeatures ?? 'all',
    ...(hyper.maxLeafNodes === undefined ? {} : { maxLeafNodes: hyper.maxLeafNodes }),
    minImpurityDecrease: hyper.minImpurityDecrease ?? 0,
    ccpAlpha: hyper.ccpAlpha ?? 0,
    ...(hyper.record === undefined ? {} : { record: hyper.record }),
  }
}

/**
 * A CART classification tree (sklearn.tree.DecisionTreeClassifier), grown by a traced algorithm one node per step.
 * Exposition: the tree record (every node's rows, class counts, impurity, every candidate split evaluated and the
 * chosen one, in growth order), the cost-complexity pruning path and, for two features, the decision surface.
 *
 * @param hyper - Criterion (default gini), splitter, depth and leaf limits, feature subsampling, best-first growth,
 * minimum decrease, pruning alpha and how many candidates per feature to record.
 * @returns The estimator.
 */
export function decisionTreeClassifier(
  hyper: Partial<DecisionTreeHyper & { readonly criterion: ClassificationCriterion }> = {},
): ExplainedEstimator<DecisionTreeHyper, FittedTree> {
  return treeEstimator('decision-tree-classifier', true, treeDefaults(hyper, 'gini'))
}

/**
 * A CART regression tree (sklearn.tree.DecisionTreeRegressor); exposition as {@link decisionTreeClassifier}, with
 * leaf means as values. `predictDistribution` is normal with each leaf's mean and standard deviation.
 *
 * @param hyper - Criterion (default squared error) and the growth options of {@link decisionTreeClassifier}.
 * @returns The estimator.
 */
export function decisionTreeRegressor(
  hyper: Partial<DecisionTreeHyper & { readonly criterion: RegressionCriterion }> = {},
): ExplainedEstimator<DecisionTreeHyper, FittedTree> {
  return treeEstimator('decision-tree-regressor', false, treeDefaults(hyper, 'squared_error'))
}

/**
 * An extremely randomised classification tree (sklearn.tree.ExtraTreeClassifier): one random threshold per feature,
 * sqrt(d) features per split.
 *
 * @param hyper - As {@link decisionTreeClassifier} (splitter random, maxFeatures sqrt by default).
 * @returns The estimator.
 */
export function extraTreeClassifier(
  hyper: Partial<DecisionTreeHyper> = {},
): ExplainedEstimator<DecisionTreeHyper, FittedTree> {
  return treeEstimator(
    'extra-tree-classifier',
    true,
    treeDefaults({ splitter: 'random', maxFeatures: 'sqrt', ...hyper }, 'gini'),
  )
}

/**
 * An extremely randomised regression tree (sklearn.tree.ExtraTreeRegressor): one random threshold per feature, all
 * features per split.
 *
 * @param hyper - As {@link decisionTreeRegressor} (splitter random by default).
 * @returns The estimator.
 */
export function extraTreeRegressor(
  hyper: Partial<DecisionTreeHyper> = {},
): ExplainedEstimator<DecisionTreeHyper, FittedTree> {
  return treeEstimator('extra-tree-regressor', false, treeDefaults({ splitter: 'random', ...hyper }, 'squared_error'))
}

/** {@link decisionTreeClassifier} as a class. */
export class DecisionTreeClassifier extends ExplainedEstimatorClass<DecisionTreeHyper, FittedTree> {
  /** @param hyper - As {@link decisionTreeClassifier}. */
  constructor(hyper: Partial<DecisionTreeHyper & { readonly criterion: ClassificationCriterion }> = {}) {
    super(decisionTreeClassifier(hyper))
  }
}

/** {@link decisionTreeRegressor} as a class. */
export class DecisionTreeRegressor extends ExplainedEstimatorClass<DecisionTreeHyper, FittedTree> {
  /** @param hyper - As {@link decisionTreeRegressor}. */
  constructor(hyper: Partial<DecisionTreeHyper & { readonly criterion: RegressionCriterion }> = {}) {
    super(decisionTreeRegressor(hyper))
  }
}

/** The candidate splits of a record node, best first (for views that show the top few). */
export function bestCandidates(node: TreeNode, k = 5): SplitCandidate[] {
  return [...node.candidates].sort((a, b) => b.decrease - a.decrease).slice(0, k)
}
