/**
 * Growing a decision tree as a stepped `Algorithm`: recursive binary splitting, one node decided per step, depth-first
 * (scikit-learn's DepthFirstTreeBuilder, node ids in preorder), breadth-first or best-first (the open leaf with the
 * largest weighted impurity decrease next, under a leaf limit, as scikit-learn's BestFirstTreeBuilder). Every node
 * keeps the rows that reach it and every candidate split evaluated there, so a view can show how the choice was made.
 * Limits: depth, samples to split, samples and weight per leaf, leaf count and minimum impurity decrease; each split may
 * look at a random subset of the features (for forests), at categorical features by subsets of their codes and at
 * missing values. `growTree` runs it to completion; `treeOfState` and `treeRecordOf` turn a state, even a partial one,
 * into a `Tree` and an exposition record.
 */

import { MdmlError } from '../base/index.ts'
import { type SplitCandidate, type TreeNode, type TreeRecord, treeRecord } from '../exposition/index.ts'
import type { FitOptions, LearnData } from '../learn/estimator.ts'
import { encodeLabels, type MatrixView, matrixView, sampleWeightsOf, targetVector } from '../learn/validation.ts'
import { foldIn, type Key, permutation } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { type Algorithm, runAlgorithm, stateFromJson, stateToJson } from '../trace/index.ts'
import {
  type Criterion,
  impurityOf,
  isClassificationCriterion,
  nodeValue,
  type SplitProblem,
  type SplitSearch,
  searchSplits,
} from './criteria.ts'
import type { Tree } from './tree.ts'

/** How many features a split may look at: all, a count, a fraction, or sqrt / log2 of the number of features. */
export type MaxFeatures = number | 'sqrt' | 'log2' | 'all'

/** The order nodes are grown in: depth-first, breadth-first, or best-first by impurity decrease. */
export type Growth = 'depth' | 'breadth' | 'best'

/** The growth options shared by every tree (scikit-learn's DecisionTree parameters). */
export interface GrowOptions {
  readonly criterion: Criterion
  readonly splitter: 'best' | 'random'
  /** The order of growth (default depth-first; always best-first when `maxLeafNodes` is set). */
  readonly growth?: Growth
  readonly maxDepth?: number
  readonly minSamplesSplit: number
  readonly minSamplesLeaf: number
  readonly minWeightFractionLeaf: number
  readonly maxFeatures: MaxFeatures
  /** Stop at this many leaves (grows best-first). */
  readonly maxLeafNodes?: number
  readonly minImpurityDecrease: number
  /** Candidates kept per feature in the record: all (default), the best few, or none (0). */
  readonly record?: number
}

/** The split a node decided on. */
export interface DecidedSplit {
  readonly feature: number
  readonly threshold: number
  readonly decrease: number
  readonly categories?: readonly number[]
  readonly missingLeft: boolean
}

/** A node of a tree in growth: born with its rows and statistics, searched when popped (or at birth, best-first). */
export interface GrownNode {
  /** The birth id (the final id is its rank in growth order, see {@link treeOfState}). */
  readonly id: number
  readonly parent: number
  readonly depth: number
  readonly isLeft: boolean
  readonly rows: Int32Array
  readonly weight: number
  readonly value: readonly number[]
  readonly impurity: number
  /** Every candidate split evaluated here (once the node has been searched). */
  readonly candidates: readonly SplitCandidate[]
  /** The split chosen, once the node has been searched and is to be split. */
  readonly split?: DecidedSplit
  /** The rows each way of the split, until the node is popped and its children are born. */
  readonly found?: { readonly left: Int32Array; readonly right: Int32Array }
  /** Child ids, -1 while the node is a leaf. */
  readonly left: number
  readonly right: number
  /** The step at which the node was decided, -1 while it waits. */
  readonly order: number
}

/** The state of {@link growTreeAlgorithm}: the nodes so far and the ones waiting. */
export interface GrowState {
  readonly nodes: readonly GrownNode[]
  /** Ids waiting: a stack (depth-first), a queue (breadth-first) or a pool (best-first). */
  readonly open: readonly number[]
  /** Nodes decided so far. */
  readonly decided: number
}

/**
 * The paths {@link growTreeAlgorithm} logs: the node just decided, its candidate splits, the tree's size and impurity,
 * and the training score (`eval/accuracy` for classification, `eval/r2` for regression).
 *
 * @example
 * ```ts
 * import type { GrowPath } from 'mdml-engine/trees'
 * const path: GrowPath = 'tree/split'
 * path // => 'tree/split'
 * ```
 */
export type GrowPath =
  | 'tree/split'
  | 'tree/candidates'
  | 'tree/leaves'
  | 'tree/depth'
  | 'tree/impurity'
  | 'eval/accuracy'
  | 'eval/r2'

const EPSILON = Number.EPSILON

function featureCount(m: MaxFeatures, d: number): number {
  if (m === 'all') return d
  if (m === 'sqrt') return Math.max(1, Math.floor(Math.sqrt(d)))
  if (m === 'log2') return Math.max(1, Math.floor(Math.log2(d)))
  if (m > 0 && m < 1) return Math.max(1, Math.floor(m * d))
  return Math.min(d, Math.max(1, Math.floor(m)))
}

/** What a tree needs to know about its data to be assembled from a state. */
export interface TreeInfo {
  readonly task: 'classification' | 'regression'
  readonly classes: readonly number[]
  readonly nFeatures: number
}

function withoutFound(node: GrownNode): GrownNode {
  const { found: _found, ...rest } = node
  return rest
}

/**
 * The growth of a tree as an algorithm. Each step decides one node: depth-first and breadth-first pop the next waiting
 * node and search its splits (its children are born with their rows and statistics); best-first has searched every
 * node at birth and pops the one whose split decreases the impurity most. Logs, when subscribed: `tree/split` (a
 * message naming the node, its feature, threshold and decrease), `tree/candidates` (every candidate split of the node
 * as rows of feature, threshold, left and right impurity, decrease), `tree/leaves`, `tree/depth`, `tree/impurity` (the
 * weighted impurity of the leaves), and `eval/accuracy` (classification) or `eval/r2` on the training rows.
 *
 * @param problem - Data, criterion and leaf limits.
 * @param options - Growth options.
 * @param info - Task, classes and feature count (for the logs that need the leaves' predictions).
 * @returns The algorithm.
 * @concept decision-trees
 * @example
 * ```ts
 * import { growTreeAlgorithm, treeOfState } from 'mdml-engine/trees'
 * import { matrixView } from 'mdml-engine/learn'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const problem = { x: matrixView([[0], [1], [2], [3]], 'demo'), y: Float64Array.of(0, 0, 1, 1), w: Float64Array.of(1, 1, 1, 1), k: 2, criterion: 'gini' as const, minSamplesLeaf: 1, minWeightLeaf: 0, totalWeight: 4 }
 * const options = { criterion: 'gini' as const, splitter: 'best' as const, minSamplesSplit: 2, minSamplesLeaf: 1, minWeightFractionLeaf: 0, maxFeatures: 'all' as const, minImpurityDecrease: 0 }
 * const info = { task: 'classification' as const, classes: [0, 1], nFeatures: 1 }
 * const run = runAlgorithm(growTreeAlgorithm(problem, options, info), { maxSteps: 100 })
 * treeOfState(run.state, info).threshold // => [1.5, -2, -2]
 * ```
 */
export function growTreeAlgorithm(
  problem: SplitProblem,
  options: GrowOptions,
  info: TreeInfo,
): Algorithm<GrowState, GrowPath> {
  const { x } = problem
  const nf = featureCount(options.maxFeatures, x.d)
  const maxDepth = options.maxDepth ?? Number.POSITIVE_INFINITY
  const growth: Growth = options.maxLeafNodes === undefined ? (options.growth ?? 'depth') : 'best'
  const maxLeaves = options.maxLeafNodes ?? Number.POSITIVE_INFINITY
  const all = Array.from({ length: x.d }, (_, j) => j)
  const featuresFor = (key: Key): number[] =>
    nf >= x.d && options.splitter === 'best' ? all : Array.from(toFlat(permutation(key, x.d)))
  const stopsHere = (rows: Int32Array, weight: number, depth: number, impurity: number): boolean =>
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
  /** Search a node's splits: its candidates and, if it is to split, the split and the rows each way. */
  const decide = (node: GrownNode, key: Key): GrownNode => {
    if (stopsHere(node.rows, node.weight, node.depth, node.impurity)) return node
    const s = search(node.rows, key)
    const b = s.best
    if (b === undefined || b.decrease + EPSILON < options.minImpurityDecrease)
      return { ...node, candidates: s.candidates }
    return {
      ...node,
      candidates: s.candidates,
      split: {
        feature: b.feature,
        threshold: b.threshold,
        decrease: b.decrease,
        ...(b.categories === undefined ? {} : { categories: b.categories }),
        missingLeft: b.missingLeft,
      },
      found: { left: b.leftRows, right: b.rightRows },
    }
  }
  const birth = (id: number, rows: Int32Array, parent: number, isLeft: boolean, depth: number): GrownNode => {
    let weight = 0
    for (const r of rows) weight += problem.w[r] ?? 0
    return {
      id,
      parent,
      depth,
      isLeft,
      rows,
      weight,
      value: nodeValue(problem, rows),
      impurity: impurityOf(problem, rows),
      candidates: [],
      left: -1,
      right: -1,
      order: -1,
    }
  }
  const allRows = Int32Array.from({ length: x.n }, (_, i) => i)
  const isBest = growth === 'best'
  // The leaves' predictions on the training rows, for the accuracy or R^2 logged each step.
  const trainingScore = (nodes: readonly GrownNode[]): number => {
    const leaves = nodes.filter((n) => n.left < 0)
    if (info.task === 'classification') {
      let right = 0
      let total = 0
      for (const n of leaves) {
        right += Math.max(...n.value)
        total += n.weight
      }
      return total > 0 ? right / total : 0
    }
    let sse = 0
    let sst = 0
    let sw = 0
    let sy = 0
    for (const n of leaves) {
      for (const r of n.rows) {
        const w = problem.w[r] ?? 0
        const yv = problem.y[r] ?? 0
        sse += w * (yv - (n.value[0] ?? 0)) ** 2
        sw += w
        sy += w * yv
      }
    }
    const mean = sw > 0 ? sy / sw : 0
    for (const n of leaves) for (const r of n.rows) sst += (problem.w[r] ?? 0) * ((problem.y[r] ?? 0) - mean) ** 2
    return sst > 0 ? 1 - sse / sst : 1
  }
  return {
    name: 'growTree',
    config: { criterion: problem.criterion, splitter: options.splitter, growth },
    init: (key) => {
      const root = birth(0, allRows, -1, false, 0)
      if (!isBest) return { nodes: [root], open: [0], decided: 0 }
      const searched = decide(root, key)
      return { nodes: [{ ...searched, order: 0 }], open: searched.split === undefined ? [] : [0], decided: 0 }
    },
    step: (s, ctx) => {
      const nodes = [...s.nodes]
      let open = [...s.open]
      let id: number
      let node: GrownNode
      if (isBest) {
        let pick = 0
        for (let i = 1; i < open.length; i++)
          if ((nodes[open[i] ?? 0]?.split?.decrease ?? -1) > (nodes[open[pick] ?? 0]?.split?.decrease ?? -1)) pick = i
        id = open.splice(pick, 1)[0] ?? 0
        node = nodes[id] as GrownNode
      } else {
        id = (growth === 'depth' ? open.pop() : open.shift()) ?? 0
        node = decide(nodes[id] as GrownNode, ctx.key)
      }
      const split = node.split
      const found = node.found
      let decided: GrownNode = withoutFound({ ...node, order: isBest ? node.id : s.decided })
      if (split !== undefined && found !== undefined) {
        const kids: GrownNode[] = []
        for (const [side, rows] of [
          [true, found.left],
          [false, found.right],
        ] as const) {
          const child = birth(nodes.length + kids.length, rows, id, side, node.depth + 1)
          kids.push(isBest ? { ...decide(child, foldIn(ctx.key, side ? 1 : 2)), order: child.id } : child)
        }
        const [l, r] = kids as [GrownNode, GrownNode]
        decided = { ...decided, left: l.id, right: r.id }
        nodes.push(l, r)
        if (isBest) open.push(...kids.filter((k) => k.split !== undefined).map((k) => k.id))
        else if (growth === 'depth') open.push(r.id, l.id)
        else open.push(l.id, r.id)
      }
      nodes[id] = decided
      // Best-first stops at the leaf limit: the nodes still waiting stay leaves.
      if (isBest && nodes.filter((n) => n.left >= 0).length >= maxLeaves - 1) open = []
      if (ctx.wants('tree/split'))
        ctx.log('tree/split', {
          node: isBest ? id : s.decided,
          depth: node.depth,
          samples: node.rows.length,
          impurity: node.impurity,
          ...(split === undefined
            ? { leaf: true }
            : {
                leaf: false,
                feature: split.feature,
                threshold: split.threshold,
                decrease: split.decrease,
                left: decided.left,
                right: decided.right,
              }),
        })
      if (ctx.wants('tree/candidates') && node.candidates.length > 0)
        ctx.log(
          'tree/candidates',
          node.candidates.map((c) => [c.feature, c.threshold, c.impurityLeft, c.impurityRight, c.decrease]),
        )
      if (ctx.wants('tree/leaves')) ctx.log('tree/leaves', nodes.filter((n) => n.left < 0).length)
      if (ctx.wants('tree/depth')) ctx.log('tree/depth', Math.max(...nodes.map((n) => n.depth)))
      if (ctx.wants('tree/impurity')) {
        let total = 0
        for (const n of nodes) if (n.left < 0) total += (n.weight / problem.totalWeight) * n.impurity
        ctx.log('tree/impurity', total)
      }
      const scorePath = info.task === 'classification' ? 'eval/accuracy' : 'eval/r2'
      if (ctx.wants(scorePath)) ctx.log(scorePath, trainingScore(nodes))
      return { nodes, open, decided: s.decided + 1 }
    },
    done: (s) => (s.open.length === 0 ? 'converged' : undefined),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as GrowState,
  }
}

/** The growth order of a state's nodes: birth ids sorted by the step they were decided (waiting nodes last). */
function growthOrder(nodes: readonly GrownNode[]): number[] {
  const key = (i: number): number => {
    const o = nodes[i]?.order ?? -1
    return o < 0 ? Number.POSITIVE_INFINITY : o
  }
  return nodes
    .map((_, i) => i)
    .sort((a, b) => {
      const ka = key(a)
      const kb = key(b)
      return ka === kb ? a - b : ka < kb ? -1 : 1
    })
}

/**
 * The tree of a state, partial or complete: nodes are numbered in growth order (preorder for depth-first, creation
 * order for best-first) and nodes still waiting are leaves.
 *
 * @param state - A state of {@link growTreeAlgorithm}.
 * @param info - Task, classes and feature count.
 * @returns The tree.
 * @example
 * ```ts
 * import { treeOfState } from 'mdml-engine/trees'
 * const root = { id: 0, parent: -1, depth: 0, isLeft: false, rows: Int32Array.of(0, 1), weight: 2, value: [1, 1], impurity: 0.5, candidates: [], left: -1, right: -1, order: -1 }
 * treeOfState({ nodes: [root], open: [0], decided: 0 }, { task: 'classification', classes: [0, 1], nFeatures: 1 }).left // => [-1]
 * ```
 */
export function treeOfState(state: GrowState, info: TreeInfo): Tree {
  const byFinal = growthOrder(state.nodes)
  const finalOf = new Map(byFinal.map((birthId, f) => [birthId, f]))
  const range = byFinal.map((birthId) => state.nodes[birthId] as GrownNode)
  const internal = (n: GrownNode): boolean => n.left >= 0 && n.split !== undefined
  return {
    task: info.task,
    classes: [...info.classes],
    nFeatures: info.nFeatures,
    feature: range.map((n) => (internal(n) ? (n.split?.feature ?? -2) : -2)),
    threshold: range.map((n) => (internal(n) ? (n.split?.threshold ?? -2) : -2)),
    left: range.map((n) => (internal(n) ? (finalOf.get(n.left) ?? -1) : -1)),
    right: range.map((n) => (internal(n) ? (finalOf.get(n.right) ?? -1) : -1)),
    value: range.map((n) => [...n.value]),
    impurity: range.map((n) => n.impurity),
    nSamples: range.map((n) => n.rows.length),
    weightedNSamples: range.map((n) => n.weight),
    missingLeft: range.map((n) => (internal(n) ? n.split?.missingLeft === true : false)),
    categories: range.map((n) => (internal(n) ? [...(n.split?.categories ?? [])] : [])),
  }
}

/**
 * The exposition record of a state: every node in growth order with its rows, value, impurity, candidates and the
 * split chosen. Nodes still waiting appear as leaves.
 *
 * @param state - A state of {@link growTreeAlgorithm}.
 * @param names - The criterion, feature names and class names.
 * @returns The record.
 * @example
 * ```ts
 * import { treeRecordOf } from 'mdml-engine/trees'
 * const root = { id: 0, parent: -1, depth: 0, isLeft: false, rows: Int32Array.of(0, 1), weight: 2, value: [1, 1], impurity: 0.5, candidates: [], left: -1, right: -1, order: -1 }
 * treeRecordOf({ nodes: [root], open: [0], decided: 0 }, { criterion: 'gini', features: ['x'] }).nodes.length // => 1
 * ```
 */
export function treeRecordOf(
  state: GrowState,
  names: { readonly criterion: string; readonly features: readonly string[]; readonly classes?: readonly string[] },
): TreeRecord {
  const byFinal = growthOrder(state.nodes)
  const finalOf = new Map(byFinal.map((birthId, f) => [birthId, f]))
  const nodes: TreeNode[] = byFinal.map((birthId, f) => {
    const n = state.nodes[birthId] as GrownNode
    const split = n.left >= 0 ? n.split : undefined
    return {
      id: f,
      parent: n.parent < 0 ? null : (finalOf.get(n.parent) ?? null),
      depth: n.depth,
      rows: Array.from(n.rows),
      value: [...n.value],
      impurity: n.impurity,
      candidates: n.candidates,
      order: f,
      ...(split === undefined
        ? {}
        : {
            split: {
              feature: split.feature,
              threshold: split.threshold,
              left: finalOf.get(n.left) ?? -1,
              right: finalOf.get(n.right) ?? -1,
              missingLeft: split.missingLeft,
              ...(split.categories === undefined ? {} : { categories: split.categories }),
            },
          }),
    }
  })
  return treeRecord({
    criterion: names.criterion,
    features: [...names.features],
    ...(names.classes === undefined ? {} : { classes: [...names.classes] }),
    nodes,
  })
}

/** The data of a tree: features (NaN allowed), targets (class codes or values), weights and classes. */
export interface TreeData {
  readonly x: MatrixView
  readonly y: Float64Array
  readonly w: Float64Array
  /** Class labels (empty for regression). */
  readonly classes: readonly number[]
}

/**
 * Prepare learning data for a tree: the features as a matrix (missing values allowed), class labels as codes or
 * targets as numbers, and the sample weights.
 *
 * @param data - The data.
 * @param classifier - Whether the targets are class labels.
 * @param who - The caller (for errors).
 * @returns The prepared data.
 * @example
 * ```ts
 * import { treeData } from 'mdml-engine/trees'
 * treeData({ x: [[1], [2]], y: [5, 7] }, true, 'demo').classes // => [5, 7]
 * ```
 */
export function treeData(data: LearnData, classifier: boolean, who: string): TreeData {
  const x = matrixView(data.x, who, { allowNaN: true })
  const w = sampleWeightsOf(data.weights, x.n, who)
  if (data.y === undefined) throw new MdmlError('value', `${who}: needs targets y`)
  if (!classifier) return { x, y: targetVector(data.y, who, x.n), w, classes: [] }
  const { classes, codes } = encodeLabels(data.y, who, x.n)
  return { x, y: Float64Array.from(codes), w, classes }
}

/**
 * The split problem of prepared data and options.
 *
 * @param data - The prepared data.
 * @param options - Growth options (criterion and leaf limits).
 * @param categorical - Indices of the categorical features.
 * @returns The problem.
 * @throws MdmlError with code `value` when the criterion does not suit the task.
 * @example
 * ```ts
 * import { splitProblem, treeData } from 'mdml-engine/trees'
 * const o = { criterion: 'gini' as const, splitter: 'best' as const, minSamplesSplit: 2, minSamplesLeaf: 1, minWeightFractionLeaf: 0, maxFeatures: 'all' as const, minImpurityDecrease: 0 }
 * splitProblem(treeData({ x: [[1], [2]], y: [0, 1] }, true, 'demo'), o).totalWeight // => 2
 * ```
 */
export function splitProblem(data: TreeData, options: GrowOptions, categorical: readonly number[] = []): SplitProblem {
  const total = data.w.reduce((a, b) => a + b, 0)
  const classification = data.classes.length > 0
  if (classification !== isClassificationCriterion(options.criterion))
    throw new MdmlError('value', `growTree: criterion ${options.criterion} does not suit this task`)
  return {
    x: data.x,
    y: data.y,
    w: data.w,
    k: data.classes.length,
    criterion: options.criterion,
    minSamplesLeaf: options.minSamplesLeaf,
    minWeightLeaf: options.minWeightFractionLeaf * total,
    totalWeight: total,
    ...(categorical.length === 0
      ? {}
      : { categorical: Array.from({ length: data.x.d }, (_, j) => categorical.includes(j)) }),
  }
}

/** A grown tree: its arrays, the nodes of its record and the final state. */
export interface GrownTree {
  readonly tree: Tree
  readonly nodes: readonly TreeNode[]
  readonly state: GrowState
}

/**
 * Grow a tree on prepared data to completion (the entry point the ensembles use), running {@link growTreeAlgorithm}.
 *
 * @param data - Features, targets, weights and classes.
 * @param options - Growth options.
 * @param run - Key, tracing and the categorical features.
 * @returns The tree, the nodes of its record and the final state.
 * @example
 * ```ts
 * import { growTree, treeData } from 'mdml-engine/trees'
 * const o = { criterion: 'squared_error' as const, splitter: 'best' as const, minSamplesSplit: 2, minSamplesLeaf: 1, minWeightFractionLeaf: 0, maxFeatures: 'all' as const, minImpurityDecrease: 0 }
 * growTree(treeData({ x: [[0], [1], [2], [3]], y: [0, 0, 5, 5] }, false, 'demo'), o).tree.threshold // => [1.5, -2, -2]
 * ```
 */
export function growTree(
  data: TreeData,
  options: GrowOptions,
  run: FitOptions & { readonly categorical?: readonly number[] } = {},
): GrownTree {
  const problem = splitProblem(data, options, run.categorical)
  const info: TreeInfo = {
    task: data.classes.length > 0 ? 'classification' : 'regression',
    classes: data.classes,
    nFeatures: data.x.d,
  }
  const r = runAlgorithm(growTreeAlgorithm(problem, options, info), {
    maxSteps: 4 * data.x.n + 10,
    ...(run.key === undefined ? {} : { key: run.key }),
    ...(run.parent !== undefined
      ? { parent: run.parent }
      : {
          ...(run.tracer === undefined ? {} : { tracer: run.tracer }),
          ...(run.policy === undefined ? {} : { policy: run.policy }),
        }),
  })
  const record = treeRecordOf(r.state, { criterion: options.criterion, features: [] })
  return { tree: treeOfState(r.state, info), nodes: record.nodes, state: r.state }
}

/**
 * The candidate splits of a record node, best first (for views that show the top few).
 *
 * @param node - A node of a tree record.
 * @param k - How many to keep (default 5).
 * @returns The candidates by decreasing impurity decrease.
 * @example
 * ```ts
 * import { bestCandidates } from 'mdml-engine/trees'
 * const c = (decrease: number) => ({ feature: 0, threshold: decrease, impurityLeft: 0, impurityRight: 0, decrease })
 * bestCandidates({ id: 0, parent: null, depth: 0, rows: [], value: [], impurity: 0, candidates: [c(1), c(3), c(2)], order: 0 }, 2).map((x) => x.decrease) // => [3, 2]
 * ```
 */
export function bestCandidates(node: TreeNode, k = 5): SplitCandidate[] {
  return [...node.candidates].sort((a, b) => b.decrease - a.decrease).slice(0, k)
}
