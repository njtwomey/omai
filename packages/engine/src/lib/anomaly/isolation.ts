/**
 * The isolation forest (Liu, Ting and Zhou 2008; as in scikit-learn's `IsolationForest`) and the extended isolation
 * forest (Hariri, Kind and Brunner 2019): random trees grown on subsamples isolate points by random axis-parallel or
 * hyperplane splits, and anomalies, being few and different, are isolated in fewer splits. Trees are grown one per
 * traced step as an `Algorithm`, and the estimators are built on the iterative base (`kind: 'outlier'`) with isolation
 * path lengths as per-point evidence.
 */

import { EULER_GAMMA, type Json, maxOf, minOf } from '../base/index.ts'
import {
  ExplainedEstimatorClass,
  type Fitted,
  type IterativeEstimator,
  type IterativeFitted,
  isMatrixView,
  iterative,
  type MatrixView,
  matrixView,
} from '../learn/index.ts'
import { foldIn, type Key, normal, subsample, uniformStream } from '../random/index.ts'
import { bool, dict, int, real } from '../space/index.ts'
import { type TensorLike, toFlat } from '../tensor/index.ts'
import {
  type Algorithm,
  type Contract,
  channels,
  defineContract,
  derived,
  Log,
  scalar,
  stateFromJson,
  stateToJson,
} from '../trace/index.ts'
import { contaminationThreshold, type OutlierVerbs, outlierExposition, outlierVerbs } from './combine.ts'

/**
 * An isolation tree node, flat: a split (feature index, or a hyperplane normal for extended trees, and threshold) or a
 * leaf with its training subsample size.
 */
export type IsolationNode =
  | { readonly leaf: true; readonly size: number }
  | {
      readonly leaf: false
      /** The feature split on (standard trees) or `-1` when `normal` is used (extended trees). */
      readonly feature: number
      readonly normal: number[] | null
      readonly threshold: number
      readonly left: number
      readonly right: number
    }

/** An isolation tree as a node list (root at index 0). */
export type IsolationTree = IsolationNode[]

/** One node visited along a point's root-to-leaf isolation path through a tree. */
export interface IsolationPathStep {
  readonly node: number
  readonly depth: number
  readonly leaf: boolean
  readonly feature: number
  readonly normal: number[] | null
  readonly threshold: number
  readonly value: number
  readonly direction: 'left' | 'right' | 'leaf'
  readonly size: number
}

function toView(x: TensorLike | MatrixView, who: string): MatrixView {
  return isMatrixView(x) ? x : matrixView(x, who, { minSamples: 2 })
}

/**
 * The average path length of an unsuccessful binary-search-tree search over `n` points (scikit-learn's
 * `_average_path_length`): `c(n) = 2 H(n - 1) - 2 (n - 1) / n`, with `c(1) = 0` and `c(2) = 1`.
 *
 * @param n - The number of points.
 * @returns The expected path length `c(n)`.
 * @example
 * ```ts
 * import { averagePathLength } from 'omai-engine/anomaly'
 * [averagePathLength(1), averagePathLength(2)] // => [0, 1]
 * ```
 */
export function averagePathLength(n: number): number {
  if (n <= 1) return 0
  if (n === 2) return 1
  return 2 * (Math.log(n - 1) + EULER_GAMMA) - (2 * (n - 1)) / n
}

/**
 * Grow one isolation tree (Liu, Ting and Zhou 2008) on a subset of rows: random features and uniform thresholds
 * between the node's minimum and maximum (or, when `extended` is true (Hariri et al. 2019), random hyperplanes through
 * a uniform point in the bounding box), until a node has a single row, is constant, or reaches the depth limit.
 *
 * @param x - The data (`TensorLike` `[n, d]` or `MatrixView`).
 * @param rows - The row indices the tree sees.
 * @param key - Random key.
 * @param options - Depth limit and whether the tree uses extended hyperplane splits.
 * @returns The grown isolation tree.
 * @example
 * ```ts
 * import { isolationTree } from 'omai-engine/anomaly'
 * import { key } from 'omai-engine/random'
 * isolationTree([[0], [1], [10]], [0, 1, 2], key(0), { maxDepth: 2 }).length > 1 // => true
 * ```
 */
export function isolationTree(
  x: TensorLike | MatrixView,
  rows: readonly number[],
  key: Key,
  options: { readonly maxDepth: number; readonly extended?: boolean },
): IsolationTree {
  const m = toView(x, 'isolationTree')
  const u = uniformStream(foldIn(key, 0))
  const z = options.extended === true ? toFlat(normal(foldIn(key, 1), [Math.max(1, 2 * rows.length) * m.d])) : []
  let zi = 0
  const nodes: IsolationTree = []
  const value = (i: number, j: number): number => m.data[i * m.d + j] ?? 0
  const grow = (members: readonly number[], depth: number): number => {
    const id = nodes.length
    nodes.push({ leaf: true, size: members.length })
    if (members.length <= 1 || depth >= options.maxDepth) return id
    if (options.extended === true) {
      const nrm = Array.from({ length: m.d }, () => z[zi++ % Math.max(1, z.length)] ?? 0)
      const lo = Array.from({ length: m.d }, (_, j) => minOf(members.map((i) => value(i, j))))
      const hi = Array.from({ length: m.d }, (_, j) => maxOf(members.map((i) => value(i, j))))
      if (lo.every((v, j) => v === hi[j])) return id
      const point = lo.map((v, j) => v + u() * ((hi[j] ?? v) - v))
      const threshold = nrm.reduce((s, w, j) => s + w * (point[j] ?? 0), 0)
      const proj = (i: number): number => nrm.reduce((s, w, j) => s + w * value(i, j), 0)
      const left = members.filter((i) => proj(i) < threshold)
      const right = members.filter((i) => proj(i) >= threshold)
      if (left.length === 0 || right.length === 0) return id
      const l = grow(left, depth + 1)
      const r = grow(right, depth + 1)
      nodes[id] = { leaf: false, feature: -1, normal: nrm, threshold, left: l, right: r }
      return id
    }
    const candidates = Array.from({ length: m.d }, (_, j) => j).filter((j) => {
      const vs = members.map((i) => value(i, j))
      return maxOf(vs) > minOf(vs)
    })
    if (candidates.length === 0) return id
    const feature = candidates[Math.min(candidates.length - 1, Math.floor(u() * candidates.length))] ?? 0
    const vs = members.map((i) => value(i, feature))
    const lo = minOf(vs)
    const hi = maxOf(vs)
    const threshold = lo + u() * (hi - lo)
    const left = members.filter((i) => value(i, feature) < threshold)
    const right = members.filter((i) => value(i, feature) >= threshold)
    if (left.length === 0 || right.length === 0) return id
    const l = grow(left, depth + 1)
    const r = grow(right, depth + 1)
    nodes[id] = { leaf: false, feature, normal: null, threshold, left: l, right: r }
    return id
  }
  grow(rows, 0)
  return nodes
}

/**
 * The path length of a point in an isolation tree: the depth of its leaf plus `c(leaf size)`.
 *
 * @param tree - The isolation tree.
 * @param point - The point `[d]`.
 * @returns The adjusted path length.
 * @example
 * ```ts
 * import { isolationTree, pathLength } from 'omai-engine/anomaly'
 * import { key } from 'omai-engine/random'
 * const t = isolationTree([[0], [0.1], [0.2], [10]], [0, 1, 2, 3], key(0), { maxDepth: 3 })
 * pathLength(t, [10]) <= pathLength(t, [0.1]) // => true
 * ```
 */
export function pathLength(tree: IsolationTree, point: readonly number[]): number {
  let node = 0
  let depth = 0
  for (;;) {
    const n = tree[node]
    if (n === undefined) return depth
    if (n.leaf) return depth + averagePathLength(n.size)
    const v = n.normal === null ? (point[n.feature] ?? 0) : n.normal.reduce((s, w, j) => s + w * (point[j] ?? 0), 0)
    node = v < n.threshold ? n.left : n.right
    depth++
  }
}

/**
 * Trace the sequence of nodes a point visits from the root of an isolation tree to its leaf.
 *
 * @param tree - The isolation tree.
 * @param point - The point `[d]`.
 * @returns The steps from root to leaf.
 * @example
 * ```ts
 * import { isolationPath, isolationTree } from 'omai-engine/anomaly'
 * import { key } from 'omai-engine/random'
 * const t = isolationTree([[0], [10]], [0, 1], key(0), { maxDepth: 2 })
 * isolationPath(t, [10]).at(-1)?.leaf // => true
 * ```
 */
export function isolationPath(tree: IsolationTree, point: readonly number[]): IsolationPathStep[] {
  const steps: IsolationPathStep[] = []
  let node = 0
  let depth = 0
  for (;;) {
    const n = tree[node]
    if (n === undefined) break
    if (n.leaf) {
      steps.push({
        node,
        depth,
        leaf: true,
        feature: -1,
        normal: null,
        threshold: 0,
        value: 0,
        direction: 'leaf',
        size: n.size,
      })
      break
    }
    const v = n.normal === null ? (point[n.feature] ?? 0) : n.normal.reduce((s, w, j) => s + w * (point[j] ?? 0), 0)
    const goLeft = v < n.threshold
    steps.push({
      node,
      depth,
      leaf: false,
      feature: n.feature,
      normal: n.normal,
      threshold: n.threshold,
      value: v,
      direction: goLeft ? 'left' : 'right',
      size: 0,
    })
    node = goLeft ? n.left : n.right
    depth++
  }
  return steps
}

/** The state of growing an isolation forest tree by tree. */
export interface ForestState {
  readonly trees: IsolationTree[]
  readonly meanPaths: number[]
  readonly scores: number[]
  readonly scoreChange: number
}

/** Options of {@link isolationForestAlgorithm}. */
export interface IsolationForestAlgorithmOptions {
  /** Number of trees to grow (default 100). */
  readonly nEstimators?: number
  /** Subsample size per tree (default `min(256, n)`). */
  readonly maxSamples?: number
  /** Whether to grow extended hyperplane trees (default false). */
  readonly extended?: boolean
}

/**
 * Growing an isolation forest as a traced {@link Algorithm}: one tree per step on its own subsample, updating the
 * running mean path lengths and anomaly scores of the training rows and observing the tree count and score change.
 *
 * @param x - The data (`TensorLike` `[n, d]` or `MatrixView`).
 * @param options - Trees, subsample size and whether the splits are hyperplanes.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { isolationForestAlgorithm } from 'omai-engine/anomaly'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const out = runAlgorithm(isolationForestAlgorithm([[0], [0.1], [0.2], [10]], { nEstimators: 10 }), { maxSteps: 10 })
 * out.state.trees.length // => 10
 * ```
 */
export function isolationForestAlgorithm(
  x: TensorLike | MatrixView,
  opts: IsolationForestAlgorithmOptions = {},
): Algorithm<ForestState> {
  const m = toView(x, 'isolationForestAlgorithm')
  const nEstimators = Math.max(1, opts.nEstimators ?? 100)
  const maxSamples = Math.min(Math.max(2, opts.maxSamples ?? 256), m.n)
  const extended = opts.extended ?? false
  const maxDepth = Math.ceil(Math.log2(Math.max(maxSamples, 2)))
  const c = Math.max(averagePathLength(maxSamples), 1e-12)
  const points = Array.from({ length: m.n }, (_, i) => Array.from(m.data.subarray(i * m.d, (i + 1) * m.d)))
  return {
    name: extended ? 'extended-isolation-forest' : 'isolation-forest',
    config: { nEstimators, maxSamples, extended },
    schema: channels({
      metrics: {
        trees: { description: 'the number of isolation trees grown so far' },
        scoreChange: { description: 'mean absolute change in anomaly scores this step', direction: 'lower' },
      },
      tensors: {
        scores: { role: 'state', description: 'anomaly score of each training row', shape: ['n'] },
        meanPaths: { role: 'state', description: 'mean path length of each training row', shape: ['n'] },
      },
    }),
    init: () => ({
      trees: [],
      meanPaths: new Array<number>(m.n).fill(c),
      scores: new Array<number>(m.n).fill(0.5),
      scoreChange: 0,
    }),
    step: (s, ctx) => {
      const rows = Array.from(subsample(foldIn(ctx.key, 0), m.n, maxSamples).data)
      const tree = isolationTree(m, rows, foldIn(ctx.key, 1), { maxDepth, extended })
      const k = s.trees.length + 1
      const meanPaths = points.map((pt, i) => {
        const h = pathLength(tree, pt)
        return k === 1 ? h : ((s.meanPaths[i] ?? 0) * (k - 1) + h) / k
      })
      const scores = meanPaths.map((h) => 2 ** (-h / c))
      const scoreChange = scores.reduce((acc, sc, i) => acc + Math.abs(sc - (s.scores[i] ?? 0.5)), 0) / Math.max(1, m.n)
      return {
        trees: [...s.trees, tree],
        meanPaths,
        scores,
        scoreChange,
      }
    },
    done: (s) => (s.trees.length >= nEstimators ? 'completed' : undefined),
    observe: (s) => ({
      metrics: { trees: s.trees.length, scoreChange: s.scoreChange },
      tensors: {
        scores: { role: 'state', value: s.scores, shape: [m.n] },
        meanPaths: { role: 'state', value: s.meanPaths, shape: [m.n] },
      },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as ForestState,
  }
}

/**
 * Extended isolation forest (Hariri, Kind and Brunner 2019) as a traced {@link Algorithm}: grows one hyperplane-split
 * isolation tree per step.
 *
 * @param x - The data (`TensorLike` `[n, d]` or `MatrixView`).
 * @param options - Number of trees (`nEstimators`) and subsample size (`maxSamples`).
 * @returns The algorithm.
 * @example
 * ```ts
 * import { extendedIsolationForestAlgorithm } from 'omai-engine/anomaly'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const out = runAlgorithm(extendedIsolationForestAlgorithm([[0, 0], [0.1, 0.1], [10, 10]], { nEstimators: 5 }), { maxSteps: 5 })
 * out.state.trees.length // => 5
 * ```
 */
export function extendedIsolationForestAlgorithm(
  x: TensorLike | MatrixView,
  options: Omit<IsolationForestAlgorithmOptions, 'extended'> = {},
): Algorithm<ForestState> {
  return isolationForestAlgorithm(x, { ...options, extended: true })
}

/** Hyperparameters of the isolation forest. */
export interface IsolationForestHyper {
  /** Number of trees (default 100). */
  readonly nEstimators: number
  /** Rows per tree, capped at the number of rows (default 256). */
  readonly maxSamples: number
  /** The expected fraction of outliers in `(0, 0.5]`, or `'auto'` (default: the paper's threshold 0.5 on the score). */
  readonly contamination: number | 'auto'
  /** Hyperplane splits (extended isolation forest) instead of axis-parallel ones (default false). */
  readonly extended: boolean
}

/** The parameters of a fitted isolation forest. */
export type IsolationForestParams = {
  readonly trees: IsolationTree[]
  readonly maxSamples: number
  /** The anomaly score above which a point is an outlier. */
  readonly threshold: number
  readonly nFeaturesIn: number
}

/** A fitted isolation forest. */
export type FittedIsolationForest = IterativeFitted<IsolationForestParams, OutlierVerbs>

function forestVerbs(p: IsolationForestParams): OutlierVerbs {
  const c = Math.max(averagePathLength(p.maxSamples), 1e-12)
  const paths = (x: TensorLike): number[] => {
    const m = matrixView(x, 'isolationForest', { nFeatures: p.nFeaturesIn })
    if (p.trees.length === 0) return new Array<number>(m.n).fill(c)
    return Array.from({ length: m.n }, (_, i) => {
      const point = Array.from(m.data.subarray(i * m.d, (i + 1) * m.d))
      return p.trees.reduce((a, t) => a + pathLength(t, point), 0) / p.trees.length
    })
  }
  return outlierVerbs((x) => paths(x).map((h) => 2 ** (-h / c)), p.threshold, p.nFeaturesIn)
}

/**
 * The paths of the isolation forest: the trees so far and the change in the scores, renamed from the algorithm’s channels by its contract, and the derived `model/isolationPath`.
 *
 * @example
 * ```ts
 * import type { IsolationForestPath } from 'omai-engine/anomaly'
 * const path: IsolationForestPath = 'forest/trees'
 * path // => 'forest/trees'
 * ```
 */
export type IsolationForestPath = 'forest/trees' | 'forest/scoreChange' | 'model/isolationPath'

/**
 * The isolation forest (sklearn.ensemble.IsolationForest; extended trees optionally) on the iterative base: trees
 * grown on subsamples one tree per step; a point's anomaly score is `2^(-E[h(x)] / c(psi))` from its mean path length.
 * Its exposition is an `anomalyScores` record with the mean path lengths as evidence and, for one or two features, a
 * `decisionSurface` of the score.
 *
 * @param hyper - Trees, rows per tree, contamination and the extended variant.
 * @returns The iterative outlier estimator.
 * @index estimator Isolation forest outlier detector.
 * @example
 * ```ts
 * import { isolationForest } from 'omai-engine/anomaly'
 * import { key } from 'omai-engine/random'
 * import { toFlat } from 'omai-engine/tensor'
 * const x = [[0], [0.1], [-0.1], [0.2], [-0.2], [20]]
 * toFlat(isolationForest({ nEstimators: 30, contamination: 0.2 }).fit({ x }, { key: key(0) }).predict([[0], [20]])) // => [1, -1]
 * ```
 */
export function isolationForest(
  hyper: Partial<IsolationForestHyper> = {},
): IterativeEstimator<IsolationForestHyper, ForestState, IsolationForestParams, OutlierVerbs, IsolationForestPath> {
  const defaults: IsolationForestHyper = {
    nEstimators: hyper.nEstimators ?? 100,
    maxSamples: hyper.maxSamples ?? 256,
    contamination: hyper.contamination ?? 'auto',
    extended: hyper.extended ?? false,
  }
  return iterative<IsolationForestHyper, ForestState, IsolationForestParams, OutlierVerbs, IsolationForestPath>({
    name: defaults.extended ? 'extended-isolation-forest' : 'isolation-forest',
    kind: 'outlier',
    defaults,
    hyperSpace: (h) =>
      dict({
        nEstimators: int(1, 500, { default: h.nEstimators, label: 'trees' }),
        maxSamples: int(2, 1024, { default: h.maxSamples, label: 'rows per tree' }),
        ...(typeof h.contamination === 'number'
          ? {
              contamination: real(0.01, 0.5, {
                label: 'contamination',
                description: 'The expected fraction of outliers: sets the threshold on the scores.',
                default: h.contamination,
              }),
            }
          : {}),
        extended: bool({
          label: 'extended',
          description: 'Split on random hyperplanes (extended isolation forest) instead of single features.',
          default: h.extended,
        }),
      }),
    algorithm: (data, h) => {
      const m = matrixView(data.x, 'isolationForest', { minSamples: 2 })
      return isolationForestAlgorithm(m, {
        nEstimators: h.nEstimators,
        maxSamples: Math.min(h.maxSamples, m.n),
        extended: h.extended,
      })
    },
    maxSteps: (h) => h.nEstimators,
    params: (s, data, h) => {
      const m = matrixView(data.x, 'isolationForest', { minSamples: 2 })
      const psi = Math.min(h.maxSamples, m.n)
      const c = h.contamination
      const threshold = s.trees.length === 0 || c === 'auto' ? 0.5 : contaminationThreshold(s.scores, c)
      return {
        trees: s.trees,
        maxSamples: psi,
        threshold,
        nFeaturesIn: m.d,
      }
    },
    verbs: forestVerbs,
    contract: (data): Contract<ForestState, IsolationForestPath> => {
      const m = matrixView(data.x, 'isolationForest', { minSamples: 2 })
      return defineContract({
        'forest/trees': scalar({ from: 'trees', description: 'the number of isolation trees grown so far' }),
        'forest/scoreChange': scalar({
          from: 'scoreChange',
          description: 'mean absolute change in anomaly scores this step',
          direction: 'lower',
        }),
        'model/isolationPath': derived<ForestState>({
          description: 'the isolation path of a chosen training row through the latest tree',
          compute: (s, args) => {
            const row = (args as { readonly row?: number } | undefined)?.row ?? 0
            const tree = s.trees.at(-1)
            if (tree === undefined) return [] as Json
            const pt = Array.from(m.data.subarray(row * m.d, (row + 1) * m.d))
            return isolationPath(tree, pt) as unknown as Json
          },
        }),
      })
    },
    records: (s, data, h) => {
      const m = matrixView(data.x, 'isolationForest', { minSamples: 2 })
      const psi = Math.min(h.maxSamples, m.n)
      const c = h.contamination
      const threshold = s.trees.length === 0 || c === 'auto' ? 0.5 : contaminationThreshold(s.scores, c)
      const params: IsolationForestParams = {
        trees: s.trees,
        maxSamples: psi,
        threshold,
        nFeaturesIn: m.d,
      }
      const fitted = {
        estimator: h.extended ? 'extended-isolation-forest' : 'isolation-forest',
        kind: 'outlier' as const,
        params,
        ...forestVerbs(params),
      }
      return outlierExposition(
        fitted,
        data.x,
        h.extended ? 'extendedIsolationForest' : 'isolationForest',
        threshold,
        { name: 'meanPathLength', values: s.meanPaths },
        s.scores,
      )
    },
    logs: { 'solver/trees': Log(), 'solver/scoreChange': Log() },
  })
}

/**
 * The extended isolation forest (Hariri, Kind and Brunner 2019) on the iterative base: splits on random hyperplanes
 * rather than coordinate axes, eliminating axis-parallel score artefacts.
 *
 * @param hyper - Trees, rows per tree and contamination.
 * @returns The iterative outlier estimator.
 * @example
 * ```ts
 * import { extendedIsolationForest } from 'omai-engine/anomaly'
 * import { key } from 'omai-engine/random'
 * import { toFlat } from 'omai-engine/tensor'
 * const x = [[0, 0], [0.1, 0.1], [-0.1, -0.1], [0.2, -0.1], [-0.1, 0.2], [20, 20]]
 * toFlat(extendedIsolationForest({ nEstimators: 30, contamination: 0.2 }).fit({ x }, { key: key(0) }).predict([[0, 0], [20, 20]])) // => [1, -1]
 * ```
 */
export function extendedIsolationForest(
  hyper: Partial<Omit<IsolationForestHyper, 'extended'>> = {},
): IterativeEstimator<IsolationForestHyper, ForestState, IsolationForestParams, OutlierVerbs, IsolationForestPath> {
  return isolationForest({ ...hyper, extended: true })
}

/**
 * The isolation forest as a scikit-learn-style class wrapper.
 *
 * @example
 * ```ts
 * import { IsolationForest } from 'omai-engine/anomaly'
 * import { key } from 'omai-engine/random'
 * import { toFlat } from 'omai-engine/tensor'
 * const x = [[0], [0.1], [-0.1], [0.2], [-0.2], [20]]
 * toFlat(new IsolationForest({ nEstimators: 30, contamination: 0.2 }).fit(x, undefined, { key: key(0) }).predict([[0], [20]])) // => [1, -1]
 * ```
 */
export class IsolationForest extends ExplainedEstimatorClass<
  IsolationForestHyper,
  Fitted<IsolationForestParams> & OutlierVerbs
> {
  /**
   * Make an unfitted isolation forest model.
   *
   * @param hyper - The hyperparameters of {@link isolationForest}.
   */
  constructor(hyper: Partial<IsolationForestHyper> = {}) {
    super(isolationForest(hyper))
  }
}

/**
 * The extended isolation forest as a scikit-learn-style class wrapper.
 *
 * @example
 * ```ts
 * import { ExtendedIsolationForest } from 'omai-engine/anomaly'
 * import { key } from 'omai-engine/random'
 * import { toFlat } from 'omai-engine/tensor'
 * const x = [[0, 0], [0.1, 0.1], [-0.1, -0.1], [0.2, -0.1], [-0.1, 0.2], [20, 20]]
 * toFlat(new ExtendedIsolationForest({ nEstimators: 30, contamination: 0.2 }).fit(x, undefined, { key: key(0) }).predict([[0, 0], [20, 20]])) // => [1, -1]
 * ```
 */
export class ExtendedIsolationForest extends ExplainedEstimatorClass<
  IsolationForestHyper,
  Fitted<IsolationForestParams> & OutlierVerbs
> {
  /**
   * Make an unfitted extended isolation forest model.
   *
   * @param hyper - The hyperparameters of {@link extendedIsolationForest}.
   */
  constructor(hyper: Partial<Omit<IsolationForestHyper, 'extended'>> = {}) {
    super(extendedIsolationForest(hyper))
  }
}
