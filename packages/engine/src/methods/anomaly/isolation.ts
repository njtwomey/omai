import {
  defineExplainedEstimator,
  type Estimator,
  EstimatorClass,
  type ExplainedFitted,
  fitRunOptions,
  type MatrixView,
  matrixView,
} from '../../lib/learn/index.ts'
import { foldIn, type Key, normal, subsample, uniformStream } from '../../lib/random/index.ts'
import { bool, dict, int, real } from '../../lib/space/index.ts'
import { type TensorLike, toFlat } from '../../lib/tensor/index.ts'
import { type Algorithm, runAlgorithm } from '../../lib/trace/index.ts'
import { contaminationThreshold, type OutlierVerbs, outlierExposition, outlierVerbs } from './combine.ts'

/**
 * An isolation tree node, flat: a split (feature, or a hyperplane normal for extended trees, and threshold) or a leaf
 * with its training size.
 */
export type IsolationNode =
  | { readonly leaf: true; readonly size: number }
  | {
      readonly leaf: false
      /** The feature split on (standard trees) or the hyperplane normal (extended trees). */
      readonly feature: number
      readonly normal: number[] | null
      readonly threshold: number
      readonly left: number
      readonly right: number
    }

/** An isolation tree as a node list (root first). */
export type IsolationTree = IsolationNode[]

const EULER = 0.5772156649015329

/**
 * The average path length of an unsuccessful binary-search-tree search over n points (sklearn's
 * `_average_path_length`): c(n) = 2 H(n - 1) - 2 (n - 1) / n, with c(1) = 0 and c(2) = 1.
 *
 * @param n - The number of points.
 * @returns c(n).
 */
export function averagePathLength(n: number): number {
  if (n <= 1) return 0
  if (n === 2) return 1
  return 2 * (Math.log(n - 1) + EULER) - (2 * (n - 1)) / n
}

/**
 * Grow one isolation tree (Liu, Ting and Zhou 2008) on some rows: random features and uniform thresholds between the
 * node's minimum and maximum (or, extended (Hariri et al. 2019), random hyperplanes through a uniform point), until a
 * node is a single row, constant, or at the depth limit.
 *
 * @param x - The data.
 * @param rows - The rows the tree sees.
 * @param key - Randomness.
 * @param options - Depth limit and whether the tree is extended.
 * @returns The tree.
 */
export function isolationTree(
  x: MatrixView,
  rows: readonly number[],
  key: Key,
  options: { readonly maxDepth: number; readonly extended?: boolean },
): IsolationTree {
  const u = uniformStream(foldIn(key, 0))
  const z = options.extended === true ? toFlat(normal(foldIn(key, 1), [Math.max(1, 2 * rows.length) * x.d])) : []
  let zi = 0
  const nodes: IsolationTree = []
  const value = (i: number, j: number): number => x.data[i * x.d + j] ?? 0
  const grow = (members: readonly number[], depth: number): number => {
    const id = nodes.length
    nodes.push({ leaf: true, size: members.length })
    if (members.length <= 1 || depth >= options.maxDepth) return id
    if (options.extended === true) {
      const nrm = Array.from({ length: x.d }, () => z[zi++ % Math.max(1, z.length)] ?? 0)
      const lo = Array.from({ length: x.d }, (_, j) => Math.min(...members.map((i) => value(i, j))))
      const hi = Array.from({ length: x.d }, (_, j) => Math.max(...members.map((i) => value(i, j))))
      if (lo.every((v, j) => v === hi[j])) return id
      const point = lo.map((v, j) => v + u() * ((hi[j] ?? v) - v))
      const threshold = nrm.reduce((s, w, j) => s + w * (point[j] ?? 0), 0)
      const proj = (i: number): number => nrm.reduce((s, w, j) => s + w * value(i, j), 0)
      const left = members.filter((i) => proj(i) < threshold)
      const right = members.filter((i) => proj(i) >= threshold)
      const l = grow(left, depth + 1)
      const r = grow(right, depth + 1)
      nodes[id] = { leaf: false, feature: -1, normal: nrm, threshold, left: l, right: r }
      return id
    }
    const candidates = Array.from({ length: x.d }, (_, j) => j).filter((j) => {
      const vs = members.map((i) => value(i, j))
      return Math.max(...vs) > Math.min(...vs)
    })
    if (candidates.length === 0) return id
    const feature = candidates[Math.min(candidates.length - 1, Math.floor(u() * candidates.length))] ?? 0
    const vs = members.map((i) => value(i, feature))
    const lo = Math.min(...vs)
    const hi = Math.max(...vs)
    const threshold = lo + u() * (hi - lo)
    const left = members.filter((i) => value(i, feature) < threshold)
    const right = members.filter((i) => value(i, feature) >= threshold)
    const l = grow(left, depth + 1)
    const r = grow(right, depth + 1)
    nodes[id] = { leaf: false, feature, normal: null, threshold, left: l, right: r }
    return id
  }
  grow(rows, 0)
  return nodes
}

/**
 * The path length of a point in an isolation tree: the depth of its leaf plus c(leaf size).
 *
 * @param tree - The tree.
 * @param point - The point.
 * @returns The adjusted path length.
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

/** The state of growing a forest tree by tree. */
export interface ForestState {
  readonly trees: IsolationTree[]
}

/**
 * Growing an isolation forest as a traced algorithm: one tree per step, each on its own subsample.
 *
 * @param x - The data.
 * @param nEstimators - Trees.
 * @param maxSamples - Rows per tree.
 * @param extended - Extended (hyperplane) trees.
 * @returns The algorithm.
 */
export function isolationForestAlgorithm(
  x: MatrixView,
  nEstimators: number,
  maxSamples: number,
  extended = false,
): Algorithm<ForestState> {
  const maxDepth = Math.ceil(Math.log2(Math.max(maxSamples, 2)))
  return {
    name: 'isolation-forest',
    config: { nEstimators, maxSamples, extended },
    init: () => ({ trees: [] }),
    step: (s, ctx) => {
      const rows = Array.from(subsample(foldIn(ctx.key, 0), x.n, maxSamples).data)
      return { trees: [...s.trees, isolationTree(x, rows, foldIn(ctx.key, 1), { maxDepth, extended })] }
    },
    done: (s) => (s.trees.length >= nEstimators ? 'completed' : undefined),
    observe: (s) => ({ metrics: { trees: s.trees.length } }),
  }
}

/** Hyperparameters of the isolation forest. */
export type IsolationForestHyper = {
  readonly nEstimators?: number
  /** Rows per tree (default min(256, n)). */
  readonly maxSamples?: number
  /** The expected fraction of outliers, or 'auto' (the original paper's threshold 0.5 on the score). */
  readonly contamination?: number | 'auto'
  /** Hyperplane splits (extended isolation forest) instead of axis-parallel ones (default false). */
  readonly extended?: boolean
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
export type FittedIsolationForest = ExplainedFitted<IsolationForestParams, OutlierVerbs>

/**
 * The isolation forest (sklearn.ensemble.IsolationForest; extended trees optionally): trees grown on subsamples
 * (traced tree by tree); a point's anomaly score is 2^(-E[h(x)] / c(psi)) from its mean path length. Its exposition is
 * an `anomalyScores` record with the mean path lengths as evidence and, for one or two features, a `decisionSurface`
 * of the score.
 *
 * @param hyper - Trees, rows per tree, contamination and the extended variant.
 * @returns The estimator.
 */
export function isolationForest(
  hyper: IsolationForestHyper = {},
): Estimator<IsolationForestHyper, FittedIsolationForest> {
  const verbs = (p: IsolationForestParams): OutlierVerbs => {
    const c = averagePathLength(p.maxSamples)
    const paths = (x: TensorLike): number[] => {
      const m = matrixView(x, 'isolationForest', { nFeatures: p.nFeaturesIn })
      return Array.from({ length: m.n }, (_, i) => {
        const point = Array.from(m.data.subarray(i * m.d, (i + 1) * m.d))
        return p.trees.reduce((a, t) => a + pathLength(t, point), 0) / p.trees.length
      })
    }
    return outlierVerbs((x) => paths(x).map((h) => 2 ** (-h / c)), p.threshold, p.nFeaturesIn)
  }
  return defineExplainedEstimator<IsolationForestHyper, IsolationForestParams, OutlierVerbs>(
    {
      name: 'isolation-forest',
      kind: 'outlier',
      hyperSpace: (h) =>
        dict({
          nEstimators: int(1, 500, { default: h.nEstimators ?? 100, label: 'trees' }),
          maxSamples: int(2, 1024, { default: h.maxSamples ?? 256, label: 'rows per tree' }),
          contamination: real(0.01, 0.5, { default: typeof h.contamination === 'number' ? h.contamination : 0.1 }),
          extended: bool({ default: h.extended ?? false }),
        }),
      fit: (data, h, options) => {
        const m = matrixView(data.x, 'isolationForest', { minSamples: 2 })
        const psi = Math.min(h.maxSamples ?? 256, m.n)
        const nTrees = h.nEstimators ?? 100
        const run = fitRunOptions(options, 'isolation-forest')
        const out = runAlgorithm(isolationForestAlgorithm(m, nTrees, psi, h.extended ?? false), {
          ...run,
          maxSteps: nTrees,
        })
        const base: IsolationForestParams = {
          trees: out.state.trees,
          maxSamples: psi,
          threshold: 0.5,
          nFeaturesIn: m.d,
        }
        const scores = toFlat(verbs(base).scoreSamples(data.x))
        const c = h.contamination ?? 'auto'
        const threshold = c === 'auto' ? 0.5 : contaminationThreshold(scores, c)
        const params = { ...base, threshold }
        const fitted = { estimator: 'isolation-forest', kind: 'outlier' as const, params, ...verbs(params) }
        const c0 = averagePathLength(psi)
        return {
          params,
          exposition: outlierExposition(fitted, data.x, 'isolationForest', threshold, {
            name: 'meanPathLength',
            values: scores.map((s) => -Math.log2(s) * c0),
          }),
        }
      },
      methods: verbs,
    },
    hyper,
  )
}

/** The isolation forest as a class. */
export class IsolationForest extends EstimatorClass<IsolationForestHyper, FittedIsolationForest> {
  /**
   * Make an unfitted model.
   *
   * @param hyper - The hyperparameters of {@link isolationForest}.
   */
  constructor(hyper: IsolationForestHyper = {}) {
    super(isolationForest(hyper))
  }
}
