// TODO: harmonic functions with class mass normalisation (Zhu, Ghahramani and Lafferty 2003) as a third variant.
import { argmaxOf, WhiteboxError } from '../../lib/base/index.ts'
import { type ClusteringSteps, clusteringSteps, type Exposition } from '../../lib/exposition/index.ts'
import { pairwiseKernels } from '../../lib/kernels/index.ts'
import {
  classNamesOf,
  decisionSurfaceOf,
  defineExplainedEstimator,
  type ExplainedEstimator,
  ExplainedEstimatorClass,
  type ExplainedFitted,
  type FitOptions,
  featureNamesOf,
  fitRunOptions,
  type LearnData,
  type MatrixView,
  matrixView,
} from '../../lib/learn/index.ts'
import { toDense } from '../../lib/numerics/sparse/index.ts'
import { kNearest, kNeighboursGraph } from '../../lib/numerics/spatial/index.ts'
import { choice, dict, int, real, type Space } from '../../lib/space/index.ts'
import { fromStorage, type Tensor, type TensorLike, toFlat } from '../../lib/tensor/index.ts'
import { type Algorithm, channels, runAlgorithm } from '../../lib/trace/index.ts'
import { semiSupervisedCodes, UNLABELLED } from './labels.ts'

/** Hyperparameters of {@link labelPropagation} and {@link labelSpreading} (scikit-learn's names and defaults). */
export interface GraphLabelHyper {
  /** The affinity: a Gaussian kernel on all pairs or a k-nearest-neighbour graph. */
  readonly kernel: 'rbf' | 'knn'
  /** The rbf kernel's gamma, exp(-gamma |x - y|^2) (default 20). */
  readonly gamma: number
  /** Neighbours of the knn kernel (default 7). */
  readonly nNeighbors: number
  /** Label spreading's clamping factor: the share of a row's labels taken from its neighbours (default 0.2). */
  readonly alpha: number
  /** Iteration cap (default 1000 for propagation, 30 for spreading). */
  readonly maxIter: number
  /** Stop when the label distributions change by less than this in total (default 1e-3). */
  readonly tol: number
  /** Most snapshots kept in the exposition (default 60; evenly thinned beyond). */
  readonly maxFrames: number
}

/** Parameters of a fitted graph-based model: the training rows, their label distributions and the transduction. */
export type GraphLabelParams = {
  readonly classes: number[]
  readonly x: number[][]
  /** labelDistributions[i][c]: row i's distribution over the classes. */
  readonly labelDistributions: number[][]
  /** The class given to every training row. */
  readonly transduction: number[]
  readonly nIter: number
  readonly kernel: 'rbf' | 'knn'
  readonly gamma: number
  readonly nNeighbors: number
}

/** Verbs of a fitted graph-based model. */
export interface GraphLabelVerbs {
  readonly predict: (x: TensorLike) => Tensor
  readonly predictProba: (x: TensorLike) => Tensor
  readonly classes: number[]
  readonly nFeaturesIn: number
}

/** A fitted label propagation or spreading model. */
export type FittedGraphLabel = ExplainedFitted<GraphLabelParams, GraphLabelVerbs>

/** The state of the label-propagation iteration. */
export interface GraphLabelState {
  /** Label distributions `[n * k]`, row-major. */
  readonly f: Float64Array
  /** Total absolute change at the last step. */
  readonly change: number
}

/** The dense affinity of the training rows, `[n * n]` row-major: rbf kernel values or knn connectivity (with self). */
function affinity(m: MatrixView, h: Pick<GraphLabelHyper, 'kernel' | 'gamma' | 'nNeighbors'>, x: TensorLike) {
  if (h.kernel === 'rbf') return Float64Array.from(toFlat(pairwiseKernels(x, undefined, 'rbf', { gamma: h.gamma })))
  const k = Math.min(h.nNeighbors, m.n)
  return Float64Array.from(toFlat(toDense(kNeighboursGraph(x, k, { includeSelf: true }))))
}

/**
 * The graph each variant iterates with (scikit-learn's `_build_graph`): propagation uses the affinity divided by its
 * column sums, spreading the symmetrically normalised affinity D^-1/2 W D^-1/2 with self-loops removed.
 */
function graphMatrix(w: Float64Array, n: number, variant: 'propagation' | 'spreading'): Float64Array {
  const g = Float64Array.from(w)
  if (variant === 'propagation') {
    const col = new Float64Array(n)
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) col[j] = (col[j] ?? 0) + (w[i * n + j] ?? 0)
    for (let i = 0; i < n; i++) {
      const s = col[i] ?? 0
      if (s > 0) for (let j = 0; j < n; j++) g[i * n + j] = (w[i * n + j] ?? 0) / s
    }
    return g
  }
  for (let i = 0; i < n; i++) g[i * n + i] = 0
  const deg = new Float64Array(n)
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) deg[j] = (deg[j] ?? 0) + (g[i * n + j] ?? 0)
  const r = Float64Array.from(deg, (d) => (d === 0 ? 1 : Math.sqrt(d)))
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) g[i * n + j] = (g[i * n + j] ?? 0) / ((r[i] ?? 1) * (r[j] ?? 1))
  return g
}

/**
 * The label-propagation iteration as a traced algorithm (Zhou et al. 2004; Zhu and Ghahramani 2002), as scikit-learn
 * runs it: propagation multiplies the label distributions by the row-normalised graph, renormalises and clamps the
 * labelled rows to their labels; spreading takes alpha of each row from its neighbours and 1 - alpha from its own
 * label. Stops when the total change falls below tol.
 *
 * @param graph - The iteration matrix `[n * n]`.
 * @param codes - A class code per row (-1 unlabelled).
 * @param k - The number of classes.
 * @param options - The variant, alpha and tol.
 * @returns The algorithm.
 */
export function graphLabelAlgorithm(
  graph: Float64Array,
  codes: readonly number[],
  k: number,
  options: { readonly variant: 'propagation' | 'spreading'; readonly alpha: number; readonly tol: number },
): Algorithm<GraphLabelState> {
  const n = codes.length
  const { variant, alpha, tol } = options
  const y0 = new Float64Array(n * k)
  for (const [i, c] of codes.entries()) if (c >= 0) y0[i * k + c] = 1
  const yStatic = variant === 'spreading' ? Float64Array.from(y0, (v) => v * (1 - alpha)) : y0
  return {
    name: `label-${variant}`,
    config: { variant, alpha, tol, rows: n, classes: k },
    schema: channels({
      metrics: { change: { description: 'total change of the label distributions', direction: 'lower' } },
    }),
    init: () => ({ f: Float64Array.from(y0), change: Number.POSITIVE_INFINITY }),
    step: (s) => {
      const f = new Float64Array(n * k)
      for (let i = 0; i < n; i++)
        for (let j = 0; j < n; j++) {
          const w = graph[i * n + j] ?? 0
          if (w === 0) continue
          for (let c = 0; c < k; c++) f[i * k + c] = (f[i * k + c] ?? 0) + w * (s.f[j * k + c] ?? 0)
        }
      for (let i = 0; i < n; i++) {
        if (variant === 'propagation') {
          if ((codes[i] ?? UNLABELLED) >= 0) {
            for (let c = 0; c < k; c++) f[i * k + c] = yStatic[i * k + c] ?? 0
            continue
          }
          let z = 0
          for (let c = 0; c < k; c++) z += f[i * k + c] ?? 0
          if (z > 0) for (let c = 0; c < k; c++) f[i * k + c] = (f[i * k + c] ?? 0) / z
        } else for (let c = 0; c < k; c++) f[i * k + c] = alpha * (f[i * k + c] ?? 0) + (yStatic[i * k + c] ?? 0)
      }
      let change = 0
      for (let i = 0; i < f.length; i++) change += Math.abs((f[i] ?? 0) - (s.f[i] ?? 0))
      return { f, change }
    },
    done: (s) => (s.change < tol ? 'converged' : undefined),
    observe: (s) => ({ metrics: Number.isFinite(s.change) ? { change: s.change } : {} }),
  }
}

/** Each row's most probable class code, or -1 when it holds no label mass yet. */
function assignment(f: Float64Array, n: number, k: number): number[] {
  return Array.from({ length: n }, (_, i) => {
    const row = f.subarray(i * k, (i + 1) * k)
    let z = 0
    for (const v of row) z += v
    return z > 0 ? argmaxOf(row) : UNLABELLED
  })
}

/** Evenly thinned snapshot indices 0..last, at most `max` of them, always keeping both ends. */
export function thinnedSteps(last: number, max: number): number[] {
  if (last + 1 <= max) return Array.from({ length: last + 1 }, (_, t) => t)
  const out = new Set<number>()
  for (let j = 0; j < max; j++) out.add(Math.round((j * last) / (max - 1)))
  return [...out].sort((a, b) => a - b)
}

/** Predicted class probabilities of new rows (scikit-learn's `predict_proba`): kernel- or neighbour-weighted sums. */
function graphProba(p: GraphLabelParams, x: TensorLike): Tensor {
  const d = p.x[0]?.length ?? 0
  const q = matrixView(x, 'labelPropagation', { nFeatures: d })
  const k = p.classes.length
  const out = new Float64Array(q.n * k)
  if (p.kernel === 'rbf') {
    const w = toFlat(pairwiseKernels(x, p.x, 'rbf', { gamma: p.gamma }))
    const n = p.x.length
    for (let i = 0; i < q.n; i++)
      for (let j = 0; j < n; j++) {
        const wij = w[i * n + j] ?? 0
        if (wij !== 0)
          for (let c = 0; c < k; c++) out[i * k + c] = (out[i * k + c] ?? 0) + wij * (p.labelDistributions[j]?.[c] ?? 0)
      }
  } else {
    const nb = toFlat(kNearest(p.x, x, Math.min(p.nNeighbors, p.x.length)).indices)
    const kk = Math.min(p.nNeighbors, p.x.length)
    for (let i = 0; i < q.n; i++)
      for (let a = 0; a < kk; a++) {
        const j = nb[i * kk + a] ?? 0
        for (let c = 0; c < k; c++) out[i * k + c] = (out[i * k + c] ?? 0) + (p.labelDistributions[j]?.[c] ?? 0)
      }
  }
  for (let i = 0; i < q.n; i++) {
    let z = 0
    for (let c = 0; c < k; c++) z += out[i * k + c] ?? 0
    for (let c = 0; c < k; c++) out[i * k + c] = z > 0 ? (out[i * k + c] ?? 0) / z : 1 / k
  }
  return fromStorage(out, [q.n, k], 'float64')
}

function graphVerbs(p: GraphLabelParams): GraphLabelVerbs {
  return {
    predictProba: (x) => graphProba(p, x),
    predict: (x) => {
      const pr = graphProba(p, x)
      const k = p.classes.length
      const flat = toFlat(pr)
      const m = pr.shape[0] ?? 0
      return fromStorage(
        Float64Array.from({ length: m }, (_, i) => p.classes[argmaxOf(flat.slice(i * k, (i + 1) * k))] ?? 0),
        [m],
        'float64',
      )
    },
    classes: p.classes,
    nFeaturesIn: p.x[0]?.length ?? 0,
  }
}

function fitGraphLabels(
  variant: 'propagation' | 'spreading',
  data: LearnData,
  h: GraphLabelHyper,
  options: FitOptions,
): { params: GraphLabelParams; exposition: Exposition[] } {
  const who = `label${variant === 'propagation' ? 'Propagation' : 'Spreading'}`
  const m = matrixView(data.x, who)
  if (!(variant === 'propagation' || (h.alpha > 0 && h.alpha < 1)))
    throw new WhiteboxError('value', `${who}: alpha must be in (0, 1)`)
  const { classes, codes } = semiSupervisedCodes(data.y ?? [], who, m.n)
  const k = classes.length
  const graph = graphMatrix(affinity(m, h, data.x), m.n, variant)
  const snapshots: number[][] = []
  const changes: number[] = []
  const algorithm = graphLabelAlgorithm(graph, codes, k, { variant, alpha: h.alpha, tol: h.tol })
  // Keep every iterate's assignment (cheap), thinned for the record afterwards.
  const tracked: Algorithm<GraphLabelState> = {
    ...algorithm,
    init: (key) => {
      const s = algorithm.init(key)
      snapshots.push(assignment(s.f, m.n, k))
      changes.push(Number.NaN)
      return s
    },
    step: (s, ctx) => {
      const next = algorithm.step(s, ctx)
      snapshots.push(assignment(next.f, m.n, k))
      changes.push(next.change)
      return next
    },
  }
  const run = runAlgorithm(tracked, { ...fitRunOptions(options, who), maxSteps: h.maxIter })
  const f = run.state.f
  const dist = Array.from({ length: m.n }, (_, i) => {
    const row = Array.from(f.subarray(i * k, (i + 1) * k))
    const z = row.reduce((a, v) => a + v, 0)
    return z > 0 ? row.map((v) => v / z) : row
  })
  const x = Array.from({ length: m.n }, (_, i) => Array.from(m.data.subarray(i * m.d, (i + 1) * m.d)))
  const params: GraphLabelParams = {
    classes,
    x,
    labelDistributions: dist,
    transduction: dist.map((r) => classes[argmaxOf(r)] ?? 0),
    nIter: run.steps,
    kernel: h.kernel,
    gamma: h.gamma,
    nNeighbors: h.nNeighbors,
  }
  const keep = thinnedSteps(snapshots.length - 1, h.maxFrames)
  const steps: ClusteringSteps = clusteringSteps({
    method: `label-${variant}`,
    assignments: keep.map((t) => snapshots[t] ?? []),
    objective: keep.map((t) => (t === 0 ? (changes[1] ?? 0) : (changes[t] ?? 0))),
    // The labelled rows: those with a label from the start.
    seeds: (snapshots[0] ?? []).flatMap((c, i) => (c >= 0 ? [i] : [])),
  })
  const verbs = graphVerbs(params)
  const surface =
    m.d === 2
      ? [
          decisionSurfaceOf({ estimator: who, kind: 'classifier', params, ...verbs }, data.x, {
            featureNames: featureNamesOf(data, m.d),
            classNames: classNamesOf(data, classes),
          }),
        ]
      : []
  return { params, exposition: [steps, ...surface] }
}

function graphEstimator(
  variant: 'propagation' | 'spreading',
  hyper: Partial<GraphLabelHyper>,
): ExplainedEstimator<GraphLabelHyper, FittedGraphLabel> {
  const name = variant === 'propagation' ? 'label-propagation' : 'label-spreading'
  return defineExplainedEstimator<GraphLabelHyper, GraphLabelParams, GraphLabelVerbs>(
    {
      name,
      kind: 'classifier',
      hyperSpace: (h: GraphLabelHyper): Space =>
        dict({
          kernel: choice(['rbf', 'knn'], { default: h.kernel }),
          gamma: real(0.1, 100, { log: true, default: h.gamma, label: 'rbf gamma' }),
          nNeighbors: int(1, 30, { default: h.nNeighbors, label: 'knn neighbours' }),
          ...(variant === 'spreading' ? { alpha: real(0.01, 0.99, { default: h.alpha, label: 'alpha' }) } : {}),
          maxIter: int(1, 1000, { default: h.maxIter, label: 'iterations' }),
        }),
      fit: (data, h, options) => fitGraphLabels(variant, data, h, options),
      methods: graphVerbs,
    },
    {
      kernel: hyper.kernel ?? 'rbf',
      gamma: hyper.gamma ?? 20,
      nNeighbors: hyper.nNeighbors ?? 7,
      alpha: hyper.alpha ?? 0.2,
      maxIter: hyper.maxIter ?? (variant === 'propagation' ? 1000 : 30),
      tol: hyper.tol ?? 1e-3,
      maxFrames: hyper.maxFrames ?? 60,
    },
  )
}

/**
 * Label propagation (sklearn.semi_supervised.LabelPropagation; Zhu and Ghahramani 2002): labels flow from the labelled
 * rows (targets -1 are unlabelled) along a similarity graph, the labelled rows clamped to their labels, until the
 * distributions settle. Exposition: a `clusteringSteps` record (each row's current label per iteration, -1 while it
 * holds no label mass, and the change per iteration) and, for two features, the decision surface.
 *
 * @param hyper - Kernel, gamma, neighbours, iteration cap, tolerance and snapshot cap.
 * @returns The estimator.
 */
export function labelPropagation(
  hyper: Partial<GraphLabelHyper> = {},
): ExplainedEstimator<GraphLabelHyper, FittedGraphLabel> {
  return graphEstimator('propagation', hyper)
}

/**
 * Label spreading (sklearn.semi_supervised.LabelSpreading; Zhou et al. 2004): like label propagation on the
 * normalised graph Laplacian, but soft: each step keeps 1 - alpha of every row's own label, so noisy labels can be
 * overruled. Exposition as {@link labelPropagation}.
 *
 * @param hyper - Kernel, gamma, neighbours, alpha, iteration cap, tolerance and snapshot cap.
 * @returns The estimator.
 */
export function labelSpreading(
  hyper: Partial<GraphLabelHyper> = {},
): ExplainedEstimator<GraphLabelHyper, FittedGraphLabel> {
  return graphEstimator('spreading', hyper)
}

/** {@link labelPropagation} as a class. */
export class LabelPropagation extends ExplainedEstimatorClass<GraphLabelHyper, FittedGraphLabel> {
  /** @param hyper - As {@link labelPropagation}. */
  constructor(hyper: Partial<GraphLabelHyper> = {}) {
    super(labelPropagation(hyper))
  }
}

/** {@link labelSpreading} as a class. */
export class LabelSpreading extends ExplainedEstimatorClass<GraphLabelHyper, FittedGraphLabel> {
  /** @param hyper - As {@link labelSpreading}. */
  constructor(hyper: Partial<GraphLabelHyper> = {}) {
    super(labelSpreading(hyper))
  }
}
