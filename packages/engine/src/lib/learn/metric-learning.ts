/**
 * Metric learning: a linear map `L` such that Euclidean distance after `L` serves nearest-neighbour classification.
 * Neighbourhood components analysis (Goldberger et al.) maximises the expected leave-one-out accuracy of a soft
 * nearest-neighbour rule, `ncaValueAndGradient` giving the objective and its exact gradient and the fit running
 * L-BFGS on it as an algorithm; large margin nearest neighbour (Weinberger and Saul) pulls each point towards its
 * target neighbours of the same class and pushes impostors of other classes beyond a margin, by gradient descent on
 * `L`. Both are transformers on the iterative base: a session logs the objective and the map step by step and
 * `fittedAt(t)` is the embedding at step `t`.
 */

import { customVjp } from '../autodiff/index.ts'
import { MdmlError } from '../base/index.ts'
import { dgemm, eigh } from '../numerics/linalg/index.ts'
import { type LbfgsState, lbfgsAlgorithm } from '../optim/index.ts'
import { foldIn, type Key, normal } from '../random/index.ts'
import { choice, dict, int, real } from '../space/index.ts'
import {
  fromStorage,
  mul,
  primalOf,
  type Tensor,
  type TensorLike,
  tensor,
  toFlat,
  type Value,
} from '../tensor/index.ts'
import { type Algorithm, channels, Log, stateFromJson, stateToJson } from '../trace/index.ts'
import { type IterativeEstimator, iterative } from './bases/index.ts'
import { fitDiscriminant } from './discriminant.ts'
import type { LearnData } from './estimator.ts'
import { encodeLabels, type MatrixView, matrixView } from './validation.ts'

/** The value and gradient of the NCA objective at a map. */
export interface NcaEvaluation {
  /** The expected number of correctly classified points under the soft neighbour rule, `sum_i p_i`. */
  readonly value: number
  /** `d value / d A`, row-major `[nComponents][nFeatures]`. */
  readonly gradient: Float64Array
  /** The soft-neighbour probability `p_i` of each point's own class. */
  readonly probabilities: Float64Array
}

/**
 * The NCA objective and its gradient: with `z_i = A x_i`, `p_ij` the softmax of `-|z_i - z_j|^2` over `j != i`, and
 * `p_i = sum over j of the same class of p_ij`, the objective is `sum_i p_i` and its gradient is
 * `2 A sum_ij (p_i p_ij - [same class] p_ij) (x_i - x_j)(x_i - x_j)'`.
 *
 * @param a - The map `[nComponents][nFeatures]`, row-major.
 * @param nComponents - Its number of rows.
 * @param x - The points.
 * @param codes - The class index of each point.
 * @returns The value, gradient and per-point probabilities.
 * @example
 * ```ts
 * import { matrixView, ncaValueAndGradient } from 'mdml-engine/learn'
 * const x = matrixView([[0], [0.1], [5], [5.1]], 'demo')
 * const r = ncaValueAndGradient([1], 1, x, [0, 0, 1, 1])
 * r.value // => 3.999999999858
 * ```
 */
export function ncaValueAndGradient(
  a: ArrayLike<number>,
  nComponents: number,
  x: MatrixView,
  codes: ArrayLike<number>,
): NcaEvaluation {
  const { n, d, data } = x
  const dp = nComponents
  const z = dgemm(data, a, n, d, dp, { transposeB: true })
  const p = new Float64Array(n * n)
  for (let i = 0; i < n; i++) {
    let lowest = Number.POSITIVE_INFINITY
    for (let j = 0; j < n; j++) {
      if (j === i) continue
      let s = 0
      for (let c = 0; c < dp; c++) s += ((z[i * dp + c] ?? 0) - (z[j * dp + c] ?? 0)) ** 2
      p[i * n + j] = s
      lowest = Math.min(lowest, s)
    }
    let total = 0
    for (let j = 0; j < n; j++) {
      if (j === i) continue
      const e = Math.exp(-((p[i * n + j] ?? 0) - lowest))
      p[i * n + j] = e
      total += e
    }
    for (let j = 0; j < n; j++) p[i * n + j] = j === i ? 0 : (p[i * n + j] ?? 0) / total
  }
  const own = new Float64Array(n)
  let value = 0
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) if (j !== i && codes[j] === codes[i]) own[i] = (own[i] ?? 0) + (p[i * n + j] ?? 0)
    value += own[i] ?? 0
  }
  // The gradient is 2 A X' (D - (M + M')) X for M_ij = p_i p_ij - [same class] p_ij, D the row-plus-column sums of M.
  const m = new Float64Array(n * n)
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++)
      m[i * n + j] = (own[i] ?? 0) * (p[i * n + j] ?? 0) - (codes[j] === codes[i] && j !== i ? (p[i * n + j] ?? 0) : 0)
  const laplacian = new Float64Array(n * n)
  for (let i = 0; i < n; i++) {
    let rows = 0
    let cols = 0
    for (let j = 0; j < n; j++) {
      rows += m[i * n + j] ?? 0
      cols += m[j * n + i] ?? 0
      laplacian[i * n + j] = -((m[i * n + j] ?? 0) + (m[j * n + i] ?? 0))
    }
    laplacian[i * n + i] = (laplacian[i * n + i] ?? 0) + rows + cols
  }
  const lx = dgemm(laplacian, data, n, n, d)
  const t = dgemm(data, lx, d, n, d, { transposeA: true })
  const gradient = dgemm(a, t, dp, d, d).map((v) => 2 * v)
  return { value, gradient, probabilities: own }
}

/** How a metric-learning map starts. */
export type MetricInit = 'auto' | 'identity' | 'pca' | 'lda' | 'random' | readonly (readonly number[])[]

const unflatten = (flat: ArrayLike<number>, rows: number, cols: number): number[][] =>
  Array.from({ length: rows }, (_, i) => Array.from({ length: cols }, (_, j) => flat[i * cols + j] ?? 0))

/** The starting map of a metric learner, as scikit-learn's `NeighborhoodComponentsAnalysis._initialize`. */
function initialMap(
  init: MetricInit,
  x: MatrixView,
  codes: Int32Array,
  classes: readonly number[],
  dp: number,
  key: Key,
): Float64Array {
  const { n, d } = x
  if (typeof init !== 'string') {
    if (init.length !== dp || init.some((r) => r.length !== d)) {
      throw new MdmlError('shape', `metric learning: the initial map must be [${dp}, ${d}]`)
    }
    return Float64Array.from(init.flat())
  }
  let kind = init
  if (kind === 'auto') kind = dp <= Math.min(d, classes.length - 1) ? 'lda' : dp < Math.min(d, n) ? 'pca' : 'identity'
  if (kind === 'identity')
    return Float64Array.from({ length: dp * d }, (_, ij) => (Math.floor(ij / d) === ij % d ? 1 : 0))
  if (kind === 'random') return Float64Array.from(toFlat(normal(key, [dp, d])))
  if (kind === 'lda') {
    const fit = fitDiscriminant(x, codes, classes, {
      covariance: 'pooled',
      shrinkage: 0,
      regParam: 0,
      mix: 1,
      unbiased: false,
      rankTolerance: 1e-10,
      nComponents: dp,
    })
    if (fit.scalings.length === 0 || (fit.scalings[0]?.length ?? 0) < dp) {
      throw new MdmlError(
        'value',
        `metric learning: the LDA initialisation has only ${fit.scalings[0]?.length ?? 0} axes for ${dp} components`,
      )
    }
    return Float64Array.from({ length: dp * d }, (_, ij) => fit.scalings[ij % d]?.[Math.floor(ij / d)] ?? 0)
  }
  // pca: the leading eigenvectors of the covariance as rows.
  const mean = new Float64Array(d)
  for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) mean[j] = (mean[j] ?? 0) + (x.data[i * d + j] ?? 0) / n
  const centred = Float64Array.from(x.data, (v, ij) => v - (mean[ij % d] ?? 0))
  const cov = dgemm(centred, centred, d, n, d, { transposeA: true })
  const e = eigh(fromStorage(cov, [d, d], 'float64'))
  const vectors = toFlat(primalOf(e.eigenvectors))
  const out = new Float64Array(dp * d)
  for (let r = 0; r < dp; r++) {
    const col = d - 1 - r
    let big = 0
    for (let j = 1; j < d; j++) if (Math.abs(vectors[j * d + col] ?? 0) > Math.abs(vectors[big * d + col] ?? 0)) big = j
    const sign = (vectors[big * d + col] ?? 0) < 0 ? -1 : 1
    for (let j = 0; j < d; j++) out[r * d + j] = sign * (vectors[j * d + col] ?? 0)
  }
  return out
}

/** The hyperparameters of {@link neighbourhoodComponentsAnalysis}. */
export interface NcaHyper {
  /** The output dimension (default the number of features). */
  readonly nComponents?: number
  readonly init: MetricInit
  /** The L-BFGS iteration cap. */
  readonly maxIter: number
  /** Stop when the gradient's largest entry falls to this. */
  readonly gradTol: number
}

/** The fitted parameters of a learnt linear map. */
export type LinearMapParams = {
  /** The map `[nComponents][nFeatures]`. */
  readonly components: number[][]
  /** The objective at this map (NCA: expected correct count, to maximise; LMNN: the loss, to minimise). */
  readonly objective: number
}

/** What a fitted metric learner offers. */
export interface LinearMapVerbs {
  readonly nFeaturesIn: number
  /** `x A'`: the points in the learnt metric. */
  readonly transform: (x: TensorLike) => Tensor
  readonly components: number[][]
}

function mapVerbs(p: LinearMapParams): LinearMapVerbs {
  const dp = p.components.length
  const d = p.components[0]?.length ?? 0
  const a = Float64Array.from(p.components.flat())
  return {
    nFeaturesIn: d,
    components: p.components,
    transform: (x) => {
      const m = matrixView(x, 'metric learning', { nFeatures: d })
      return fromStorage(dgemm(m.data, a, m.n, d, dp, { transposeB: true }), [m.n, dp], 'float64')
    },
  }
}

function trainingOf(
  data: LearnData,
  who: string,
): { readonly x: MatrixView; readonly classes: readonly number[]; readonly codes: Int32Array } {
  const x = matrixView(data.x, who)
  const { classes, codes } = encodeLabels(data.y ?? [], who, x.n)
  return { x, classes, codes }
}

/**
 * The NCA fit as an algorithm: L-BFGS on the negative objective, one iteration per step, from a starting map. It logs
 * the metrics `objective` (the expected number of correct points, rising) and `gradNorm`, and the tensor `transform`
 * (the map, `[nComponents, nFeatures]`).
 *
 * @param data - Features and class labels.
 * @param hyper - `nComponents`, `init`, `maxIter`, `gradTol`.
 * @param key - Draws the `random` start.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { ncaAlgorithm } from 'mdml-engine/learn'
 * import { key } from 'mdml-engine/random'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const x = [[0, 5], [0.2, 1], [0.1, -3], [3, 4], [3.1, 0], [2.9, -4]]
 * const alg = ncaAlgorithm({ x, y: [0, 0, 0, 1, 1, 1] }, { init: 'identity', maxIter: 20, gradTol: 1e-6, nComponents: 1 }, key(0))
 * runAlgorithm(alg, { maxSteps: 30 }).state.value < -4 // => true
 * ```
 */
export function ncaAlgorithm(data: LearnData, hyper: NcaHyper, key: Key): Algorithm<LbfgsState> {
  const { x, classes, codes } = trainingOf(data, 'nca')
  const dp = hyper.nComponents ?? x.d
  const start = initialMap(hyper.init, x, codes, classes, dp, foldIn(key, 7))
  const loss = customVjp({
    name: 'nca',
    forward: (a) => {
      const r = ncaValueAndGradient(toFlat(primalOf(a as Value)), dp, x, codes)
      return tensor(-r.value)
    },
    vjp: (g, [a]) => {
      const r = ncaValueAndGradient(toFlat(primalOf(a as Value)), dp, x, codes)
      return [
        mul(
          g,
          fromStorage(
            r.gradient.map((v) => -v),
            [dp, x.d],
            'float64',
          ),
        ),
      ]
    },
  })
  const inner = lbfgsAlgorithm((a) => loss(a as Value), fromStorage(start, [dp, x.d], 'float64'), {
    gradTol: hyper.gradTol,
  })
  return {
    ...inner,
    name: 'nca',
    config: { nComponents: dp, maxIter: hyper.maxIter, gradTol: hyper.gradTol },
    schema: channels({
      metrics: {
        objective: {
          description: 'expected number of points classified right by the soft neighbour rule',
          direction: 'higher',
        },
        gradNorm: { description: 'largest entry of the objective’s gradient', direction: 'lower' },
      },
      tensors: { transform: { role: 'param', description: 'the linear map', shape: [dp, x.d] } },
    }),
    observe: (s) => ({
      metrics: { objective: -s.value, gradNorm: s.grad.reduce((m, v) => Math.max(m, Math.abs(v)), 0) },
      tensors: { transform: { role: 'param', value: s.x, shape: [dp, x.d] } },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as LbfgsState,
  }
}

const NCA_DEFAULTS: NcaHyper = { init: 'auto', maxIter: 50, gradTol: 1e-5 }

/**
 * Neighbourhood components analysis (`sklearn.neighbors.NeighborhoodComponentsAnalysis`): a linear map learnt by
 * maximising the expected leave-one-out accuracy of the soft nearest-neighbour rule, started from the LDA axes, the
 * principal components, the identity or a random map. `transform` embeds points in the learnt metric (to hand to a
 * k-nearest-neighbour classifier, or to draw the neighbourhoods sharpening step by step).
 *
 * @param hyper - `nComponents`, `init`, `maxIter`, `gradTol`.
 * @returns The estimator.
 * @reference J. Goldberger, S. Roweis, G. Hinton and R. Salakhutdinov, Neighbourhood components analysis, NIPS (2004).
 * @example
 * ```ts
 * import { neighbourhoodComponentsAnalysis } from 'mdml-engine/learn'
 * const x = [[0, 5], [0.2, 1], [0.1, -3], [3, 4], [3.1, 0], [2.9, -4]]
 * const f = neighbourhoodComponentsAnalysis({ nComponents: 1, init: 'identity' }).fit({ x, y: [0, 0, 0, 1, 1, 1] })
 * f.params.objective > 5.5 // => true
 * ```
 */
export function neighbourhoodComponentsAnalysis(
  hyper: Partial<NcaHyper> = {},
): IterativeEstimator<NcaHyper, LbfgsState, LinearMapParams, LinearMapVerbs> {
  return iterative<NcaHyper, LbfgsState, LinearMapParams, LinearMapVerbs>(
    {
      name: 'nca',
      kind: 'transformer',
      defaults: NCA_DEFAULTS,
      hyperSpace: (h) =>
        dict({
          gradTol: real(Math.min(1e-9, h.gradTol), Math.max(1e-2, h.gradTol), { default: h.gradTol, log: true }),
          ...(typeof h.init === 'string'
            ? { init: choice(['auto', 'identity', 'pca', 'lda', 'random'] as const, { default: h.init }) }
            : {}),
        }),
      algorithm: (data, h, key) => ncaAlgorithm(data, h, key),
      maxSteps: (h) => h.maxIter,
      params: (s, data, h) => {
        const dp = h.nComponents ?? matrixView(data.x, 'nca').d
        const d = matrixView(data.x, 'nca').d
        return { components: unflatten(s.x, dp, d), objective: -s.value }
      },
      verbs: mapVerbs,
      logs: { 'solver/objective': Log(), 'solver/gradNorm': Log(), 'solver/transform': Log({ detail: 'full' }) },
    },
    { ...NCA_DEFAULTS, ...hyper },
  )
}

/** The hyperparameters of {@link largeMarginNearestNeighbour}. */
export interface LmnnHyper {
  /** Target neighbours per point. */
  readonly nNeighbors: number
  /** The weight of the push (impostor) term against the pull term, in `(0, 1)`. */
  readonly mu: number
  readonly nComponents?: number
  readonly init: Exclude<MetricInit, 'auto'> | 'auto'
  readonly maxIter: number
  /** The initial step. */
  readonly learningRate: number
  /** Stop when the loss changes by less than this, relatively, in a step. */
  readonly tol: number
}

/** The state of the LMNN algorithm. */
export type LmnnState = {
  /** The map `[nComponents][nFeatures]`, row-major. */
  readonly a: number[]
  readonly loss: number
  readonly gradient: number[]
  /** Triplets `(i, j, l)` whose impostor `l` is within the margin of target neighbour `j` of `i`. */
  readonly impostors: number
  readonly step: number
  /** Whether the last step was accepted. */
  readonly accepted: boolean
  readonly change: number
}

/** The pull-and-push loss of LMNN and its gradient at a map, with the number of active impostors. */
function lmnnEvaluate(
  a: ArrayLike<number>,
  dp: number,
  x: MatrixView,
  codes: ArrayLike<number>,
  targets: readonly (readonly number[])[],
  mu: number,
): { readonly loss: number; readonly gradient: number[]; readonly impostors: number } {
  const { n, d, data } = x
  const z = dgemm(data, a, n, d, dp, { transposeB: true })
  const dist = (i: number, j: number): number => {
    let s = 0
    for (let c = 0; c < dp; c++) s += ((z[i * dp + c] ?? 0) - (z[j * dp + c] ?? 0)) ** 2
    return s
  }
  const weights = new Float64Array(n * n) // coefficient of x_ij x_ij' in the gradient, antisymmetrised below
  let pull = 0
  let push = 0
  let impostors = 0
  for (let i = 0; i < n; i++)
    for (const j of targets[i] ?? []) {
      const dij = dist(i, j)
      pull += dij
      weights[i * n + j] = (weights[i * n + j] ?? 0) + (1 - mu)
      for (let l = 0; l < n; l++) {
        if (codes[l] === codes[i]) continue
        const slack = 1 + dij - dist(i, l)
        if (slack > 0) {
          push += slack
          impostors++
          weights[i * n + j] = (weights[i * n + j] ?? 0) + mu
          weights[i * n + l] = (weights[i * n + l] ?? 0) - mu
        }
      }
    }
  // sum_ij w_ij (x_i - x_j)(x_i - x_j)' = X' (diag(r + c) - (W + W')) X
  const laplacian = new Float64Array(n * n)
  for (let i = 0; i < n; i++) {
    let rows = 0
    let cols = 0
    for (let j = 0; j < n; j++) {
      rows += weights[i * n + j] ?? 0
      cols += weights[j * n + i] ?? 0
      laplacian[i * n + j] = -((weights[i * n + j] ?? 0) + (weights[j * n + i] ?? 0))
    }
    laplacian[i * n + i] = (laplacian[i * n + i] ?? 0) + rows + cols
  }
  const t = dgemm(data, dgemm(laplacian, data, n, n, d), d, n, d, { transposeA: true })
  return { loss: (1 - mu) * pull + mu * push, gradient: Array.from(dgemm(a, t, dp, d, d), (v) => 2 * v), impostors }
}

/**
 * Large margin nearest neighbour as an algorithm (Weinberger and Saul): each point's `k` nearest same-class points
 * (in the starting metric) are its target neighbours; the loss pulls them close (`(1 - mu)` times the sum of squared
 * distances after the map) and pushes impostors, points of other classes that intrude within a unit margin of a
 * target neighbour, away (`mu` times the hinge `[1 + d_ij - d_il]_+`). Gradient descent on the map with an adaptive
 * step (grown by 1% on a decrease of the loss, halved and the step rejected otherwise). It logs the metrics
 * `objective` (the loss), `impostors` and `step`, and the tensor `transform`.
 *
 * @param data - Features and class labels.
 * @param hyper - `nNeighbors`, `mu`, `nComponents`, `init`, `maxIter`, `learningRate`, `tol`.
 * @param key - Draws a `random` start.
 * @returns The algorithm.
 * @reference K. Weinberger and L. Saul, Distance metric learning for large margin nearest neighbor classification,
 * JMLR 10 (2009).
 * @example
 * ```ts
 * import { lmnnAlgorithm } from 'mdml-engine/learn'
 * import { key } from 'mdml-engine/random'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const x = [[0, 5], [0.2, 1], [0.1, -3], [3, 4], [3.1, 0], [2.9, -4]]
 * const alg = lmnnAlgorithm({ x, y: [0, 0, 0, 1, 1, 1] }, { nNeighbors: 1, mu: 0.5, init: 'identity', maxIter: 50, learningRate: 1e-3, tol: 1e-9 }, key(0))
 * const end = runAlgorithm(alg, { maxSteps: 60 }).state
 * end.loss < alg.init(key(0)).loss // => true
 * ```
 */
export function lmnnAlgorithm(data: LearnData, hyper: LmnnHyper, key: Key): Algorithm<LmnnState> {
  const { x, classes, codes } = trainingOf(data, 'lmnn')
  const dp = hyper.nComponents ?? x.d
  const k = Math.min(hyper.nNeighbors, x.n - 1)
  const start = initialMap(hyper.init, x, codes, classes, dp, foldIn(key, 7))
  // Target neighbours: the k nearest of the same class under the starting metric.
  const z0 = dgemm(x.data, start, x.n, x.d, dp, { transposeB: true })
  const targets = Array.from({ length: x.n }, (_, i) => {
    const same = Array.from({ length: x.n }, (_, j) => j).filter((j) => j !== i && codes[j] === codes[i])
    const dist = (j: number): number => {
      let s = 0
      for (let c = 0; c < dp; c++) s += ((z0[i * dp + c] ?? 0) - (z0[j * dp + c] ?? 0)) ** 2
      return s
    }
    return same.sort((p, q) => dist(p) - dist(q) || p - q).slice(0, k)
  })
  const at = (a: ArrayLike<number>, step: number, accepted: boolean, change: number): LmnnState => {
    const e = lmnnEvaluate(a, dp, x, codes, targets, hyper.mu)
    return { a: Array.from(a), loss: e.loss, gradient: e.gradient, impostors: e.impostors, step, accepted, change }
  }
  return {
    name: 'lmnn',
    config: { nNeighbors: k, mu: hyper.mu, nComponents: dp, learningRate: hyper.learningRate },
    schema: channels({
      metrics: {
        objective: {
          description: 'the LMNN loss: weighted pull of target neighbours plus hinge of impostors',
          direction: 'lower',
        },
        impostors: { description: 'impostor triplets inside the margin', direction: 'lower' },
        step: { description: 'the gradient step size', direction: 'higher' },
      },
      tensors: { transform: { role: 'param', description: 'the linear map', shape: [dp, x.d] } },
    }),
    init: () => at(start, hyper.learningRate, true, Number.POSITIVE_INFINITY),
    step: (s) => {
      const proposal = s.a.map((v, i) => v - s.step * (s.gradient[i] ?? 0))
      const next = at(proposal, s.step * 1.01, true, 0)
      if (next.loss < s.loss) return { ...next, change: (s.loss - next.loss) / Math.max(Math.abs(s.loss), 1e-300) }
      return { ...s, step: s.step * 0.5, accepted: false, change: Number.POSITIVE_INFINITY }
    },
    done: (s) => (s.accepted && s.change < hyper.tol ? 'converged' : s.step < 1e-14 ? 'converged' : undefined),
    observe: (s) => ({
      metrics: { objective: s.loss, impostors: s.impostors, step: s.step },
      tensors: { transform: { role: 'param', value: s.a, shape: [dp, x.d] } },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as LmnnState,
  }
}

const LMNN_DEFAULTS: LmnnHyper = {
  nNeighbors: 3,
  mu: 0.5,
  init: 'identity',
  maxIter: 200,
  learningRate: 1e-3,
  tol: 1e-7,
}

/**
 * Large margin nearest neighbour (`metric-learn`'s LMNN): a linear map that pulls each point's target neighbours in
 * and pushes impostors beyond a margin, so that k-nearest-neighbour classification in the learnt metric improves.
 * Fitted by {@link lmnnAlgorithm}, one gradient step per iteration.
 *
 * @param hyper - `nNeighbors`, `mu`, `nComponents`, `init`, `maxIter`, `learningRate`, `tol`.
 * @returns The estimator.
 * @example
 * ```ts
 * import { largeMarginNearestNeighbour } from 'mdml-engine/learn'
 * const x = [[0, 5], [0.2, 1], [0.1, -3], [3, 4], [3.1, 0], [2.9, -4]]
 * const f = largeMarginNearestNeighbour({ nNeighbors: 1, learningRate: 1e-3 }).fit({ x, y: [0, 0, 0, 1, 1, 1] })
 * f.params.components.length // => 2
 * ```
 */
export function largeMarginNearestNeighbour(
  hyper: Partial<LmnnHyper> = {},
): IterativeEstimator<LmnnHyper, LmnnState, LinearMapParams, LinearMapVerbs> {
  return iterative<LmnnHyper, LmnnState, LinearMapParams, LinearMapVerbs>(
    {
      name: 'lmnn',
      kind: 'transformer',
      defaults: LMNN_DEFAULTS,
      hyperSpace: (h) =>
        dict({
          nNeighbors: int(1, Math.max(10, h.nNeighbors), { default: h.nNeighbors }),
          mu: real(0.01, 0.99, { default: h.mu }),
          learningRate: real(Math.min(1e-6, h.learningRate), Math.max(1, h.learningRate), {
            default: h.learningRate,
            log: true,
          }),
        }),
      algorithm: (data, h, key) => lmnnAlgorithm(data, h, key),
      maxSteps: (h) => h.maxIter,
      params: (s, data, h) => {
        const d = matrixView(data.x, 'lmnn').d
        const dp = h.nComponents ?? d
        return { components: unflatten(s.a, dp, d), objective: s.loss }
      },
      verbs: mapVerbs,
      logs: { 'solver/objective': Log(), 'solver/impostors': Log(), 'solver/transform': Log({ detail: 'full' }) },
    },
    { ...LMNN_DEFAULTS, ...hyper },
  )
}
