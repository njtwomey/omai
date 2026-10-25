import { grad } from '../autodiff/index.ts'
import { WhiteboxError } from '../base/index.ts'
import { dijkstra, graph } from '../graph/index.ts'
import { keyFor, runOptionsFor } from '../learn/_shared.ts'
import type { FitOptions } from '../learn/estimator.ts'
import { columnOf, matrixView } from '../learn/validation.ts'
import { foldIn, normal } from '../random/index.ts'
import {
  add,
  asTensor,
  div,
  fromStorage,
  log,
  mul,
  relu,
  sliceAxis,
  sqrt,
  square,
  squeeze,
  sub,
  sum,
  type Tensor,
  type TensorLike,
  toFlat,
  type Value,
} from '../tensor/index.ts'
import { type Algorithm, channels, runAlgorithm } from '../trace/index.ts'
import type { DifferentiableModel } from './gradients.ts'

function rowOf(x: TensorLike, who: string): Float64Array {
  const v = Float64Array.from(toFlat(asTensor(x)))
  if (v.length === 0) throw new WhiteboxError('shape', `${who}: empty point`)
  return v
}

/**
 * Median absolute deviations of the training features (Wachter et al.'s feature scales), 1 where a feature has none.
 *
 * @param training - Rows `[n, d]`.
 * @returns One scale per feature.
 * @example
 * ```ts
 * import * as explain from 'whitebox-engine/explain'
 * explain.medianAbsoluteDeviations([[0, 1], [1, 1], [4, 1]]) // => [1, 1]
 * ```
 */
export function medianAbsoluteDeviations(training: TensorLike): number[] {
  const m = matrixView(training, 'medianAbsoluteDeviations')
  const median = (v: Float64Array): number => {
    const s = Float64Array.from(v).sort()
    const h = (s.length - 1) / 2
    return ((s[Math.floor(h)] ?? 0) + (s[Math.ceil(h)] ?? 0)) / 2
  }
  return Array.from({ length: m.d }, (_, j) => {
    const c = columnOf(m, j)
    const med = median(c)
    const mad = median(c.map((v) => Math.abs(v - med)))
    return mad > 0 ? mad : 1
  })
}

/** Options of {@link wachterCounterfactual}. */
export interface WachterOptions extends FitOptions {
  /** The desired model output (for a probability, e.g. 0.5 + margin). */
  readonly target: number
  /** Accept when |f(x') - target| is at most this (default 0.05). */
  readonly tolerance?: number
  /** Feature scales for the weighted L1 distance (default ones; see {@link medianAbsoluteDeviations}). */
  readonly scales?: readonly number[]
  /** The starting weight of the prediction loss (default 0.1), multiplied by 2 each outer round. */
  readonly lambda?: number
  /** Outer rounds (default 20) and proximal-gradient iterations per round (default 200). */
  readonly maxRounds?: number
  readonly maxIterations?: number
  /** Features that may not change. */
  readonly immutable?: readonly number[]
}

/** A counterfactual explanation. */
export interface Counterfactual {
  /** The counterfactual point. */
  readonly x: readonly number[]
  /** The model's output there. */
  readonly prediction: number
  /** Its weighted L1 distance from the original point. */
  readonly distance: number
  /** Whether the target was reached within tolerance. */
  readonly valid: boolean
}

function scalarModel(model: DifferentiableModel, d: number): (v: Float64Array) => number {
  return (v) => toFlat(model(fromStorage(Float64Array.from(v), [1, d], 'float64')))[0] ?? 0
}

/**
 * Wachter counterfactuals (Wachter, Mittelstadt and Russell 2017): the closest point, in the MAD-weighted L1
 * distance, at which the model's output reaches a target, from argmin lambda (f(x') - y')^2 + sum_j |x'_j - x_j| / MAD_j
 * with lambda doubled until the target is met. Each subproblem is solved by proximal gradient descent (the L1 term's
 * proximal map is soft thresholding towards x, which also keeps sparse changes); immutable features are fixed. A
 * traced algorithm with one step per lambda.
 *
 * @param model - The differentiable model (one output per row; the first column is used).
 * @param x - The original point `[d]`.
 * @param options - target, tolerance, scales, lambda, maxRounds, maxIterations, immutable and tracing.
 * @returns The counterfactual.
 * @concept counterfactual-explanations
 * @example
 * ```ts
 * import * as explain from 'whitebox-engine/explain'
 * import { matmul } from 'whitebox-engine/tensor'
 * const score = (v) => matmul(v, [[1], [2]])
 * const cf = explain.wachterCounterfactual(score, [0, 0], { target: 2, tolerance: 0.01 })
 * cf.valid // => true
 * Math.abs(cf.x[0]) < 0.01 // => true
 * ```
 */
export function wachterCounterfactual(
  model: DifferentiableModel,
  x: TensorLike,
  options: WachterOptions,
): Counterfactual {
  const x0 = rowOf(x, 'wachterCounterfactual')
  const d = x0.length
  const scales = options.scales ?? new Array<number>(d).fill(1)
  if (scales.length !== d) throw new WhiteboxError('shape', 'wachterCounterfactual: one scale per feature')
  const fixed = new Set(options.immutable ?? [])
  const tol = options.tolerance ?? 0.05
  const f = scalarModel(model, d)
  const gradF = grad((v: Value) => sum(model(v)))
  const distance = (v: Float64Array): number =>
    v.reduce((s, t, j) => s + Math.abs(t - (x0[j] ?? 0)) / (scales[j] ?? 1), 0)
  type State = { readonly point: Float64Array; readonly lambda: number; readonly value: number }
  const inner = (start: Float64Array, lambda: number): Float64Array => {
    let p = Float64Array.from(start)
    let step = 1
    const objective = (v: Float64Array): number => lambda * (f(v) - options.target) ** 2 + distance(v)
    for (let it = 0; it < (options.maxIterations ?? 200); it++) {
      const fv = f(p)
      const g = toFlat(gradF(fromStorage(Float64Array.from(p), [1, d], 'float64')) as Tensor).map(
        (t) => 2 * lambda * (fv - options.target) * t,
      )
      const current = objective(p)
      let next = p
      for (let bt = 0; bt < 30; bt++) {
        next = Float64Array.from(p, (v, j) => {
          if (fixed.has(j)) return x0[j] ?? 0
          const z = v - step * (g[j] ?? 0)
          const shrink = step / (scales[j] ?? 1)
          const delta = z - (x0[j] ?? 0)
          return (x0[j] ?? 0) + Math.sign(delta) * Math.max(Math.abs(delta) - shrink, 0)
        })
        if (objective(next) <= current + 1e-15) break
        step /= 2
      }
      const moved = next.reduce((s, v, j) => Math.max(s, Math.abs(v - (p[j] ?? 0))), 0)
      p = next
      step *= 1.5
      if (moved < 1e-10) break
    }
    return p
  }
  const algorithm: Algorithm<State> = {
    name: 'wachter-counterfactual',
    config: { features: d, target: options.target },
    schema: channels({
      metrics: {
        lambda: { description: 'the weight of the prediction loss' },
        gap: { description: "|f(x') - target|", direction: 'lower' },
      },
    }),
    init: () => ({ point: Float64Array.from(x0), lambda: options.lambda ?? 0.1, value: f(x0) }),
    step: (s) => {
      const point = inner(s.point, s.lambda)
      return { point, lambda: s.lambda * 2, value: f(point) }
    },
    done: (s) => (Math.abs(s.value - options.target) <= tol ? 'converged' : undefined),
    observe: (s) => ({ metrics: { lambda: s.lambda, gap: Math.abs(s.value - options.target) } }),
  }
  const { state } = runAlgorithm(algorithm, {
    ...runOptionsFor(options, 'wachterCounterfactual'),
    maxSteps: options.maxRounds ?? 20,
  })
  return {
    x: Array.from(state.point),
    prediction: state.value,
    distance: distance(state.point),
    valid: Math.abs(state.value - options.target) <= tol,
  }
}

/** Options of {@link diverseCounterfactuals}. */
export interface DiverseOptions extends FitOptions {
  /** The number of counterfactuals (default 4). */
  readonly count?: number
  /** The desired class: outputs above 0.5 (1) or below (0) for a probability model (default 1). */
  readonly desired?: 0 | 1
  /** Weights of proximity (default 0.5) and diversity (default 1). */
  readonly proximityWeight?: number
  readonly diversityWeight?: number
  /** Feature scales for distances (default ones). */
  readonly scales?: readonly number[]
  /** Gradient iterations (default 500) and the learning rate (default 0.05). */
  readonly maxIterations?: number
  readonly learningRate?: number
  /** The initial spread around x (default 0.5). */
  readonly spread?: number
}

/** The result of {@link diverseCounterfactuals}. */
export interface DiverseCounterfactuals {
  readonly points: readonly (readonly number[])[]
  readonly predictions: readonly number[]
  /** Which reach the desired class. */
  readonly valid: readonly boolean[]
  /** det(K) of the diversity kernel K_ij = 1 / (1 + dist(c_i, c_j)): larger is more diverse. */
  readonly diversity: number
}

/**
 * Diverse counterfactuals (Mothilal, Sharma and Tan 2020, DiCE): k points found jointly by gradient descent (Adam) on
 * the hinge loss of reaching the desired class, plus their mean scaled-L1 distance to x, minus the log-determinant of
 * the kernel K_ij = 1 / (1 + dist(c_i, c_j)) (a determinantal point process that spreads them out). A traced algorithm
 * with one step per iteration.
 *
 * @param model - The differentiable model returning a probability per row.
 * @param x - The original point `[d]`.
 * @param options - count, desired, weights, scales, maxIterations, learningRate, spread, key and tracing.
 * @returns The counterfactuals, their predictions and validity, and the diversity.
 * @example
 * ```ts
 * import * as explain from 'whitebox-engine/explain'
 * import { matmul, sigmoid } from 'whitebox-engine/tensor'
 * const proba = (v) => sigmoid(matmul(v, [[2], [2]]))
 * const r = explain.diverseCounterfactuals(proba, [-1, -1], { count: 3 })
 * r.valid // => [true, true, true]
 * ```
 */
export function diverseCounterfactuals(
  model: DifferentiableModel,
  x: TensorLike,
  options: DiverseOptions = {},
): DiverseCounterfactuals {
  const x0 = rowOf(x, 'diverseCounterfactuals')
  const d = x0.length
  const k = options.count ?? 4
  if (!(Number.isInteger(k) && k >= 1)) throw new WhiteboxError('value', 'diverseCounterfactuals: count must be >= 1')
  const scales = fromStorage(Float64Array.from(options.scales ?? new Array<number>(d).fill(1)), [1, d], 'float64')
  const xt = fromStorage(
    Float64Array.from({ length: k * d }, (_, i) => x0[i % d] ?? 0),
    [k, d],
    'float64',
  )
  const desired = options.desired ?? 1
  const wProx = options.proximityWeight ?? 0.5
  const wDiv = options.diversityWeight ?? 1
  const eps = 1e-6
  const loss = (c: Value): Value => {
    const p = model(c)
    // Hinge on the logit: max(0, 1 - s z) with z = logit(p), s = +1 for class 1.
    const z = sub(log(add(p, eps)), log(add(sub(1, p), eps)))
    const margin = sub(1, mul(desired === 1 ? 1 : -1, z))
    const hinge = sum(relu(margin))
    const prox = sum(div(sqrt(add(square(sub(c, xt)), 1e-9)), scales))
    let diversity: Value = fromStorage(new Float64Array([0]), [], 'float64')
    if (k > 1) diversity = logDetKernel(c, scales, k)
    return sub(add(mul(hinge, 1 / k), mul(prox, wProx / k)), mul(diversity, wDiv))
  }
  const dl = grad(loss)
  const key = keyFor(options, 'diverseCounterfactuals')
  type State = { readonly c: Float64Array; readonly m: Float64Array; readonly v: Float64Array; readonly t: number }
  const lr = options.learningRate ?? 0.05
  const algorithm: Algorithm<State> = {
    name: 'diverse-counterfactuals',
    config: { count: k, features: d },
    schema: channels({ metrics: { loss: { description: 'the DiCE objective', direction: 'lower' } } }),
    init: () => {
      const z = toFlat(normal(foldIn(key, 0), [k, d]))
      return {
        c: Float64Array.from({ length: k * d }, (_, i) => (x0[i % d] ?? 0) + (options.spread ?? 0.5) * (z[i] ?? 0)),
        m: new Float64Array(k * d),
        v: new Float64Array(k * d),
        t: 0,
      }
    },
    step: (s) => {
      const g = toFlat(dl(fromStorage(Float64Array.from(s.c), [k, d], 'float64')) as Tensor)
      const t = s.t + 1
      const m = Float64Array.from(s.m, (mv, i) => 0.9 * mv + 0.1 * (g[i] ?? 0))
      const v = Float64Array.from(s.v, (vv, i) => 0.999 * vv + 0.001 * (g[i] ?? 0) ** 2)
      const c = Float64Array.from(
        s.c,
        (cv, i) => cv - (lr * ((m[i] ?? 0) / (1 - 0.9 ** t))) / (Math.sqrt((v[i] ?? 0) / (1 - 0.999 ** t)) + 1e-8),
      )
      return { c, m, v, t }
    },
    observe: (s) => ({
      metrics: { loss: toFlat(loss(fromStorage(Float64Array.from(s.c), [k, d], 'float64')))[0] ?? 0 },
    }),
  }
  const { state } = runAlgorithm(algorithm, {
    ...runOptionsFor(options, 'diverseCounterfactuals'),
    maxSteps: options.maxIterations ?? 500,
  })
  const final = fromStorage(Float64Array.from(state.c), [k, d], 'float64')
  const out = model(final) as Tensor
  const width = out.shape.length === 2 ? (out.shape[1] ?? 1) : 1
  const predictions = toFlat(out).filter((_, i) => i % width === 0)
  return {
    points: Array.from({ length: k }, (_, i) => Array.from(state.c.subarray(i * d, (i + 1) * d))),
    predictions,
    valid: predictions.map((p) => (desired === 1 ? p > 0.5 : p < 0.5)),
    diversity: k > 1 ? Math.exp(toFlat(logDetKernel(final, scales, k))[0] ?? 0) : 1,
  }
}

/** log det of K_ij = 1 / (1 + dist(c_i, c_j)) by a Cholesky-free Gaussian elimination on values. */
function logDetKernel(c: Value, scales: Tensor, k: number): Value {
  const rows = Array.from({ length: k }, (_, i) => squeeze(sliceAxis(c, 0, i, i + 1), 0))
  const a: Value[][] = rows.map((ri) =>
    rows.map((rj) => div(1, add(1, sum(div(sqrt(add(square(sub(ri, rj)), 1e-9)), squeeze(scales, 0)))))),
  )
  // Gaussian elimination without pivoting: the kernel is positive definite.
  let logdet: Value = fromStorage(new Float64Array([0]), [], 'float64')
  for (let p = 0; p < k; p++) {
    const rowP = a[p] as Value[]
    const pivot = rowP[p] as Value
    logdet = add(logdet, log(pivot))
    for (let r = p + 1; r < k; r++) {
      const rowR = a[r] as Value[]
      const factor = div(rowR[p] as Value, pivot)
      for (let col = p; col < k; col++) rowR[col] = sub(rowR[col] as Value, mul(factor, rowP[col] as Value))
    }
  }
  return logdet
}

/** Options of {@link faceCounterfactual}. */
export interface FaceOptions {
  /** Neighbours per point in the graph (default 5). */
  readonly kNeighbors?: number
  /** The predicted label (or a predicate on the model's output) the end point must have. */
  readonly accept: (prediction: number) => boolean
  /** Only end at points whose k-NN density estimate is at least this quantile of the data's (default 0.1). */
  readonly densityQuantile?: number
  /** Feature scales for distances (default ones). */
  readonly scales?: readonly number[]
}

/** A FACE path: a feasible sequence of data points from x to a counterfactual. */
export interface FacePath {
  /** Row indices of the data along the path (x itself excluded). */
  readonly path: readonly number[]
  readonly counterfactual: readonly number[]
  /** The path's total density-weighted length. */
  readonly cost: number
}

/**
 * FACE: feasible and actionable counterfactual explanations (Poyiadzi et al. 2020): build a k-nearest-neighbour graph
 * of the data whose edge costs are distances divided by the density at their midpoint, so cheap paths stay in
 * populated regions, and return the cheapest path from x to a dense point the model labels as wanted.
 *
 * @param predict - The model's output for rows `[m, d]` (`[m]`).
 * @param x - The original point `[d]`.
 * @param data - Candidate points `[n, d]` (the training data).
 * @param options - accept, kNeighbors, densityQuantile and scales.
 * @returns The path, or undefined when no acceptable point is reachable.
 * @example
 * ```ts
 * import * as explain from 'whitebox-engine/explain'
 * import { tensor } from 'whitebox-engine/tensor'
 * const sign = (x) => tensor(Array.from({ length: x.shape[0] }, (_, i) => (x.data[i] > 2 ? 1 : 0)))
 * const data = [[0], [0.5], [1], [1.5], [2], [2.5], [3]]
 * explain.faceCounterfactual(sign, [0.1], data, { accept: (p) => p === 1, kNeighbors: 2 }).counterfactual // => [2.5]
 * ```
 */
export function faceCounterfactual(
  predict: (x: Tensor) => Tensor,
  x: TensorLike,
  data: TensorLike,
  options: FaceOptions,
): FacePath | undefined {
  const x0 = rowOf(x, 'faceCounterfactual')
  const m = matrixView(data, 'faceCounterfactual', { nFeatures: x0.length })
  const d = m.d
  const scales = options.scales ?? new Array<number>(d).fill(1)
  const pts = [
    Array.from(x0),
    ...Array.from({ length: m.n }, (_, i) => Array.from(m.data.subarray(i * d, (i + 1) * d))),
  ]
  const n = pts.length
  const kk = Math.min(options.kNeighbors ?? 5, n - 1)
  const dist = (a: readonly number[], b: readonly number[]): number =>
    Math.sqrt(a.reduce((s, v, j) => s + ((v - (b[j] ?? 0)) / (scales[j] ?? 1)) ** 2, 0))
  const knn = pts.map((p, i) =>
    pts
      .map((q, j) => ({ j, dd: dist(p, q) }))
      .filter((e) => e.j !== i)
      .sort((a, b) => a.dd - b.dd || a.j - b.j)
      .slice(0, kk),
  )
  // A k-NN density estimate: inverse of the distance to the k-th neighbour.
  const density = (p: readonly number[]): number => {
    const dd = pts.map((q) => dist(p, q)).sort((a, b) => a - b)
    return 1 / ((dd[kk] ?? dd[dd.length - 1] ?? 1) + 1e-12)
  }
  const edges: [number, number, number][] = []
  knn.forEach((list, i) => {
    for (const { j, dd } of list) {
      const mid = (pts[i] ?? []).map((v, t) => (v + (pts[j]?.[t] ?? 0)) / 2)
      edges.push([i, j, dd / density(mid)])
    }
  })
  const g = graph(edges, { nodes: n, duplicates: 'min' })
  const r = dijkstra(g, 0)
  const preds = toFlat(predict(fromStorage(m.data, [m.n, d], 'float64')))
  const dens = pts.slice(1).map(density)
  const sorted = [...dens].sort((a, b) => a - b)
  const cut = sorted[Math.floor((options.densityQuantile ?? 0.1) * (sorted.length - 1))] ?? 0
  let best = -1
  for (let i = 1; i < n; i++) {
    if (!options.accept(preds[i - 1] ?? Number.NaN) || (dens[i - 1] ?? 0) < cut) continue
    const c = r.distance[i] ?? Number.POSITIVE_INFINITY
    if (Number.isFinite(c) && (best < 0 || c < (r.distance[best] ?? Number.POSITIVE_INFINITY))) best = i
  }
  if (best < 0) return undefined
  const path: number[] = []
  for (let v = best, guard = 0; v > 0 && guard <= n; v = r.predecessor[v] ?? -1, guard++) path.push(v - 1)
  path.reverse()
  return { path, counterfactual: pts[best] ?? [], cost: r.distance[best] ?? Number.NaN }
}

/** Options of {@link linearRecourse}. */
export interface RecourseOptions {
  /** The cost per unit change of each feature (default ones). */
  readonly costs?: readonly number[]
  /** Bounds on each feature's change, `[lower, upper]` (default unbounded). */
  readonly bounds?: readonly (readonly [number, number])[]
  /** Features that may not change. */
  readonly immutable?: readonly number[]
  /** The score to reach, w . x + b >= margin (default 0). */
  readonly margin?: number
}

/** A recourse action. */
export interface Recourse {
  /** The change to each feature. */
  readonly action: readonly number[]
  /** Its total cost, sum of cost_j |a_j|. */
  readonly cost: number
  /** Whether the margin is reachable within the bounds. */
  readonly feasible: boolean
}

/**
 * Actionable recourse for a linear classifier (Ustun, Spangher and Liu 2019, with linear costs): the cheapest change
 * a with w . (x + a) + b >= margin, each a_j within its bounds and immutable features fixed. With linear costs this is a
 * fractional knapsack, solved exactly by spending on features in order of |w_j| / cost_j.
 *
 * @param weights - The classifier's coefficients `[d]`.
 * @param intercept - Its intercept.
 * @param x - The point `[d]`.
 * @param options - costs, bounds, immutable and margin.
 * @returns The action, its cost and feasibility.
 * @example
 * ```ts
 * import * as explain from 'whitebox-engine/explain'
 * // Score 2 x0 + x1 - 4 at (0, 0); raising x0 is twice as effective per unit as x1.
 * explain.linearRecourse([2, 1], -4, [0, 0], { bounds: [[0, 1], [0, 5]] }) // => { action: [1, 2], cost: 3, feasible: true }
 * ```
 */
export function linearRecourse(
  weights: readonly number[],
  intercept: number,
  x: TensorLike,
  options: RecourseOptions = {},
): Recourse {
  const x0 = rowOf(x, 'linearRecourse')
  const d = x0.length
  if (weights.length !== d) throw new WhiteboxError('shape', 'linearRecourse: one weight per feature')
  const costs = options.costs ?? new Array<number>(d).fill(1)
  if (costs.some((c) => !(c > 0))) throw new WhiteboxError('value', 'linearRecourse: costs must be positive')
  const fixed = new Set(options.immutable ?? [])
  const score = weights.reduce((s, w, j) => s + w * (x0[j] ?? 0), intercept)
  let need = (options.margin ?? 0) - score
  const action = new Array<number>(d).fill(0)
  if (need <= 0) return { action, cost: 0, feasible: true }
  const order = weights
    .map((w, j) => ({ j, w, eff: Math.abs(w) / (costs[j] ?? 1) }))
    .filter((e) => e.w !== 0 && !fixed.has(e.j))
    .sort((a, b) => b.eff - a.eff || a.j - b.j)
  let cost = 0
  for (const { j, w } of order) {
    if (need <= 1e-15) break
    const [lo, hi] = options.bounds?.[j] ?? [Number.NEGATIVE_INFINITY, Number.POSITIVE_INFINITY]
    const room = w > 0 ? hi : -lo
    if (!(room > 0)) continue
    const step = Math.min(room, need / Math.abs(w))
    action[j] = w > 0 ? step : -step
    need -= step * Math.abs(w)
    cost += step * (costs[j] ?? 1)
  }
  return { action, cost, feasible: need <= 1e-12 }
}
