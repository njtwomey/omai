/**
 * Learned shapelets (Grabocka, Schilling, Wistuba and Schmidt-Thieme 2014): instead of searching the training series for
 * discriminative subsequences, the shapelets are parameters. A series is described by its soft-minimum distance to each
 * shapelet, M = sum_j D_j exp(alpha D_j) / sum_j exp(alpha D_j) over the windows j (alpha negative, so the weights
 * concentrate on the nearest window and M is a smooth stand-in for the minimum), a softmax classifier on those distances
 * gives the class probabilities, and the shapelets and the classifier weights are fitted together on the regularised
 * cross-entropy. The objective is smooth, so the fit takes its solver from the slot of `smoothSolvers`: the stochastic
 * `gradient` mode with Adam is the default (the paper's stochastic gradient descent on mini-batches of series), and any
 * other solver reaches a minimum of the same function. The shapelets start at k-means centres of the windows.
 */

import { maxOf, OmaiError } from '../base/index.ts'
import { initialCentres, lloydAlgorithm } from '../cluster/index.ts'
import { analyticObjective, type SmoothProblem, slotAlgorithm, smoothSolverConfig } from '../optim/index.ts'
import { foldIn, type Key, key as makeKey } from '../random/index.ts'
import type { SpaceValue } from '../space/index.ts'
import type { Value } from '../tensor/index.ts'
import { type Algorithm, runAlgorithm } from '../trace/index.ts'

/** Training data: series of one length, and the class index (0, 1, ...) of each. */
export interface LearnedShapeletData {
  readonly series: readonly ArrayLike<number>[]
  readonly labels: readonly number[]
  readonly nClasses: number
}

/** The shape of the model. */
export interface LearnedShapeletConfig {
  /** The number of shapelets (default 3). */
  readonly nShapelets?: number
  /** The length of every shapelet (default 7). */
  readonly length?: number
  /** The soft-minimum sharpness, negative (default -30). */
  readonly alpha?: number
  /** The L2 penalty on the classifier weights (default 1e-3). */
  readonly ridge?: number
}

interface ShapeletDimensions {
  readonly n: number
  readonly T: number
  readonly K: number
  readonly L: number
  readonly C: number
  readonly alpha: number
  readonly ridge: number
}

function shapeOf(data: LearnedShapeletData, config: LearnedShapeletConfig): ShapeletDimensions {
  const n = data.series.length
  const T = data.series[0]?.length ?? 0
  const K = config.nShapelets ?? 3
  const L = config.length ?? 7
  const alpha = config.alpha ?? -30
  if (n === 0 || data.series.some((s) => s.length !== T))
    throw new OmaiError('shape', 'learnedShapelets: series of one length')
  if (data.labels.length !== n) throw new OmaiError('shape', 'learnedShapelets: one label per series')
  if (!(Number.isInteger(L) && L >= 2 && L <= T))
    throw new OmaiError('value', `learnedShapelets: length must be in 2..${T}`)
  if (!(Number.isInteger(K) && K >= 1))
    throw new OmaiError('value', 'learnedShapelets: nShapelets must be a positive integer')
  if (!(alpha < 0)) throw new OmaiError('value', 'learnedShapelets: alpha must be negative')
  if (data.labels.some((l) => !(Number.isInteger(l) && l >= 0 && l < data.nClasses)))
    throw new OmaiError('value', 'learnedShapelets: labels must be class indices below nClasses')
  return { n, T, K, L, C: data.nClasses, alpha, ridge: config.ridge ?? 1e-3 }
}

/** The squared distances of every window of a series to a shapelet, averaged over the shapelet's length. */
function windowDistances(x: ArrayLike<number>, s: ArrayLike<number>, L: number, out: Float64Array): void {
  const J = x.length - L + 1
  for (let j = 0; j < J; j++) {
    let d = 0
    for (let l = 0; l < L; l++) d += ((x[j + l] ?? 0) - (s[l] ?? 0)) ** 2
    out[j] = d / L
  }
}

/** The soft minimum of distances `d` at sharpness alpha, with the weights of the windows (for the gradient). */
function softMin(d: Float64Array, J: number, alpha: number, weights: Float64Array): number {
  let lo = Number.POSITIVE_INFINITY
  for (let j = 0; j < J; j++) lo = Math.min(lo, d[j] ?? 0)
  let z = 0
  for (let j = 0; j < J; j++) {
    const e = Math.exp(alpha * ((d[j] ?? 0) - lo))
    weights[j] = e
    z += e
  }
  let m = 0
  for (let j = 0; j < J; j++) {
    weights[j] = (weights[j] ?? 0) / z
    m += (d[j] ?? 0) * (weights[j] ?? 0)
  }
  return m
}

/**
 * The soft-minimum distance from a series to a shapelet: the mean squared difference of each window to the shapelet,
 * combined as sum_j D_j exp(alpha D_j) / sum_j exp(alpha D_j). It lies between the smallest and the largest window
 * distance and tends to the smallest (the `shapeletDistance`) as alpha goes to minus infinity.
 *
 * @param series - The series.
 * @param shapelet - The shapelet.
 * @param alpha - The sharpness (negative; default -30).
 * @returns The soft-minimum distance.
 * @example
 * ```ts
 * import { softMinDistance } from 'omai-engine/timeseries'
 * softMinDistance([0, 1, 2, 1, 0, 0], [1, 2, 1], -1000) // => 0
 * ```
 */
export function softMinDistance(series: ArrayLike<number>, shapelet: ArrayLike<number>, alpha = -30): number {
  const L = shapelet.length
  const J = series.length - L + 1
  if (J < 1) throw new OmaiError('shape', 'softMinDistance: the shapelet is longer than the series')
  const d = new Float64Array(J)
  windowDistances(series, shapelet, L, d)
  return softMin(d, J, alpha, new Float64Array(J))
}

/** The packed parameters: the shapelets row by row, then the classifier weights class by class (bias first). */
function split(p: ArrayLike<number>, s: ShapeletDimensions): { shapelets: Float64Array; weights: Float64Array } {
  const flat = Float64Array.from(p as ArrayLike<number>)
  return { shapelets: flat.subarray(0, s.K * s.L), weights: flat.subarray(s.K * s.L) }
}

/** The mean cross-entropy of the rows in `rows` (all by default), the penalty, and the gradient of their sum. */
function evaluate(
  data: LearnedShapeletData,
  s: ShapeletDimensions,
  p: ArrayLike<number>,
  rows?: readonly number[],
): { value: number; gradient: Float64Array; accuracy: number } {
  const { shapelets, weights } = split(p, s)
  const grad = new Float64Array(s.K * s.L + s.C * (s.K + 1))
  const J = s.T - s.L + 1
  const d = new Float64Array(J)
  const w = new Float64Array(J)
  const idx = rows ?? Array.from({ length: s.n }, (_, i) => i)
  const nb = idx.length
  let loss = 0
  let correct = 0
  for (const i of idx) {
    const x = data.series[i] as ArrayLike<number>
    const m = new Float64Array(s.K)
    // the per-shapelet window weights times (1 + alpha (D_j - M)), kept for the shapelet gradient
    const kernels: Float64Array[] = []
    for (let k = 0; k < s.K; k++) {
      const sk = shapelets.subarray(k * s.L, (k + 1) * s.L)
      windowDistances(x, sk, s.L, d)
      m[k] = softMin(d, J, s.alpha, w)
      kernels.push(Float64Array.from(w, (wj, j) => wj * (1 + s.alpha * ((d[j] ?? 0) - (m[k] ?? 0)))))
    }
    const z = new Float64Array(s.C)
    let top = Number.NEGATIVE_INFINITY
    for (let c = 0; c < s.C; c++) {
      let v = weights[c * (s.K + 1)] ?? 0
      for (let k = 0; k < s.K; k++) v += (weights[c * (s.K + 1) + 1 + k] ?? 0) * (m[k] ?? 0)
      z[c] = v
      top = Math.max(top, v)
    }
    let norm = 0
    for (let c = 0; c < s.C; c++) norm += Math.exp((z[c] ?? 0) - top)
    const lse = top + Math.log(norm)
    const y = data.labels[i] as number
    loss += lse - (z[y] ?? 0)
    let best = 0
    for (let c = 0; c < s.C; c++) if ((z[c] ?? 0) > (z[best] ?? 0)) best = c
    if (best === y) correct++
    const dM = new Float64Array(s.K)
    for (let c = 0; c < s.C; c++) {
      const r = Math.exp((z[c] ?? 0) - lse) - (c === y ? 1 : 0)
      const base = s.K * s.L + c * (s.K + 1)
      grad[base] = (grad[base] ?? 0) + r / nb
      for (let k = 0; k < s.K; k++) {
        grad[base + 1 + k] = (grad[base + 1 + k] ?? 0) + (r * (m[k] ?? 0)) / nb
        dM[k] = (dM[k] ?? 0) + r * (weights[c * (s.K + 1) + 1 + k] ?? 0)
      }
    }
    for (let k = 0; k < s.K; k++) {
      const sk = shapelets.subarray(k * s.L, (k + 1) * s.L)
      const ker = kernels[k] as Float64Array
      for (let l = 0; l < s.L; l++) {
        let g = 0
        for (let j = 0; j < J; j++) g += (ker[j] ?? 0) * (-2 / s.L) * ((x[j + l] ?? 0) - (sk[l] ?? 0))
        grad[k * s.L + l] = (grad[k * s.L + l] ?? 0) + ((dM[k] ?? 0) * g) / nb
      }
    }
  }
  let penalty = 0
  for (let c = 0; c < s.C; c++)
    for (let k = 0; k < s.K; k++) {
      const at = s.K * s.L + c * (s.K + 1) + 1 + k
      penalty += 0.5 * s.ridge * (weights[c * (s.K + 1) + 1 + k] ?? 0) ** 2
      grad[at] = (grad[at] ?? 0) + s.ridge * (weights[c * (s.K + 1) + 1 + k] ?? 0)
    }
  return { value: loss / nb + penalty, gradient: grad, accuracy: correct / nb }
}

/**
 * The starting shapelets: k-means centres (k-means++ seeds, Lloyd iterations) of all windows of the training series of
 * the shapelet length, as Grabocka et al. initialise.
 *
 * @param data - The training series.
 * @param config - Number and length of the shapelets.
 * @param key - The key for the seeding.
 * @returns The shapelets `[nShapelets][length]`.
 * @example
 * ```ts
 * import { initialShapelets } from 'omai-engine/timeseries'
 * import { key } from 'omai-engine/random'
 * const series = [[0, 0, 1, 2, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]]
 * initialShapelets({ series, labels: [1, 0], nClasses: 2 }, { nShapelets: 2, length: 3 }, key(0)).length // => 2
 * ```
 */
export function initialShapelets(data: LearnedShapeletData, config: LearnedShapeletConfig, key: Key): number[][] {
  const s = shapeOf(data, config)
  const windows: number[][] = []
  for (const x of data.series)
    for (let j = 0; j + s.L <= s.T; j++) windows.push(Array.from({ length: s.L }, (_, l) => x[j + l] ?? 0))
  const seeds = initialCentres(windows, Math.min(s.K, windows.length), 'k-means++', key)
  const run = runAlgorithm(lloydAlgorithm(windows, seeds), { maxSteps: 25, key })
  return run.state.centres
}

/**
 * The learned-shapelets objective as a smooth problem for the solver slot: the mean cross-entropy of the softmax
 * classifier on the soft-minimum distances plus `ridge / 2` times the squared classifier weights (the biases are free),
 * over the packed parameters (shapelets row by row, then each class's bias and weights), with the analytic gradient
 * through the soft minimum. A batch of series indices gives the mean over those series, so the stochastic solvers draw
 * mini-batches of series.
 *
 * @param data - The training series.
 * @param config - Number and length of the shapelets, sharpness, ridge.
 * @param start - The starting shapelets (default {@link initialShapelets} with key 0).
 * @returns The problem.
 * @example
 * ```ts
 * import { learnedShapeletProblem } from 'omai-engine/timeseries'
 * const series = [[0, 0, 1, 2, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]]
 * learnedShapeletProblem({ series, labels: [1, 0], nClasses: 2 }, { nShapelets: 2, length: 3 }).x0.length // => 12
 * ```
 */
export function learnedShapeletProblem(
  data: LearnedShapeletData,
  config: LearnedShapeletConfig = {},
  start?: readonly (readonly number[])[],
): SmoothProblem {
  const s = shapeOf(data, config)
  const shapelets = start ?? initialShapelets(data, config, makeKey(0))
  const objectiveOf = (batch?: readonly number[]): ((x: Value) => Value) =>
    analyticObjective({
      value: (p) => evaluate(data, s, p, batch).value,
      gradient: (p) => evaluate(data, s, p, batch).gradient,
    })
  const full = objectiveOf()
  return {
    objective: ((x: unknown, batch?: readonly number[]) =>
      (batch === undefined ? full : objectiveOf(batch))(x as Value)) as SmoothProblem['objective'],
    x0: [...shapelets.flat(), ...new Array<number>(s.C * (s.K + 1)).fill(0)],
    batches: { n: s.n },
  }
}

/** The state of {@link learnedShapeletsAlgorithm}. */
export type LearnedShapeletsState = {
  readonly solver: unknown
  /** The packed parameters. */
  readonly parameters: number[]
  readonly objective: number
  readonly gradient: number
  /** The training accuracy at the parameters. */
  readonly accuracy: number
  readonly evaluations: number
  readonly stalled: boolean
}

/** Options of {@link learnedShapeletsAlgorithm}. */
export interface LearnedShapeletsOptions extends LearnedShapeletConfig {
  /** The solver, a value of `smoothSolvers.space` (default the stochastic gradient mode with Adam, batches of 16). */
  readonly solver?: SpaceValue
  /** Stop when the largest gradient entry falls below this (default 1e-6). */
  readonly tol?: number
  /** The key for the starting shapelets and the batches. */
  readonly key?: Key
  /** Starting shapelets (default k-means centres of the windows). */
  readonly start?: readonly (readonly number[])[]
}

/** The default solver: Adam at learning rate 0.05 on mini-batches of 16 series. */
export const learnedShapeletSolver: SpaceValue = smoothSolverConfig({
  kind: 'gradient',
  optimiser: { kind: 'adam' },
  learningRate: 0.05,
  batch: 16,
  gradTol: 1e-8,
})

/**
 * Learned shapelets as a traced {@link Algorithm} over the solver slot: each step is one step of the chosen solver on
 * {@link learnedShapeletProblem}. It observes the objective, the training accuracy, the gradient norm and the
 * `shapelets` tensor `[nShapelets, length]` (so the shapelets can be watched taking shape).
 *
 * @param data - The training series.
 * @param options - Shape, sharpness, ridge, solver, tolerance, key and start.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { learnedShapeletsAlgorithm } from 'omai-engine/timeseries'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const series = [[0, 0, 1, 2, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 2, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]]
 * const alg = learnedShapeletsAlgorithm({ series, labels: [1, 0, 1, 0], nClasses: 2 }, { nShapelets: 1, length: 3 })
 * runAlgorithm(alg, { maxSteps: 3 }).steps // => 3
 * ```
 */
export function learnedShapeletsAlgorithm(
  data: LearnedShapeletData,
  options: LearnedShapeletsOptions = {},
): Algorithm<LearnedShapeletsState> {
  const s = shapeOf(data, options)
  const k = options.key ?? makeKey(0)
  const problem = learnedShapeletProblem(data, options, options.start ?? initialShapelets(data, options, foldIn(k, 1)))
  const solver = options.solver ?? learnedShapeletSolver
  return slotAlgorithm<LearnedShapeletsState>({
    name: 'learned-shapelets',
    config: { nShapelets: s.K, length: s.L, alpha: s.alpha, ridge: s.ridge, solver } as never,
    schema: {
      metrics: {
        objective: {
          description: 'the mean cross-entropy plus the weight penalty at the parameters',
          direction: 'lower',
        },
        accuracy: { description: 'the accuracy on the training series', direction: 'higher' },
        gradient: {
          description: 'the largest gradient entry (of the last batch for the stochastic solver)',
          direction: 'lower',
        },
      },
      tensors: {
        shapelets: { role: 'param', description: 'the shapelets', shape: [s.K, s.L] },
        weights: {
          role: 'param',
          description: 'the classifier biases and weights, class by class',
          shape: [s.C, s.K + 1],
        },
      },
    },
    problem,
    solver,
    tol: options.tol ?? 1e-6,
    key: k,
    view: (r) => ({
      parameters: Array.from(r.x),
      objective: evaluate(data, s, r.x).value,
      gradient: r.grad.reduce((a, b) => Math.max(a, Math.abs(b)), 0),
      accuracy: evaluate(data, s, r.x).accuracy,
      evaluations: 0,
      stalled: false,
    }),
    observe: (st) => ({
      metrics: { objective: st.objective, accuracy: st.accuracy, gradient: st.gradient },
      tensors: {
        shapelets: { role: 'param', value: st.parameters.slice(0, s.K * s.L), shape: [s.K, s.L] },
        weights: { role: 'param', value: st.parameters.slice(s.K * s.L), shape: [s.C, s.K + 1] },
      },
    }),
  })
}

/** A fitted learned-shapelets classifier. */
export interface LearnedShapelets {
  readonly shapelets: number[][]
  /** The classifier: row c holds the bias then the weight on each shapelet's soft-minimum distance. */
  readonly weights: number[][]
  readonly alpha: number
  readonly nClasses: number
  /** The training accuracy and objective at the fit. */
  readonly accuracy: number
  readonly objective: number
}

/**
 * Fit a learned-shapelets classifier ({@link learnedShapeletsAlgorithm} run for `maxSteps` steps).
 *
 * @param data - The training series.
 * @param options - Shape, solver, key.
 * @param maxSteps - The step budget (default 400).
 * @returns The shapelets, the classifier, and the training accuracy and objective.
 * @example
 * ```ts
 * import { fitLearnedShapelets } from 'omai-engine/timeseries'
 * const series = [[0, 0, 1, 2, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 2, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]]
 * fitLearnedShapelets({ series, labels: [1, 0, 1, 0], nClasses: 2 }, { nShapelets: 1, length: 3 }, 20).shapelets[0]?.length // => 3
 * ```
 */
export function fitLearnedShapelets(
  data: LearnedShapeletData,
  options: LearnedShapeletsOptions = {},
  maxSteps = 400,
): LearnedShapelets {
  const s = shapeOf(data, options)
  const run = runAlgorithm(learnedShapeletsAlgorithm(data, options), {
    maxSteps,
    key: foldIn(options.key ?? makeKey(0), 2),
  })
  const p = run.state.parameters
  return {
    shapelets: Array.from({ length: s.K }, (_, k) => p.slice(k * s.L, (k + 1) * s.L)),
    weights: Array.from({ length: s.C }, (_, c) => p.slice(s.K * s.L + c * (s.K + 1), s.K * s.L + (c + 1) * (s.K + 1))),
    alpha: s.alpha,
    nClasses: s.C,
    accuracy: run.state.accuracy,
    objective: run.state.objective,
  }
}

/**
 * The class probabilities and the predicted class of each series under a fitted model.
 *
 * @param model - The fitted classifier.
 * @param series - The series (of the training length).
 * @returns The probabilities `[n][classes]` and the predicted classes.
 * @example
 * ```ts
 * import { predictLearnedShapelets } from 'omai-engine/timeseries'
 * const model = { shapelets: [[1, 2, 1]], weights: [[2, -5], [-2, 5]], alpha: -30, nClasses: 2, accuracy: 1, objective: 0 }
 * // a series containing the shapelet is near it (class 0 here); a flat one is far (class 1)
 * predictLearnedShapelets(model, [[0, 1, 2, 1, 0], [0, 0, 0, 0, 0]]).classes // => [0, 1]
 * ```
 */
export function predictLearnedShapelets(
  model: LearnedShapelets,
  series: readonly ArrayLike<number>[],
): { readonly probabilities: number[][]; readonly classes: number[] } {
  const probabilities = series.map((x) => {
    const m = model.shapelets.map((sk) => softMinDistance(x, sk, model.alpha))
    const z = model.weights.map((w) => (w[0] ?? 0) + m.reduce((a, v, k) => a + v * (w[1 + k] ?? 0), 0))
    const top = maxOf(z)
    const e = z.map((v) => Math.exp(v - top))
    const sum = e.reduce((a, v) => a + v, 0)
    return e.map((v) => v / sum)
  })
  return { probabilities, classes: probabilities.map((p) => p.reduce((b, v, c) => (v > (p[b] ?? 0) ? c : b), 0)) }
}
