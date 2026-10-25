/**
 * Example-based explanations: the data points that represent a dataset (MMD-critic's prototypes) and those it
 * represents badly (its criticisms), the nearest training examples of a query, training examples whose loss gradients
 * point the same way as a test point's (gradient similarity), and trust scores (how much closer a point is to its
 * predicted class's dense core than to any other class's).
 */

import { grad } from '../autodiff/index.ts'
import { WhiteboxError } from '../base/index.ts'
import type { LearnData } from '../learn/estimator.ts'
import { matrixView, targetVector } from '../learn/validation.ts'
import { cholesky, choleskyLogdet } from '../numerics/linalg/index.ts'
import { asTensor, fromStorage, type Tensor, type TensorLike, toFlat, type Value } from '../tensor/index.ts'
import type { ExampleLoss } from './valuation.ts'

function rowsOfView(x: TensorLike, who: string): { n: number; d: number; rows: number[][] } {
  const m = matrixView(x, who)
  return {
    n: m.n,
    d: m.d,
    rows: Array.from({ length: m.n }, (_, i) => Array.from(m.data.subarray(i * m.d, (i + 1) * m.d))),
  }
}

function sqDist(a: readonly number[], b: readonly number[]): number {
  let s = 0
  for (let j = 0; j < a.length; j++) s += ((a[j] ?? 0) - (b[j] ?? 0)) ** 2
  return s
}

/** Options of {@link mmdCritic}. */
export interface MmdCriticOptions {
  /** Prototypes chosen (default 4). */
  readonly nPrototypes?: number
  /** Criticisms chosen (default 2). */
  readonly nCriticisms?: number
  /** The RBF kernel's gamma, k(a, b) = exp(-gamma |a - b|^2) (default 1 / d). */
  readonly gamma?: number
  /** Add log det K_CC to the criticisms' objective so they spread out (default true). */
  readonly regularise?: boolean
}

/** Prototypes and criticisms of a dataset. */
export interface MmdCriticResult {
  /** Row indices of the prototypes, in the order chosen. */
  readonly prototypes: readonly number[]
  /** Row indices of the criticisms, in the order chosen. */
  readonly criticisms: readonly number[]
  /** The witness function at every row: positive where the data are denser than the prototypes cover. */
  readonly witness: readonly number[]
  /** The squared MMD between the data and the prototypes after each prototype is added. */
  readonly mmd: readonly number[]
}

/**
 * MMD-critic (Kim, Khanna and Koyejo 2016): prototypes chosen greedily to minimise the maximum mean discrepancy
 * between the data and the prototypes under an RBF kernel, then criticisms, the points where the witness function
 * w(x) = mean_i k(x, x_i) - mean_j k(x, z_j) is largest in magnitude (the data the prototypes fail to represent), with
 * a log-determinant term that keeps them apart.
 *
 * @param x - The data `[n, d]`.
 * @param options - nPrototypes, nCriticisms, gamma and regularise.
 * @returns The prototypes, criticisms, witness values and the MMD as prototypes were added.
 * @example
 * ```ts
 * import * as explain from 'whitebox-engine/explain'
 * // Two clusters: one prototype each; the outlier is the criticism.
 * const x = [[0, 0], [0.1, 0], [0, 0.1], [5, 5], [5.1, 5], [5, 5.1], [2.5, -3]]
 * const r = explain.mmdCritic(x, { nPrototypes: 2, nCriticisms: 1, gamma: 1 })
 * [...r.prototypes].sort() // => [0, 3]
 * r.criticisms // => [6]
 * ```
 */
export function mmdCritic(x: TensorLike, options: MmdCriticOptions = {}): MmdCriticResult {
  const { n, d, rows } = rowsOfView(x, 'mmdCritic')
  const gamma = options.gamma ?? 1 / Math.max(1, d)
  const np = Math.min(options.nPrototypes ?? 4, n)
  const nc = Math.min(options.nCriticisms ?? 2, n)
  const K = rows.map((a) => rows.map((b) => Math.exp(-gamma * sqDist(a, b))))
  const colMean = K.map((r) => r.reduce((s, v) => s + v, 0) / n)
  const kAll = colMean.reduce((s, v) => s + v, 0) / n
  const prototypes: number[] = []
  const mmd: number[] = []
  const chosen = new Set<number>()
  for (let t = 0; t < np; t++) {
    let best = -1
    let bestValue = Number.NEGATIVE_INFINITY
    for (let c = 0; c < n; c++) {
      if (chosen.has(c)) continue
      const s = [...prototypes, c]
      let cross = 0
      for (const j of s) cross += colMean[j] ?? 0
      let within = 0
      for (const a of s) for (const b of s) within += K[a]?.[b] ?? 0
      // J(S) = 2/|S| sum_j mean_i k(x_i, z_j) - 1/|S|^2 sum k(z, z'); the MMD^2 is kAll - J(S).
      const value = (2 * cross) / s.length - within / (s.length * s.length)
      if (value > bestValue + 1e-15) {
        bestValue = value
        best = c
      }
    }
    prototypes.push(best)
    chosen.add(best)
    mmd.push(kAll - bestValue)
  }
  const witness = rows.map((_, i) => {
    let p = 0
    for (const j of prototypes) p += K[i]?.[j] ?? 0
    return (colMean[i] ?? 0) - p / Math.max(1, prototypes.length)
  })
  const criticisms: number[] = []
  const used = new Set(prototypes)
  const regularise = options.regularise ?? true
  for (let t = 0; t < nc; t++) {
    let best = -1
    let bestValue = Number.NEGATIVE_INFINITY
    for (let c = 0; c < n; c++) {
      if (used.has(c)) continue
      const s = [...criticisms, c]
      let value = s.reduce((acc, j) => acc + Math.abs(witness[j] ?? 0), 0)
      if (regularise) value += logDet(s.map((a) => s.map((b) => K[a]?.[b] ?? 0)))
      if (value > bestValue + 1e-15) {
        bestValue = value
        best = c
      }
    }
    if (best < 0) break
    criticisms.push(best)
    used.add(best)
  }
  return { prototypes, criticisms, witness, mmd }
}

/** log det of a small symmetric positive semidefinite matrix (-Infinity when it is singular). */
function logDet(a: readonly (readonly number[])[]): number {
  try {
    return toFlat(choleskyLogdet(cholesky(a.map((r) => [...r]))) as Tensor)[0] ?? Number.NEGATIVE_INFINITY
  } catch {
    return Number.NEGATIVE_INFINITY
  }
}

/** The nearest examples of a query. */
export interface NearestExamples {
  /** Row indices of the data, nearest first. */
  readonly indices: readonly number[]
  readonly distances: readonly number[]
}

/**
 * The k nearest examples of a query in the data (k-NN explanations: "rows like this one were labelled so"), by the
 * Euclidean distance of features divided by their scales.
 *
 * @param query - The point `[d]`.
 * @param x - The data `[n, d]`.
 * @param options - k (default 5) and scales (default ones).
 * @returns Their indices and distances, nearest first.
 * @example
 * ```ts
 * import * as explain from 'whitebox-engine/explain'
 * explain.nearestExamples([0, 0], [[3, 4], [1, 0], [0, 2]], { k: 2 }) // => { indices: [1, 2], distances: [1, 2] }
 * ```
 */
export function nearestExamples(
  query: TensorLike,
  x: TensorLike,
  options: { readonly k?: number; readonly scales?: readonly number[] } = {},
): NearestExamples {
  const q = toFlat(asTensor(query))
  const { rows } = rowsOfView(x, 'nearestExamples')
  const s = options.scales
  const dist = rows.map((r, i) => ({
    i,
    v: Math.sqrt(r.reduce((acc, v, j) => acc + ((v - (q[j] ?? 0)) / (s?.[j] ?? 1)) ** 2, 0)),
  }))
  dist.sort((a, b) => a.v - b.v || a.i - b.i)
  const top = dist.slice(0, Math.min(options.k ?? 5, dist.length))
  return { indices: top.map((e) => e.i), distances: top.map((e) => e.v) }
}

/** Options of {@link trustScores}. */
export interface TrustScoreOptions {
  /** The fraction of each class's training rows dropped as low density (default 0.1). */
  readonly alpha?: number
  /** Neighbours of the k-NN radius that measures density (default 10). */
  readonly kFilter?: number
  /** Distance to the k-th nearest point of a class's high-density set (default 1). */
  readonly k?: number
}

/**
 * Trust scores (Jiang, Kim, Guan and Gupta 2018; alibi's TrustScore): per class, keep the training rows of high
 * density (dropping the fraction alpha with the largest distance to their kFilter-th same-class neighbour); a test
 * point's score is its distance to the nearest other class's set over its distance to its predicted class's set.
 * Above 1 the prediction agrees with the neighbourhood; below 1 a different class is closer.
 *
 * @param train - Training data with class labels.
 * @param x - Test points `[m, d]`.
 * @param predicted - The model's predicted label for each test point.
 * @param options - alpha, kFilter and k.
 * @returns The scores and, for each point, the closest other class.
 * @example
 * ```ts
 * import * as explain from 'whitebox-engine/explain'
 * const train = { x: [[0], [0.2], [0.4], [5], [5.2], [5.4]], y: [0, 0, 0, 1, 1, 1] }
 * const r = explain.trustScores(train, [[0.1], [0.1]], [0, 1], { alpha: 0, kFilter: 1 })
 * r.scores.map((s) => s > 1) // => [true, false]
 * ```
 */
export function trustScores(
  train: LearnData,
  x: TensorLike,
  predicted: readonly number[],
  options: TrustScoreOptions = {},
): { readonly scores: readonly number[]; readonly closestOther: readonly number[] } {
  const tr = rowsOfView(train.x, 'trustScores')
  if (train.y === undefined) throw new WhiteboxError('value', 'trustScores: the training data need labels')
  const y = targetVector(train.y, 'trustScores', tr.n)
  const test = rowsOfView(x, 'trustScores')
  if (predicted.length !== test.n) throw new WhiteboxError('shape', 'trustScores: one predicted label per test point')
  const alpha = options.alpha ?? 0.1
  const kf = options.kFilter ?? 10
  const k = options.k ?? 1
  const classes = Array.from(new Set(y)).sort((a, b) => a - b)
  const sets = new Map<number, number[][]>()
  for (const c of classes) {
    const members = tr.rows.filter((_, i) => y[i] === c)
    const radius = members.map((p) => {
      const d = members.map((q) => Math.sqrt(sqDist(p, q))).sort((a, b) => a - b)
      return d[Math.min(kf, d.length - 1)] ?? 0
    })
    const order = radius.map((r, i) => ({ r, i })).sort((a, b) => a.r - b.r || a.i - b.i)
    const keep = Math.max(1, Math.ceil(members.length * (1 - alpha)))
    sets.set(
      c,
      order.slice(0, keep).map((e) => members[e.i] ?? []),
    )
  }
  const distTo = (p: readonly number[], c: number): number => {
    const d = (sets.get(c) ?? []).map((q) => Math.sqrt(sqDist(p, q))).sort((a, b) => a - b)
    return d[Math.min(k - 1, d.length - 1)] ?? Number.POSITIVE_INFINITY
  }
  const scores: number[] = []
  const closestOther: number[] = []
  test.rows.forEach((p, i) => {
    const own = distTo(p, predicted[i] ?? 0)
    let other = Number.POSITIVE_INFINITY
    let which = -1
    for (const c of classes) {
      if (c === predicted[i]) continue
      const v = distTo(p, c)
      if (v < other) {
        other = v
        which = c
      }
    }
    scores.push(own === 0 ? Number.POSITIVE_INFINITY : other / own)
    closestOther.push(which)
  })
  return { scores, closestOther }
}

/** Options of {@link gradientSimilarity}. */
export interface GradientSimilarityOptions {
  /** `dot` (TracIn-like influence), `cosine` (default) or `gradCosine` (alibi's grad_cos: dot over the train norm). */
  readonly similarity?: 'dot' | 'cosine' | 'gradCosine'
  /** Training examples kept per test point (default 5). */
  readonly k?: number
}

/**
 * Gradient similarity (Charpiat et al. 2019; alibi's GradientSimilarity): explain a prediction by the training
 * examples whose loss gradient with respect to the parameters points the same way as the test example's, so a step on
 * one would move the other's loss too. Scores are dot products, cosines or the dot over the training gradient's norm.
 *
 * @param loss - The per-example loss of the parameters.
 * @param params - The fitted parameters.
 * @param train - Training data with targets.
 * @param test - Test data with targets (the label explained).
 * @param options - similarity and k.
 * @returns For each test point, all scores and the top k training rows.
 * @example
 * ```ts
 * import * as explain from 'whitebox-engine/explain'
 * import { dot, square, sub } from 'whitebox-engine/tensor'
 * // Squared error of a linear model through the origin: gradients 2 (w x - y) x.
 * const loss = (w, x, y) => square(sub(dot(w, x), y))
 * const r = explain.gradientSimilarity(loss, [1], { x: [[1], [-1]], y: [2, -2] }, { x: [[2]], y: [3] })
 * r.top[0] // => [0, 1]
 * ```
 */
export function gradientSimilarity(
  loss: ExampleLoss,
  params: TensorLike,
  train: LearnData,
  test: LearnData,
  options: GradientSimilarityOptions = {},
): { readonly scores: readonly (readonly number[])[]; readonly top: readonly (readonly number[])[] } {
  const tr = rowsOfView(train.x, 'gradientSimilarity')
  const te = rowsOfView(test.x, 'gradientSimilarity')
  if (train.y === undefined || test.y === undefined)
    throw new WhiteboxError('value', 'gradientSimilarity: train and test need targets')
  const ytr = targetVector(train.y, 'gradientSimilarity', tr.n)
  const yte = targetVector(test.y, 'gradientSimilarity', te.n)
  const gradOf = (x: readonly number[], y: number): number[] =>
    toFlat(
      grad((p: Value) => loss(p, fromStorage(Float64Array.from(x), [x.length], 'float64'), y))(
        asTensor(params),
      ) as Tensor,
    )
  const gTrain = tr.rows.map((r, i) => gradOf(r, ytr[i] ?? 0))
  const norm = (g: readonly number[]): number => Math.sqrt(g.reduce((s, v) => s + v * v, 0))
  const kind = options.similarity ?? 'cosine'
  const scores = te.rows.map((r, i) => {
    const g = gradOf(r, yte[i] ?? 0)
    const ng = norm(g)
    return gTrain.map((h) => {
      const dot = h.reduce((s, v, j) => s + v * (g[j] ?? 0), 0)
      if (kind === 'dot') return dot
      const nh = norm(h)
      if (kind === 'gradCosine') return nh === 0 ? 0 : dot / nh
      return nh === 0 || ng === 0 ? 0 : dot / (nh * ng)
    })
  })
  const k = Math.min(options.k ?? 5, tr.n)
  const top = scores.map((s) =>
    s
      .map((v, j) => ({ v, j }))
      .sort((a, b) => b.v - a.v || a.j - b.j)
      .slice(0, k)
      .map((e) => e.j),
  )
  return { scores, top }
}
