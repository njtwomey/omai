/**
 * Metrics for generative models computed from features or embeddings: the Inception score, the Frechet (FID) and kernel
 * (KID) Inception distances, precision, recall, density and coverage of generated samples, CLIPScore for image-text
 * agreement and the Vendi diversity score.
 */

import { MdmlError, type MetricResult, type MetricSpec } from '../base/index.ts'
import { eigvalsh, funmSymmetric } from '../numerics/linalg/index.ts'
import { type Key, key as makeKey, permutation, split } from '../random/index.ts'
import { matmul, relu, sqrt, type TensorLike, tensor, toFlat } from '../tensor/index.ts'
import { defineMetric, type MetricFunction, metricResult, shapedArray } from './metric.ts'

const spec = (
  name: string,
  direction: 'higher' | 'lower',
  range: readonly [number, number] = [0, Infinity],
): MetricSpec => ({
  name,
  direction,
  range,
  inputs: 'values',
  category: 'generative',
  concept: 'generative-evaluation',
})

/** Rows of a feature matrix `[N, D]`. */
function rows(name: string, x: TensorLike): { n: number; d: number; data: Float64Array } {
  const m = shapedArray(x)
  if (m.shape.length !== 2) throw new MdmlError('shape', `${name}: features are [N, D]`)
  return { n: m.shape[0] ?? 0, d: m.shape[1] ?? 0, data: m.data }
}

/** Mean and covariance (ddof 1) of feature rows. */
function moments(f: { n: number; d: number; data: Float64Array }): { mu: Float64Array; cov: number[][] } {
  const mu = new Float64Array(f.d)
  for (let i = 0; i < f.n; i++) for (let j = 0; j < f.d; j++) mu[j] = (mu[j] ?? 0) + (f.data[i * f.d + j] ?? 0) / f.n
  const cov = Array.from({ length: f.d }, () => new Array<number>(f.d).fill(0))
  for (let i = 0; i < f.n; i++) {
    for (let a = 0; a < f.d; a++) {
      const da = (f.data[i * f.d + a] ?? 0) - (mu[a] ?? 0)
      const row = cov[a] as number[]
      for (let b = 0; b < f.d; b++)
        row[b] = (row[b] ?? 0) + (da * ((f.data[i * f.d + b] ?? 0) - (mu[b] ?? 0))) / (f.n - 1)
    }
  }
  return { mu, cov }
}

/**
 * The Inception score (Salimans et al., 2016) from class probabilities p(y | x) of generated samples: exp(E_x
 * KL(p(y | x) || p(y))) on each of `splits` consecutive chunks, with the mean and standard deviation over chunks.
 * mdml ships no classifier; pass the softmax outputs of Inception (or any classifier).
 *
 * @param probs - Class probabilities `[N, C]`.
 * @param options - splits (default 10).
 * @returns The mean score, with the standard deviation in the breakdown.
 * @concept generative-evaluation
 * @example
 * ```ts
 * import { inceptionScore } from 'mdml-engine/metrics'
 * Number(inceptionScore([[1, 0], [0, 1]], { splits: 1 }).estimate.value.toFixed(12)) // => 2
 * ```
 */
export const inceptionScore: MetricFunction<[TensorLike, { readonly splits?: number }?]> = defineMetric(
  spec('inceptionScore', 'higher', [1, Infinity]),
  (probs: TensorLike, options: { readonly splits?: number } = {}): MetricResult => {
    const p = rows('inceptionScore', probs)
    const splits = options.splits ?? 10
    const scores: number[] = []
    for (let s = 0; s < splits; s++) {
      const lo = Math.floor((s * p.n) / splits)
      const hi = Math.floor(((s + 1) * p.n) / splits)
      const marginal = new Float64Array(p.d)
      for (let i = lo; i < hi; i++)
        for (let j = 0; j < p.d; j++) marginal[j] = (marginal[j] ?? 0) + (p.data[i * p.d + j] ?? 0) / (hi - lo)
      let kl = 0
      for (let i = lo; i < hi; i++) {
        for (let j = 0; j < p.d; j++) {
          const v = p.data[i * p.d + j] ?? 0
          if (v > 0) kl += (v * (Math.log(v) - Math.log(marginal[j] ?? 1))) / (hi - lo)
        }
      }
      scores.push(Math.exp(kl))
    }
    const mean = scores.reduce((a, b) => a + b, 0) / splits
    const std = Math.sqrt(scores.reduce((a, b) => a + (b - mean) ** 2, 0) / splits)
    return metricResult('inceptionScore', mean, p.n, { std })
  },
)

/**
 * The Frechet distance between Gaussians fitted to two feature sets (Heusel et al., 2017; pytorch-fid's
 * calculate_frechet_distance): |mu_1 - mu_2|^2 + tr(S_1 + S_2 - 2 (S_1 S_2)^(1/2)), with tr (S_1 S_2)^(1/2) computed as
 * the sum of square roots of the eigenvalues of S_1^(1/2) S_2 S_1^(1/2). Pass Inception pool features for FID.
 *
 * @param realFeatures - Features of real samples `[N, D]`.
 * @param fakeFeatures - Features of generated samples `[M, D]`.
 * @returns The metric result.
 * @example
 * ```ts
 * import { frechetDistance } from 'mdml-engine/metrics'
 * frechetDistance([[0, 0], [2, 0], [0, 2]], [[0, 0], [2, 0], [0, 2]]).estimate.value < 1e-12 // => true
 * ```
 */
export const frechetDistance: MetricFunction<[TensorLike, TensorLike]> = defineMetric(
  spec('frechetDistance', 'lower'),
  (realFeatures: TensorLike, fakeFeatures: TensorLike): MetricResult => {
    const a = moments(rows('frechetDistance', realFeatures))
    const b = moments(rows('frechetDistance', fakeFeatures))
    const root = funmSymmetric(tensor(a.cov), (w) => sqrt(relu(w)))
    const inner = matmul(matmul(root, tensor(b.cov)), root)
    const sym = shapedArray(inner)
    const d = a.mu.length
    // Symmetrise against rounding before the eigenvalues.
    const s = Array.from({ length: d }, (_, i) =>
      Array.from({ length: d }, (_, j) => ((sym.data[i * d + j] ?? 0) + (sym.data[j * d + i] ?? 0)) / 2),
    )
    const traceRoot = toFlat(eigvalsh(s)).reduce((acc, v) => acc + Math.sqrt(Math.max(v, 0)), 0)
    let diff = 0
    let tr = 0
    for (let i = 0; i < d; i++) {
      diff += ((a.mu[i] ?? 0) - (b.mu[i] ?? 0)) ** 2
      tr += (a.cov[i]?.[i] ?? 0) + (b.cov[i]?.[i] ?? 0)
    }
    return metricResult('frechetDistance', diff + tr - 2 * traceRoot, d)
  },
)

/** The unbiased MMD^2 of two feature sets under the polynomial kernel (x . y / d + 1)^3 (torchmetrics' poly_mmd). */
function polyMmd(
  x: { n: number; d: number; data: Float64Array },
  y: { n: number; d: number; data: Float64Array },
  ix: readonly number[],
  iy: readonly number[],
  degree: number,
  gamma: number,
  coef: number,
): number {
  const k = (a: Float64Array, i: number, b: Float64Array, j: number): number => {
    let dot = 0
    for (let t = 0; t < x.d; t++) dot += (a[i * x.d + t] ?? 0) * (b[j * x.d + t] ?? 0)
    return (gamma * dot + coef) ** degree
  }
  const m = ix.length
  const n = iy.length
  let kxx = 0
  let kyy = 0
  let kxy = 0
  for (const [p, i] of ix.entries()) for (const [q, j] of ix.entries()) if (p !== q) kxx += k(x.data, i, x.data, j)
  for (const [p, i] of iy.entries()) for (const [q, j] of iy.entries()) if (p !== q) kyy += k(y.data, i, y.data, j)
  for (const i of ix) for (const j of iy) kxy += k(x.data, i, y.data, j)
  return kxx / (m * (m - 1)) + kyy / (n * (n - 1)) - (2 * kxy) / (m * n)
}

/** Options of {@link kernelInceptionDistance}. */
export interface KidOptions {
  /** The number of random subsets (default 100), each of `subsetSize` samples (default 1000, capped at the set sizes). */
  readonly subsets?: number
  readonly subsetSize?: number
  /** The polynomial kernel (gamma x . y + coef)^degree: degree 3, gamma 1 / D, coef 1 by default. */
  readonly degree?: number
  readonly gamma?: number
  readonly coef?: number
  /** The key for drawing subsets (default key(0)); unused when one subset covers both sets. */
  readonly key?: Key
}

/**
 * The kernel Inception distance (Binkowski et al., 2018; torchmetrics' KernelInceptionDistance) from features: the
 * unbiased MMD^2 between real and generated features under a cubic polynomial kernel, averaged over random subsets
 * (mean and standard deviation). With one subset as large as both sets it is the MMD^2 of all the features.
 *
 * @param realFeatures - Features of real samples `[N, D]`.
 * @param fakeFeatures - Features of generated samples `[M, D]`.
 * @param options - subsets, subsetSize, kernel parameters and key.
 * @returns The mean over subsets, with the standard deviation in the breakdown.
 * @example
 * ```ts
 * import { kernelInceptionDistance } from 'mdml-engine/metrics'
 * const f = [[0, 1], [1, 0], [1, 1]]
 * // The unbiased estimate of a set against itself is slightly negative.
 * kernelInceptionDistance(f, f, { subsets: 1 }).estimate.value < 0 // => true
 * ```
 */
export const kernelInceptionDistance: MetricFunction<[TensorLike, TensorLike, KidOptions?]> = defineMetric(
  spec('kernelInceptionDistance', 'lower', [-Infinity, Infinity]),
  (realFeatures: TensorLike, fakeFeatures: TensorLike, options: KidOptions = {}): MetricResult => {
    const x = rows('kernelInceptionDistance', realFeatures)
    const y = rows('kernelInceptionDistance', fakeFeatures)
    const size = Math.min(options.subsetSize ?? 1000, x.n, y.n)
    const subsets = options.subsets ?? 100
    const degree = options.degree ?? 3
    const gamma = options.gamma ?? 1 / x.d
    const coef = options.coef ?? 1
    const values: number[] = []
    const all = (n: number): number[] => Array.from({ length: n }, (_, i) => i)
    if (subsets === 1 && (options.subsetSize ?? 1000) >= Math.max(x.n, y.n))
      values.push(polyMmd(x, y, all(x.n), all(y.n), degree, gamma, coef))
    else {
      for (const k of split(options.key ?? makeKey(0), subsets)) {
        const [kx, ky] = split(k, 2)
        const ix = toFlat(permutation(kx ?? k, x.n)).slice(0, size)
        const iy = toFlat(permutation(ky ?? k, y.n)).slice(0, size)
        values.push(polyMmd(x, y, ix, iy, degree, gamma, coef))
      }
    }
    const mean = values.reduce((a, b) => a + b, 0) / values.length
    const std = Math.sqrt(values.reduce((a, b) => a + (b - mean) ** 2, 0) / values.length)
    return metricResult('kernelInceptionDistance', mean, values.length, { std })
  },
)

/** Euclidean distances between rows. */
function distances(
  a: { n: number; d: number; data: Float64Array },
  b: { n: number; d: number; data: Float64Array },
): Float64Array {
  const out = new Float64Array(a.n * b.n)
  for (let i = 0; i < a.n; i++) {
    for (let j = 0; j < b.n; j++) {
      let s = 0
      for (let t = 0; t < a.d; t++) s += ((a.data[i * a.d + t] ?? 0) - (b.data[j * b.d + t] ?? 0)) ** 2
      out[i * b.n + j] = Math.sqrt(s)
    }
  }
  return out
}

/** The distance of each row to its k-th nearest neighbour in its own set (self excluded). */
function radii(f: { n: number; d: number; data: Float64Array }, k: number): Float64Array {
  const d = distances(f, f)
  return Float64Array.from({ length: f.n }, (_, i) => {
    const row = Array.from(d.subarray(i * f.n, (i + 1) * f.n)).sort((x, y) => x - y)
    return row[k] ?? Number.POSITIVE_INFINITY
  })
}

/** Precision, recall, density and coverage of generated samples. */
export interface Prdc {
  readonly precision: number
  readonly recall: number
  readonly density: number
  readonly coverage: number
}

/**
 * Precision and recall of generative models (Kynkaanniemi et al., 2019) with density and coverage (Naeem et al.,
 * 2020; the prdc package): manifolds are unions of k-nearest-neighbour balls; precision is the fraction of fakes in
 * the real manifold, recall of reals in the fake manifold, density the mean number of real balls containing a fake
 * over k, coverage the fraction of real balls containing a fake.
 *
 * @param realFeatures - Features of real samples `[N, D]`.
 * @param fakeFeatures - Features of generated samples `[M, D]`.
 * @param options - nearestK (default 5).
 * @returns The four scores.
 * @example
 * ```ts
 * import { precisionRecallDensityCoverage } from 'mdml-engine/metrics'
 * const f = [[0, 0], [1, 0], [0, 1], [1, 1]]
 * precisionRecallDensityCoverage(f, f, { nearestK: 1 }).coverage // => 1
 * ```
 */
export function precisionRecallDensityCoverage(
  realFeatures: TensorLike,
  fakeFeatures: TensorLike,
  options: { readonly nearestK?: number } = {},
): Prdc {
  const k = options.nearestK ?? 5
  const r = rows('precisionRecallDensityCoverage', realFeatures)
  const f = rows('precisionRecallDensityCoverage', fakeFeatures)
  const rr = radii(r, k)
  const fr = radii(f, k)
  const d = distances(r, f)
  let precision = 0
  let density = 0
  for (let j = 0; j < f.n; j++) {
    let inside = 0
    for (let i = 0; i < r.n; i++) if ((d[i * f.n + j] ?? 0) < (rr[i] ?? 0)) inside++
    if (inside > 0) precision++
    density += inside
  }
  let recall = 0
  let coverage = 0
  for (let i = 0; i < r.n; i++) {
    let any = false
    let nearest = Number.POSITIVE_INFINITY
    for (let j = 0; j < f.n; j++) {
      const v = d[i * f.n + j] ?? 0
      if (v < (fr[j] ?? 0)) any = true
      nearest = Math.min(nearest, v)
    }
    if (any) recall++
    if (nearest < (rr[i] ?? 0)) coverage++
  }
  return { precision: precision / f.n, recall: recall / r.n, density: density / (k * f.n), coverage: coverage / r.n }
}

/**
 * CLIPScore from embeddings (Hessel et al., 2021; torchmetrics' CLIPScore): the mean over pairs of
 * max(100 cos(image, text), 0). mdml ships no CLIP model; pass its image and text embeddings.
 *
 * @param imageEmbeddings - Image embeddings `[N, D]`.
 * @param textEmbeddings - Matching text embeddings `[N, D]`.
 * @returns The metric result, in [0, 100].
 * @example
 * ```ts
 * import { clipScore } from 'mdml-engine/metrics'
 * clipScore([[1, 0]], [[1, 0]]).estimate.value // => 100
 * ```
 */
export const clipScore: MetricFunction<[TensorLike, TensorLike]> = defineMetric(
  spec('clipScore', 'higher', [0, 100]),
  (imageEmbeddings: TensorLike, textEmbeddings: TensorLike): MetricResult => {
    const a = rows('clipScore', imageEmbeddings)
    const b = rows('clipScore', textEmbeddings)
    let total = 0
    for (let i = 0; i < a.n; i++) {
      let dot = 0
      let na = 0
      let nb = 0
      for (let t = 0; t < a.d; t++) {
        const x = a.data[i * a.d + t] ?? 0
        const y = b.data[i * b.d + t] ?? 0
        dot += x * y
        na += x * x
        nb += y * y
      }
      total += Math.max((100 * dot) / (Math.sqrt(na) * Math.sqrt(nb)), 0)
    }
    return metricResult('clipScore', total / a.n, a.n)
  },
)

/**
 * The Vendi score (Friedman and Dieng, 2023): exp of the Shannon entropy (order q = 1, or the Renyi entropy of order
 * q) of the eigenvalues of K / n for a similarity matrix with unit diagonal, the effective number of distinct samples.
 * Given embeddings, K is their cosine similarity.
 *
 * @param x - Embeddings `[N, D]`, or a similarity matrix `[N, N]` with `similarity: true`.
 * @param options - similarity (x is already K) and q (default 1).
 * @returns The metric result, in [1, N].
 * @example
 * ```ts
 * import { vendiScore } from 'mdml-engine/metrics'
 * Number(vendiScore([[1, 0], [0, 1]]).estimate.value.toFixed(6)) // => 2
 * ```
 */
export const vendiScore: MetricFunction<[TensorLike, { readonly similarity?: boolean; readonly q?: number }?]> =
  defineMetric(
    spec('vendiScore', 'higher', [1, Infinity]),
    (x: TensorLike, options: { readonly similarity?: boolean; readonly q?: number } = {}): MetricResult => {
      const m = rows('vendiScore', x)
      let k: number[][]
      if (options.similarity === true) {
        k = Array.from({ length: m.n }, (_, i) => Array.from(m.data.subarray(i * m.d, (i + 1) * m.d)))
      } else {
        const norms = Array.from({ length: m.n }, (_, i) => {
          let s = 0
          for (let t = 0; t < m.d; t++) s += (m.data[i * m.d + t] ?? 0) ** 2
          return Math.sqrt(s)
        })
        k = Array.from({ length: m.n }, (_, i) =>
          Array.from({ length: m.n }, (_, j) => {
            let dot = 0
            for (let t = 0; t < m.d; t++) dot += (m.data[i * m.d + t] ?? 0) * (m.data[j * m.d + t] ?? 0)
            return dot / ((norms[i] ?? 1) * (norms[j] ?? 1))
          }),
        )
      }
      const lambdas = toFlat(eigvalsh(k.map((row) => row.map((v) => v / m.n)))).filter((v) => v > 0)
      const q = options.q ?? 1
      const value =
        q === 1
          ? Math.exp(-lambdas.reduce((a, l) => a + l * Math.log(l), 0))
          : q === Number.POSITIVE_INFINITY
            ? 1 / lambdas.reduce((a, l) => Math.max(a, l), 0)
            : lambdas.reduce((a, l) => a + l ** q, 0) ** (1 / (1 - q))
      return metricResult('vendiScore', value, m.n)
    },
  )
