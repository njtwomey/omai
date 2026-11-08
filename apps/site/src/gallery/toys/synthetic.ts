/**
 * Made-up data for the record recipes: seeded normals and uniforms, rows of them, clouds of blobs, and a toy credit
 * model (six features, one interaction) that the attribution, dependence and fairness recipes explain. Nothing here is
 * fitted; it exists so a recipe can show a view on data whose truth is known.
 */

import type { PredictFunction } from 'omai-engine/explain'
import { key, normal, uniform } from 'omai-engine/random'
import { asTensor, type Tensor, type TensorLike, tensor, toArray, toFlat } from 'omai-engine/tensor'

/** `n` standard normals (times `scale`) from a seed. */
export function normals(seed: number, n: number, scale = 1): number[] {
  return (toArray(normal(key(seed), [n])) as number[]).map((v) => v * scale)
}

/** `n` uniforms in [0, 1) from a seed. */
export function uniforms(seed: number, n: number): number[] {
  return toArray(uniform(key(seed), [n])) as number[]
}

/** `n` rows of `d` standard normals. */
export function gaussianRows(seed: number, n: number, d: number): number[][] {
  const flat = normals(seed, n * d)
  return Array.from({ length: n }, (_, i) => flat.slice(i * d, (i + 1) * d))
}

/** Rows of `k` Gaussian blobs in the plane (centres on a circle), with the blob of each row. */
export function blobs(
  seed: number,
  k: number,
  perBlob: number,
  spread = 0.6,
): { readonly points: number[][]; readonly labels: number[] } {
  const noise = normals(seed, 2 * k * perBlob, spread)
  const points: number[][] = []
  const labels: number[] = []
  for (let c = 0; c < k; c++)
    for (let i = 0; i < perBlob; i++) {
      const j = c * perBlob + i
      const angle = (2 * Math.PI * c) / k + 0.4
      points.push([3 * Math.cos(angle) + (noise[2 * j] ?? 0), 3 * Math.sin(angle) + (noise[2 * j + 1] ?? 0)])
      labels.push(c)
    }
  return { points, labels }
}

/** The toy credit model's features. */
export const CREDIT_FEATURES: readonly string[] = ['income', 'debt', 'age', 'tenure', 'late payments', 'savings']

/** The toy credit model's log odds of repaying: income and savings help, debt and late payments hurt, and debt hurts more when income is low. */
export function creditScore(x: readonly number[]): number {
  const [income = 0, debt = 0, age = 0, tenure = 0, late = 0, savings = 0] = x
  return (
    0.9 * income -
    0.7 * debt +
    0.15 * age +
    0.25 * tenure -
    0.8 * late +
    0.5 * savings -
    0.5 * debt * Math.min(0, income)
  )
}

/** The credit model as a batch predictor (one log odds per row), the form the explainers take. */
export const creditPredict: PredictFunction = (x) => {
  const d = CREDIT_FEATURES.length
  const n = x.shape[0] ?? 0
  return tensor(Array.from({ length: n }, (_, i) => creditScore(Array.from(x.data.slice(i * d, (i + 1) * d)))))
}

/** `n` applicants of the toy credit model: standard normal features. */
export function creditRows(seed: number, n: number): number[][] {
  return gaussianRows(seed, n, CREDIT_FEATURES.length)
}

/** A sigmoid. */
export const sigmoid = (z: number): number => 1 / (1 + Math.exp(-z))

/** A fitted regressor in the form the inspection tools take, from a function of one row. */
export function regressor(f: (x: readonly number[]) => number, d: number): FittedLike {
  return {
    estimator: 'toy',
    kind: 'regressor',
    params: {},
    predict: (x) => {
      const t = asTensor(x)
      const flat = toFlat(t)
      return tensor(Array.from({ length: t.shape[0] ?? 0 }, (_, i) => f(Array.from(flat.slice(i * d, (i + 1) * d)))))
    },
  }
}

/** The shape of a fitted estimator the inspection tools accept. */
export interface FittedLike {
  readonly estimator: string
  readonly kind: 'regressor'
  readonly params: Record<string, never>
  readonly predict: (x: TensorLike) => Tensor
}

/** Scored rows of two groups: labels, a score per row (a probability) and the group (0 majority, 1 minority). */
export interface ScoredRows {
  readonly scores: number[]
  readonly labels: number[]
  readonly groups: number[]
}

/**
 * `n` made-up decisions about two groups. The minority (30%) has a lower base rate, and its scores are pushed down by
 * `gap` on the log-odds scale, so a single threshold selects it less often and misses more of its positives.
 */
export function scoredGroups(seed: number, n: number, gap: number): ScoredRows {
  const u = uniforms(seed, 2 * n)
  const z = normals(seed + 1, n)
  const groups = Array.from({ length: n }, (_, i) => ((u[i] ?? 0) < 0.3 ? 1 : 0))
  const labels = groups.map((g, i) => ((u[n + i] ?? 0) < (g === 1 ? 0.3 : 0.45) ? 1 : 0))
  const scores = groups.map((g, i) => sigmoid(1.6 * ((labels[i] ?? 0) - 0.5) * 2 + (z[i] ?? 0) - gap * g))
  return { scores, labels, groups }
}

/** A regression problem with sparse truth: `n` rows of `d` features, y = 3·x0 − 2·x1 + 1.5·x4 plus noise. */
export function sparseRegression(seed: number, n: number, d: number, noise: number): { x: number[][]; y: number[] } {
  const x = gaussianRows(seed, n, d)
  const e = normals(seed + 50, n, noise)
  const y = x.map((r, i) => 3 * (r[0] ?? 0) - 2 * (r[1] ?? 0) + 1.5 * (r[4] ?? 0) + (e[i] ?? 0))
  return { x, y }
}

/** A two-class problem in the plane: two interleaved crescents, each label flipped with probability `flip`. */
export function crescents(seed: number, n: number, flip: number): { x: number[][]; y: number[] } {
  const u = uniforms(seed, 3 * n)
  const e = normals(seed + 9, 2 * n, 0.18)
  const y = Array.from({ length: n }, (_, i) => i % 2)
  const x = y.map((c, i) => {
    const t = Math.PI * (u[i] ?? 0)
    return c === 0
      ? [Math.cos(t) + (e[2 * i] ?? 0), Math.sin(t) + (e[2 * i + 1] ?? 0)]
      : [1 - Math.cos(t) + (e[2 * i] ?? 0), 0.4 - Math.sin(t) + (e[2 * i + 1] ?? 0)]
  })
  return { x, y: y.map((c, i) => ((u[n + i] ?? 1) < flip ? 1 - c : c)) }
}
