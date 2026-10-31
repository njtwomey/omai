/**
 * Made-up data for the record recipes: seeded normals and uniforms, rows of them, clouds of blobs, and a toy credit
 * model (six features, one interaction) that the attribution, dependence and fairness recipes explain. Nothing here is
 * fitted; it exists so a recipe can show a view on data whose truth is known.
 */

import type { PredictFunction } from 'mdml-engine/explain'
import { key, normal, uniform } from 'mdml-engine/random'
import { asTensor, type Tensor, type TensorLike, tensor, toArray, toFlat } from 'mdml-engine/tensor'

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
