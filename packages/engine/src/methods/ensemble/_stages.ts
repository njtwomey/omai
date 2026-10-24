// Staged scores shared by the ensembles: the score after each member, from per-member outputs on the training rows.
import { argmaxOf } from '../../lib/base/index.ts'
import { accuracyScore, r2Score } from '../../lib/metrics/index.ts'

/** Per-member outputs on n rows: probabilities `[n * k]` (classification) or values `[n]` (regression). */
export type MemberOutputs = readonly Float64Array[]

/** Accuracy (k > 0) or R^2 of a running average of member outputs after each member, optionally on a subset of rows. */
export function stagedAverageScores(
  outputs: MemberOutputs,
  y: ArrayLike<number>,
  k: number,
  weights?: readonly number[],
): number[] {
  const n = y.length
  const width = Math.max(k, 1)
  const sum = new Float64Array(n * width)
  let total = 0
  return outputs.map((o, m) => {
    const w = weights?.[m] ?? 1
    total += w
    for (let i = 0; i < sum.length; i++) sum[i] = (sum[i] ?? 0) + w * (o[i] ?? 0)
    return scoreOf(sum, total, y, k)
  })
}

/** Accuracy of argmax rows (k > 0) or R^2 of values / total. */
export function scoreOf(sum: Float64Array, total: number, y: ArrayLike<number>, k: number): number {
  const n = y.length
  if (k > 0) {
    const pred = Array.from({ length: n }, (_, i) => argmaxOf(sum.subarray(i * k, (i + 1) * k)))
    return accuracyScore(Array.from(y), pred).estimate.value
  }
  return r2Score(
    Array.from(y),
    Array.from(sum, (v) => v / total),
  ).estimate.value
}

/**
 * Out-of-bag scores after each member: each row is predicted by the members whose bootstrap missed it (rows no member
 * has missed yet are left out of that stage's score).
 */
export function stagedOobScores(
  outputs: MemberOutputs,
  inBag: readonly Uint8Array[],
  y: ArrayLike<number>,
  k: number,
): number[] {
  const n = y.length
  const width = Math.max(k, 1)
  const sum = new Float64Array(n * width)
  const count = new Float64Array(n)
  return outputs.map((o, m) => {
    const bag = inBag[m] ?? new Uint8Array(n)
    for (let i = 0; i < n; i++) {
      if (bag[i]) continue
      count[i] = (count[i] ?? 0) + 1
      for (let c = 0; c < width; c++) sum[i * width + c] = (sum[i * width + c] ?? 0) + (o[i * width + c] ?? 0)
    }
    const rows = Array.from({ length: n }, (_, i) => i).filter((i) => (count[i] ?? 0) > 0)
    if (rows.length < 2) return Number.NaN
    if (k > 0) {
      const pred = rows.map((i) => argmaxOf(sum.subarray(i * k, (i + 1) * k)))
      return accuracyScore(
        rows.map((i) => y[i] ?? 0),
        pred,
      ).estimate.value
    }
    return r2Score(
      rows.map((i) => y[i] ?? 0),
      rows.map((i) => (sum[i] ?? 0) / (count[i] ?? 1)),
    ).estimate.value
  })
}
