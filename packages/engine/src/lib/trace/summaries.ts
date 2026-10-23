import type { TensorSummary } from './events.ts'

/**
 * Summary statistics of a tensor's values: count, mean, standard deviation (two-pass), extremes, Euclidean norm,
 * fraction of zeros and a 10-bin histogram. Non-finite values are excluded from the moments and counted as NaN mean
 * when all values are non-finite.
 *
 * @param values - The values.
 * @param bins - Histogram bins (default 10).
 * @returns The summary.
 * @example
 * ```ts
 * import { summarise } from 'whitebox-engine/trace'
 * summarise([0, 3, 4]).norm // => 5
 * ```
 */
export function summarise(values: ArrayLike<number>, bins = 10): TensorSummary {
  const finite: number[] = []
  let zeros = 0
  for (let i = 0; i < values.length; i++) {
    const v = values[i] ?? Number.NaN
    if (v === 0) zeros += 1
    if (Number.isFinite(v)) finite.push(v)
  }
  const n = finite.length
  let sum = 0
  let min = Infinity
  let max = -Infinity
  let sq = 0
  for (const v of finite) {
    sum += v
    sq += v * v
    if (v < min) min = v
    if (v > max) max = v
  }
  const mean = n === 0 ? Number.NaN : sum / n
  let dev = 0
  for (const v of finite) dev += (v - mean) ** 2
  const width = max > min ? (max - min) / bins : 1
  const edges = Array.from({ length: bins + 1 }, (_, i) => (n === 0 ? 0 : min + i * width))
  const counts = new Array<number>(bins).fill(0)
  for (const v of finite) {
    const b = max > min ? Math.min(bins - 1, Math.floor((v - min) / width)) : 0
    counts[b] = (counts[b] ?? 0) + 1
  }
  return {
    n: values.length,
    mean,
    std: n > 0 ? Math.sqrt(dev / n) : Number.NaN,
    min: n > 0 ? min : Number.NaN,
    max: n > 0 ? max : Number.NaN,
    norm: Math.sqrt(sq),
    zeros: values.length > 0 ? zeros / values.length : 0,
    histogram: { edges, counts },
  }
}
