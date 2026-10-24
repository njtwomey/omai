import { WhiteboxError } from '../base/index.ts'
import { categorical, foldIn, type Key, permutation, randint, uniform } from '../random/index.ts'

/**
 * The order in which rows are visited in one epoch (a torch `Sampler`): a function of the epoch number, so each epoch
 * of a random sampler is reshuffled from its own derived key and every epoch is reproducible.
 *
 * @concept datasets
 */
export type Sampler = (epoch: number) => Int32Array

/**
 * Visit rows `0, 1, ..., n - 1` in order every epoch.
 *
 * @param n - The number of rows.
 * @returns The sampler.
 * @example
 * ```ts
 * import { sequentialSampler } from 'whitebox-engine/data'
 * Array.from(sequentialSampler(3)(0)) // => [0, 1, 2]
 * ```
 */
export function sequentialSampler(n: number): Sampler {
  return () => Int32Array.from({ length: n }, (_, i) => i)
}

/** Options of {@link randomSampler}. */
export interface RandomSamplerOptions {
  /** Draw with replacement (default false: a permutation). */
  readonly replacement?: boolean
  /** Rows per epoch (default `n`; more than `n` needs replacement). */
  readonly numSamples?: number
}

/**
 * Visit rows in a random order, reshuffled every epoch (torch `RandomSampler`).
 *
 * @param n - The number of rows.
 * @param key - The base key; epoch `e` uses `foldIn(key, e)`.
 * @param options - Replacement and samples per epoch.
 * @returns The sampler.
 * @throws WhiteboxError with code `value` when more samples than rows are asked for without replacement.
 * @example
 * ```ts
 * import { randomSampler } from 'whitebox-engine/data'
 * import { key } from 'whitebox-engine/random'
 * Array.from(randomSampler(4, key(0))(0)).sort() // => [0, 1, 2, 3]
 * ```
 */
export function randomSampler(n: number, key: Key, options: RandomSamplerOptions = {}): Sampler {
  const m = options.numSamples ?? n
  if (options.replacement !== true && m > n)
    throw new WhiteboxError('value', `randomSampler: ${m} samples from ${n} rows need replacement`)
  return (epoch) => {
    const k = foldIn(key, epoch)
    if (options.replacement === true) return Int32Array.from(randint(k, [m], 0, n).data)
    return permutation(k, n).data.slice(0, m)
  }
}

/**
 * Draw rows with probability proportional to their weights (torch `WeightedRandomSampler`), for example inverse class
 * frequencies to balance classes.
 *
 * @param weights - A non-negative weight per row.
 * @param numSamples - Rows per epoch.
 * @param key - The base key.
 * @param options - Whether to draw with replacement (default true).
 * @returns The sampler.
 * @throws WhiteboxError with code `value` when too few rows have positive weight for sampling without replacement.
 * @example
 * ```ts
 * import { weightedRandomSampler } from 'whitebox-engine/data'
 * import { key } from 'whitebox-engine/random'
 * Array.from(weightedRandomSampler([0, 1, 0], 3, key(0))(0)) // => [1, 1, 1]
 * ```
 */
export function weightedRandomSampler(
  weights: readonly number[],
  numSamples: number,
  key: Key,
  options: { readonly replacement?: boolean } = {},
): Sampler {
  const positive = weights.filter((w) => w > 0).length
  if (options.replacement === false && numSamples > positive)
    throw new WhiteboxError(
      'value',
      `weightedRandomSampler: ${numSamples} draws from ${positive} rows without replacement`,
    )
  return (epoch) => {
    const k = foldIn(key, epoch)
    if (options.replacement !== false) return Int32Array.from(categorical(k, weights, [numSamples]).data)
    // Without replacement: Efraimidis-Spirakis keys u^(1/w), the largest numSamples.
    const u = uniform(k, [weights.length]).data
    const scored = weights.map((w, i) => [w > 0 ? Math.log(u[i] ?? 0) / w : -Infinity, i] as const)
    scored.sort((a, b) => b[0] - a[0] || a[1] - b[1])
    return Int32Array.from(scored.slice(0, numSamples), (s) => s[1])
  }
}

/**
 * Visit a fixed subset of rows in a random order every epoch (torch `SubsetRandomSampler`).
 *
 * @param indices - The rows to visit.
 * @param key - The base key.
 * @returns The sampler.
 * @example
 * ```ts
 * import { subsetRandomSampler } from 'whitebox-engine/data'
 * import { key } from 'whitebox-engine/random'
 * Array.from(subsetRandomSampler([5, 7], key(0))(0)).sort() // => [5, 7]
 * ```
 */
export function subsetRandomSampler(indices: readonly number[], key: Key): Sampler {
  return (epoch) => {
    const p = permutation(foldIn(key, epoch), indices.length).data
    return Int32Array.from(p, (i) => indices[i] ?? 0)
  }
}

/**
 * Inverse-frequency weights per row, `n / (k * count(class))` (sklearn's `compute_sample_weight('balanced')`), for
 * class-balanced sampling or weighting.
 *
 * @param labels - The class of each row.
 * @returns A weight per row.
 * @example
 * ```ts
 * import { balancedWeights } from 'whitebox-engine/data'
 * balancedWeights([0, 0, 0, 1]) // => [0.6666666666666666, 0.6666666666666666, 0.6666666666666666, 2]
 * ```
 */
export function balancedWeights(labels: readonly number[]): number[] {
  const counts = new Map<number, number>()
  for (const c of labels) counts.set(c, (counts.get(c) ?? 0) + 1)
  const k = counts.size
  return labels.map((c) => labels.length / (k * (counts.get(c) ?? 1)))
}

/**
 * Group a sampler's order into batches of row indices (torch `BatchSampler`).
 *
 * @param sampler - The row order.
 * @param batchSize - Rows per batch.
 * @param dropLast - Drop a final batch smaller than `batchSize` (default false).
 * @returns A function from the epoch to its batches of indices.
 * @example
 * ```ts
 * import { batchSampler, sequentialSampler } from 'whitebox-engine/data'
 * batchSampler(sequentialSampler(5), 2)(0).map((b) => Array.from(b)) // => [[0, 1], [2, 3], [4]]
 * ```
 */
export function batchSampler(sampler: Sampler, batchSize: number, dropLast = false): (epoch: number) => Int32Array[] {
  if (!(batchSize >= 1)) throw new WhiteboxError('value', `batchSampler: batch size ${batchSize} must be at least 1`)
  return (epoch) => {
    const order = sampler(epoch)
    const out: Int32Array[] = []
    for (let start = 0; start < order.length; start += batchSize) {
      const stop = Math.min(order.length, start + batchSize)
      if (dropLast && stop - start < batchSize) break
      out.push(order.slice(start, stop))
    }
    return out
  }
}
