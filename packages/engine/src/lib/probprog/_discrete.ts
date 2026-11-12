// Shared pieces of the discrete engines: posterior marginals on integer supports that need not start at zero, and
// flat batch indices of plated sites.
import { categorical, type Distribution, defineDistribution } from '../dists/index.ts'
import { add, reshape, sub, tensor, type Value } from '../tensor/index.ts'

/**
 * A categorical distribution over low, ..., low + K - 1 (a posterior marginal of a discrete site).
 *
 * @param probs - Probabilities `[...batch, K]` (each row sums to one).
 * @param low - The smallest state.
 * @returns The distribution.
 */
export function shiftedCategorical(probs: Value, low: number): Distribution {
  const base = categorical({ probs })
  if (low === 0) return base
  const k = probs.shape[probs.shape.length - 1] ?? 0
  return defineDistribution({
    name: 'categorical',
    params: base.params,
    support: { kind: 'integerInterval', low, high: low + k - 1 },
    batchShape: base.batchShape,
    logProb: (x) => base.logProb(sub(x, low)),
    cdf: (x) => base.cdf(sub(x, low)),
    ppf: (q) => add(base.ppf(q), low),
    sample: (key, shape) => add(base.sample(key, shape), low),
    mean: () => add(base.mean(), low),
    variance: () => base.variance(),
    mode: () => add(base.mode(), low),
    entropy: () => base.entropy(),
  })
}

/** The multi-index of flat position `i` in a row-major shape. */
export function unravel(i: number, shape: readonly number[]): number[] {
  const out = new Array<number>(shape.length).fill(0)
  let r = i
  for (let a = shape.length - 1; a >= 0; a--) {
    const n = shape[a] ?? 1
    out[a] = r % n
    r = Math.floor(r / n)
  }
  return out
}

/** A `[...shape, K]` probability tensor from per-element probability rows. */
export function probabilityTensor(rows: readonly (readonly number[])[], shape: readonly number[], k: number): Value {
  return reshape(tensor(rows.map((r) => r.slice(0, k))), [...shape, k])
}
