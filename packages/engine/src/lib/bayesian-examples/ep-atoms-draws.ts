/**
 * The seeded draws the atoms generate their data from: a stream of named draws (normals, uniforms, coin flips, counts) of
 * which the first `n` are the same whatever `n` is, so adding a measurement to an atom keeps the ones it had.
 */

import { bernoulli, foldIn, key, normal, poisson, uniform } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'

/** The most draws one call makes (the size of the longest atom). */
const MOST = 64

/** A seeded stream of draws; each method makes its own draws from the seed, so the order of calls does not matter. */
export interface AtomDraws {
  /** `n` standard normals. */
  readonly normals: (n: number, stream?: number) => number[]
  /** `n` uniforms on `[low, high)`. */
  readonly uniforms: (n: number, low?: number, high?: number, stream?: number) => number[]
  /** `n` coin flips (0 or 1) with probability `p` of 1. */
  readonly flips: (n: number, p: number, stream?: number) => number[]
  /** `n` Poisson counts with a rate each (a number, or one per draw). */
  readonly counts: (n: number, rate: number | readonly number[], stream?: number) => number[]
}

/**
 * The draws of a seed.
 *
 * @param seed - The seed.
 * @returns The stream.
 * @example
 * ```ts
 * import { atomDraws } from 'omai-engine/bayesian-examples'
 * atomDraws(3).normals(2)[1] === atomDraws(3).normals(5)[1] // => true
 * ```
 */
export function atomDraws(seed: number): AtomDraws {
  const at = (stream: number, salt: number): ReturnType<typeof key> => foldIn(foldIn(key(seed), salt), stream)
  return {
    normals: (n, stream = 0) => Array.from(toFlat(normal(at(stream, 1), [MOST])).slice(0, n)),
    uniforms: (n, low = 0, high = 1, stream = 0) =>
      Array.from(toFlat(uniform(at(stream, 2), [MOST], { low, high })).slice(0, n)),
    flips: (n, p, stream = 0) => Array.from(toFlat(bernoulli(at(stream, 3), p, [MOST])).slice(0, n), Number),
    counts: (n, rate, stream = 0) => {
      const rates = typeof rate === 'number' ? Array.from({ length: n }, () => rate) : [...rate]
      return rates.slice(0, n).map((r, i) => Number(toFlat(poisson(foldIn(at(stream, 4), i), r, [1]))[0] ?? 0))
    },
  }
}
