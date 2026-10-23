import { WhiteboxError } from '../base/index.ts'
import { fromStorage, type Tensor } from '../tensor/index.ts'
import { unitDoubles } from './_unit.ts'
import { foldIn, type Key } from './key.ts'

const PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

function radicalInverse(i: number, base: number): number {
  let f = 1
  let r = 0
  let n = i
  while (n > 0) {
    f /= base
    r += f * (n % base)
    n = Math.floor(n / base)
  }
  return r
}

/**
 * Points of the Halton low-discrepancy sequence (radical inverses in the first primes), skipping the origin.
 *
 * @param n - Number of points.
 * @param d - Dimension (at most 25).
 * @returns A float64 tensor `[n, d]` in the unit cube.
 * @concept quasi-random
 * @example
 * ```ts
 * import { halton } from 'whitebox-engine/random'
 * Array.from(halton(2, 2).data) // => [0.5, 0.333333, 0.25, 0.666667]
 * ```
 */
export function halton(n: number, d: number): Tensor<'float64'> {
  if (d > PRIMES.length) throw new WhiteboxError('value', `halton: dimension at most ${PRIMES.length}`)
  const out = new Float64Array(n * d)
  for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) out[i * d + j] = radicalInverse(i + 1, PRIMES[j] ?? 2)
  return fromStorage(out, [n, d], 'float64')
}

/**
 * A Latin hypercube sample: each axis is split into `n` strata and every stratum holds exactly one point.
 *
 * @param k - The random key.
 * @param n - Number of points.
 * @param d - Dimension.
 * @returns A float64 tensor `[n, d]` in the unit cube.
 * @concept quasi-random
 * @example
 * ```ts
 * import { key, latinHypercube } from 'whitebox-engine/random'
 * const x = latinHypercube(key(0), 4, 1)
 * Array.from(x.data, (v) => Math.floor(v * 4)).sort() // => [0, 1, 2, 3]
 * ```
 */
export function latinHypercube(k: Key, n: number, d: number): Tensor<'float64'> {
  const out = new Float64Array(n * d)
  for (let j = 0; j < d; j++) {
    const kj = foldIn(k, j)
    const u = unitDoubles(foldIn(kj, 0), 2 * n)
    // Shuffle the strata with the first n uniforms, jitter within them with the rest.
    const strata = Array.from({ length: n }, (_, i) => i)
    for (let i = n - 1; i > 0; i--) {
      const r = Math.floor((u[i] ?? 0) * (i + 1))
      ;[strata[i], strata[r]] = [strata[r] as number, strata[i] as number]
    }
    for (let i = 0; i < n; i++) out[i * d + j] = ((strata[i] ?? 0) + (u[n + i] ?? 0)) / n
  }
  return fromStorage(out, [n, d], 'float64')
}
