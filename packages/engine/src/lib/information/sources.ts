/**
 * Sources to code, with the truth about them: a memoryless source drawing symbols from a distribution, and a Markov
 * source whose symbol depends on the last. Each comes with its exact entropy (rate) and the information content of the
 * sample it produced, the numbers a compressor can approach and not beat on average: an order-0 coder reaches the
 * entropy of a memoryless source, and misses the lower entropy rate of a Markov source.
 */

import { MdmlError } from '../base/index.ts'
import { solve } from '../numerics/linalg/index.ts'
import { categorical, foldIn, type Key, uniform } from '../random/index.ts'
import { type TensorLike, toFlat } from '../tensor/index.ts'
import { float } from './_shared.ts'

/** A sample from a source with the truth about it. */
export interface SourceSample {
  /** The symbols. */
  readonly symbols: readonly number[]
  /** The source's entropy (rate) in bits per symbol. */
  readonly entropy: number
  /** The information content of this sample, `-sum log2 p`, in bits (conditional on the previous symbol for a Markov source). */
  readonly ideal: number
}

function normalised(p: TensorLike, name: string): number[] {
  const xs = Array.from(toFlat(float(p)))
  const total = xs.reduce((a, b) => a + b, 0)
  if (xs.length === 0 || xs.some((x) => !(x >= 0)) || !(total > 0))
    throw new MdmlError('value', `${name}: need a non-empty vector of non-negative weights`)
  return xs.map((x) => x / total)
}

const bits = (p: readonly number[]): number => p.reduce((a, x) => a - (x > 0 ? x * Math.log2(x) : 0), 0)

/**
 * `n` independent symbols from a distribution, with its entropy H = -sum p log2 p (bits per symbol) and the information
 * content of the sample. By the law of large numbers `ideal / n` tends to H; no lossless code does better on average.
 *
 * @param key - The random key.
 * @param probs - Symbol probabilities (or weights).
 * @param n - The number of symbols.
 * @returns The symbols, the entropy and the sample's information content.
 * @throws MdmlError with code `value` for bad probabilities.
 * @concept source-entropy
 * @example
 * ```ts
 * import { memorylessSource } from 'mdml-engine/information'
 * import { key } from 'mdml-engine/random'
 * const s = memorylessSource(key(0), [0.5, 0.25, 0.25], 10)
 * [s.symbols.length, s.entropy] // => [10, 1.5]
 * ```
 */
export function memorylessSource(key: Key, probs: TensorLike, n: number): SourceSample {
  const p = normalised(probs, 'memorylessSource')
  const symbols = Array.from(categorical(key, p, [n]).data)
  return { symbols, entropy: bits(p), ideal: symbols.reduce((a, s) => a - Math.log2(p[s] ?? 1), 0) }
}

/** A sample from a Markov source. */
export interface MarkovSample extends SourceSample {
  /** The stationary distribution of the chain. */
  readonly stationary: readonly number[]
}

/**
 * `n` symbols of a first-order Markov source with the given transition matrix (rows: the last symbol), started from the
 * stationary distribution. The entropy rate H = sum_i pi_i H(P_i) (bits per symbol), with pi the stationary distribution,
 * is below the entropy of the stationary distribution itself when the symbols depend on each other: a coder that
 * ignores the dependence pays the larger number.
 *
 * @param key - The random key.
 * @param transition - Row-stochastic transition probabilities.
 * @param n - The number of symbols.
 * @returns The symbols, the entropy rate, the stationary distribution and the sample's conditional information content.
 * @throws MdmlError with code `value` for a non-square matrix or rows that do not sum to one.
 * @concept source-entropy
 * @example
 * ```ts
 * import { markovSource } from 'mdml-engine/information'
 * import { key } from 'mdml-engine/random'
 * const s = markovSource(key(0), [[0.9, 0.1], [0.1, 0.9]], 5)
 * [s.stationary, Math.round(s.entropy * 1e6) / 1e6] // => [[0.5, 0.5], 0.468996]
 * ```
 */
export function markovSource(key: Key, transition: readonly (readonly number[])[], n: number): MarkovSample {
  const m = transition.length
  if (m === 0 || transition.some((r) => r.length !== m))
    throw new MdmlError('value', 'markovSource: the transition matrix must be square')
  const rows = transition.map((r) => normalised(r, 'markovSource'))
  if (transition.some((r) => Math.abs(r.reduce((a, b) => a + b, 0) - 1) > 1e-9))
    throw new MdmlError('value', 'markovSource: the rows of the transition matrix must sum to 1')
  // The stationary distribution solves pi (P - I) = 0 with the sum of pi equal to 1.
  const a = Array.from({ length: m }, (_, i) =>
    Array.from({ length: m }, (_, j) => (rows[j]?.[i] ?? 0) - (i === j ? 1 : 0)),
  )
  a[m - 1] = Array<number>(m).fill(1)
  const stationary = Array.from(
    toFlat(
      solve(
        a,
        Array.from({ length: m }, (_, i) => (i === m - 1 ? 1 : 0)),
      ),
    ),
  )
  const draws = uniform(foldIn(key, 1), [Math.max(n, 1)]).data
  const pick = (p: readonly number[], x: number): number => {
    let acc = 0
    for (let i = 0; i < p.length; i++) {
      acc += p[i] ?? 0
      if (x < acc) return i
    }
    return p.length - 1
  }
  const symbols: number[] = []
  let ideal = 0
  let last = pick(stationary, draws[0] ?? 0)
  for (let t = 0; t < n; t++) {
    const next = t === 0 ? last : pick(rows[last] ?? [], draws[t] ?? 0)
    ideal -= Math.log2(t === 0 ? (stationary[next] ?? 1) : (rows[last]?.[next] ?? 1))
    symbols.push(next)
    last = next
  }
  return { symbols, entropy: stationary.reduce((acc, pi, i) => acc + pi * bits(rows[i] ?? []), 0), ideal, stationary }
}
