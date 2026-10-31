/**
 * The Relief family of feature weights (Kira and Rendell 1992; Kononenko 1994): features are scored by how well they
 * separate an instance from its nearest neighbours of other classes (misses) while agreeing with its nearest neighbours of the
 * same class (hits). Because the neighbours are found in the full feature space, Relief credits features that matter in
 * interaction, such as the two halves of an XOR, which every univariate score misses. `reliefAlgorithm` is one sampled
 * instance per step; k = 1 with two classes is the original Relief and larger k with class-prior weighted misses is ReliefF.
 */

import { type Json, MdmlError } from '../base/index.ts'
import { type Key, key as makeKey, permutation, randint } from '../random/index.ts'
import { type TensorLike, toFlat } from '../tensor/index.ts'
import { type Algorithm, channels, runAlgorithm } from '../trace/index.ts'
import { encodeLabels, matrixView } from './validation.ts'

/** Options of {@link reliefAlgorithm} and {@link relief}. */
export interface ReliefOptions {
  /** Neighbours per class: 1 for the original Relief, 10 is ReliefF's usual choice (default 10). */
  readonly k?: number
  /** Number of sampled instances m (default n: every instance once). More than n samples with replacement. */
  readonly iterations?: number
  /** Which features are discrete (differences 0 or 1); the others are scaled by their range. */
  readonly discrete?: readonly boolean[]
}

/** State of {@link reliefAlgorithm}. */
export interface ReliefState {
  readonly t: number
  /** The weight of each feature after t sampled instances. */
  readonly weights: readonly number[]
  /** The instance sampled at each step: a random permutation (m <= n) or m draws with replacement, fixed by the run's key. */
  readonly order: readonly number[]
}

/**
 * Relief and ReliefF as a stepped algorithm, one sampled instance per step. For the sampled instance R, the weight of
 * feature f changes by - sum over its k nearest hits H of diff_f(R, H) / (m k) + sum over each other class C of
 * [P(C) / (1 - P(class(R)))] sum over its k nearest misses M of diff_f(R, M) / (m k), where diff is the range-scaled absolute
 * difference (or 0/1 for a discrete feature) and nearness is the sum of the diffs. Without sampling noise (m = n, which visits
 * each instance once) the weights are deterministic.
 *
 * @param x - The features `[n, d]`.
 * @param y - The class labels `[n]`.
 * @param options - k, the number of iterations and the discrete mask.
 * @returns The algorithm over {@link ReliefState}.
 * @concept feature-selection
 * @example
 * ```ts
 * import { reliefAlgorithm } from 'mdml-engine/learn'
 * reliefAlgorithm([[0], [1], [10], [11]], [0, 0, 1, 1], { k: 1 }).name // => 'relief'
 * ```
 */
export function reliefAlgorithm(x: TensorLike, y: TensorLike, options: ReliefOptions = {}): Algorithm<ReliefState> {
  const m = matrixView(x, 'relief')
  const { n, d } = m
  const labels = encodeLabels(y, 'relief', n)
  const classOf = labels.codes
  const nClass = labels.classes.length
  if (nClass < 2) throw new MdmlError('value', 'relief: needs at least two classes')
  const k = options.k ?? 10
  if (!(Number.isInteger(k) && k >= 1)) throw new MdmlError('value', 'relief: k must be a positive integer')
  const iterations = options.iterations ?? n
  const prior = new Float64Array(nClass)
  for (const c of classOf) prior[c] = (prior[c] ?? 0) + 1 / n
  const range = new Float64Array(d)
  for (let j = 0; j < d; j++) {
    let lo = Number.POSITIVE_INFINITY
    let hi = Number.NEGATIVE_INFINITY
    for (let i = 0; i < n; i++) {
      const v = m.data[i * d + j] ?? 0
      lo = Math.min(lo, v)
      hi = Math.max(hi, v)
    }
    range[j] = hi - lo || 1
  }
  const diff = (j: number, a: number, b: number): number => {
    const u = m.data[a * d + j] ?? 0
    const v = m.data[b * d + j] ?? 0
    return options.discrete?.[j] === true ? (u === v ? 0 : 1) : Math.abs(u - v) / (range[j] ?? 1)
  }
  return {
    name: 'relief',
    config: { n, d, k, iterations } as Json,
    schema: channels({
      metrics: { maxWeight: { description: 'the largest feature weight so far' } },
      tensors: { weights: { role: 'param', description: 'feature weights after the sampled instances so far' } },
    }),
    init: (key) => ({
      t: 0,
      weights: new Array<number>(d).fill(0),
      order:
        iterations <= n
          ? Array.from(toFlat(permutation(key, n)).slice(0, iterations))
          : Array.from(toFlat(randint(key, [iterations], 0, n))),
    }),
    step: (s) => {
      const r = s.order[s.t] ?? 0
      const dist = new Float64Array(n)
      for (let i = 0; i < n; i++) {
        if (i === r) {
          dist[i] = Number.POSITIVE_INFINITY
          continue
        }
        let acc = 0
        for (let j = 0; j < d; j++) acc += diff(j, r, i)
        dist[i] = acc
      }
      const w = Float64Array.from(s.weights)
      const own = classOf[r] ?? 0
      for (let c = 0; c < nClass; c++) {
        const members = Array.from({ length: n }, (_, i) => i).filter((i) => i !== r && classOf[i] === c)
        members.sort((a, b) => (dist[a] ?? 0) - (dist[b] ?? 0) || a - b)
        const near = members.slice(0, k)
        if (near.length === 0) continue
        const scale = c === own ? -1 : (prior[c] ?? 0) / (1 - (prior[own] ?? 0))
        for (const i of near)
          for (let j = 0; j < d; j++) w[j] = (w[j] ?? 0) + (scale * diff(j, r, i)) / (iterations * near.length)
      }
      return { t: s.t + 1, weights: Array.from(w), order: s.order }
    },
    done: (s) => (s.t >= iterations ? 'completed' : undefined),
    observe: (s) => ({
      metrics: { maxWeight: Math.max(...s.weights) },
      tensors: { weights: { role: 'param', value: s.weights } },
    }),
    serialise: (s) => ({ t: s.t, weights: [...s.weights], order: [...s.order] }),
    deserialise: (j) => j as unknown as ReliefState,
  }
}

/**
 * Relief / ReliefF feature weights ({@link reliefAlgorithm} run to the end).
 *
 * @param x - The features `[n, d]`.
 * @param y - The class labels.
 * @param options - k, iterations, discrete mask, and the seed through `key`.
 * @returns The weight of each feature (larger is more relevant) and the ranking.
 * @concept feature-selection
 * @example
 * ```ts
 * import { relief } from 'mdml-engine/learn'
 * relief([[0], [1], [10], [11]], [0, 0, 1, 1], { k: 1 }).weights.map((v) => Math.round(v * 10000) / 10000) // => [0.7727]
 * ```
 */
export function relief(
  x: TensorLike,
  y: TensorLike,
  options: ReliefOptions & { readonly key?: Key } = {},
): { readonly weights: readonly number[]; readonly ranking: readonly number[] } {
  const { key, ...rest } = options
  const alg = reliefAlgorithm(x, y, rest)
  const iterations = rest.iterations ?? matrixView(x, 'relief').n
  const r = runAlgorithm(alg, { maxSteps: iterations, key: key ?? makeKey(0) })
  const weights = r.state.weights
  return { weights, ranking: Array.from(weights.keys()).sort((a, b) => (weights[b] ?? 0) - (weights[a] ?? 0)) }
}
