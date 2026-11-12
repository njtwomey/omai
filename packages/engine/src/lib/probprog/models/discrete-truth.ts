/**
 * Truths for discrete quantities and for models whose exact posterior is a weighted enumeration: `pmfOracle` is the marginal
 * oracle of a quantity on a finite set of values (a hidden state, a count), with a right-continuous step CDF and the smallest
 * value whose CDF reaches `p` as the quantile function; `hmmSmoothing` is forward-backward on a discrete-state chain with
 * Gaussian emissions, written out here and independent of every engine of the library (it also returns the exact
 * distribution of the number of steps spent in a state, by a forward pass that tracks the count).
 */

import { ones, toArray, toFlat, zeros } from '../../tensor/index.ts'
import type { MarginalOracle } from '../validation/index.ts'

/**
 * The oracle of a quantity with a finite support.
 *
 * @param name - The quantity's name.
 * @param values - The values it takes.
 * @param probs - Their probabilities (normalised here).
 * @returns The oracle: mean, sd, a step CDF and the quantile function (the smallest value whose CDF is at least `p`).
 * @concept posterior-validation
 * @example
 * ```ts
 * import { pmfOracle } from 'omai-engine/probprog/models'
 * const o = pmfOracle('z', [0, 1, 2], [0.2, 0.5, 0.3])
 * [o.mean, o.cdf(1), o.quantile(0.5), o.quantile(0.9)] // => [1.1, 0.7, 1, 2]
 * ```
 */
export function pmfOracle(name: string, values: readonly number[], probs: readonly number[]): MarginalOracle {
  const total = probs.reduce((a, p) => a + p, 0)
  const order = values.map((_, i) => i).sort((a, b) => (values[a] as number) - (values[b] as number))
  const x = order.map((i) => values[i] as number)
  const p = order.map((i) => (probs[i] as number) / total)
  const mean = p.reduce((a, q, i) => a + q * (x[i] as number), 0)
  const variance = p.reduce((a, q, i) => a + q * ((x[i] as number) - mean) ** 2, 0)
  return {
    name,
    mean,
    sd: Math.sqrt(variance),
    cdf: (v) => p.reduce((a, q, i) => ((x[i] as number) <= v ? a + q : a), 0),
    quantile: (q) => {
      let acc = 0
      for (let i = 0; i < x.length; i++) {
        acc += p[i] as number
        if (acc >= q - 1e-12) return x[i] as number
      }
      return x[x.length - 1] as number
    },
  }
}

/** The smoothing of a Gaussian-emission hidden Markov chain: marginals, evidence and the distribution of the visits to a state. */
export interface HmmSmoothing {
  /** `marginals[t][k]` is p(z_t = k | y). */
  readonly marginals: readonly (readonly number[])[]
  readonly logEvidence: number
  /** `visits[k][c]` is the probability that exactly `c` of the steps are in state `k`, given y. */
  readonly visits: readonly (readonly number[])[]
}

/**
 * Forward-backward on a discrete-state chain with Gaussian emissions `y_t ~ N(means[z_t], sd)`: the posterior marginal of each
 * state, the log evidence, and the exact distribution of the number of steps in each state (a forward pass over (state,
 * count), so the dependence between the steps is kept, which independent marginals would lose).
 *
 * @param y - The observations.
 * @param initial - The distribution of the first state.
 * @param transition - Row `i` is the distribution of the next state given state `i`.
 * @param means - The emission means.
 * @param sd - The emission standard deviation.
 * @returns The smoothing.
 * @concept hidden-markov-models
 * @example
 * ```ts
 * import { hmmSmoothing } from 'omai-engine/probprog/models'
 * const s = hmmSmoothing([0, 1], [0.5, 0.5], [[0.9, 0.1], [0.1, 0.9]], [0, 1], 1)
 * [s.marginals.length, Math.abs(s.marginals[0]?.reduce((a, v) => a + v, 0) - 1) < 1e-12] // => [2, true]
 * ```
 */
export function hmmSmoothing(
  y: readonly number[],
  initial: readonly number[],
  transition: readonly (readonly number[])[],
  means: readonly number[],
  sd: number,
): HmmSmoothing {
  const T = y.length
  const K = initial.length
  const lik = (t: number, k: number): number =>
    Math.exp(-0.5 * (((y[t] as number) - (means[k] as number)) / sd) ** 2) / (sd * Math.sqrt(2 * Math.PI))
  const row = (i: number): readonly number[] => transition[i] as readonly number[]
  const alpha: number[][] = []
  const scale: number[] = []
  for (let t = 0; t < T; t++) {
    // banned-ok: reference: an exact answer written out in plain arithmetic to check the library against
    const a = Array.from({ length: K }, (_, k) => {
      const prior =
        t === 0
          ? (initial[k] as number)
          : (alpha[t - 1] as number[]).reduce((s, v, i) => s + v * (row(i)[k] as number), 0)
      return prior * lik(t, k)
    })
    scale[t] = a.reduce((s, v) => s + v, 0)
    alpha[t] = a.map((v) => v / (scale[t] as number))
  }
  const beta = toArray(ones([T, K])) as number[][]
  for (let t = T - 2; t >= 0; t--)
    beta[t] = Array.from(
      { length: K },
      (_, i) =>
        row(i).reduce((s, v, j) => s + v * lik(t + 1, j) * ((beta[t + 1] as number[])[j] as number), 0) /
        (scale[t + 1] as number),
    )
  const marginals = alpha.map((a, t) => a.map((v, k) => v * ((beta[t] as number[])[k] as number)))
  // The count of visits to each state: f[k'][c] after step t is the scaled probability of (state k', c visits to the target).
  // banned-ok: reference: an exact answer written out in plain arithmetic to check the library against
  const visits = Array.from({ length: K }, (_, target) => {
    let f = toArray(zeros([K, T + 1])) as number[][]
    for (let t = 0; t < T; t++) {
      const next = toArray(zeros([K, T + 1])) as number[][]
      for (let k = 0; k < K; k++) {
        const hit = k === target ? 1 : 0
        for (let c = 0; c <= T - hit; c++) {
          const prior =
            t === 0
              ? c === 0
                ? (initial[k] as number)
                : 0
              : f.reduce((s, fi, i) => s + ((fi[c] as number) ?? 0) * (row(i)[k] as number), 0)
          ;(next[k] as number[])[c + hit] =
            ((next[k] as number[])[c + hit] as number) + (prior * lik(t, k)) / (scale[t] as number)
        }
      }
      f = next
    }
    const pmf = toFlat(zeros([T + 1]))
    for (const fk of f)
      fk.forEach((v, c) => {
        pmf[c] = (pmf[c] as number) + v
      })
    return pmf
  })
  return { marginals, logEvidence: scale.reduce((s, v) => s + Math.log(v), 0), visits }
}
