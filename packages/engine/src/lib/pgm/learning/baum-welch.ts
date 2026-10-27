/**
 * Baum-Welch, expectation-maximisation for hidden Markov models (Baum et al. 1970; hmmlearn's loop): each step runs
 * forward-backward on every training sequence under the current parameters (the E-step: state posteriors, pairwise
 * posteriors and the log likelihood), then re-estimates the start, transition and emission parameters from the expected
 * counts (the M-step). Emissions are categorical or Gaussian (diagonal or full covariances with hmmlearn's covariance
 * prior). Parameters are plain data, the algorithm keeps no past, and `hmmDecode` reads a sequence under any
 * parameters: log likelihood, state posteriors and the Viterbi path.
 */

import { MdmlError } from '../../base/index.ts'
import { initialCentres } from '../../cluster/index.ts'
import { type Key, uniform } from '../../random/index.ts'
import { type Algorithm, channels, iterate } from '../../trace/index.ts'
import { forwardBackward, viterbi } from '../inference/chain.ts'
import { type HiddenMarkovModel, hiddenMarkovModel, hmmChain } from '../models/state-space.ts'

/** One observation: a symbol index (categorical emissions) or a vector (Gaussian emissions; a number is a 1-vector). */
export type HmmObservation = number | readonly number[]

/** A training or test sequence: one observation per time step. */
export type HmmSequence = readonly HmmObservation[]

/**
 * The parameters of a hidden Markov model as plain JSON: the start distribution, `transition[i][j] = P(z_{t+1} = j |
 * z_t = i)`, and the emissions: a table `emission[k][s] = P(x = s | z = k)` for categorical, a mean and a covariance
 * per state for Gaussian.
 *
 * @concept hidden-markov-models
 */
export type HmmParameters =
  | { kind: 'categorical'; start: number[]; transition: number[][]; emission: number[][] }
  | { kind: 'gaussian'; start: number[]; transition: number[][]; means: number[][]; covariances: number[][][] }

/** Options of {@link baumWelchAlgorithm}. */
export interface BaumWelchOptions {
  /** Gaussian emissions: diagonal (default, hmmlearn's) or full covariances. */
  readonly covarianceType?: 'diag' | 'full'
  /** hmmlearn's covariance prior, added to the scatter before dividing by the state occupancy (default 1e-2). */
  readonly covariancePrior?: number
  /** Converged when the log likelihood gains less than this in a step (default 1e-2, as hmmlearn). */
  readonly tol?: number
}

/**
 * The state of {@link baumWelchAlgorithm}: the parameters now, the log likelihood of the sequences under the
 * parameters the last step started from (hmmlearn's convention: the quantity EM improved) and how much it gained.
 */
export type BaumWelchState = {
  params: HmmParameters
  logLikelihood: number | null
  gain: number | null
}

const normalised = (row: readonly number[]): number[] => {
  const z = row.reduce((a, v) => a + Math.max(v, 0), 0)
  return row.map((v) => (z > 0 ? Math.max(v, 0) / z : 1 / row.length))
}

const vector = (o: HmmObservation): readonly number[] => (typeof o === 'number' ? [o] : o)

/**
 * A hidden Markov model from its parameters (state names `s0`, `s1`, ...), validated.
 *
 * @param params - The parameters.
 * @returns The core model, ready for `hmmChain`, `forwardBackward` and `hmmSample`.
 * @throws MdmlError with code `value` for rows that are not distributions or covariances that are not positive definite.
 * @concept hidden-markov-models
 * @example
 * ```ts
 * import { learning } from 'mdml-engine/pgm'
 * const params = { kind: 'categorical', start: [1, 0], transition: [[0.9, 0.1], [0.2, 0.8]], emission: [[0.7, 0.3], [0.1, 0.9]] }
 * learning.hmmFromParameters(params).states // => ['s0', 's1']
 * ```
 */
export function hmmFromParameters(params: HmmParameters): HiddenMarkovModel {
  const states = params.start.map((_, k) => `s${k}`)
  const norm = (rows: readonly (readonly number[])[]): number[][] => rows.map(normalised)
  return hiddenMarkovModel({
    states,
    initial: normalised(params.start),
    transition: norm(params.transition),
    emission:
      params.kind === 'categorical'
        ? {
            kind: 'categorical',
            symbols: (params.emission[0] ?? []).map((_, s) => String(s)),
            probabilities: norm(params.emission),
          }
        : { kind: 'gaussian', means: params.means, covariances: params.covariances },
  })
}

/**
 * A sequence read under hidden Markov model parameters: its log likelihood, the posterior of each state at each time
 * (forward-backward) and the most probable path (Viterbi).
 *
 * @param params - The parameters.
 * @param sequence - The observations, a symbol index or a vector per step.
 * @returns The log likelihood, the posteriors `[T][K]` and the Viterbi path.
 * @throws MdmlError with code `value` for a symbol or vector the model cannot emit.
 * @concept hidden-markov-models
 * @example
 * ```ts
 * import { learning } from 'mdml-engine/pgm'
 * const params = { kind: 'categorical', start: [0.5, 0.5], transition: [[0.9, 0.1], [0.1, 0.9]], emission: [[0.9, 0.1], [0.1, 0.9]] }
 * learning.hmmDecode(params, [0, 0, 1, 1]).path // => [0, 0, 1, 1]
 * ```
 */
export function hmmDecode(
  params: HmmParameters,
  sequence: HmmSequence,
): { readonly logLikelihood: number; readonly posteriors: number[][]; readonly path: number[] } {
  const chain = hmmChain(
    hmmFromParameters(params),
    sequence.map((o) => (params.kind === 'categorical' ? o : vector(o))),
  )
  const fb = forwardBackward(chain)
  return { logLikelihood: fb.logLikelihood, posteriors: fb.posteriors, path: viterbi(chain).path }
}

/** Options of {@link initialHmmParameters}. */
export interface HmmInitOptions {
  /** The number of hidden states. */
  readonly states: number
  readonly emission: 'categorical' | 'gaussian'
  /** Categorical emissions: the number of symbols (default the largest symbol seen plus one). */
  readonly symbols?: number
  /** Gaussian emissions: zero the initial covariances' off-diagonal (`diag`, default) or keep the data's (`full`). */
  readonly covarianceType?: 'diag' | 'full'
  /** Gaussian emissions: added to the data variance for the initial covariances (default 1e-3). */
  readonly minCovar?: number
  /** The initial probability of staying in a state (default `1 / states`: uniform transitions, as hmmlearn). */
  readonly stay?: number
}

/**
 * Starting parameters for Baum-Welch, as hmmlearn makes them: a uniform start distribution, uniform transitions (or
 * sticky ones with `stay`), random emission rows (categorical), or k-means++ seeds as means with the data's variance
 * as every covariance (Gaussian). States are numbered by what they favour (the expected symbol, the first coordinate of
 * the mean), so the numbering does not depend on the key.
 *
 * @param sequences - The training sequences.
 * @param options - States, emission family and the shape of the start.
 * @param key - The random key.
 * @returns The parameters.
 * @throws MdmlError with code `value` for fewer than one state, categorical data that are not symbol indices, or no data.
 * @concept hidden-markov-models
 * @example
 * ```ts
 * import { learning } from 'mdml-engine/pgm'
 * import { key } from 'mdml-engine/random'
 * const p = learning.initialHmmParameters([[0, 1, 2, 1, 0]], { states: 2, emission: 'categorical' }, key(0))
 * p.transition // => [[0.5, 0.5], [0.5, 0.5]]
 * ```
 */
export function initialHmmParameters(
  sequences: readonly HmmSequence[],
  options: HmmInitOptions,
  key: Key,
): HmmParameters {
  const K = options.states
  if (!(Number.isInteger(K) && K >= 1))
    throw new MdmlError('value', 'initialHmmParameters: states must be a positive integer')
  const rows = sequences.flat().map(vector)
  if (rows.length === 0) throw new MdmlError('value', 'initialHmmParameters: no observations')
  const stay = options.stay ?? 1 / K
  const transition = Array.from({ length: K }, (_, i) =>
    Array.from({ length: K }, (_, j) => (K === 1 ? 1 : i === j ? stay : (1 - stay) / (K - 1))),
  )
  const start = new Array<number>(K).fill(1 / K)
  if (options.emission === 'categorical') {
    let largest = 0
    for (const r of rows) {
      const v = r[0] ?? 0
      if (!(Number.isInteger(v) && v >= 0))
        throw new MdmlError('value', 'initialHmmParameters: symbols must be integers >= 0')
      largest = Math.max(largest, v)
    }
    const S = options.symbols ?? largest + 1
    const u = uniform(key, [K, S]).data
    const emission = Array.from({ length: K }, (_, k) =>
      normalised(Array.from({ length: S }, (_, s) => 0.1 + (u[k * S + s] ?? 0))),
    )
    // Numbered by the symbol each state favours, so the numbering is the same from one key to the next.
    const favoured = (row: readonly number[]): number => row.reduce((a, p, s) => a + p * s, 0)
    return { kind: 'categorical', start, transition, emission: emission.sort((a, b) => favoured(a) - favoured(b)) }
  }
  const d = rows[0]?.length ?? 1
  const means = initialCentres(rows, Math.min(K, rows.length), 'k-means++', key)
  const mu = Array.from({ length: d }, (_, a) => rows.reduce((s, r) => s + (r[a] ?? 0), 0) / rows.length)
  const minCovar = options.minCovar ?? 1e-3
  const covariance = Array.from({ length: d }, (_, a) =>
    Array.from({ length: d }, (_, b) => {
      const c =
        rows.reduce((s, r) => s + ((r[a] ?? 0) - (mu[a] ?? 0)) * ((r[b] ?? 0) - (mu[b] ?? 0)), 0) /
        Math.max(rows.length - 1, 1)
      return a === b ? c + minCovar : options.covarianceType === 'full' ? c : 0
    }),
  )
  // Numbered by the first coordinate of the mean, so the numbering is the same from one key to the next.
  means.sort((a, b) => (a[0] ?? 0) - (b[0] ?? 0))
  return { kind: 'gaussian', start, transition, means, covariances: means.map(() => covariance.map((r) => [...r])) }
}

// One EM iteration: the E-step on every sequence, then the M-step from the expected counts.
function emStep(
  params: HmmParameters,
  observations: readonly (readonly HmmObservation[])[],
  options: Required<BaumWelchOptions>,
): { params: HmmParameters; logLikelihood: number } {
  const K = params.start.length
  const model = hmmFromParameters(params)
  const categorical = params.kind === 'categorical'
  const S = categorical ? (params.emission[0]?.length ?? 0) : 0
  const d = params.kind === 'gaussian' ? (params.means[0]?.length ?? 1) : 1
  const startCounts = new Array<number>(K).fill(0)
  const trans = Array.from({ length: K }, () => new Array<number>(K).fill(0))
  const post = new Array<number>(K).fill(0)
  const symbolCounts = Array.from({ length: K }, () => new Array<number>(S).fill(0))
  const obs = Array.from({ length: K }, () => new Array<number>(d).fill(0))
  const obs2 = Array.from({ length: K }, () => Array.from({ length: d }, () => new Array<number>(d).fill(0)))
  let logLikelihood = 0
  for (const seq of observations) {
    const fb = forwardBackward(hmmChain(model, seq))
    logLikelihood += fb.logLikelihood
    for (let k = 0; k < K; k++) startCounts[k] = (startCounts[k] ?? 0) + (fb.posteriors[0]?.[k] ?? 0)
    for (const xi of fb.pairwise)
      for (let i = 0; i < K; i++)
        for (let j = 0; j < K; j++) (trans[i] as number[])[j] = (trans[i]?.[j] ?? 0) + (xi[i]?.[j] ?? 0)
    for (const [t, g] of fb.posteriors.entries()) {
      const row = categorical ? [] : vector(seq[t] ?? 0)
      const symbol = categorical ? (seq[t] as number) : 0
      for (let k = 0; k < K; k++) {
        const gk = g[k] ?? 0
        post[k] = (post[k] ?? 0) + gk
        if (categorical) (symbolCounts[k] as number[])[symbol] = (symbolCounts[k]?.[symbol] ?? 0) + gk
        else
          for (let a = 0; a < d; a++) {
            ;(obs[k] as number[])[a] = (obs[k]?.[a] ?? 0) + gk * (row[a] ?? 0)
            for (let b = 0; b < d; b++)
              ((obs2[k] as number[][])[a] as number[])[b] =
                (obs2[k]?.[a]?.[b] ?? 0) + gk * (row[a] ?? 0) * (row[b] ?? 0)
          }
      }
    }
  }
  const start = normalised(startCounts)
  const transition = trans.map(normalised)
  if (categorical)
    return { params: { kind: 'categorical', start, transition, emission: symbolCounts.map(normalised) }, logLikelihood }
  const means = obs.map((o, k) => o.map((v) => v / (post[k] ?? 1)))
  const covariances = means.map((mu, k) => {
    const n = post[k] ?? 0
    const scatter = Array.from({ length: d }, (_, a) =>
      Array.from(
        { length: d },
        (_, b) =>
          (obs2[k]?.[a]?.[b] ?? 0) -
          (obs[k]?.[a] ?? 0) * (mu[b] ?? 0) -
          (mu[a] ?? 0) * (obs[k]?.[b] ?? 0) +
          (mu[a] ?? 0) * (mu[b] ?? 0) * n,
      ),
    )
    // hmmlearn: diagonal (prior + scatter) / max(n, 1e-5); full (prior + scatter) / n, the prior on every entry.
    if (options.covarianceType === 'diag')
      return scatter.map((r, a) => r.map((v, b) => (a === b ? (options.covariancePrior + v) / Math.max(n, 1e-5) : 0)))
    return scatter.map((r) => r.map((v) => (options.covariancePrior + v) / n))
  })
  return { params: { kind: 'gaussian', start, transition, means, covariances }, logLikelihood }
}

/**
 * Baum-Welch as a traced {@link Algorithm}: one EM iteration per step, from the given parameters, converged when the
 * log likelihood gains less than `tol`. The state keeps no past (the tape does): `observe` gives the metric
 * `logLikelihood` (of the parameters each step started from, so it is non-decreasing) and the parameter tensors
 * `start` `[k]`, `transition` `[k, k]` and `emission` (the categorical table `[k, symbols]`, or the Gaussian means
 * `[k, d]`).
 *
 * @param sequences - The training sequences (a symbol index or a vector per step).
 * @param start - The starting parameters, see {@link initialHmmParameters}.
 * @param options - Covariance type and prior (Gaussian), and the tolerance.
 * @returns The algorithm; run it with `runAlgorithm`.
 * @concept hidden-markov-models
 * @example
 * ```ts
 * import { learning } from 'mdml-engine/pgm'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const start = { kind: 'categorical', start: [0.5, 0.5], transition: [[0.6, 0.4], [0.4, 0.6]], emission: [[0.7, 0.3], [0.2, 0.8]] }
 * const alg = learning.baumWelchAlgorithm([[0, 0, 1, 1, 0, 0, 1, 1]], start, { tol: 1e-6 })
 * runAlgorithm(alg, { maxSteps: 30 }).state.params.kind // => 'categorical'
 * ```
 */
export function baumWelchAlgorithm(
  sequences: readonly HmmSequence[],
  start: HmmParameters,
  options: BaumWelchOptions = {},
): Algorithm<BaumWelchState> {
  const opts: Required<BaumWelchOptions> = {
    covarianceType: options.covarianceType ?? 'diag',
    covariancePrior: options.covariancePrior ?? 1e-2,
    tol: options.tol ?? 1e-2,
  }
  const K = start.start.length
  const columns = start.kind === 'categorical' ? (start.emission[0]?.length ?? 0) : (start.means[0]?.length ?? 1)
  const observations = sequences.map((s) => s.map((o) => (start.kind === 'categorical' ? o : vector(o))))
  return {
    name: 'baum-welch',
    config: { states: K, emission: start.kind, covarianceType: opts.covarianceType, tol: opts.tol },
    schema: channels({
      metrics: {
        logLikelihood: {
          description: 'log likelihood of the sequences under the parameters the step started from',
          direction: 'higher',
        },
      },
      tensors: {
        start: { role: 'param', description: 'the start distribution', shape: [K] },
        transition: {
          role: 'param',
          description: 'the transition matrix, row i the distribution after state i',
          shape: [K, K],
        },
        emission: {
          role: 'param',
          description:
            start.kind === 'categorical'
              ? 'the emission table, row k the symbols of state k'
              : 'the Gaussian means, row k of state k',
          shape: [K, columns],
        },
      },
    }),
    init: () => ({ params: start, logLikelihood: null, gain: null }),
    step: (s) => {
      const next = emStep(s.params, observations, opts)
      return {
        params: next.params,
        logLikelihood: next.logLikelihood,
        gain: s.logLikelihood === null ? null : next.logLikelihood - s.logLikelihood,
      }
    },
    done: (s) => (s.gain !== null && s.gain < opts.tol ? 'converged' : undefined),
    observe: (s) => ({
      metrics: s.logLikelihood === null ? {} : { logLikelihood: s.logLikelihood },
      tensors: {
        start: { role: 'param', value: s.params.start, shape: [K] },
        transition: { role: 'param', value: s.params.transition.flat(), shape: [K, K] },
        emission: {
          role: 'param',
          value: (s.params.kind === 'categorical' ? s.params.emission : s.params.means).flat(),
          shape: [K, columns],
        },
      },
    }),
    serialise: (s) => ({ params: s.params, logLikelihood: s.logLikelihood, gain: s.gain }),
    deserialise: (j) => j as unknown as BaumWelchState,
  }
}

/** Options of {@link baumWelch}: the start, the algorithm and the budget. */
export interface BaumWelchFitOptions extends BaumWelchOptions, Omit<HmmInitOptions, 'states' | 'emission'> {
  readonly states: number
  readonly emission: 'categorical' | 'gaussian'
  readonly key: Key
  /** Iteration budget (default 100). */
  readonly maxIter?: number
}

/**
 * Fit a hidden Markov model by Baum-Welch from {@link initialHmmParameters}: the parameters, the log likelihood of every
 * iteration and whether it converged.
 *
 * @param sequences - The training sequences.
 * @param options - States, emissions, key, iteration budget, tolerance and the start's shape.
 * @returns The fitted parameters, the log likelihoods per iteration, the iterations run and the convergence flag.
 * @concept hidden-markov-models
 * @example
 * ```ts
 * import { learning } from 'mdml-engine/pgm'
 * import { key } from 'mdml-engine/random'
 * const x = [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1]
 * learning.baumWelch([x], { states: 2, emission: 'categorical', key: key(0), maxIter: 50 }).params.kind // => 'categorical'
 * ```
 */
export function baumWelch(
  sequences: readonly HmmSequence[],
  options: BaumWelchFitOptions,
): {
  readonly params: HmmParameters
  readonly logLikelihoods: readonly number[]
  readonly iterations: number
  readonly converged: boolean
} {
  const start = initialHmmParameters(sequences, options, options.key)
  const algorithm = baumWelchAlgorithm(sequences, start, options)
  const logLikelihoods: number[] = []
  let last: { readonly t: number; readonly state: BaumWelchState; readonly status?: string } = {
    t: 0,
    state: algorithm.init(options.key),
  }
  for (const item of iterate(algorithm, { key: options.key, maxSteps: options.maxIter ?? 100 })) {
    last = item
    if (item.state.logLikelihood !== null) logLikelihoods.push(item.state.logLikelihood)
  }
  return { params: last.state.params, logLikelihoods, iterations: last.t, converged: last.status === 'converged' }
}
