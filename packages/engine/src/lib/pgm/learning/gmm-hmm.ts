/**
 * Hidden Markov models with Gaussian-mixture emissions (hmmlearn's `GMMHMM`): each hidden state emits from its own
 * mixture of Gaussians, so a state can cover several clusters of the observation space, which a single Gaussian per
 * state cannot. The E-step is the ordinary forward-backward pass with the mixture density as the emission and the
 * posterior of each mixture component inside each state; the M-step re-estimates the start, the transitions, the mixture
 * weights, the means and the covariances (diagonal or full) from the expected counts, so the log likelihood never
 * decreases (with a zero covariance prior). With one component per state it is the Gaussian Baum-Welch exactly.
 */

import { MdmlError, maxOf } from '../../base/index.ts'
import { initialCentres } from '../../cluster/index.ts'
import { foldIn, type Key, normal } from '../../random/index.ts'
import { toFlat } from '../../tensor/index.ts'
import { type Algorithm, channels, stateFromJson, stateToJson } from '../../trace/index.ts'
import { choleskyLower } from '../_dense.ts'
import { type ChainModel, forwardBackward } from '../inference/chain.ts'
import type { HmmSequence } from './baum-welch.ts'

const NEG = Number.NEGATIVE_INFINITY
const vec = (o: number | readonly number[]): readonly number[] => (typeof o === 'number' ? [o] : o)

/**
 * The parameters of a Gaussian-mixture hidden Markov model: the start distribution, `transition[i][j]`, and for each
 * state k its M mixture `weights[k]`, `means[k][m]` and `covariances[k][m]`.
 *
 * @concept hidden-markov-models
 */
export type GmmHmmParameters = {
  start: number[]
  transition: number[][]
  weights: number[][]
  means: number[][][]
  covariances: number[][][][]
}

/** A mixture component with its Cholesky factor and log normaliser, for evaluating the density. */
interface Component {
  readonly mean: readonly number[]
  readonly chol: readonly (readonly number[])[]
  readonly logNorm: number
}

function component(mean: readonly number[], cov: readonly (readonly number[])[]): Component {
  const chol = choleskyLower(cov)
  const logdet = 2 * chol.reduce((a, row, i) => a + Math.log(row[i] ?? 1), 0)
  return { mean, chol, logNorm: -0.5 * (mean.length * Math.log(2 * Math.PI) + logdet) }
}

/** log N(x; mean, cov) by forward substitution with the Cholesky factor. */
function logDensity(c: Component, x: readonly number[]): number {
  const d = c.mean.length
  const z = new Array<number>(d).fill(0)
  let q = 0
  for (let i = 0; i < d; i++) {
    let s = (x[i] ?? 0) - (c.mean[i] ?? 0)
    for (let j = 0; j < i; j++) s -= (c.chol[i]?.[j] ?? 0) * (z[j] ?? 0)
    z[i] = s / (c.chol[i]?.[i] ?? 1)
    q += (z[i] ?? 0) ** 2
  }
  return c.logNorm - 0.5 * q
}

/** The components of every state, and the log emission of a sequence plus the log of each component's joint term. */
function emissions(
  params: GmmHmmParameters,
  x: readonly (readonly number[])[],
): { chain: ChainModel; joint: number[][][] } {
  const comps = params.means.map((ms, k) => ms.map((m, c) => component(m, params.covariances[k]?.[c] ?? [])))
  const joint = x.map((o) =>
    comps.map((cs, k) => cs.map((c, m) => Math.log(params.weights[k]?.[m] ?? 0) + logDensity(c, o))),
  )
  return {
    chain: {
      logInitial: params.start.map(Math.log),
      logTransition: params.transition.map((r) => r.map(Math.log)),
      logEmission: joint.map((perState) => perState.map((terms) => logsumexp(terms))),
    },
    joint,
  }
}

function logsumexp(xs: readonly number[]): number {
  const m = maxOf(xs)
  if (m === NEG) return NEG
  return m + Math.log(xs.reduce((a, v) => a + Math.exp(v - m), 0))
}

/**
 * The log likelihood of sequences under a Gaussian-mixture hidden Markov model.
 *
 * @param params - The parameters.
 * @param sequences - Sequences of vectors (a number is a 1-vector).
 * @returns The summed log likelihood.
 * @concept hidden-markov-models
 * @example
 * ```ts
 * import { learning } from 'mdml-engine/pgm'
 * const p = { start: [1], transition: [[1]], weights: [[0.5, 0.5]], means: [[[-1], [1]]], covariances: [[[[1]], [[1]]]] }
 * Math.round(learning.gmmHmmLogLikelihood(p, [[0]]) * 1e6) / 1e6 // => -1.418939
 * ```
 */
export function gmmHmmLogLikelihood(params: GmmHmmParameters, sequences: readonly HmmSequence[]): number {
  return sequences.reduce((a, s) => a + forwardBackward(emissions(params, s.map(vec)).chain).logLikelihood, 0)
}

/** Options of {@link gmmHmmAlgorithm}. */
export interface GmmHmmOptions {
  /** Diagonal (default) or full covariances. */
  readonly covarianceType?: 'diag' | 'full'
  /** Added to the scatter of each component before dividing by its weight, on the diagonal (default 1e-2, as hmmlearn's HMM; 0 for exact EM). */
  readonly covariancePrior?: number
  /** Added to every variance after the update (default 1e-6). */
  readonly minCovar?: number
  /** Converged when the log likelihood gains less than this in a step (default 1e-4). */
  readonly tol?: number
}

/** The state of {@link gmmHmmAlgorithm}. */
export type GmmHmmState = {
  params: GmmHmmParameters
  /** The log likelihood under the parameters the last step started from (the quantity EM improved). */
  logLikelihood: number | null
  gain: number | null
}

/**
 * Expectation-maximisation for a Gaussian-mixture hidden Markov model as a traced {@link Algorithm}: each step runs
 * forward-backward on every sequence with the mixture densities as emissions, splits each state's posterior among its
 * components, and re-estimates everything in closed form. It observes the log likelihood and the `weights` tensor.
 *
 * @param sequences - The training sequences.
 * @param start - The starting parameters, see {@link initialGmmHmmParameters}.
 * @param options - Covariance type and prior, variance floor and tolerance.
 * @returns The algorithm.
 * @concept hidden-markov-models
 * @example
 * ```ts
 * import { learning } from 'mdml-engine/pgm'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const p = { start: [0.5, 0.5], transition: [[0.5, 0.5], [0.5, 0.5]], weights: [[1], [1]], means: [[[0]], [[3]]], covariances: [[[[1]]], [[[1]]]] }
 * runAlgorithm(learning.gmmHmmAlgorithm([[0, 0.2, 3, 3.1]], p), { maxSteps: 3 }).steps // => 3
 * ```
 */
export function gmmHmmAlgorithm(
  sequences: readonly HmmSequence[],
  start: GmmHmmParameters,
  options: GmmHmmOptions = {},
): Algorithm<GmmHmmState> {
  const K = start.start.length
  const M = start.weights[0]?.length ?? 1
  const D = start.means[0]?.[0]?.length ?? 1
  const full = options.covarianceType === 'full'
  const prior = options.covariancePrior ?? 1e-2
  const floor = options.minCovar ?? 1e-6
  const tol = options.tol ?? 1e-4
  const data = sequences.map((s) => s.map(vec))
  const norm = (row: readonly number[]): number[] => {
    const z = row.reduce((a, v) => a + v, 0)
    return z > 0 ? row.map((v) => v / z) : row.map(() => 1 / row.length)
  }
  return {
    name: 'gmm-hmm-em',
    config: { states: K, mixtures: M, covarianceType: full ? 'full' : 'diag', covariancePrior: prior, tol },
    schema: channels({
      metrics: {
        logLikelihood: {
          description: 'log likelihood of the sequences under the parameters the step started from',
          direction: 'higher',
        },
      },
      tensors: { weights: { role: 'param', description: 'the mixture weights of each state', shape: [K, M] } },
    }),
    init: () => ({ params: start, logLikelihood: null, gain: null }),
    step: (s) => {
      const startCounts = new Array<number>(K).fill(0)
      const moves = Array.from({ length: K }, () => new Array<number>(K).fill(0))
      const n = Array.from({ length: K }, () => new Array<number>(M).fill(0))
      const sum = Array.from({ length: K }, () => Array.from({ length: M }, () => new Array<number>(D).fill(0)))
      const outer = Array.from({ length: K }, () =>
        Array.from({ length: M }, () => Array.from({ length: D }, () => new Array<number>(D).fill(0))),
      )
      let logLikelihood = 0
      for (const x of data) {
        const { chain, joint } = emissions(s.params, x)
        const fb = forwardBackward(chain)
        logLikelihood += fb.logLikelihood
        for (let k = 0; k < K; k++) startCounts[k] = (startCounts[k] ?? 0) + (fb.posteriors[0]?.[k] ?? 0)
        for (const xi of fb.pairwise)
          for (let i = 0; i < K; i++)
            for (let j = 0; j < K; j++) (moves[i] as number[])[j] = (moves[i]?.[j] ?? 0) + (xi[i]?.[j] ?? 0)
        for (const [t, o] of x.entries())
          for (let k = 0; k < K; k++) {
            const total = logsumexp(joint[t]?.[k] ?? [])
            for (let m = 0; m < M; m++) {
              const g = (fb.posteriors[t]?.[k] ?? 0) * Math.exp((joint[t]?.[k]?.[m] ?? NEG) - total)
              ;(n[k] as number[])[m] = (n[k]?.[m] ?? 0) + g
              for (let a = 0; a < D; a++) {
                ;((sum[k] as number[][])[m] as number[])[a] = (sum[k]?.[m]?.[a] ?? 0) + g * (o[a] ?? 0)
                for (let b = 0; b < D; b++)
                  (((outer[k] as number[][][])[m] as number[][])[a] as number[])[b] =
                    (outer[k]?.[m]?.[a]?.[b] ?? 0) + g * (o[a] ?? 0) * (o[b] ?? 0)
              }
            }
          }
      }
      const means = sum.map((perState, k) => perState.map((v, m) => v.map((q) => q / Math.max(n[k]?.[m] ?? 0, 1e-300))))
      const covariances = means.map((perState, k) =>
        perState.map((mu, m) => {
          const w = n[k]?.[m] ?? 0
          return Array.from({ length: D }, (_, a) =>
            Array.from({ length: D }, (_, b) => {
              const scatter = (outer[k]?.[m]?.[a]?.[b] ?? 0) - w * (mu[a] ?? 0) * (mu[b] ?? 0)
              if (a !== b) return full ? scatter / Math.max(w, 1e-5) : 0
              return (prior + scatter) / Math.max(w, 1e-5) + floor
            }),
          )
        }),
      )
      const params: GmmHmmParameters = {
        start: norm(startCounts),
        transition: moves.map(norm),
        weights: n.map(norm),
        means,
        covariances,
      }
      return { params, logLikelihood, gain: s.logLikelihood === null ? null : logLikelihood - s.logLikelihood }
    },
    done: (s) => (s.gain !== null && s.gain < tol ? 'converged' : undefined),
    observe: (s) => ({
      metrics: s.logLikelihood === null ? {} : { logLikelihood: s.logLikelihood },
      tensors: { weights: { role: 'param', value: s.params.weights.flat(), shape: [K, M] } },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as GmmHmmState,
  }
}

/**
 * Starting parameters for {@link gmmHmmAlgorithm}: the state means are k-means++ seeds, each state's first component
 * sits there with the data's variance, the others are shifted by a keyed Gaussian jitter of one standard deviation so
 * that the components can separate; weights and transitions are uniform. States are numbered by the first coordinate of
 * their mean. With one component per state the start is the Gaussian Baum-Welch start.
 *
 * @param sequences - The training sequences.
 * @param options - The number of states and of mixture components.
 * @param key - The key for the seeds and the jitter.
 * @returns The parameters.
 * @throws MdmlError with code `value` for no data or fewer than one state or component.
 * @concept hidden-markov-models
 * @example
 * ```ts
 * import { learning } from 'mdml-engine/pgm'
 * import { key } from 'mdml-engine/random'
 * const p = learning.initialGmmHmmParameters([[0, 0.1, 5, 5.1, -4, -4.1]], { states: 2, mixtures: 2 }, key(0))
 * p.weights // => [[0.5, 0.5], [0.5, 0.5]]
 * ```
 */
export function initialGmmHmmParameters(
  sequences: readonly HmmSequence[],
  options: { readonly states: number; readonly mixtures: number },
  key: Key,
): GmmHmmParameters {
  const { states: K, mixtures: M } = options
  if (!(Number.isInteger(K) && K >= 1 && Number.isInteger(M) && M >= 1))
    throw new MdmlError('value', 'initialGmmHmmParameters: states and mixtures must be positive integers')
  const rows = sequences.flat().map(vec)
  if (rows.length === 0) throw new MdmlError('value', 'initialGmmHmmParameters: no observations')
  const D = rows[0]?.length ?? 1
  const centres = initialCentres(rows, Math.min(K, rows.length), 'k-means++', key).sort(
    (a, b) => (a[0] ?? 0) - (b[0] ?? 0),
  )
  const mu = Array.from({ length: D }, (_, a) => rows.reduce((s, r) => s + (r[a] ?? 0), 0) / rows.length)
  const variance = Array.from({ length: D }, (_, a) =>
    Math.max(rows.reduce((s, r) => s + ((r[a] ?? 0) - (mu[a] ?? 0)) ** 2, 0) / Math.max(rows.length - 1, 1), 1e-3),
  )
  const jitter = Array.from(toFlat(normal(foldIn(key, 5), [K, M, D])))
  return {
    start: new Array<number>(K).fill(1 / K),
    transition: Array.from({ length: K }, () => new Array<number>(K).fill(1 / K)),
    weights: Array.from({ length: K }, () => new Array<number>(M).fill(1 / M)),
    means: Array.from({ length: K }, (_, k) =>
      Array.from({ length: M }, (_, m) =>
        Array.from(
          { length: D },
          (_, a) =>
            (centres[k]?.[a] ?? 0) + (m === 0 ? 0 : Math.sqrt(variance[a] ?? 1) * (jitter[(k * M + m) * D + a] ?? 0)),
        ),
      ),
    ),
    covariances: Array.from({ length: K }, () =>
      Array.from({ length: M }, () =>
        Array.from({ length: D }, (_, a) => Array.from({ length: D }, (_, b) => (a === b ? (variance[a] ?? 1) : 0))),
      ),
    ),
  }
}
