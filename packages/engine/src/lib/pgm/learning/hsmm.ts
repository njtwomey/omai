/**
 * The hidden semi-Markov model (Yu 2010; Murphy 2002): a hidden Markov model whose states last for a random number of
 * steps with an explicit duration distribution instead of the geometric one that a self-transition implies. A path is a
 * sequence of segments, each a state (never the same as the one before: the transition matrix has a zero diagonal),
 * a duration from that state's distribution and that many emissions from the state. The final segment is right-censored:
 * it may run past the end of the data. The forward-backward pass runs over durations in log space (O(T K D) for a
 * duration cap D), the EM loop re-estimates the start, the transitions, the emissions and the duration distributions
 * (nonparametric, shifted Poisson or shifted negative binomial, the last two fitted exactly on the truncated support).
 */

import { logSumExpOf, OmaiError } from '../../base/index.ts'
import { brentq } from '../../numerics/roots/index.ts'
import { fminbound } from '../../optim/index.ts'
import { foldIn, type Key, normal, uniform } from '../../random/index.ts'
import { toFlat } from '../../tensor/index.ts'
import { type Algorithm, channels, stateFromJson, stateToJson } from '../../trace/index.ts'
import { choleskyLower } from '../_dense.ts'
import { hmmChain } from '../models/state-space.ts'
import {
  type BaumWelchOptions,
  emissionMStep,
  type HmmObservation,
  type HmmParameters,
  type HmmSequence,
  hmmFromParameters,
} from './baum-welch.ts'

const NEG = Number.NEGATIVE_INFINITY
const lse = (xs: readonly number[]): number => (xs.length === 0 ? NEG : logSumExpOf(xs))
const vector = (o: HmmObservation): readonly number[] => (typeof o === 'number' ? [o] : o)

/**
 * The parameters of a hidden semi-Markov model: a hidden Markov model's `start`, `transition` (zero diagonal: a segment
 * is followed by a different state) and emissions, and `duration[k][tau - 1]`, the probability that a segment of state k
 * lasts `tau` steps, for `tau` from 1 to the cap `D = duration[k].length`.
 *
 * @concept hidden-markov-models
 */
export type HsmmParameters = HmmParameters & { duration: number[][] }

/** The family of the duration distributions the EM loop fits. */
export type DurationFamily = 'nonparametric' | 'poisson' | 'negative-binomial'

/**
 * The duration table of the geometric distribution that a self-transition probability implies, capped at `cap`: the
 * probability of lasting tau steps is a^(tau - 1) (1 - a) for tau below the cap, and the last entry holds the rest of
 * the tail, so that the survival function is a^(tau - 1) at every tau up to the cap. With the transitions renormalised
 * to leave the state (`transition_ij / (1 - transition_ii)`) the semi-Markov model is the hidden Markov model it came
 * from, for sequences no longer than the cap.
 *
 * @param stay - The self-transition probability of each state.
 * @param cap - The largest duration.
 * @returns The duration table `[k][cap]`.
 * @example
 * ```ts
 * import { learning } from 'omai-engine/pgm'
 * learning.geometricDurations([0.5], 3) // => [[0.5, 0.25, 0.25]]
 * ```
 */
export function geometricDurations(stay: readonly number[], cap: number): number[][] {
  return stay.map((a) => Array.from({ length: cap }, (_, u) => (u < cap - 1 ? a ** u * (1 - a) : a ** u)))
}

/**
 * The hidden semi-Markov parameters of a hidden Markov model: the same start and emissions, the transitions with the
 * self-loop removed and renormalised, and the geometric durations the self-loops imply ({@link geometricDurations}).
 *
 * @param params - Hidden Markov model parameters.
 * @param cap - The duration cap (at least the longest sequence to be read, for the two to agree).
 * @returns The semi-Markov parameters.
 * @example
 * ```ts
 * import { learning } from 'omai-engine/pgm'
 * const p = { kind: 'categorical', start: [0.5, 0.5], transition: [[0.5, 0.5], [0.25, 0.75]], emission: [[0.9, 0.1], [0.2, 0.8]] } as const
 * learning.hsmmFromHmm(p, 2).transition // => [[0, 1], [1, 0]]
 * ```
 */
export function hsmmFromHmm(params: HmmParameters, cap: number): HsmmParameters {
  const stay = params.transition.map((row, i) => row[i] ?? 0)
  const transition = params.transition.map((row, i) => {
    const z = 1 - (stay[i] ?? 0)
    return row.map((v, j) => (j === i ? 0 : z > 0 ? v / z : 0))
  })
  return { ...params, transition, duration: geometricDurations(stay, cap) }
}

/** What one sequence says about the model: the quantities the M-step needs, as expectations under the posterior. */
export interface HsmmPosterior {
  /** log p(sequence). */
  readonly logLikelihood: number
  /** The probability that each state starts the sequence. */
  readonly start: number[]
  /** The expected number of moves from state i to state j. */
  readonly transition: number[][]
  /** The expected number of segments of state k lasting tau steps, `[k][tau - 1]`; the final segment spreads over the lengths it may have. */
  readonly duration: number[][]
  /** The posterior probability of each state at each step, `[T][K]`. */
  readonly occupancy: number[][]
}

/**
 * Forward-backward for a hidden semi-Markov model over one sequence: the log likelihood (the sum over all segmentations,
 * the last segment right-censored) and the posterior expectations of the start, the transitions, the durations and the
 * state occupancy.
 *
 * @param params - The parameters.
 * @param sequence - The observations.
 * @returns The log likelihood and the expected counts.
 * @throws OmaiError with code `value` for an empty sequence or an observation the emissions cannot read.
 * @concept hidden-markov-models
 * @example
 * ```ts
 * import { learning } from 'omai-engine/pgm'
 * const p = { kind: 'categorical', start: [1, 0], transition: [[0, 1], [1, 0]], emission: [[1, 0], [0, 1]], duration: [[0, 1], [1, 0]] } as const
 * learning.hsmmPosterior(p, [0, 0, 1]).occupancy // => [[1, 0], [1, 0], [0, 1]]
 * ```
 */
export function hsmmPosterior(params: HsmmParameters, sequence: HmmSequence): HsmmPosterior {
  const T = sequence.length
  if (T === 0) throw new OmaiError('value', 'hsmm: empty sequence')
  const K = params.start.length
  const D = params.duration[0]?.length ?? 0
  const obs = sequence.map((o) => (params.kind === 'categorical' ? o : vector(o)))
  const lb = hmmChain(hmmFromParameters(params), obs).logEmission
  // cumulative emission log probability, with the impossible steps (log 0) counted apart so that differences stay finite
  const cum = Array.from({ length: K }, (_, j) => {
    const c = new Float64Array(T + 1)
    const zeros = new Int32Array(T + 1)
    for (let t = 0; t < T; t++) {
      const v = lb[t]?.[j] ?? NEG
      c[t + 1] = (c[t] ?? 0) + (v === NEG ? 0 : v)
      zeros[t + 1] = (zeros[t] ?? 0) + (v === NEG ? 1 : 0)
    }
    return { c, zeros }
  })
  const seg = (j: number, a: number, b: number): number =>
    (cum[j]?.zeros[b] ?? 0) - (cum[j]?.zeros[a] ?? 0) > 0 ? NEG : (cum[j]?.c[b] ?? 0) - (cum[j]?.c[a] ?? 0)
  const lA = params.transition.map((r) => r.map(Math.log))
  const ld = params.duration.map((r) => r.map(Math.log))
  // log survival: lS[j][tau - 1] = log sum_{u >= tau} d_j(u)
  const lS = ld.map((row) => row.map((_, u) => lse(row.slice(u))))
  const aStar = Array.from({ length: T }, () => new Array<number>(K).fill(NEG))
  const aEnd = Array.from({ length: T }, () => new Array<number>(K).fill(NEG))
  for (let j = 0; j < K; j++) (aStar[0] as number[])[j] = Math.log(params.start[j] ?? 0)
  for (let t = 1; t < T; t++) {
    for (let i = 0; i < K; i++) {
      const terms: number[] = []
      for (let tau = 1; tau <= Math.min(D, t); tau++)
        terms.push((aStar[t - tau]?.[i] ?? NEG) + (ld[i]?.[tau - 1] ?? NEG) + seg(i, t - tau, t))
      ;(aEnd[t] as number[])[i] = lse(terms)
    }
    for (let j = 0; j < K; j++) {
      ;(aStar[t] as number[])[j] = lse(Array.from({ length: K }, (_, i) => (aEnd[t]?.[i] ?? NEG) + (lA[i]?.[j] ?? NEG)))
    }
  }
  const bStar = Array.from({ length: T }, () => new Array<number>(K).fill(NEG))
  const bEnd = Array.from({ length: T }, () => new Array<number>(K).fill(NEG))
  for (let s = T - 1; s >= 0; s--) {
    for (let j = 0; j < K; j++) {
      const terms: number[] = []
      if (T - s <= D) terms.push((lS[j]?.[T - s - 1] ?? NEG) + seg(j, s, T))
      for (let tau = 1; tau <= Math.min(D, T - s - 1); tau++)
        terms.push((ld[j]?.[tau - 1] ?? NEG) + seg(j, s, s + tau) + (bEnd[s + tau]?.[j] ?? NEG))
      ;(bStar[s] as number[])[j] = lse(terms)
    }
    if (s >= 1)
      for (let i = 0; i < K; i++)
        (bEnd[s] as number[])[i] = lse(
          Array.from({ length: K }, (_, j) => (lA[i]?.[j] ?? NEG) + (bStar[s]?.[j] ?? NEG)),
        )
  }
  const logLikelihood = lse(Array.from({ length: K }, (_, j) => (aStar[0]?.[j] ?? NEG) + (bStar[0]?.[j] ?? NEG)))
  if (!Number.isFinite(logLikelihood))
    throw new OmaiError('value', 'hsmm: the sequence has probability zero under the model')
  const start = Array.from({ length: K }, (_, j) =>
    Math.exp((aStar[0]?.[j] ?? NEG) + (bStar[0]?.[j] ?? NEG) - logLikelihood),
  )
  const transition = Array.from({ length: K }, () => new Array<number>(K).fill(0))
  for (let t = 1; t < T; t++)
    for (let i = 0; i < K; i++)
      for (let j = 0; j < K; j++)
        (transition[i] as number[])[j] =
          (transition[i]?.[j] ?? 0) +
          Math.exp((aEnd[t]?.[i] ?? NEG) + (lA[i]?.[j] ?? NEG) + (bStar[t]?.[j] ?? NEG) - logLikelihood)
  const duration = Array.from({ length: K }, () => new Array<number>(D).fill(0))
  const diff = Array.from({ length: T + 1 }, () => new Array<number>(K).fill(0))
  const cover = (j: number, a: number, b: number, p: number): void => {
    ;(diff[a] as number[])[j] = (diff[a]?.[j] ?? 0) + p
    ;(diff[b] as number[])[j] = (diff[b]?.[j] ?? 0) - p
  }
  for (let s = 0; s < T; s++)
    for (let j = 0; j < K; j++) {
      const as = aStar[s]?.[j] ?? NEG
      if (as === NEG) continue
      if (T - s <= D) {
        const p = Math.exp(as + (lS[j]?.[T - s - 1] ?? NEG) + seg(j, s, T) - logLikelihood)
        if (p > 0) {
          // the final segment's length is unknown beyond T - s: its duration posterior is d_j(tau) / S_j(T - s) there
          for (let tau = T - s; tau <= D; tau++)
            (duration[j] as number[])[tau - 1] =
              (duration[j]?.[tau - 1] ?? 0) + p * Math.exp((ld[j]?.[tau - 1] ?? NEG) - (lS[j]?.[T - s - 1] ?? NEG))
          cover(j, s, T, p)
        }
      }
      for (let tau = 1; tau <= Math.min(D, T - s - 1); tau++) {
        const p = Math.exp(
          as + (ld[j]?.[tau - 1] ?? NEG) + seg(j, s, s + tau) + (bEnd[s + tau]?.[j] ?? NEG) - logLikelihood,
        )
        if (p > 0) {
          ;(duration[j] as number[])[tau - 1] = (duration[j]?.[tau - 1] ?? 0) + p
          cover(j, s, s + tau, p)
        }
      }
    }
  const occupancy: number[][] = []
  const run = new Array<number>(K).fill(0)
  for (let t = 0; t < T; t++) {
    for (let j = 0; j < K; j++) run[j] = (run[j] ?? 0) + (diff[t]?.[j] ?? 0)
    occupancy.push([...run])
  }
  return { logLikelihood, start, transition, duration, occupancy }
}

/**
 * The log likelihood of sequences under a hidden semi-Markov model.
 *
 * @param params - The parameters.
 * @param sequences - The sequences.
 * @returns The summed log likelihood.
 * @concept hidden-markov-models
 * @example
 * ```ts
 * import { learning } from 'omai-engine/pgm'
 * const p = { kind: 'categorical', start: [1, 0], transition: [[0, 1], [1, 0]], emission: [[1, 0], [0, 1]], duration: [[0, 1], [1, 0]] } as const
 * learning.hsmmLogLikelihood(p, [[0, 0, 1]]) // => 0
 * ```
 */
export function hsmmLogLikelihood(params: HsmmParameters, sequences: readonly HmmSequence[]): number {
  return sequences.reduce((a, s) => a + hsmmPosterior(params, s).logLikelihood, 0)
}

/**
 * The shifted Poisson duration table on 1..cap: the probability of lasting tau steps is proportional to
 * lambda^(tau - 1) / (tau - 1)!, normalised over the cap (so the mean can be anything the cap allows).
 *
 * @param lambda - The Poisson rate of tau - 1.
 * @param cap - The largest duration.
 * @returns The duration pmf, length `cap`.
 * @example
 * ```ts
 * import { learning } from 'omai-engine/pgm'
 * learning.poissonDurations(1, 3).map((p) => Math.round(p * 1000) / 1000) // => [0.4, 0.4, 0.2]
 * ```
 */
export function poissonDurations(lambda: number, cap: number): number[] {
  let logFactorial = 0
  const w = Array.from({ length: cap }, (_, u) => {
    if (u > 0) logFactorial += Math.log(u)
    return u * Math.log(lambda) - logFactorial
  })
  const z = lse(w)
  return w.map((v) => Math.exp(v - z))
}

/**
 * The shifted negative binomial duration table on 1..cap: the probability of lasting tau steps is proportional to
 * Gamma(tau - 1 + r) / (Gamma(r) (tau - 1)!) (1 - p)^(tau - 1), normalised over the cap. At r = 1 it is the geometric
 * distribution; a smaller r makes it more spread than a Poisson of the same mean.
 *
 * @param r - The size.
 * @param p - The success probability.
 * @param cap - The largest duration.
 * @returns The duration pmf, length `cap`.
 * @example
 * ```ts
 * import { learning } from 'omai-engine/pgm'
 * learning.negativeBinomialDurations(1, 0.5, 3).map((p) => Math.round(p * 1000) / 1000) // => [0.571, 0.286, 0.143]
 * ```
 */
export function negativeBinomialDurations(r: number, p: number, cap: number): number[] {
  // log Gamma(u + r) / (Gamma(r) u!) accumulates as the sum of log((r + k) / (k + 1)) over k < u
  let logChoose = 0
  const w = Array.from({ length: cap }, (_, u) => {
    if (u > 0) logChoose += Math.log((r + u - 1) / u)
    return logChoose + u * Math.log(1 - p)
  })
  const z = lse(w)
  return w.map((v) => Math.exp(v - z))
}

const meanOf = (pmf: readonly number[]): number => pmf.reduce((a, v, u) => a + v * u, 0)

/**
 * The maximum-likelihood rate of the shifted Poisson duration distribution truncated at the cap, for (expected) counts
 * of each duration: the rate whose truncated mean equals the counts' mean (exact, by root finding; the truncation makes
 * the untruncated mean a biased answer).
 *
 * @param counts - The (expected) number of segments of duration 1, 2, ..., cap.
 * @returns The rate.
 * @example
 * ```ts
 * import { learning } from 'omai-engine/pgm'
 * Math.round(learning.fitPoissonDurations(learning.poissonDurations(3, 10)) * 1e6) / 1e6 // => 3
 * ```
 */
export function fitPoissonDurations(counts: readonly number[]): number {
  const total = counts.reduce((a, v) => a + v, 0)
  const m = counts.reduce((a, v, u) => a + v * u, 0) / total
  const cap = counts.length
  if (m <= 1e-12) return 1e-6
  if (m >= (cap - 1) * (1 - 1e-9)) return 50 * cap
  const f = (s: number): number => meanOf(poissonDurations(Math.exp(s), cap)) - m
  return Math.exp(brentq(f, -25, Math.log(50 * cap), { xtol: 1e-14, maxIterations: 500 }).root)
}

/**
 * The maximum-likelihood shifted negative binomial duration distribution truncated at the cap, for (expected) counts of
 * each duration: alternating exact maximisation over p (the truncated mean matches the counts' mean, by root finding)
 * and a bounded search over r that is kept only if it raises the likelihood, so the likelihood never decreases from the
 * starting size.
 *
 * @param counts - The (expected) number of segments of duration 1, 2, ..., cap.
 * @param r0 - The starting size.
 * @returns The size and success probability.
 * @example
 * ```ts
 * import { learning } from 'omai-engine/pgm'
 * const fit = learning.fitNegativeBinomialDurations(learning.negativeBinomialDurations(3, 0.4, 30), 1)
 * [Math.round(fit.r * 100) / 100, Math.round(fit.p * 100) / 100] // => [3, 0.4]
 * ```
 */
export function fitNegativeBinomialDurations(counts: readonly number[], r0: number): { r: number; p: number } {
  const total = counts.reduce((a, v) => a + v, 0)
  const m = counts.reduce((a, v, u) => a + v * u, 0) / total
  const cap = counts.length
  const loglik = (r: number, p: number): number => {
    const pmf = negativeBinomialDurations(r, p, cap)
    return counts.reduce((a, c, u) => a + (c > 0 ? c * Math.log(Math.max(pmf[u] ?? 0, 1e-300)) : 0), 0)
  }
  const pFor = (r: number): number => {
    if (m <= 1e-12) return 1 - 1e-9
    // the truncated mean rises with q = 1 - p: solve on the logit scale of q, where it is smooth
    const f = (a: number): number => meanOf(negativeBinomialDurations(r, 1 / (1 + Math.exp(a)), cap)) - m
    return 1 / (1 + Math.exp(brentq(f, -30, 30, { xtol: 1e-13, maxIterations: 500 }).root))
  }
  let r = r0
  let p = pFor(r)
  for (let round = 0; round < 3; round++) {
    const found = fminbound((lr) => -loglik(Math.exp(lr), pFor(Math.exp(lr))), {
      bounds: [Math.log(0.05), Math.log(500)],
      tol: 1e-9,
    }).x
    const rNew = Math.exp(found)
    const pNew = pFor(rNew)
    if (loglik(rNew, pNew) > loglik(r, p)) {
      r = rNew
      p = pNew
    }
  }
  return { r, p }
}

/** Options of {@link hsmmAlgorithm}. */
export interface HsmmOptions extends Pick<BaumWelchOptions, 'covarianceType' | 'covariancePrior'> {
  /** The family the duration distributions are fitted in (default nonparametric: the expected counts). */
  readonly durationFamily?: DurationFamily
  /** Converged when the log likelihood gains less than this in a step (default 1e-6). */
  readonly tol?: number
}

/** The state of {@link hsmmAlgorithm}. */
export type HsmmState = {
  params: HsmmParameters
  /** The negative binomial size of each state (empty for the other families). */
  sizes: number[]
  /** The log likelihood under the parameters the last step started from (the quantity EM improved). */
  logLikelihood: number | null
  gain: number | null
}

/**
 * Expectation-maximisation for a hidden semi-Markov model as a traced {@link Algorithm}: each step runs
 * {@link hsmmPosterior} on every sequence, then re-estimates the start distribution and the transitions from the
 * expected counts, the emissions from the state occupancies, and the durations from the expected duration counts
 * (the counts themselves, or the maximum-likelihood shifted Poisson or negative binomial fitted to them on the
 * truncated support). The log likelihood of the parameters each step started from never decreases. It observes the
 * log likelihood and the `transition` and `duration` tensors.
 *
 * @param sequences - The training sequences.
 * @param start - The starting parameters.
 * @param options - The duration family, covariance options and tolerance.
 * @returns The algorithm.
 * @concept hidden-markov-models
 * @example
 * ```ts
 * import { learning } from 'omai-engine/pgm'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const p = { kind: 'categorical', start: [0.5, 0.5], transition: [[0, 1], [1, 0]], emission: [[0.8, 0.2], [0.3, 0.7]], duration: [[0.5, 0.5], [0.5, 0.5]] } as const
 * const alg = learning.hsmmAlgorithm([[0, 0, 1, 1, 0, 0, 1, 1]], p)
 * runAlgorithm(alg, { maxSteps: 5 }).steps // => 5
 * ```
 */
export function hsmmAlgorithm(
  sequences: readonly HmmSequence[],
  start: HsmmParameters,
  options: HsmmOptions = {},
): Algorithm<HsmmState> {
  const K = start.start.length
  const D = start.duration[0]?.length ?? 0
  const family = options.durationFamily ?? 'nonparametric'
  const tol = options.tol ?? 1e-6
  const emissionOptions = {
    ...(options.covarianceType === undefined ? {} : { covarianceType: options.covarianceType }),
    ...(options.covariancePrior === undefined ? {} : { covariancePrior: options.covariancePrior }),
  }
  const observations = sequences.map((s) => s.map((o) => (start.kind === 'categorical' ? o : vector(o))))
  const norm = (row: readonly number[]): number[] => {
    const z = row.reduce((a, v) => a + v, 0)
    return z > 0 ? row.map((v) => v / z) : row.map(() => 1 / row.length)
  }
  return {
    name: 'hsmm-em',
    config: { states: K, durationCap: D, emission: start.kind, durationFamily: family, tol },
    schema: channels({
      metrics: {
        logLikelihood: {
          description: 'log likelihood of the sequences under the parameters the step started from',
          direction: 'higher',
        },
      },
      tensors: {
        transition: { role: 'param', description: 'the transition matrix between distinct states', shape: [K, K] },
        duration: {
          role: 'param',
          description: 'the duration distributions, row k the lengths of state k',
          shape: [K, D],
        },
      },
    }),
    init: () => {
      const sizes = family === 'negative-binomial' ? start.duration.map(initialSize) : []
      return { params: start, sizes, logLikelihood: null, gain: null }
    },
    step: (s) => {
      const posts = observations.map((o) => hsmmPosterior(s.params, o))
      const logLikelihood = posts.reduce((a, p) => a + p.logLikelihood, 0)
      const first = new Array<number>(K).fill(0)
      const moves = Array.from({ length: K }, () => new Array<number>(K).fill(0))
      const lasts = Array.from({ length: K }, () => new Array<number>(D).fill(0))
      for (const p of posts)
        for (let i = 0; i < K; i++) {
          first[i] = (first[i] ?? 0) + (p.start[i] ?? 0)
          for (let j = 0; j < K; j++) (moves[i] as number[])[j] = (moves[i]?.[j] ?? 0) + (p.transition[i]?.[j] ?? 0)
          for (let u = 0; u < D; u++) (lasts[i] as number[])[u] = (lasts[i]?.[u] ?? 0) + (p.duration[i]?.[u] ?? 0)
        }
      const transition = moves.map((row, i) =>
        row.reduce((a, v) => a + v, 0) > 0 ? norm(row) : (s.params.transition[i] ?? row),
      )
      const sizes = [...s.sizes]
      const duration = lasts.map((row, k) => {
        if (row.reduce((a, v) => a + v, 0) <= 1e-300) return s.params.duration[k] ?? row
        if (family === 'nonparametric') return norm(row)
        if (family === 'poisson') return poissonDurations(fitPoissonDurations(row), D)
        const fit = fitNegativeBinomialDurations(row, s.sizes[k] ?? 1)
        sizes[k] = fit.r
        return negativeBinomialDurations(fit.r, fit.p, D)
      })
      const emission = emissionMStep(
        s.params,
        observations,
        posts.map((p) => p.occupancy),
        emissionOptions,
      )
      const params = { ...s.params, ...emission, start: norm(first), transition, duration } as HsmmParameters
      return { params, sizes, logLikelihood, gain: s.logLikelihood === null ? null : logLikelihood - s.logLikelihood }
    },
    done: (s) => (s.gain !== null && s.gain < tol ? 'converged' : undefined),
    observe: (s) => ({
      metrics: s.logLikelihood === null ? {} : { logLikelihood: s.logLikelihood },
      tensors: {
        transition: { role: 'param', value: s.params.transition.flat(), shape: [K, K] },
        duration: { role: 'param', value: s.params.duration.flat(), shape: [K, D] },
      },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as HsmmState,
  }
}

/** A negative binomial size matching the mean and variance of a duration table (1 when the variance does not exceed the mean). */
function initialSize(pmf: readonly number[]): number {
  const m = meanOf(pmf)
  const v = pmf.reduce((a, p, u) => a + p * (u - m) ** 2, 0)
  return v > m + 1e-9 ? Math.min(500, Math.max(0.05, (m * m) / (v - m))) : 10
}

/**
 * A draw from a hidden semi-Markov model: segments of states with durations until `length` steps are filled, the
 * emissions drawn per step. The last segment is cut at the length (as the model's censoring assumes).
 *
 * @param params - The parameters.
 * @param length - The number of steps.
 * @param key - The random key.
 * @returns The state of each step, the segment boundaries (the first step of each segment) and the observations.
 * @example
 * ```ts
 * import { learning } from 'omai-engine/pgm'
 * import { key } from 'omai-engine/random'
 * const p = { kind: 'categorical', start: [1, 0], transition: [[0, 1], [1, 0]], emission: [[1, 0], [0, 1]], duration: [[0, 1], [1, 0]] } as const
 * learning.hsmmSample(p, 5, key(0)).states // => [0, 0, 1, 0, 0]
 * ```
 */
export function hsmmSample(
  params: HsmmParameters,
  length: number,
  key: Key,
): { readonly states: number[]; readonly starts: number[]; readonly observations: HmmObservation[] } {
  const u = Array.from(toFlat(uniform(foldIn(key, 0), [3 * length + 3])))
  let next = 0
  const draw = (p: readonly number[]): number => {
    const r = u[next++] ?? 0
    let acc = 0
    for (let i = 0; i < p.length; i++) {
      acc += p[i] ?? 0
      if (r < acc) return i
    }
    return p.length - 1
  }
  const states: number[] = []
  const starts: number[] = []
  let z = draw(params.start)
  while (states.length < length) {
    starts.push(states.length)
    const tau = draw(params.duration[z] ?? []) + 1
    for (let i = 0; i < tau && states.length < length; i++) states.push(z)
    z = draw(params.transition[z] ?? [])
  }
  const d = params.kind === 'gaussian' ? (params.means[0]?.length ?? 1) : 0
  const noise = params.kind === 'gaussian' ? Array.from(toFlat(normal(foldIn(key, 1), [length, d]))) : []
  const observations = states.map((k, t): HmmObservation => {
    if (params.kind === 'categorical') return draw(params.emission[k] ?? [])
    const chol = choleskyLower(params.covariances[k] ?? [])
    return (params.means[k] ?? []).map(
      (mu, a) => mu + (chol[a] ?? []).reduce((acc, c, b) => acc + c * (noise[t * d + b] ?? 0), 0),
    )
  })
  return { states, starts, observations }
}
