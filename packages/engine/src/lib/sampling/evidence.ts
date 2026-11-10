/**
 * Estimators of the marginal likelihood (the evidence) from samples, each with its own inputs and its own error estimate:
 * bridge sampling (Meng and Wong 1996; Gronau et al. 2017) from posterior draws and a fitted Gaussian proposal;
 * thermodynamic integration and the stepping-stone estimator (Xie et al. 2011) from the likelihoods of samples at a ladder of
 * power posteriors, such as the particles of tempered SMC at each temperature; and the harmonic mean, which is here only to
 * be refused (its variance is infinite for the usual models, and it is the textbook example of an estimator that looks
 * fine and is wrong). Importance sampling's evidence is `logEvidence` of `importance.ts`; nested sampling and AIS have their
 * own algorithms.
 */

import { type Estimate, logSumExpOf, OmaiError } from '../base/index.ts'
import { foldIn, type Key, normal, uniform } from '../random/index.ts'
import { lowerCholesky } from './_linear.ts'
import { ess } from './diagnostics/index.ts'
import { type ChainProposal, proposal } from './target.ts'

/** An estimate of a log evidence with its (always present) standard error. */
export type EvidenceEstimate = Estimate & { readonly standardError: number }

/** A Gaussian fitted to draws: the mean `[dim]` and covariance `[dim, dim]` (row-major). */
export interface GaussianFit {
  readonly mean: Float64Array
  readonly covariance: Float64Array
}

/**
 * The mean and covariance of draws `[n, dim]`, equally weighted or by log weights (reliability weights: the covariance is
 * divided by `1 - sum w^2`, which is the n - 1 of equal weights).
 *
 * @param draws - Row-major `[n, dim]`.
 * @param dim - The dimension.
 * @param logWeights - Optional unnormalised log weights `[n]`.
 * @returns The fit.
 * @throws OmaiError with code `value` for fewer than two draws.
 * @concept bridge-sampling
 * @example
 * ```ts
 * import { fitGaussian } from 'omai-engine/sampling'
 * Array.from(fitGaussian(Float64Array.of(0, 2, 4), 1).mean) // => [2]
 * ```
 */
export function fitGaussian(draws: Float64Array, dim: number, logWeights?: ArrayLike<number>): GaussianFit {
  const n = draws.length / dim
  if (!(n >= 2)) throw new OmaiError('value', 'fitGaussian: need at least two draws')
  const w = new Float64Array(n)
  if (logWeights === undefined) w.fill(1 / n)
  else {
    const lw = Float64Array.from(logWeights)
    const z = logSumExpOf(lw)
    for (let i = 0; i < n; i++) w[i] = Math.exp((lw[i] ?? 0) - z)
  }
  const mean = new Float64Array(dim)
  for (let i = 0; i < n; i++)
    for (let a = 0; a < dim; a++) mean[a] = (mean[a] ?? 0) + (w[i] ?? 0) * (draws[i * dim + a] ?? 0)
  const cov = new Float64Array(dim * dim)
  let w2 = 0
  for (let i = 0; i < n; i++) {
    w2 += (w[i] ?? 0) ** 2
    for (let a = 0; a < dim; a++)
      for (let b = 0; b < dim; b++)
        cov[a * dim + b] =
          (cov[a * dim + b] ?? 0) +
          (w[i] ?? 0) * ((draws[i * dim + a] ?? 0) - (mean[a] ?? 0)) * ((draws[i * dim + b] ?? 0) - (mean[b] ?? 0))
  }
  const denom = 1 - w2
  return { mean, covariance: cov.map((v) => v / denom) }
}

/** Options of {@link gaussianProposal}. */
export interface GaussianProposalOptions {
  /** A factor on the covariance (default 1): above 1 gives the proposal heavier effective tails than the fit. */
  readonly inflate?: number
  /** Mix in a second proposal (the prior) with this weight, so the weights stay bounded where the Gaussian is thin. */
  readonly defensive?: { readonly prior: ChainProposal; readonly weight: number }
}

/**
 * A Gaussian proposal from a fit, as a `ChainProposal` (draws and log density), optionally mixed with the prior as a
 * defensive component: `(1 - w) N(mean, inflate x covariance) + w prior`.
 *
 * @param fit - The mean and covariance.
 * @param options - Covariance inflation and the defensive mixture.
 * @returns The proposal.
 * @throws OmaiError with code `value` for a defensive weight outside [0, 1) or a non-positive inflation.
 * @concept importance-sampling
 * @example
 * ```ts
 * import { gaussianProposal } from 'omai-engine/sampling'
 * const q = gaussianProposal({ mean: Float64Array.of(0), covariance: Float64Array.of(1) })
 * Math.abs(q.logProb([0]) + 0.5 * Math.log(2 * Math.PI)) < 1e-12 // => true
 * ```
 */
export function gaussianProposal(fit: GaussianFit, options: GaussianProposalOptions = {}): ChainProposal {
  const dim = fit.mean.length
  const inflate = options.inflate ?? 1
  const def = options.defensive
  if (!(inflate > 0)) throw new OmaiError('value', 'gaussianProposal: the inflation must be positive')
  if (def !== undefined && !(def.weight >= 0 && def.weight < 1))
    throw new OmaiError('value', 'gaussianProposal: the defensive weight must lie in [0, 1)')
  const L = lowerCholesky(
    fit.covariance.map((v) => v * inflate),
    dim,
    'gaussianProposal',
  )
  let logDet = 0
  for (let a = 0; a < dim; a++) logDet += 2 * Math.log(L[a * dim + a] ?? 1)
  const normalLog = (x: ArrayLike<number>): number => {
    // z = L^-1 (x - mean) by forward substitution.
    const z = new Float64Array(dim)
    let q = 0
    for (let a = 0; a < dim; a++) {
      let s = (x[a] ?? 0) - (fit.mean[a] ?? 0)
      for (let b = 0; b < a; b++) s -= (L[a * dim + b] ?? 0) * (z[b] ?? 0)
      z[a] = s / (L[a * dim + a] ?? 1)
      q += (z[a] ?? 0) ** 2
    }
    return -0.5 * (dim * Math.log(2 * Math.PI) + logDet + q)
  }
  const draw = (key: Key): Float64Array => {
    const e = normal(key, [dim]).data
    const out = new Float64Array(dim)
    for (let a = 0; a < dim; a++) {
      let s = fit.mean[a] ?? 0
      for (let b = 0; b <= a; b++) s += (L[a * dim + b] ?? 0) * (e[b] ?? 0)
      out[a] = s
    }
    return out
  }
  if (def === undefined || def.weight === 0) return proposal({ dim, sample: draw, logProb: normalLog })
  const w = def.weight
  return proposal({
    dim,
    sample: (key) => {
      const u = uniform(foldIn(key, 0), [1]).data[0] ?? 0
      return u < w ? def.prior.sample(foldIn(key, 1)) : draw(foldIn(key, 1))
    },
    logProb: (x) => {
      const a = Math.log(1 - w) + normalLog(x)
      const b = Math.log(w) + def.prior.logProb(x)
      return logSumExpOf([a, b])
    },
  })
}

/** Options of {@link bridgeSamplingFromRatios}. */
export interface BridgeOptions {
  /** The most iterations of the fixed-point scheme (default 1000). */
  readonly maxIterations?: number
  /** Stop when log of the estimate changes by less than this (default 1e-10). */
  readonly tol?: number
}

/** The outcome of bridge sampling. */
export interface BridgeResult {
  readonly logEvidence: number
  /** The error of the log evidence: the delta-method relative error, with the posterior draws' autocorrelation. */
  readonly standardError: number
  readonly iterations: number
  readonly converged: boolean
  /** The effective number of posterior draws the error used. */
  readonly posteriorEss: number
}

/**
 * The optimal bridge estimator on log ratios (Meng and Wong 1996, the iterative scheme of Gronau et al. 2017): with
 * `l1_i = log p~(x_i) - log q(x_i)` at posterior draws and `l2_j = log p~(y_j) - log q(y_j)` at draws from the proposal q,
 * iterate `r <- mean_j[e^{l2_j} / (s1 e^{l2_j} + s2 r)] / mean_i[1 / (s1 e^{l1_i} + s2 r)]`, with `s1 = n1 / (n1 + n2)` and
 * `s2 = n2 / (n1 + n2)`, to the fixed point, which is the evidence. The error is the delta-method relative error of the two
 * means, with the posterior draws' effective size in place of their number when they are a Markov chain.
 *
 * @param l1 - The log ratios at the posterior draws.
 * @param l2 - The log ratios at the proposal draws.
 * @param options - Iteration cap and tolerance.
 * @returns The log evidence, its error and the iterations.
 * @throws OmaiError with code `value` for fewer than two draws in either set.
 * @concept bridge-sampling
 * @reference X.-L. Meng and W. H. Wong, Simulating ratios of normalizing constants via a simple identity (Statistica Sinica 1996).
 * @example
 * ```ts
 * import { bridgeSamplingFromRatios } from 'omai-engine/sampling'
 * // p~ = q: every ratio is one, so the evidence is one.
 * bridgeSamplingFromRatios([0, 0, 0, 0], [0, 0, 0, 0]).logEvidence // => 0
 * ```
 */
export function bridgeSamplingFromRatios(
  l1: ArrayLike<number>,
  l2: ArrayLike<number>,
  options: BridgeOptions = {},
): BridgeResult {
  const n1 = l1.length
  const n2 = l2.length
  if (!(n1 >= 2 && n2 >= 2))
    throw new OmaiError('value', 'bridgeSamplingFromRatios: need at least two draws in each set')
  const maxIterations = options.maxIterations ?? 1000
  const tol = options.tol ?? 1e-10
  const s1 = n1 / (n1 + n2)
  const s2 = n2 / (n1 + n2)
  const sorted = Float64Array.from(l1).sort()
  const star = sorted[Math.floor(n1 / 2)] ?? 0
  const e1 = Float64Array.from(l1, (v) => Math.exp(v - star))
  const e2 = Float64Array.from(l2, (v) => Math.exp(v - star))
  // The fixed point in the shifted scale (r' = r e^{-star}).
  let r = Math.max(1e-300, e2.reduce((a, b) => a + b, 0) / n2)
  let iterations = 0
  let converged = false
  for (; iterations < maxIterations; iterations++) {
    let num = 0
    for (let j = 0; j < n2; j++) num += (e2[j] ?? 0) / (s1 * (e2[j] ?? 0) + s2 * r)
    let den = 0
    for (let i = 0; i < n1; i++) den += 1 / (s1 * (e1[i] ?? 0) + s2 * r)
    const next = num / n2 / (den / n1)
    const change = Math.abs(Math.log(next) - Math.log(r))
    r = next
    if (change < tol) {
      converged = true
      iterations += 1
      break
    }
  }
  const a = Float64Array.from(e2, (v) => v / (s1 * v + s2 * r))
  const b = Float64Array.from(e1, (v) => 1 / (s1 * v + s2 * r))
  const meanOf = (v: Float64Array): number => v.reduce((x, y) => x + y, 0) / v.length
  const varOf = (v: Float64Array): number => {
    const m = meanOf(v)
    return v.reduce((x, y) => x + (y - m) ** 2, 0) / (v.length - 1)
  }
  const essB = n1 >= 8 ? ess(Array.from(b), { method: 'identity' }) : n1
  const n1eff = Math.min(n1, Number.isFinite(essB) && essB > 0 ? essB : n1)
  const re2 = varOf(a) / (n2 * meanOf(a) ** 2) + varOf(b) / (n1eff * meanOf(b) ** 2)
  return { logEvidence: Math.log(r) + star, standardError: Math.sqrt(re2), iterations, converged, posteriorEss: n1eff }
}

/** Options of {@link bridgeSampling}. */
export interface BridgeSamplingOptions extends BridgeOptions {
  /** The key of the proposal draws. */
  readonly key: Key
  /** The fraction of the posterior draws used to fit the proposal; the rest are the posterior sample of the estimator (default 0.5). */
  readonly fitFraction?: number
  /** A factor on the fitted covariance (default 1). */
  readonly inflate?: number
}

/**
 * Bridge sampling from posterior draws: fit a Gaussian to the first part of the draws, draw as many points from it as there
 * remain posterior draws, and run the optimal bridge ({@link bridgeSamplingFromRatios}) between the two samples. It needs
 * only the unnormalised log posterior on the (unconstrained) coordinates the draws live on, and is the estimator of choice
 * when the posterior is unimodal and near-Gaussian on that scale, where its variance is far below the harmonic mean's and
 * importance sampling's; it degrades when the Gaussian misses the posterior's tails or one of several modes.
 *
 * @param logTarget - The unnormalised log posterior (prior times likelihood, Jacobians included) at a point.
 * @param draws - Posterior draws, row-major `[n, dim]`, in sampling order.
 * @param dim - The dimension.
 * @param options - The key, the fit fraction, the covariance inflation and the iteration controls.
 * @returns The estimate of the log evidence with its error.
 * @throws OmaiError with code `value` for fewer than eight draws.
 * @concept bridge-sampling
 * @index function Bridge sampling evidence from posterior draws and a fitted Gaussian proposal.
 * @example
 * ```ts
 * import { bridgeSampling } from 'omai-engine/sampling'
 * import { key, normal } from 'omai-engine/random'
 * // An unnormalised N(0, 1): Z = sqrt(2 pi).
 * const r = bridgeSampling((x) => -0.5 * (x[0] ?? 0) ** 2, normal(key(0), [400]).data, 1, { key: key(1) })
 * Math.abs(r.logEvidence - 0.5 * Math.log(2 * Math.PI)) < 0.1 // => true
 * ```
 */
export function bridgeSampling(
  logTarget: (x: Float64Array) => number,
  draws: Float64Array,
  dim: number,
  options: BridgeSamplingOptions,
): BridgeResult {
  const n = draws.length / dim
  if (!(n >= 8)) throw new OmaiError('value', 'bridgeSampling: need at least eight posterior draws')
  const nFit = Math.max(2, Math.min(n - 2, Math.floor(n * (options.fitFraction ?? 0.5))))
  const fit = fitGaussian(draws.subarray(0, nFit * dim), dim)
  const q = gaussianProposal(fit, options.inflate === undefined ? {} : { inflate: options.inflate })
  const n1 = n - nFit
  const ratio = (x: Float64Array): number => {
    const v = logTarget(x) - q.logProb(x)
    return Number.isNaN(v) ? Number.NEGATIVE_INFINITY : v
  }
  const l1 = new Float64Array(n1)
  for (let i = 0; i < n1; i++) l1[i] = ratio(draws.subarray((nFit + i) * dim, (nFit + i + 1) * dim))
  const ys = q.sampleMany(options.key, n1)
  const l2 = new Float64Array(n1)
  for (let j = 0; j < n1; j++) l2[j] = ratio(ys.subarray(j * dim, (j + 1) * dim))
  return bridgeSamplingFromRatios(l1, l2, options)
}

/**
 * The harmonic mean estimator of the evidence, `1 / mean(1 / L(theta_i))` over posterior draws: **refused**. Its variance
 * is infinite whenever the prior has mass where the likelihood is small relative to its posterior average, which is the usual
 * case; it converges to the evidence at a rate nobody can verify from the draws, it is typically biased far upwards by
 * a few lucky draws, and its error estimate is worthless. Radford Neal called it "the worst Monte Carlo method ever". Use
 * bridge sampling, nested sampling, the stepping-stone estimator or AIS. The call computes it only when the caller
 * writes `acknowledgeInfiniteVariance: true`, which exists to show the failure in a comparison.
 *
 * @param logLikelihoods - log L(theta_i) at posterior draws.
 * @param options - `acknowledgeInfiniteVariance` must be true to get a number.
 * @returns The estimate, whose standard error is the (misleadingly small) delta-method one.
 * @throws OmaiError with code `value` unless the infinite variance is acknowledged.
 * @concept marginal-likelihood
 * @reference R. M. Neal, Contribution to the discussion of Newton and Raftery (JRSS B 1994); Newton and Raftery, Approximate Bayesian inference with the weighted likelihood bootstrap (JRSS B 1994).
 * @example
 * ```ts
 * import { harmonicMeanEvidence } from 'omai-engine/sampling'
 * harmonicMeanEvidence([-1, -1], { acknowledgeInfiniteVariance: true }).value // => -1
 * ```
 */
export function harmonicMeanEvidence(
  logLikelihoods: ArrayLike<number>,
  options: { readonly acknowledgeInfiniteVariance?: boolean } = {},
): Estimate {
  if (options.acknowledgeInfiniteVariance !== true)
    throw new OmaiError(
      'value',
      'harmonicMeanEvidence: the harmonic mean estimator has infinite variance for the usual models and is refused; use bridge sampling, nested sampling, the stepping-stone estimator or AIS (pass acknowledgeInfiniteVariance: true to compute it anyway, to show how it fails)',
    )
  const n = logLikelihoods.length
  const neg = Float64Array.from(logLikelihoods, (v) => -v)
  const value = -(logSumExpOf(neg) - Math.log(n))
  const w = neg.map((v) => Math.exp(v - logSumExpOf(neg) + Math.log(n)))
  let s = 0
  for (const v of w) s += (v - 1) ** 2
  return {
    value,
    standardError: n > 1 ? Math.sqrt(s / (n - 1) / n) : Number.NaN,
    method: 'harmonic mean (infinite variance)',
  }
}

/** Samples at one rung of a ladder of power posteriors `p(theta) L(theta)^beta`: the temperature and the log likelihoods of the samples. */
export interface LadderRung {
  readonly beta: number
  readonly logLikelihoods: ArrayLike<number>
}

function checkLadder(rungs: readonly LadderRung[], name: string): void {
  if (rungs.length < 2) throw new OmaiError('value', `${name}: need at least two rungs`)
  rungs.forEach((r, i) => {
    if (i > 0 && !(r.beta > (rungs[i - 1]?.beta ?? 0)))
      throw new OmaiError('value', `${name}: the temperatures must increase`)
  })
}

const meanVar = (v: ArrayLike<number>): { mean: number; variance: number } => {
  const n = v.length
  let m = 0
  for (let i = 0; i < n; i++) m += (v[i] ?? 0) / n
  let s = 0
  for (let i = 0; i < n; i++) s += ((v[i] ?? 0) - m) ** 2
  return { mean: m, variance: n > 1 ? s / (n - 1) : 0 }
}

/**
 * Thermodynamic integration (Gelman and Meng 1998; Friel and Pettitt 2008): `log Z = integral_0^1 E_beta[log L] d beta`
 * over the ladder, by the trapezoid rule on the rungs' mean log likelihoods. The rungs' samples need to be draws from the power
 * posteriors; the particles of tempered SMC at each temperature are. The error adds the Monte Carlo error of each mean, weighted by
 * its trapezoid weight, as if the samples were independent; the discretisation error of the rule is not in it, and is the
 * reason to use a fine ladder where `E_beta[log L]` changes fast.
 *
 * @param rungs - The temperatures from the first (typically 0) to the last (1) with the log likelihoods of the samples.
 * @returns The log evidence of the tempered posterior at the last temperature, relative to the first.
 * @throws OmaiError with code `value` for fewer than two rungs or temperatures that do not increase.
 * @concept thermodynamic-integration
 * @index function Thermodynamic integration of the mean log likelihood over a ladder of power posteriors.
 * @example
 * ```ts
 * import { thermodynamicIntegration } from 'omai-engine/sampling'
 * // E_beta[log L] = -1 at every rung: log Z = -1.
 * thermodynamicIntegration([{ beta: 0, logLikelihoods: [-1, -1] }, { beta: 1, logLikelihoods: [-1, -1] }]).value // => -1
 * ```
 */
export function thermodynamicIntegration(rungs: readonly LadderRung[]): EvidenceEstimate {
  checkLadder(rungs, 'thermodynamicIntegration')
  let value = 0
  let var2 = 0
  rungs.forEach((r, k) => {
    const lo = rungs[k - 1]?.beta
    const hi = rungs[k + 1]?.beta
    // The trapezoid weights: half the interval on each side of the rung.
    const w = ((hi ?? r.beta) - (lo ?? r.beta)) / 2
    const { mean, variance } = meanVar(r.logLikelihoods)
    value += w * mean
    var2 += (w * w * variance) / Math.max(1, r.logLikelihoods.length)
  })
  return { value, standardError: Math.sqrt(var2), method: 'thermodynamic integration' }
}

/**
 * The stepping-stone estimator (Xie et al. 2011): `log Z = sum_k log mean_i exp((beta_k - beta_{k-1}) log L(theta_i^{(k-1)}))`,
 * each factor the importance-sampling estimate of the ratio of neighbouring power-posterior normalisers from samples at the
 * lower rung. It has lower variance and bias than thermodynamic integration for the same ladder, and is what the tempered SMC
 * sampler's evidence is when the samples are its particles. The error is the delta-method error of each factor, added in
 * quadrature as if the rungs were independent.
 *
 * @param rungs - The temperatures from the first (typically 0) to the last (1) with the log likelihoods of the samples.
 * @returns The log evidence of the tempered posterior at the last temperature, relative to the first.
 * @throws OmaiError with code `value` for fewer than two rungs or temperatures that do not increase.
 * @concept thermodynamic-integration
 * @reference W. Xie, P. O. Lewis, Y. Fan, L. Kuo and M.-H. Chen, Improving marginal likelihood estimation for Bayesian phylogenetic model selection (Systematic Biology 2011).
 * @index function Stepping-stone estimator of the evidence over a ladder of power posteriors.
 * @example
 * ```ts
 * import { steppingStone } from 'omai-engine/sampling'
 * // A constant log likelihood of -1: log Z = -1.
 * steppingStone([{ beta: 0, logLikelihoods: [-1, -1] }, { beta: 1, logLikelihoods: [-1, -1] }]).value // => -1
 * ```
 */
export function steppingStone(rungs: readonly LadderRung[]): EvidenceEstimate {
  checkLadder(rungs, 'steppingStone')
  let value = 0
  let var2 = 0
  for (let k = 1; k < rungs.length; k++) {
    const below = rungs[k - 1] as LadderRung
    const delta = (rungs[k]?.beta ?? 0) - below.beta
    const x = Float64Array.from(below.logLikelihoods, (l) => (Number.isNaN(l) ? Number.NEGATIVE_INFINITY : delta * l))
    const n = x.length
    const top = x.reduce((a, b) => Math.max(a, b), Number.NEGATIVE_INFINITY)
    const logMean = logSumExpOf(x) - Math.log(n)
    value += logMean
    const r = x.map((v) => Math.exp(v - top))
    const { mean, variance } = meanVar(r)
    var2 += mean > 0 ? variance / (n * mean * mean) : 0
  }
  return { value, standardError: Math.sqrt(var2), method: 'stepping stone' }
}
