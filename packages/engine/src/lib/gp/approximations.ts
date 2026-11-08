/**
 * The two classical Gaussian approximations of the posterior of a latent Gaussian process under a non-Gaussian
 * likelihood, each as a stepped algorithm: the Laplace approximation (a Newton iteration on the latent function, one
 * per step, Rasmussen and Williams' Algorithm 3.1) and expectation propagation (site updates of a probit likelihood,
 * one site per step, Algorithm 3.5), with the approximate log marginal likelihood of each and its gradient in the
 * kernel's hyperparameters (Algorithm 5.1 for the Laplace approximation and its EP counterpart).
 */

import { OmaiError } from '../base/index.ts'
import { gramGradient, type Kernel, withTheta } from '../kernels/index.ts'
import { cholesky, choSolve, dgemv, solveTriangular } from '../numerics/linalg/index.ts'
import { expit, log1pexp, logNdtr, normalLogPdf } from '../numerics/special/index.ts'
import { asTensor, fromStorage, primalOf, type TensorLike, toFlat, type Value } from '../tensor/index.ts'
import { type Algorithm, channels, stateFromJson, stateToJson } from '../trace/index.ts'

/** The likelihoods of the classifier: the logistic sigmoid and the normal CDF (probit) of the latent function. */
export type ClassLikelihood = 'logit' | 'probit'

const flat = (v: Value): Float64Array => Float64Array.from(toFlat(primalOf(v)))
const matrix = (a: ArrayLike<number>, n: number): Value => fromStorage(Float64Array.from(a), [n, n], 'float64')
const vec = (a: ArrayLike<number>): Value => fromStorage(Float64Array.from(a), [a.length], 'float64')

/** The terms of log p(t | f) in f for labels t = +1 or -1 and their derivatives, elementwise. */
export interface LikelihoodTerms {
  readonly logp: Float64Array
  /** d log p / d f. */
  readonly grad: Float64Array
  /** W = -d^2 log p / d f^2 (at least zero for both likelihoods). */
  readonly w: Float64Array
  /** d^3 log p / d f^3. */
  readonly third: Float64Array
}

/** The Mills-type ratio N(z) / Phi(z), stable in the lower tail. */
function millsRatio(z: Float64Array): Float64Array {
  const logPdf = flat(normalLogPdf(vec(z)))
  const logCdf = flat(logNdtr(vec(z)))
  return logPdf.map((v, i) => Math.exp(v - (logCdf[i] ?? 0)))
}

/**
 * The log likelihood of labels t = +1 or -1 given the latent function f and its first three derivatives, elementwise: the
 * ingredients of the Newton iteration, `w` being minus the second derivative.
 *
 * @param likelihood - `logit` or `probit`.
 * @param f - The latent values.
 * @param t - The labels, +1 or -1.
 * @returns `logp`, `grad`, `w` and `third`, each of the length of f.
 * @concept gaussian-process-classification
 * @example
 * ```ts
 * import { latentTerms } from 'omai-engine/gp'
 * const { w, grad } = latentTerms('logit', Float64Array.of(0), Float64Array.of(1))
 * [w[0], grad[0]] // => [0.25, 0.5]
 * ```
 */
export function latentTerms(likelihood: ClassLikelihood, f: Float64Array, t: Float64Array): LikelihoodTerms {
  const z = f.map((v, i) => v * (t[i] ?? 1))
  if (likelihood === 'logit') {
    const pi = flat(expit(vec(f)))
    return {
      logp: z.map((v) => -(toFlat(log1pexp(-v))[0] ?? 0)),
      grad: pi.map((p, i) => ((t[i] ?? 1) + 1) / 2 - p),
      w: pi.map((p) => p * (1 - p)),
      third: pi.map((p) => -p * (1 - p) * (1 - 2 * p)),
    }
  }
  const r = millsRatio(z)
  const logCdf = flat(logNdtr(vec(z)))
  return {
    logp: logCdf,
    grad: r.map((v, i) => (t[i] ?? 1) * v),
    w: r.map((v, i) => v * v + (z[i] ?? 0) * v),
    third: r.map((v, i) => {
      const zi = z[i] ?? 0
      return (t[i] ?? 1) * (2 * v * v * (zi + v) - v + zi * v * (zi + v))
    }),
  }
}

const labelsOf = (y: TensorLike, n: number, who: string): Float64Array => {
  const v = Float64Array.from(toFlat(primalOf(asTensor(y))))
  if (v.length !== n) throw new OmaiError('shape', `${who}: ${v.length} labels for ${n} points`)
  if (v.some((l) => l !== 0 && l !== 1)) throw new OmaiError('value', `${who}: labels must be 0 or 1`)
  return v.map((l) => 2 * l - 1)
}

const gramOf = (K: TensorLike): { readonly data: Float64Array; readonly n: number } => {
  const t = primalOf(asTensor(K))
  const n = t.shape[0] ?? 0
  if (t.shape.length !== 2 || t.shape[1] !== n) throw new OmaiError('shape', 'the Gram matrix must be square')
  return { data: Float64Array.from(toFlat(t)), n }
}

/** sum_i log L_ii of a factor. */
function logDiagonal(L: Value, n: number): number {
  const l = flat(L)
  let s = 0
  for (let i = 0; i < n; i++) s += Math.log(l[i * n + i] ?? 1)
  return s
}

/** B = I + W^1/2 K W^1/2 and its Cholesky factor. */
function laplaceFactor(K: Float64Array, n: number, sw: Float64Array): Value {
  const B = Float64Array.from(K, (v, idx) => (sw[Math.floor(idx / n)] ?? 0) * v * (sw[idx % n] ?? 0))
  for (let i = 0; i < n; i++) B[i * n + i] = (B[i * n + i] ?? 0) + 1
  return cholesky(matrix(B, n))
}

/** The state of {@link laplaceAlgorithm}. */
export interface LaplaceState {
  /** The latent function at the training points. */
  readonly f: Float64Array
  /** K^-1 f, the coefficients of the latent mean: f = K a. */
  readonly a: Float64Array
  /** The log marginal likelihood of the approximation at this iterate (minus infinity before the first step). */
  readonly logMarginalLikelihood: number
  readonly iterations: number
  readonly converged: boolean
}

/** Options of {@link laplaceAlgorithm}. */
export interface LaplaceOptions {
  readonly likelihood?: ClassLikelihood
  /** Stop when the log marginal likelihood improves by less than this (default 1e-10, scikit-learn's). */
  readonly tol?: number
}

/**
 * The Laplace approximation of Gaussian-process classification as a stepped algorithm: each step is one Newton
 * iteration for the mode of p(f | y) in the stable form of Rasmussen and Williams' Algorithm 3.1, with
 * B = I + W^1/2 K W^1/2, b = W f + d log p, a = b - W^1/2 B^-1 W^1/2 K b and f = K a. It records the approximate log
 * marginal likelihood -1/2 a^T f + log p(y | f) - sum log diag(chol B) and the size of the step.
 *
 * @param gram - The `[n, n]` kernel matrix of the training points.
 * @param labels - Binary labels `[n]`, 0 or 1.
 * @param options - The likelihood and the tolerance.
 * @returns The algorithm.
 * @throws OmaiError `value` when a label is not 0 or 1.
 * @concept gaussian-process-classification
 * @reference C. E. Rasmussen and C. K. I. Williams, Gaussian Processes for Machine Learning (2006), section 3.4.
 * @example
 * ```ts
 * import { laplaceAlgorithm } from 'omai-engine/gp'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const alg = laplaceAlgorithm([[1, 0.5], [0.5, 1]], [0, 1])
 * const r = runAlgorithm(alg, { maxSteps: 50 })
 * [r.status, r.state.f[0] < 0 && r.state.f[1] > 0] // => ['converged', true]
 * ```
 */
export function laplaceAlgorithm(
  gram: TensorLike,
  labels: TensorLike,
  options: LaplaceOptions = {},
): Algorithm<LaplaceState> {
  const { data: K, n } = gramOf(gram)
  const t = labelsOf(labels, n, 'laplaceAlgorithm')
  const likelihood = options.likelihood ?? 'logit'
  const tol = options.tol ?? 1e-10
  return {
    name: 'laplace',
    config: { likelihood, tol },
    schema: channels({
      metrics: {
        logMarginalLikelihood: {
          description: 'the Laplace approximation of the log marginal likelihood',
          direction: 'higher',
        },
        change: { description: 'the improvement of the log marginal likelihood in the step' },
      },
      tensors: { latent: { role: 'param', description: 'the latent function at the training points', shape: ['n'] } },
    }),
    init: () => ({
      f: new Float64Array(n),
      a: new Float64Array(n),
      logMarginalLikelihood: Number.NEGATIVE_INFINITY,
      iterations: 0,
      converged: false,
    }),
    step: (s) => {
      const lik = latentTerms(likelihood, s.f, t)
      const sw = lik.w.map(Math.sqrt)
      const L = laplaceFactor(K, n, sw)
      const b = lik.w.map((w, i) => w * (s.f[i] ?? 0) + (lik.grad[i] ?? 0))
      const Kb = dgemv(K, n, n, b)
      const solved = flat(choSolve(L, vec(sw.map((v, i) => v * (Kb[i] ?? 0)))))
      const a = b.map((v, i) => v - (sw[i] ?? 0) * (solved[i] ?? 0))
      const f = dgemv(K, n, n, a)
      const logdet = logDiagonal(L, n)
      const quad = a.reduce((sum, v, i) => sum + v * (f[i] ?? 0), 0)
      const logp = latentTerms(likelihood, f, t).logp.reduce((x, y) => x + y, 0)
      const lml = -0.5 * quad + logp - logdet
      return {
        f,
        a,
        logMarginalLikelihood: lml,
        iterations: s.iterations + 1,
        converged: lml - s.logMarginalLikelihood < tol,
      }
    },
    done: (s) => (s.converged ? 'converged' : undefined),
    observe: (s) => ({
      metrics: {
        ...(Number.isFinite(s.logMarginalLikelihood) ? { logMarginalLikelihood: s.logMarginalLikelihood } : {}),
      },
      tensors: { latent: { role: 'param', value: s.f } },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as LaplaceState,
  }
}

/** The approximate log marginal likelihood of a classifier and its gradient in theta. */
export interface ApproximateMarginal {
  readonly value: number
  /** d value / d theta, in the order of `theta(kernel)`. */
  readonly gradient: Float64Array
}

/** A Laplace posterior at the mode, with the approximate log marginal likelihood and its gradient. */
export interface LaplacePosterior extends ApproximateMarginal {
  /** The latent function at the training points. */
  readonly f: Float64Array
  /** K^-1 f. */
  readonly a: Float64Array
  readonly iterations: number
}

function runToConvergence<S extends { readonly converged: boolean }>(alg: Algorithm<S>, maxSteps: number): S {
  const ctx = {} as never
  let s = alg.init({} as never)
  for (let i = 0; i < maxSteps && !s.converged; i++) s = alg.step(s, ctx)
  return s
}

/**
 * The Laplace approximation to the log marginal likelihood of a Gaussian-process classifier and its gradient with
 * respect to the kernel's log-hyperparameters (Rasmussen and Williams' Algorithm 5.1, scikit-learn's
 * `GaussianProcessClassifier.log_marginal_likelihood(theta, eval_gradient=True)`): the explicit dependence of the
 * likelihood on K and the implicit one through the mode.
 *
 * @param kernel - The kernel.
 * @param x - Inputs `[n, d]`.
 * @param y - Labels `[n]`, 0 or 1.
 * @param options - The likelihood, a theta to evaluate at, the iteration cap (default 100) and tolerance of the mode search.
 * @returns The value, the gradient and the mode.
 * @concept gaussian-process-classification
 * @example
 * ```ts
 * import { laplaceMarginal } from 'omai-engine/gp'
 * import { rbfKernel } from 'omai-engine/kernels'
 * const { value, gradient } = laplaceMarginal(rbfKernel(), [[-1], [-0.5], [0.5], [1]], [0, 0, 1, 1])
 * [value < 0, gradient.length] // => [true, 1]
 * ```
 */
export function laplaceMarginal(
  kernel: Kernel,
  x: TensorLike,
  y: TensorLike,
  options: {
    readonly likelihood?: ClassLikelihood
    readonly theta?: TensorLike
    readonly maxIterations?: number
    readonly tol?: number
  } = {},
): LaplacePosterior {
  const at = options.theta === undefined ? kernel : withTheta(kernel, options.theta)
  const { gram: Kv, gradient } = gramGradient(at, x)
  const { data: K, n } = gramOf(Kv)
  const p = gradient.shape[2] ?? 0
  const t = labelsOf(y, n, 'laplaceMarginal')
  const likelihood = options.likelihood ?? 'logit'
  const alg = laplaceAlgorithm(
    Kv,
    t.map((v) => (v + 1) / 2),
    { likelihood, tol: options.tol ?? 1e-10 },
  )
  const s = runToConvergence(alg, options.maxIterations ?? 100)
  const lik = latentTerms(likelihood, s.f, t)
  const sw = lik.w.map(Math.sqrt)
  const L = laplaceFactor(K, n, sw)
  // R = W^1/2 B^-1 W^1/2, C = L^-1 W^1/2 K and s2 = 1/2 (diag K - diag C^T C) d^3 log p / d f^3 (Algorithm 5.1).
  const swDiag = Float64Array.from({ length: n * n }, (_, idx) =>
    idx % n === Math.floor(idx / n) ? (sw[idx % n] ?? 0) : 0,
  )
  const Bi = flat(choSolve(L, matrix(swDiag, n)))
  const R = Float64Array.from({ length: n * n }, (_, idx) => (sw[Math.floor(idx / n)] ?? 0) * (Bi[idx] ?? 0))
  const swK = Float64Array.from(K, (v, idx) => (sw[Math.floor(idx / n)] ?? 0) * v)
  const C = flat(solveTriangular(L, matrix(swK, n), { lower: true }))
  const s2 = Float64Array.from({ length: n }, (_, j) => {
    let colsq = 0
    for (let i = 0; i < n; i++) colsq += (C[i * n + j] ?? 0) ** 2
    return 0.5 * ((K[j * n + j] ?? 0) - colsq) * (lik.third[j] ?? 0)
  })
  const dK = flat(gradient)
  const grad = new Float64Array(p)
  for (let j = 0; j < p; j++) {
    const Cj = Float64Array.from({ length: n * n }, (_, idx) => dK[idx * p + j] ?? 0)
    const Cja = dgemv(Cj, n, n, s.a)
    let quad = 0
    let trace = 0
    for (let i = 0; i < n; i++) quad += (s.a[i] ?? 0) * (Cja[i] ?? 0)
    for (let i = 0; i < n * n; i++) trace += (R[i] ?? 0) * (Cj[i] ?? 0)
    const bj = dgemv(Cj, n, n, lik.grad)
    const KRb = dgemv(K, n, n, dgemv(R, n, n, bj))
    let third = 0
    for (let i = 0; i < n; i++) third += (s2[i] ?? 0) * ((bj[i] ?? 0) - (KRb[i] ?? 0))
    grad[j] = 0.5 * quad - 0.5 * trace + third
  }
  return { value: s.logMarginalLikelihood, gradient: grad, f: s.f, a: s.a, iterations: s.iterations }
}

/** The state of {@link epAlgorithm}: the site parameters and the Gaussian approximation they define. */
export interface EpState {
  /** The site precisions tau~_i (zero before a site's first update). */
  readonly tau: Float64Array
  /** The site precision-means nu~_i = tau~_i mu~_i. */
  readonly nu: Float64Array
  /** The posterior mean mu and covariance Sigma (row-major) of the approximation. */
  readonly mu: Float64Array
  readonly sigma: Float64Array
  /** The site to update next. */
  readonly site: number
  /** Completed sweeps over all sites. */
  readonly sweeps: number
  /** The largest change of a site parameter in the sweep so far. */
  readonly change: number
  /** The EP approximation of the log marginal likelihood after the last completed sweep (minus infinity before). */
  readonly logMarginalLikelihood: number
  readonly converged: boolean
  /** The site updated by the last step and its cavity moments, for the explanation. */
  readonly last: { readonly site: number; readonly cavityMean: number; readonly cavityVariance: number } | null
}

/** Options of {@link epAlgorithm}. */
export interface EpOptions {
  /** Stop when a whole sweep changes no site parameter by more than this (default 1e-8). */
  readonly tol?: number
  /** Damping of the site updates, in [0, 1): 0 is the plain update (default 0). */
  readonly damping?: number
}

function epMoments(
  state: EpState,
  K: Float64Array,
  n: number,
): { readonly sigma: Float64Array; readonly mu: Float64Array } {
  // Sigma = K - V^T V with V = L^-1 S^1/2 K and B = I + S^1/2 K S^1/2, as Algorithm 3.5 recomputes it each sweep.
  const s = state.tau.map((v) => Math.sqrt(Math.max(v, 0)))
  const B = Float64Array.from(K, (v, idx) => (s[Math.floor(idx / n)] ?? 0) * v * (s[idx % n] ?? 0))
  for (let i = 0; i < n; i++) B[i * n + i] = (B[i * n + i] ?? 0) + 1
  const L = cholesky(matrix(B, n))
  const sK = Float64Array.from(K, (v, idx) => (s[Math.floor(idx / n)] ?? 0) * v)
  const V = flat(solveTriangular(L, matrix(sK, n), { lower: true }))
  const sigma = Float64Array.from(K)
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) {
      let acc = 0
      for (let k = 0; k < n; k++) acc += (V[k * n + i] ?? 0) * (V[k * n + j] ?? 0)
      sigma[i * n + j] = (K[i * n + j] ?? 0) - acc
    }
  return { sigma, mu: dgemv(sigma, n, n, state.nu) }
}

/**
 * Expectation propagation for Gaussian-process classification with the probit likelihood, as a stepped algorithm: each
 * step updates one site (Rasmussen and Williams' Algorithm 3.5). The cavity distribution of the site is the posterior
 * with the site removed; its product with the exact likelihood term is moment-matched by a Gaussian, whose difference from
 * the cavity becomes the new site; the posterior is updated by a rank-one change, and recomputed exactly after each sweep
 * over the sites, when the log marginal likelihood is recorded.
 *
 * @param gram - The `[n, n]` kernel matrix of the training points.
 * @param labels - Binary labels `[n]`, 0 or 1.
 * @param options - Tolerance and damping.
 * @returns The algorithm.
 * @throws OmaiError `value` when a label is not 0 or 1.
 * @concept gaussian-process-classification
 * @reference T. Minka, Expectation propagation for approximate Bayesian inference (2001); Rasmussen and Williams (2006), section 3.6.
 * @example
 * ```ts
 * import { epAlgorithm } from 'omai-engine/gp'
 * import { runAlgorithm } from 'omai-engine/trace'
 * // One observation: the marginal likelihood of a probit label under a zero-mean prior is exactly 1/2.
 * const r = runAlgorithm(epAlgorithm([[2]], [1]), { maxSteps: 20 })
 * Math.exp(r.state.logMarginalLikelihood).toFixed(6) // => '0.500000'
 * ```
 */
export function epAlgorithm(gram: TensorLike, labels: TensorLike, options: EpOptions = {}): Algorithm<EpState> {
  const { data: K, n } = gramOf(gram)
  const t = labelsOf(labels, n, 'epAlgorithm')
  const tol = options.tol ?? 1e-8
  const damping = options.damping ?? 0
  if (!(damping >= 0 && damping < 1))
    throw new OmaiError('domain', `epAlgorithm: damping must be in [0, 1), got ${damping}`)
  return {
    name: 'ep',
    config: { tol, damping },
    schema: channels({
      metrics: {
        logMarginalLikelihood: {
          description: 'the EP approximation of the log marginal likelihood after a sweep',
          direction: 'higher',
        },
        change: { description: 'the largest change of a site parameter in the sweep so far' },
        sweeps: { description: 'completed sweeps over the sites' },
      },
      tensors: {
        sitePrecision: { role: 'param', description: 'the site precisions tau~', shape: ['n'] },
        siteMean: { role: 'param', description: 'the site precision-means nu~', shape: ['n'] },
        latentMean: { role: 'state', description: 'the posterior mean of the latent function', shape: ['n'] },
      },
      messages: { site: { description: 'the site updated, with its cavity mean and variance' } },
    }),
    init: () => ({
      tau: new Float64Array(n),
      nu: new Float64Array(n),
      mu: new Float64Array(n),
      sigma: Float64Array.from(K),
      site: 0,
      sweeps: 0,
      change: 0,
      logMarginalLikelihood: Number.NEGATIVE_INFINITY,
      converged: false,
      last: null,
    }),
    step: (s) => {
      const i = s.site
      const sii = s.sigma[i * n + i] ?? 1
      const tauCavity = 1 / sii - (s.tau[i] ?? 0)
      const nuCavity = (s.mu[i] ?? 0) / sii - (s.nu[i] ?? 0)
      let tau = s.tau
      let nu = s.nu
      let sigma = s.sigma
      let mu = s.mu
      let change = s.change
      let last = s.last
      if (tauCavity > 0) {
        const vCav = 1 / tauCavity
        const mCav = nuCavity * vCav
        const ti = t[i] ?? 1
        const z = (ti * mCav) / Math.sqrt(1 + vCav)
        const [r] = millsRatio(Float64Array.of(z))
        const ratio = r ?? 0
        const mHat = mCav + (ti * vCav * ratio) / Math.sqrt(1 + vCav)
        const vHat = vCav - ((vCav * vCav * ratio) / (1 + vCav)) * (z + ratio)
        const dTau = (1 / vHat - tauCavity - (s.tau[i] ?? 0)) * (1 - damping)
        const newTau = Math.max((s.tau[i] ?? 0) + dTau, 0)
        const dTauApplied = newTau - (s.tau[i] ?? 0)
        const newNu = (1 - damping) * (mHat / vHat - nuCavity) + damping * (s.nu[i] ?? 0)
        tau = Float64Array.from(s.tau)
        nu = Float64Array.from(s.nu)
        tau[i] = newTau
        nu[i] = newNu
        change = Math.max(change, Math.abs(dTauApplied), Math.abs(newNu - (s.nu[i] ?? 0)))
        // Rank-one update of the posterior covariance, then the mean from the site precision-means.
        const col = Float64Array.from({ length: n }, (_, k) => s.sigma[k * n + i] ?? 0)
        const factor = dTauApplied / (1 + dTauApplied * sii)
        sigma = Float64Array.from(s.sigma)
        for (let a = 0; a < n; a++)
          for (let b = 0; b < n; b++)
            sigma[a * n + b] = (sigma[a * n + b] ?? 0) - factor * (col[a] ?? 0) * (col[b] ?? 0)
        mu = dgemv(sigma, n, n, nu)
        last = { site: i, cavityMean: mCav, cavityVariance: vCav }
      }
      if (i + 1 < n) return { ...s, tau, nu, sigma, mu, site: i + 1, change, last }
      // The end of a sweep: Sigma and mu recomputed exactly, the likelihood recorded, and the stopping rule tested.
      const swept: EpState = { ...s, tau, nu, sigma, mu, site: 0, sweeps: s.sweeps + 1, change: 0, last }
      const exact = epMoments(swept, K, n)
      const full: EpState = { ...swept, sigma: exact.sigma, mu: exact.mu }
      const lml = epLogEvidence(full, K, n, t)
      return { ...full, logMarginalLikelihood: lml, converged: change < tol }
    },
    done: (s) => (s.converged ? 'converged' : undefined),
    observe: (s) => ({
      metrics: {
        ...(Number.isFinite(s.logMarginalLikelihood) ? { logMarginalLikelihood: s.logMarginalLikelihood } : {}),
        change: s.change,
        sweeps: s.sweeps,
      },
      tensors: {
        sitePrecision: { role: 'param', value: s.tau },
        siteMean: { role: 'param', value: s.nu },
        latentMean: { role: 'state', value: s.mu },
      },
      ...(s.last === null ? {} : { messages: [{ kind: 'site', payload: s.last }] }),
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as EpState,
  }
}

/** The EP log marginal likelihood at a state whose Sigma and mu are exact (Rasmussen and Williams' equation 3.65). */
function epLogEvidence(s: EpState, K: Float64Array, n: number, t: Float64Array): number {
  const tau = s.tau.map((v) => Math.max(v, 1e-12))
  const sigmaTilde = tau.map((v) => 1 / v)
  const muTilde = s.nu.map((v, i) => v / (tau[i] ?? 1))
  const A = Float64Array.from(K)
  for (let i = 0; i < n; i++) A[i * n + i] = (A[i * n + i] ?? 0) + (sigmaTilde[i] ?? 0)
  const L = cholesky(matrix(A, n))
  const alpha = flat(choSolve(L, vec(muTilde)))
  let value = -logDiagonal(L, n)
  for (let i = 0; i < n; i++) value -= 0.5 * (muTilde[i] ?? 0) * (alpha[i] ?? 0)
  const sii = Float64Array.from({ length: n }, (_, i) => s.sigma[i * n + i] ?? 1)
  const z = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    const tauCavity = 1 / (sii[i] ?? 1) - (s.tau[i] ?? 0)
    const vCav = 1 / Math.max(tauCavity, 1e-12)
    const mCav = ((s.mu[i] ?? 0) / (sii[i] ?? 1) - (s.nu[i] ?? 0)) * vCav
    z[i] = ((t[i] ?? 1) * mCav) / Math.sqrt(1 + vCav)
    const total = vCav + (sigmaTilde[i] ?? 0)
    value += 0.5 * Math.log(total) + (mCav - (muTilde[i] ?? 0)) ** 2 / (2 * total)
  }
  for (const v of flat(logNdtr(vec(z)))) value += v
  return value
}

/** An EP posterior: the site parameters at convergence with the approximate log marginal likelihood and its gradient. */
export interface EpPosterior extends ApproximateMarginal {
  readonly tau: Float64Array
  readonly nu: Float64Array
  readonly sweeps: number
}

/**
 * The EP approximation to the log marginal likelihood of a probit Gaussian-process classifier and its gradient in the
 * kernel's log-hyperparameters, 1/2 b^T (dK) b - 1/2 tr((K + Sigma~)^-1 dK) with b = (K + Sigma~)^-1 mu~, exact at the
 * EP fixed point (Rasmussen and Williams, section 5.5.2).
 *
 * @param kernel - The kernel.
 * @param x - Inputs `[n, d]`.
 * @param y - Labels `[n]`, 0 or 1.
 * @param options - A theta to evaluate at, the sweep cap (default 200) and tolerance of the site updates.
 * @returns The value, the gradient and the sites.
 * @concept gaussian-process-classification
 * @example
 * ```ts
 * import { epMarginal } from 'omai-engine/gp'
 * import { rbfKernel } from 'omai-engine/kernels'
 * const { value, gradient } = epMarginal(rbfKernel(), [[-1], [-0.5], [0.5], [1]], [0, 0, 1, 1])
 * [value < 0, gradient.length] // => [true, 1]
 * ```
 */
export function epMarginal(
  kernel: Kernel,
  x: TensorLike,
  y: TensorLike,
  options: {
    readonly theta?: TensorLike
    readonly maxSweeps?: number
    readonly tol?: number
    readonly damping?: number
  } = {},
): EpPosterior {
  const at = options.theta === undefined ? kernel : withTheta(kernel, options.theta)
  const { gram: Kv, gradient } = gramGradient(at, x)
  const { data: K, n } = gramOf(Kv)
  const p = gradient.shape[2] ?? 0
  const t = labelsOf(y, n, 'epMarginal')
  const alg = epAlgorithm(
    Kv,
    t.map((v) => (v + 1) / 2),
    {
      tol: options.tol ?? 1e-8,
      damping: options.damping ?? 0,
    },
  )
  const ctx = {} as never
  let s = alg.init({} as never)
  for (let i = 0, cap = (options.maxSweeps ?? 200) * n; i < cap && !s.converged; i++) s = alg.step(s, ctx)
  const tau = s.tau.map((v) => Math.max(v, 1e-12))
  const A = Float64Array.from(K)
  for (let i = 0; i < n; i++) A[i * n + i] = (A[i * n + i] ?? 0) + 1 / (tau[i] ?? 1)
  const L = cholesky(matrix(A, n))
  const muTilde = s.nu.map((v, i) => v / (tau[i] ?? 1))
  const b = flat(choSolve(L, vec(muTilde)))
  const inverse = flat(
    choSolve(
      L,
      matrix(
        Float64Array.from({ length: n * n }, (_, idx) => (idx % n === Math.floor(idx / n) ? 1 : 0)),
        n,
      ),
    ),
  )
  const dK = flat(gradient)
  const grad = new Float64Array(p)
  for (let j = 0; j < p; j++) {
    const Cj = Float64Array.from({ length: n * n }, (_, idx) => dK[idx * p + j] ?? 0)
    const Cb = dgemv(Cj, n, n, b)
    let quad = 0
    let trace = 0
    for (let i = 0; i < n; i++) quad += (b[i] ?? 0) * (Cb[i] ?? 0)
    for (let i = 0; i < n * n; i++) trace += (inverse[i] ?? 0) * (Cj[i] ?? 0)
    grad[j] = 0.5 * quad - 0.5 * trace
  }
  return { value: s.logMarginalLikelihood, gradient: grad, tau: s.tau, nu: s.nu, sweeps: s.sweeps }
}
