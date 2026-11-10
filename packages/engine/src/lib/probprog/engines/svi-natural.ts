/**
 * Natural-gradient stochastic variational inference. For an exponential-family guide the natural gradient of the ELBO in
 * the natural parameters is its ordinary gradient in the mean parameters (Hoffman et al. 2013), so a step of size one on a
 * conjugate model is the coordinate-ascent (CAVI) update, and a Robbins-Monro rate turns it into stochastic variational
 * inference on minibatches. Two algorithms: `conjugateNaturalGradientAlgorithm`, for programs whose latent sites are
 * conjugate priors of their observations (the library's `exponentialFamily` and conjugate pairs give the natural parameters
 * and the update), and `gaussianNaturalGradientAlgorithm`, for any model with a Gaussian guide (mean-field or full-rank) on
 * the unconstrained scale, where the gradients are those of any ELBO estimator and only the update is natural.
 */

import { valueAndGrad } from '../../autodiff/index.ts'
import { type Json, OmaiError } from '../../base/index.ts'
import { type Distribution, exponentialFamily } from '../../dists/index.ts'
import { cholesky, inv } from '../../numerics/linalg/index.ts'
import { foldIn, permutation } from '../../random/index.ts'
import {
  add,
  fromStorage,
  matmul,
  mul,
  primalOf,
  sliceAxis,
  sub,
  type Tensor,
  take,
  toFlat,
  transpose,
  type Value,
} from '../../tensor/index.ts'
import { type Algorithm, channels } from '../../trace/index.ts'
import { unexpanded } from '../_expand.ts'
import { analyse, type ProgramAnalysis } from '../analysis.ts'
import { logJoint } from '../density.ts'
import { type ElboEstimator, traceElbo } from '../elbo.ts'
import { type AutoGuide, autoMultivariateNormal, autoNormal } from '../guides.ts'
import type { Model } from '../primitives.ts'
import { conjugateData } from './conjugate.ts'
import { initialParams } from './svi.ts'

/** The step size of a natural-gradient step at step `t`: `rate0 (t + delay)^-forgetting` (Robbins-Monro for `forgetting` in (0.5, 1]). */
export interface NaturalRate {
  /** The size of the first step (default 1: the CAVI update on a conjugate model). */
  readonly rate0?: number
  /** Added to the step count before the power (default 1). */
  readonly delay?: number
  /** The exponent (default 0.7; 0 is a constant rate). */
  readonly forgetting?: number
}

/**
 * The natural-gradient rate at a step: `rate0 (t + delay)^-forgetting`, 1 at the first step with the defaults.
 *
 * @param rate - The schedule.
 * @param t - The step (from 0).
 * @returns The rate.
 * @concept variational-inference
 * @example
 * ```ts
 * import { naturalRate } from 'omai-engine/probprog'
 * [naturalRate({}, 0), naturalRate({ forgetting: 1 }, 3)] // => [1, 0.25]
 * ```
 */
export function naturalRate(rate: NaturalRate, t: number): number {
  return (rate.rate0 ?? 1) * (t + (rate.delay ?? 1)) ** -(rate.forgetting ?? 0.7)
}

// --- Conjugate-exponential models: natural-gradient steps are convex combinations of natural parameters -----------------------

/** The state of {@link conjugateNaturalGradientAlgorithm}. */
export interface ConjugateNaturalState {
  /** The natural parameters of each conjugate site's guide, by site. */
  readonly natural: Readonly<Record<string, readonly Value[]>>
  readonly rate: number
  /** The length of the natural gradient (the gap between the minibatch's target and the current parameters). */
  readonly naturalGradNorm: number
  /** A Monte Carlo estimate of the ELBO of the guide. */
  readonly elbo: number
}

/** Options of {@link conjugateNaturalGradientAlgorithm}. */
export interface ConjugateNaturalOptions extends NaturalRate {
  /** Observations per step (default all of them: each step is then batch CAVI). */
  readonly batchSize?: number
  /** Draws of the guide per ELBO estimate (default 8). */
  readonly elboDraws?: number
  /** Stop `converged` when the natural gradient's length falls below this (default 0: off). */
  readonly tol?: number
}

interface ConjugateParts {
  readonly analysis: ProgramAnalysis
  readonly priors: Readonly<Record<string, Distribution>>
}

const stackedNatural = (d: Distribution): readonly Value[] => exponentialFamily(unexpanded(d)).natural

/**
 * Natural-gradient SVI on a conjugate-exponential program (Hoffman, Blei, Wang and Paisley 2013): each latent site is a
 * conjugate prior of its observations, its guide is that prior's family in natural parameters, and a step moves them
 * towards `prior + (N / B) sum_{minibatch} T(x)` by the rate: `lambda <- (1 - rho) lambda + rho (eta_prior + N/B stats)`.
 * With the whole data and rate one this is the CAVI update, the exact posterior; with a minibatch and a Robbins-Monro
 * rate it is stochastic variational inference. The natural parameters come from `exponentialFamily` and the conjugate
 * update from the library's conjugate pairs (the sufficient statistics of the minibatch are the difference between the
 * posterior's and the prior's natural parameters); the record has the natural gradient's length and an ELBO estimate.
 *
 * @param model - The program.
 * @param data - Its data.
 * @param options - Minibatch size, the rate schedule, ELBO draws and tolerance.
 * @returns The algorithm.
 * @throws OmaiError with code `value` when the program is not fully conjugate.
 * @concept variational-inference
 * @index algorithm Natural-gradient SVI on conjugate models: minibatch conjugate updates with a Robbins-Monro rate.
 * @example
 * ```ts
 * import { conjugateNaturalGradientAlgorithm, observe, plate, sample } from 'omai-engine/probprog'
 * import { gamma, poisson } from 'omai-engine/dists'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const model = () => { const r = sample('r', gamma(1, 1)); plate('n', 3, () => observe('k', poisson(r), [1, 4, 2])) }
 * runAlgorithm(conjugateNaturalGradientAlgorithm(model, undefined), { seed: 0, maxSteps: 1 }).steps // => 1
 * ```
 */
export function conjugateNaturalGradientAlgorithm<D = void>(
  model: Model<D>,
  data: D,
  options: ConjugateNaturalOptions = {},
): Algorithm<ConjugateNaturalState> {
  const tol = options.tol ?? 0
  const draws = options.elboDraws ?? 8
  let parts: ConjugateParts | undefined
  const partsOf = (): ConjugateParts => {
    if (parts !== undefined) return parts
    const analysis = analyse(model, data)
    if (!analysis.fullyConjugate)
      throw new OmaiError('value', 'conjugateNaturalGradientAlgorithm: the program is not fully conjugate')
    const priors: Record<string, Distribution> = {}
    for (const c of analysis.conjugate) {
      const s = analysis.prototype.sites.get(c.prior)
      if (s?.type === 'sample') priors[c.prior] = s.fn
    }
    parts = { analysis, priors }
    return parts
  }
  const marginalsOf = (natural: Readonly<Record<string, readonly Value[]>>): Record<string, Distribution> => {
    const { priors } = partsOf()
    const out: Record<string, Distribution> = {}
    for (const [site, eta] of Object.entries(natural))
      out[site] = exponentialFamily(unexpanded(priors[site] as Distribution)).fromNatural(eta)
    return out
  }
  const elboOf = (natural: Readonly<Record<string, readonly Value[]>>, key: ReturnType<typeof foldIn>): number => {
    const q = marginalsOf(natural)
    let total = 0
    for (let i = 0; i < draws; i++) {
      const values: Record<string, Value> = {}
      let logQ = 0
      let j = 0
      for (const [site, d] of Object.entries(q)) {
        const z = primalOf(d.sample(foldIn(key, i * 64 + j++), []))
        values[site] = z
        logQ += toFlat(primalOf(d.logProb(z))).reduce((a, b) => a + b, 0)
      }
      total += (toFlat(primalOf(logJoint(model, data, values).logJoint))[0] ?? 0) - logQ
    }
    return total / draws
  }
  return {
    name: 'svi-conjugate-natural',
    config: {
      batchSize: options.batchSize ?? null,
      rate0: options.rate0 ?? 1,
      delay: options.delay ?? 1,
      forgetting: options.forgetting ?? 0.7,
    } as Json,
    schema: channels({
      metrics: {
        elbo: { description: 'Monte Carlo estimate of the evidence lower bound', direction: 'higher' },
        naturalGradNorm: {
          description: 'length of the natural gradient: the gap to the minibatch target',
          direction: 'lower',
        },
        rate: { description: 'the step size of the natural-gradient step' },
      },
    }),
    init: (key) => {
      const { priors } = partsOf()
      const natural = Object.fromEntries(Object.entries(priors).map(([site, d]) => [site, stackedNatural(d)]))
      return { natural, rate: Number.NaN, naturalGradNorm: Number.NaN, elbo: elboOf(natural, foldIn(key, 99)) }
    },
    step: (s, ctx) => {
      const { analysis, priors } = partsOf()
      const rho = naturalRate(options, ctx.t)
      const next: Record<string, readonly Value[]> = {}
      let norm2 = 0
      analysis.conjugate.forEach((c, ci) => {
        const prior = priors[c.prior] as Distribution
        const { data: rows, fixed } = conjugateData(analysis.prototype, c)
        const n = rows.shape[0] ?? 0
        const B = Math.min(options.batchSize ?? n, n)
        const batch = B >= n ? rows : take(rows, sliceAxis(permutation(foldIn(ctx.key, ci), n), 0, 0, B))
        const post = c.pair.posterior(prior, batch, fixed)
        const eta0 = stackedNatural(prior)
        const etaB = stackedNatural(post)
        const current = s.natural[c.prior] as readonly Value[]
        const scale = n / B
        next[c.prior] = eta0.map((e0, i) => {
          const target = add(e0, mul(sub(etaB[i] as Value, e0), scale))
          const lam = current[i] as Value
          for (const x of toFlat(primalOf(sub(target, lam)))) norm2 += x * x
          return primalOf(add(mul(lam, 1 - rho), mul(target, rho)))
        })
      })
      const norm = Math.sqrt(norm2)
      return { natural: next, rate: rho, naturalGradNorm: norm, elbo: elboOf(next, foldIn(ctx.key, 98)) }
    },
    done: (s) => (tol > 0 && Number.isFinite(s.naturalGradNorm) && s.naturalGradNorm <= tol ? 'converged' : undefined),
    observe: (s) => ({
      metrics: Number.isNaN(s.rate) ? {} : { elbo: s.elbo, naturalGradNorm: s.naturalGradNorm, rate: s.rate },
    }),
  }
}

/**
 * The guide distribution of each conjugate site at a state of {@link conjugateNaturalGradientAlgorithm}: the prior's family
 * with the state's natural parameters.
 *
 * @param model - The program.
 * @param data - Its data.
 * @param state - A state.
 * @returns The marginal of each site.
 * @throws OmaiError with code `value` when the program is not fully conjugate.
 * @concept variational-inference
 * @example
 * ```ts
 * import { conjugateNaturalGradientAlgorithm, conjugateNaturalMarginals, observe, plate, sample } from 'omai-engine/probprog'
 * import { gamma, poisson } from 'omai-engine/dists'
 * import { item } from 'omai-engine/tensor'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const model = () => { const r = sample('r', gamma(1, 1)); plate('n', 3, () => observe('k', poisson(r), [1, 4, 2])) }
 * const run = runAlgorithm(conjugateNaturalGradientAlgorithm(model, undefined), { seed: 0, maxSteps: 1 })
 * // One full-batch step of rate one is the exact posterior, Gamma(1 + 7, 1 + 3), of mean 2.
 * item(conjugateNaturalMarginals(model, undefined, run.state)['r']?.mean() as never) // => 2
 * ```
 */
export function conjugateNaturalMarginals<D = void>(
  model: Model<D>,
  data: D,
  state: ConjugateNaturalState,
): Record<string, Distribution> {
  const a = analyse(model, data)
  if (!a.fullyConjugate) throw new OmaiError('value', 'conjugateNaturalMarginals: the program is not fully conjugate')
  const out: Record<string, Distribution> = {}
  for (const c of a.conjugate) {
    const s = a.prototype.sites.get(c.prior)
    if (s?.type === 'sample')
      out[c.prior] = exponentialFamily(unexpanded(s.fn)).fromNatural(state.natural[c.prior] as readonly Value[])
  }
  return out
}

// --- Gaussian guides: the natural-gradient step in the mean parameters ---------------------------------------------------------

/** The result of a Gaussian natural-gradient step. */
export interface GaussianStep {
  readonly loc: Float64Array
  /** The new scales (diagonal guide) or the covariance matrix row by row (full-rank guide). */
  readonly scale: Float64Array
  /** The length of the natural gradient (the gradient in the mean parameters). */
  readonly naturalGradNorm: number
}

/**
 * One natural-gradient step of a diagonal Gaussian guide, from the gradient of the ELBO in the location and the scale. For
 * q = N(mu, sigma^2) the natural gradient in the natural parameters (mu / sigma^2, -1 / (2 sigma^2)) is the ordinary gradient
 * in the mean parameters (mu, mu^2 + sigma^2): the precision becomes `1 / sigma^2 - 2 rho dL/d(sigma^2)` and the
 * precision-weighted mean moves by `rho` times the gradient in the first mean parameter. The rate is cut, per coordinate, so
 * that the precision at least keeps a tenth of its value. With rate one and exact gradients of a conjugate model, the result
 * is the exact posterior.
 *
 * @param loc - The locations.
 * @param scale - The scales.
 * @param gradLoc - The gradient of the ELBO in the locations.
 * @param gradScale - The gradient of the ELBO in the scales.
 * @param rate - The step size.
 * @param floor - The share of the precision that a step keeps at least (default 0.05).
 * @returns The new locations and scales and the natural gradient's length.
 * @concept variational-inference
 * @example
 * ```ts
 * import { naturalStepDiagonal } from 'omai-engine/probprog'
 * // E[f'] = -a (mu - c), E[f''] = -a for a conjugate normal factor with a = 4, c = 0.5: the first step lands on N(c, 1 / a).
 * const mu = 0.1, sigma = 0.8, a = 4, c = 0.5
 * const gMu = -a * (mu - c)
 * const gSigma = 2 * sigma * (0.5 * -a + 0.5 / (sigma * sigma))
 * const r = naturalStepDiagonal([mu], [sigma], [gMu], [gSigma], 1)
 * [Math.round((r.loc[0] ?? 0) * 1e9) / 1e9, Math.round((r.scale[0] ?? 0) * 1e9) / 1e9] // => [0.5, 0.5]
 * ```
 */
export function naturalStepDiagonal(
  loc: ArrayLike<number>,
  scale: ArrayLike<number>,
  gradLoc: ArrayLike<number>,
  gradScale: ArrayLike<number>,
  rate: number,
  floor = 0.05,
): GaussianStep {
  const n = loc.length
  const newLoc = new Float64Array(n)
  const newScale = new Float64Array(n)
  let norm2 = 0
  for (let i = 0; i < n; i++) {
    const mu = loc[i] ?? 0
    const sigma = scale[i] ?? 1
    const P = 1 / (sigma * sigma)
    // dL / d(sigma^2) and the gradient in the first mean parameter.
    const g2 = (gradScale[i] ?? 0) / (2 * sigma)
    const g1 = (gradLoc[i] ?? 0) - 2 * mu * g2
    norm2 += g1 * g1 + g2 * g2
    const rho = g2 > 0 ? Math.min(rate, ((1 - floor) * P) / (2 * g2)) : rate
    const P2 = P - 2 * rho * g2
    newLoc[i] = (mu * P + rho * g1) / P2
    newScale[i] = 1 / Math.sqrt(P2)
  }
  return { loc: newLoc, scale: newScale, naturalGradNorm: Math.sqrt(norm2) }
}

/**
 * One natural-gradient step of a full-rank Gaussian guide, from the gradient of the ELBO in the mean and in the covariance
 * matrix: the precision becomes `Sigma^-1 - 2 rho S` for S the gradient in the covariance, and the precision-weighted mean
 * moves by `rho (g_mu - 2 S mu)`. The rate is halved until the new precision is positive definite.
 *
 * @param loc - The mean.
 * @param covariance - The covariance, `[D, D]`.
 * @param gradLoc - The gradient of the ELBO in the mean.
 * @param gradCovariance - The gradient of the ELBO in the covariance (symmetric), `[D, D]`.
 * @param rate - The step size.
 * @returns The new mean and covariance (row by row) and the natural gradient's length.
 * @concept variational-inference
 * @example
 * ```ts
 * import { naturalStepFullRank } from 'omai-engine/probprog'
 * import { tensor } from 'omai-engine/tensor'
 * // E[f'] = -A (mu - c), E[f''] = -A: the first step of rate one lands on N(c, A^-1) for the Gaussian factor with precision A.
 * const A = [[2, 0.5], [0.5, 1]], c = [1, -1], mu = [0, 0], Sigma = [[1, 0], [0, 1]]
 * const gMu = [-(A[0][0] * (mu[0] - c[0]) + A[0][1] * (mu[1] - c[1])), -(A[1][0] * (mu[0] - c[0]) + A[1][1] * (mu[1] - c[1]))]
 * // dL/dSigma = (-A + Sigma^-1) / 2
 * const gS = tensor([[(-2 + 1) / 2, -0.25], [-0.25, (-1 + 1) / 2]])
 * const r = naturalStepFullRank(tensor(mu), tensor(Sigma), tensor(gMu), gS, 1)
 * [Math.round((r.loc[0] ?? 0) * 1e9) / 1e9, Math.round((r.loc[1] ?? 0) * 1e9) / 1e9] // => [1, -1]
 * ```
 */
export function naturalStepFullRank(
  loc: Tensor,
  covariance: Tensor,
  gradLoc: Tensor,
  gradCovariance: Tensor,
  rate: number,
): GaussianStep {
  const D = loc.shape[0] ?? 0
  const S = mul(add(gradCovariance, transpose(gradCovariance)), 0.5)
  const P = primalOf(inv(covariance))
  const g1 = sub(gradLoc, mul(matmul(S, loc), 2))
  const lam1 = matmul(P, loc)
  let rho = rate
  for (let k = 0; k < 40; k++) {
    const P2 = sub(P, mul(S, 2 * rho))
    try {
      cholesky(P2)
    } catch {
      rho /= 2
      continue
    }
    const cov2 = primalOf(inv(P2))
    const sym = primalOf(mul(add(cov2, transpose(cov2)), 0.5))
    const mean2 = primalOf(matmul(sym, add(lam1, mul(g1, rho))))
    const norm = Math.sqrt([...toFlat(g1), ...toFlat(S)].reduce((a, x) => a + x * x, 0))
    return { loc: Float64Array.from(toFlat(mean2)), scale: Float64Array.from(toFlat(sym)), naturalGradNorm: norm }
  }
  return { loc: Float64Array.from(toFlat(loc)), scale: Float64Array.from(toFlat(covariance)), naturalGradNorm: 0 * D }
}

/** The family of Gaussian guide a natural-gradient run fits. */
export type GaussianFamily = 'meanField' | 'fullRank'

/** The state of {@link gaussianNaturalGradientAlgorithm}: the guide's parameters on the constrained scale. */
export interface GaussianNaturalState {
  readonly params: Readonly<Record<string, Value>>
  readonly elboState: unknown
  readonly elbo: number
  readonly smoothed: number
  readonly previous: number
  readonly rate: number
  readonly naturalGradNorm: number
}

/** Options of {@link gaussianNaturalGradientAlgorithm}. */
export interface GaussianNaturalOptions extends NaturalRate {
  /** The Gaussian guide: a normal per site (`meanField`, default) or one over the flat latent vector (`fullRank`). */
  readonly family?: GaussianFamily
  /** The ELBO estimator whose gradient is preconditioned (default `traceElbo`; the guide must be reparameterised). */
  readonly elbo?: ElboEstimator
  /** Stop `converged` when the smoothed ELBO settles by this relative amount (default 0: off). */
  readonly tol?: number
  /** Initial guide parameters (constrained), over the guide's own starting values. */
  readonly initParams?: Readonly<Record<string, Tensor>>
  /** The initial scale of the guide (default 0.5: a small scale makes the first steps' curvature estimates noisy). */
  readonly initScale?: number
  /** The share of the precision a diagonal step keeps at least (default 0.05). */
  readonly floor?: number
}

/**
 * Natural-gradient SVI with a Gaussian guide on the unconstrained scale, for any differentiable model. Each step takes the
 * gradient of an ELBO estimator in the guide's location and scale (or covariance) and moves the natural parameters along it by
 * {@link naturalStepDiagonal} or {@link naturalStepFullRank}: the update is the ordinary gradient step in the mean
 * parameters, so it is invariant to the parametrisation and, on a conjugate Gaussian model, converges in far fewer steps than
 * the same gradient in the guide's own parameters. The guide is `autoNormal` or `autoMultivariateNormal`, so its posterior reads as
 * theirs. Records the ELBO, the natural gradient's length and the rate.
 *
 * @param model - The model.
 * @param data - Its data.
 * @param options - The family, estimator, rate schedule and tolerance.
 * @returns The algorithm; the guide is `family === 'fullRank' ? autoMultivariateNormal(model) : autoNormal(model)`.
 * @concept variational-inference
 * @index algorithm Natural-gradient SVI with a mean-field or full-rank Gaussian guide: the step in the mean parameters.
 * @example
 * ```ts
 * import { gaussianNaturalGradientAlgorithm, observe, sample } from 'omai-engine/probprog'
 * import { normal } from 'omai-engine/dists'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const model = () => { const m = sample('m', normal(0, 1)); observe('y', normal(m, 1), 2) }
 * runAlgorithm(gaussianNaturalGradientAlgorithm(model, undefined), { seed: 0, maxSteps: 3 }).steps // => 3
 * ```
 */
export function gaussianNaturalGradientAlgorithm<D = void>(
  model: Model<D>,
  data: D,
  options: GaussianNaturalOptions = {},
): Algorithm<GaussianNaturalState> {
  const family = options.family ?? 'meanField'
  const estimator = options.elbo ?? traceElbo()
  const tol = options.tol ?? 0
  const guideOptions = { initScale: options.initScale ?? 0.5 }
  const guide: AutoGuide<D> =
    family === 'fullRank' ? autoMultivariateNormal(model, guideOptions) : autoNormal(model, guideOptions)
  return {
    name: 'svi-natural',
    config: {
      family,
      elbo: estimator.name,
      rate0: options.rate0 ?? 1,
      delay: options.delay ?? 5,
      forgetting: options.forgetting ?? 0.7,
      tol,
    } as Json,
    schema: channels({
      metrics: {
        elbo: { description: 'estimated evidence lower bound', direction: 'higher' },
        naturalGradNorm: {
          description: 'length of the natural gradient (the gradient in the mean parameters)',
          direction: 'lower',
        },
        rate: { description: 'the step size of the natural-gradient step' },
      },
    }),
    init: (key) => {
      const init = initialParams(key, model, guide, data).values
      const params: Record<string, Value> = {}
      for (const [k, v] of Object.entries(init)) params[k] = options.initParams?.[k] ?? v
      return {
        params,
        elboState: estimator.init(),
        elbo: Number.NaN,
        smoothed: Number.NaN,
        previous: Number.NaN,
        rate: Number.NaN,
        naturalGradNorm: Number.NaN,
      }
    },
    step: (s, ctx) => {
      const rho = naturalRate({ delay: 5, ...options }, ctx.t)
      let est: { readonly elbo: number; readonly state: unknown } = { elbo: Number.NaN, state: s.elboState }
      const params: Record<string, Value> = {}
      if (family === 'fullRank') {
        const L = s.params['auto_scale_tril'] as Value
        const cov = primalOf(matmul(L, transpose(L)))
        const { grad } = valueAndGrad((p: Record<string, Value>) => {
          const r = estimator.estimate(
            ctx.key,
            model,
            guide,
            data,
            { auto_loc: p['auto_loc'] as Value, auto_scale_tril: cholesky(p['cov'] as Value) },
            s.elboState,
          )
          est = r
          return r.loss
        })({ auto_loc: s.params['auto_loc'] as Value, cov })
        const g = grad as Record<string, Value>
        const step = naturalStepFullRank(
          s.params['auto_loc'] as Tensor,
          cov,
          mul(g['auto_loc'] as Value, -1) as Tensor,
          mul(g['cov'] as Value, -1) as Tensor,
          rho,
        )
        const D = s.params['auto_loc']?.shape[0] ?? 0
        params['auto_loc'] = fromStorage(step.loc, [D], 'float64')
        params['auto_scale_tril'] = primalOf(cholesky(fromStorage(step.scale, [D, D], 'float64')))
        return finish(s, params, est, rho, step.naturalGradNorm)
      }
      const { grad } = valueAndGrad((p: Record<string, Value>) => {
        const r = estimator.estimate(ctx.key, model, guide, data, p, s.elboState)
        est = r
        return r.loss
      })(s.params as Record<string, Value>)
      const g = grad as Record<string, Value>
      let norm2 = 0
      for (const name of Object.keys(s.params)) {
        if (!name.endsWith('_auto_loc')) continue
        const base = name.slice(0, -'_auto_loc'.length)
        const loc = s.params[name] as Value
        const scaleName = `${base}_auto_scale`
        const step = naturalStepDiagonal(
          toFlat(primalOf(loc)),
          toFlat(primalOf(s.params[scaleName] as Value)),
          toFlat(primalOf(mul(g[name] as Value, -1))),
          toFlat(primalOf(mul(g[scaleName] as Value, -1))),
          rho,
          options.floor ?? 0.05,
        )
        norm2 += step.naturalGradNorm ** 2
        params[name] = fromStorage(step.loc, [...loc.shape], 'float64')
        params[scaleName] = fromStorage(step.scale, [...loc.shape], 'float64')
      }
      return finish(s, params, est, rho, Math.sqrt(norm2))
    },
    done: (s) => {
      if (Number.isNaN(s.elbo)) return undefined
      if (!Number.isFinite(s.elbo)) return 'diverged'
      if (
        tol > 0 &&
        Number.isFinite(s.previous) &&
        Math.abs(s.smoothed - s.previous) <= tol * Math.max(1, Math.abs(s.smoothed))
      )
        return 'converged'
      return undefined
    },
    observe: (s) => ({
      metrics: Number.isNaN(s.elbo) ? {} : { elbo: s.elbo, naturalGradNorm: s.naturalGradNorm, rate: s.rate },
    }),
  }
}

function finish(
  s: GaussianNaturalState,
  params: Record<string, Value>,
  est: { readonly elbo: number; readonly state: unknown },
  rate: number,
  naturalGradNorm: number,
): GaussianNaturalState {
  const smoothed = Number.isNaN(s.smoothed) ? est.elbo : 0.9 * s.smoothed + 0.1 * est.elbo
  return { params, elboState: est.state, elbo: est.elbo, smoothed, previous: s.smoothed, rate, naturalGradNorm }
}
