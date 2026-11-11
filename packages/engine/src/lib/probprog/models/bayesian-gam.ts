/**
 * A Bayesian additive model of one covariate: `y = f(x) + noise` with `f` a penalised regression spline (the basis and the
 * wiggliness penalty of `omai-engine/gam`), the smoothness prior the penalty implies (the coefficients are Gaussian with the
 * penalty times a smoothing parameter as their precision: the random-walk or P-spline prior) and priors on the log smoothing
 * parameter and the log noise standard deviation, so the amount of smoothing is inferred along with the curve. The model is a
 * spec of the model layer whose data carry everything that depends on the rows (the design in the eigenbasis of the penalty
 * and its eigenvalues), so one definition serves any data and runs on every inference method; the coefficients are written
 * non-centred (`gamma = z / sqrt(precision)` with `z` standard normal), which is the same prior and removes the funnel
 * between the smoothing parameter and the coefficients that a sampler or a guide would otherwise meet. This file has the
 * constructor (`bayesianGam`), what a posterior says of the curve (`bayesianGamFit`: mean, credible band, sampled curves and
 * the draws of the hyperparameters), the exact reference (`bayesianGamReference`: given the smoothing parameter and the noise
 * the coefficients are Gaussian in closed form, so quadrature over those two gives the posterior of the curve) and the method
 * of a kind that adds the fit to its tape (`bayesianGamMethod`).
 */

import { type Json, OmaiError, quantileOf } from '../../base/index.ts'
import { buildDesign, evaluateDesign, type GamTerm, type SmoothKind, s as smoothTerm } from '../../gam/index.ts'
import { dpotrf, dpotrs, dsyev } from '../../numerics/linalg/index.ts'
import { lgamma, ndtr } from '../../numerics/special/index.ts'
import { type Key, key as makeKey } from '../../random/index.ts'
import { bs, ps } from '../../stats/smoothing/index.ts'
import { fromStorage, item, toFlat, type Value } from '../../tensor/index.ts'
import { type Config, derived } from '../../trace/index.ts'
import { type MethodSpace, methodSpace } from '../inference/method-space.ts'
import { type AnyInferenceMethod, methodKinds } from '../inference/models.ts'
import * as d from '../model/dists/index.ts'
import { type DataInput, defineModel, type ModelDefinition } from '../model/index.ts'
import type { Posterior } from '../posterior.ts'

/** What a Bayesian GAM is built from: the rows and the choices of the model. */
export interface BayesianGamOptions {
  /** The spline basis (default `ps`, a P-spline: B-splines with a difference penalty). */
  readonly basis?: SmoothKind
  /** The basis dimension before centring (default 10). */
  readonly k?: number
  /** The degree of a `ps` or `bs` basis (default 3). */
  readonly degree?: number
  /** The order of the difference penalty of a `ps` basis (default 2). */
  readonly order?: number
  /** The prior of the log smoothing parameter, normal (default mean 0, standard deviation 3: the penalty is scaled so its largest eigenvalue is 1). */
  readonly logLambdaPrior?: { readonly mean: number; readonly sd: number }
  /** The prior of the log noise standard deviation of the standardised response, normal (default mean -1, standard deviation 1). */
  readonly logSigmaPrior?: { readonly mean: number; readonly sd: number }
  /** The prior standard deviation of the intercept and the directions the penalty does not touch (a line), in standard deviations of the response (default 5). */
  readonly nullScale?: number
  /**
   * The form of the hyperpriors of the penalty prior. `logNormal` (default): the log smoothing parameter and the log noise
   * standard deviation are normal latents, and the coefficients are non-centred (built on `exp`, so only the samplers and
   * variational families run on it). `gammaNoise`: the smoothing parameter is a knob (`lambda`) and the noise precision
   * `tau = 1 / sigma^2` is a gamma latent (`noiseShape`, `noiseRate`), the coefficients centred with known scales: a
   * normal-gamma model that variational message passing runs. `fixed`: the smoothing and the noise (`noiseSd`) are both knobs,
   * a linear-Gaussian model with an exact Gaussian posterior that expectation propagation and exact Gaussian inference run.
   * `gammaSmooth`: both are inferred in the precision idiom, the smoothing parameter `lambda` a gamma latent (`lambdaShape`,
   * `lambdaRate`) that scales the precision of the coefficients (`lambda d_j + n_j`, written for the coefficients in the
   * eigenbasis of the penalty) and the noise precision a gamma latent: the model message passing runs in full (the Bayesian
   * GAM under expectation propagation: one dense Gaussian vector with a shared random precision multiplier).
   */
  readonly hyper?: 'logNormal' | 'gammaNoise' | 'fixed' | 'gammaSmooth'
  /** The shape and rate of the gamma prior of the smoothing parameter, for `gammaSmooth` (defaults 1 and 0.05: mean 20, relative to the largest eigenvalue of the penalty). */
  readonly lambdaShape?: number
  readonly lambdaRate?: number
  /** The smoothing parameter of `gammaNoise` and `fixed`, relative to the largest eigenvalue of the penalty (default 10). */
  readonly lambda?: number
  /** The shape and rate of the gamma prior of the noise precision of the standardised response, for `gammaNoise` (defaults 1 and 0.05: mean 20). */
  readonly noiseShape?: number
  readonly noiseRate?: number
  /** The known noise standard deviation of the standardised response, for `fixed` (default 0.3). */
  readonly noiseSd?: number
  /**
   * The prior of the coefficients: `penalty` (default), the smoothness prior with its smoothing parameter, or `spikeSlab`, in
   * which each coefficient of a local basis (`ps` or `bs`, B-splines without a constraint) is switched on (a wide normal) or off
   * (a narrow one around zero), so a basis function can drop out of the fit.
   */
  readonly prior?: 'penalty' | 'spikeSlab' | 'ard'
  /** The prior probability that a coefficient is in the slab, for `spikeSlab` (default 0.5). */
  readonly inclusion?: number
  /** The standard deviation of the slab, in standard deviations of the response, for `spikeSlab` (default 1). */
  readonly slabScale?: number
  /** The standard deviation of the spike as a fraction of the slab's, for `spikeSlab` (default 0.03). */
  readonly spikeRatio?: number
  /**
   * For `ard` (automatic relevance determination on the coefficients of the same local basis): each coefficient has its own
   * precision `alpha_j`, a gamma variable with this shape (default 1) and rate (`alphaRate`, default 0.5), and a large
   * `alpha_j` prunes the basis function.
   */
  readonly alphaShape?: number
  readonly alphaRate?: number
  /**
   * For `ard`: how much of the precision is the relevance's, from 0 (every coefficient has the unit ridge precision: a plain
   * smooth fit that uses every basis) to 1 (the ARD precision `alpha_j` alone, default): `s alpha_j + (1 - s)`. Between is a
   * prior that prunes less; the sparsity can be raised run after run to see bases switch off.
   */
  readonly sparsity?: number
}

/** A Bayesian GAM bound to rows: the model, its data and the maps from coefficients to curves. */
export interface BayesianGam {
  /** The model definition: priors and likelihood; the rows are in `data`. */
  readonly definition: ModelDefinition
  /** The data of the model (nested numbers: JSON). */
  readonly data: DataInput
  /** The rows the model was built on, as given. */
  readonly rows: { readonly x: readonly number[]; readonly y: readonly number[] }
  /** The number of coefficients, the intercept included. */
  readonly size: number
  /** How many directions the penalty touches: the others are the intercept and a line. */
  readonly penalised: number
  /** The mean and the standard deviation the response was standardised with. */
  readonly standardisation: { readonly mean: number; readonly sd: number }
  /** The priors the model carries. */
  readonly priors: {
    readonly logLambda: { readonly mean: number; readonly sd: number }
    readonly logSigma: { readonly mean: number; readonly sd: number }
  }
  /** The prior of the coefficients. */
  readonly prior: 'penalty' | 'spikeSlab' | 'ard'
  /** Whether the basis is local: B-splines that are non-negative, sum to one and vanish away from their knots (so each coefficient is the height of one hump). */
  readonly local: boolean
  /** The ARD prior's settings (zero for the others). */
  readonly ard: { readonly sparsity: number; readonly shape: number; readonly rate: number }
  /** The hyperprior form of the penalty prior and its settings (`logNormal` for the other priors). */
  readonly hyper: {
    readonly kind: 'logNormal' | 'gammaNoise' | 'fixed' | 'gammaSmooth'
    readonly lambda: number
    readonly lambdaShape: number
    readonly lambdaRate: number
    readonly noiseShape: number
    readonly noiseRate: number
    readonly noiseSd: number
  }
  /** The spike-and-slab prior's settings (zero for the penalty prior): inclusion, slab and spike standard deviations. */
  readonly slab: { readonly inclusion: number; readonly slab: number; readonly spike: number }
  /** The design rows `[m][size]` (in the eigenbasis) at covariate values: the curve is this times the coefficients. */
  design(x: readonly number[]): number[][]
  /**
   * The basis functions `[m][size]` themselves at covariate values (column `j` is `B_j(x)`; for the penalty prior column 0 is the
   * constant): the curve is this times `original(gamma)`.
   */
  parts(x: readonly number[]): number[][]
  /** The coefficients of the basis functions from those the model samples (the eigenbasis rotated back; the same for the spike and slab). */
  original(gamma: ArrayLike<number>): number[]
  /** The precision of each coefficient (eigenbasis) at a log smoothing parameter, in standardised units. */
  precision(logLambda: number): number[]
}

const PENALISED = 1e-9

/**
 * A Bayesian GAM on rows `(x, y)`: the smooth is built by `omai-engine/gam` (`s` with a basis), its penalty is split into
 * eigendirections (the ones with a positive eigenvalue are smoothed, the rest and the intercept get a wide prior), and the
 * model of the spec is `logLambda ~ N`, `logSigma ~ N`, `z ~ N(0, 1)`, `gamma = z / sqrt(exp(logLambda) d + n0)`,
 * `y ~ N(X gamma, exp(logSigma))` on the standardised response.
 *
 * @param x - The covariate, one value per row.
 * @param y - The response.
 * @param options - The basis, its dimension, and the priors.
 * @returns The model, its data and the maps to curves.
 * @throws OmaiError with code `shape` for different lengths, `value` for fewer than 5 rows or a non-positive prior scale.
 * @concept generalized-additive-models
 * @index function Bayesian GAM of one covariate as a model spec: penalty or spike-slab prior.
 * @example
 * ```ts
 * import { bayesianGam } from 'omai-engine/probprog/models'
 * const x = Array.from({ length: 30 }, (_, i) => i / 29)
 * const g = bayesianGam(x, x.map((v) => Math.sin(6 * v)), { k: 8 })
 * [g.size, g.penalised] // => [8, 6]
 * ```
 */
export function bayesianGam(x: readonly number[], y: readonly number[], options: BayesianGamOptions = {}): BayesianGam {
  if (x.length !== y.length) throw new OmaiError('shape', 'bayesianGam: x and y differ in length')
  if (x.length < 5) throw new OmaiError('value', 'bayesianGam: needs at least 5 rows')
  if (options.prior === 'spikeSlab' || options.prior === 'ard') return localGam(x, y, options)
  const logLambda = options.logLambdaPrior ?? { mean: 0, sd: 3 }
  const logSigma = options.logSigmaPrior ?? { mean: -1, sd: 1 }
  const nullScale = options.nullScale ?? 5
  if (!(logLambda.sd > 0 && logSigma.sd > 0 && nullScale > 0))
    throw new OmaiError('value', 'bayesianGam: the prior scales are positive')
  const kind = options.basis ?? 'ps'
  const k = options.k ?? 10
  const degree = options.degree ?? 3
  const order = options.order ?? 2
  // The basis the coefficients are on. A local one (B-splines: `ps`, `bs`) is used raw, a partition of unity with no separate
  // intercept, so every coefficient is the height of one hump. The others (`cr`, `tp`, global bases) come from the gam term,
  // centred, with the intercept as its own column.
  const local = kind === 'ps' || kind === 'bs'
  let evaluateRaw: (at: readonly number[]) => number[][]
  let p: number
  let S: Float64Array
  if (local) {
    const basis = kind === 'ps' ? ps(x, { k, degree, order }) : bs(x, { df: k, degree, intercept: true })
    evaluateRaw = rowsOf(basis)
    p = evaluateRaw([x[0] ?? 0])[0]?.length ?? 0
    S = Float64Array.from(toFlat(basis.penalties[0] as Value))
  } else {
    const term: GamTerm = smoothTerm(0, { basis: kind, k, degree, order })
    const built = buildDesign(
      x.map((v) => [v]),
      [term],
    )
    const block = built.terms[0]
    if (block === undefined || block.penalties.length === 0)
      throw new OmaiError('value', 'bayesianGam: the basis has no penalty')
    p = built.p
    const w = block.width
    // The penalty of the block, with a zero row and column for the intercept.
    S = new Float64Array(p * p)
    for (let a = 0; a < w; a++)
      for (let c = 0; c < w; c++) S[(1 + a) * p + 1 + c] = block.penalties[0]?.matrix[a * w + c] ?? 0
    evaluateRaw = (at) => {
      const grid = evaluateDesign(
        built,
        at.map((v) => [v]),
      )
      return Array.from({ length: grid.n }, (_, i) => Array.from({ length: p }, (_, j) => grid.X[i * p + j] ?? 0))
    }
  }
  // The penalty's eigendirections, scaled so the largest eigenvalue is one: the smoothing parameter is then relative. The
  // directions it does not touch (a constant and a line) get the wide prior.
  const eig = dsyev(S, p)
  const top = Math.max(...eig.w)
  const diagonal = Array.from(eig.w, (e) => (e > PENALISED * top ? e / top : 0))
  const rotate = (rows: number[][]): number[][] =>
    rows.map((row) =>
      Array.from({ length: p }, (_, j) => row.reduce((acc, v, c) => acc + v * (eig.v[c * p + j] ?? 0), 0)),
    )
  const mean = y.reduce((a, v) => a + v, 0) / y.length
  const sd = Math.sqrt(y.reduce((a, v) => a + (v - mean) ** 2, 0) / y.length) || 1
  const nullPrecision = diagonal.map((e) => (e === 0 ? 1 / nullScale ** 2 : 0))
  const hyperKind = options.hyper ?? 'logNormal'
  const lambda0 = options.lambda ?? 10
  const noiseShape = options.noiseShape ?? 1
  const noiseRate = options.noiseRate ?? 0.05
  const noiseSd = options.noiseSd ?? 0.3
  const lambdaShape = options.lambdaShape ?? 1
  const lambdaRate = options.lambdaRate ?? 0.05
  if (!(lambda0 > 0 && noiseShape > 0 && noiseRate > 0 && noiseSd > 0 && lambdaShape > 0 && lambdaRate > 0))
    throw new OmaiError(
      'value',
      'bayesianGam: the smoothing parameter, the gamma parameters and the noise are positive',
    )
  const fixedScale = diagonal.map((e, j) => 1 / Math.sqrt(lambda0 * e + (nullPrecision[j] ?? 0)))
  const definitionOf = (): ModelDefinition => {
    if (hyperKind === 'gammaSmooth') {
      // region gam:gamma-smooth
      // The precision idiom throughout: lambda and tau are gamma latents, the coefficients (in the
      // eigenbasis of the penalty) have the precision lambda d_j + n_j, the noise the precision tau.
      const model = defineModel('bayesian-gam-gamma-smooth', (m) => {
        const row = m.range('row')
        const coef = m.range('coef')
        const X = m.data('X', { dims: [row, coef], role: 'covariate' })
        const penalty = m.data('penalty', {
          dims: [coef],
          doc: 'the eigenvalue of the penalty along each coefficient (0: not smoothed)',
        })
        const flat = m.data('nullPrecision', {
          dims: [coef],
          doc: 'the prior precision of a coefficient the penalty does not touch',
        })
        const y = m.data('y', { dims: [row], role: 'response' })
        const lam = m.latent('lambda', d.gamma(lambdaShape, lambdaRate), {
          doc: 'the smoothing parameter: how heavily wiggliness is penalised',
        })
        const tau = m.latent('tau', d.gamma(noiseShape, noiseRate), {
          doc: 'the noise precision, 1 / sigma squared',
        })
        const gamma = m.latent('gamma', d.normal(0, lam.mul(penalty).add(flat).pow(-0.5)), {
          dims: [coef],
          doc: 'the spline coefficients, with the penalty as the precision of their prior',
        })
        m.observe('y', d.normal(X.matmul(gamma), tau.pow(-0.5)), y)
      })
      // endregion
      return model
    }
    if (hyperKind !== 'logNormal') {
      // region gam:known-scale
      // Centred, with the scale of each coefficient known: a Gaussian vector with a known covariance.
      const model = defineModel(`bayesian-gam-${hyperKind}`, (m) => {
        const row = m.range('row')
        const coef = m.range('coef')
        const X = m.data('X', { dims: [row, coef], role: 'covariate' })
        const scale = m.data('scale', {
          dims: [coef],
          doc: 'the prior standard deviation of each coefficient',
        })
        const y = m.data('y', { dims: [row], role: 'response' })
        const gamma = m.latent('gamma', d.normal(0, scale), {
          dims: [coef],
          doc: 'the spline coefficients, with the penalty as the precision of their prior',
        })
        // endregion
        if (hyperKind === 'fixed') {
          // region gam:known-scale:fixed
          m.observe('y', d.normal(X.matmul(gamma), noiseSd), y)
          // endregion
          return
        }
        // region gam:known-scale:learned
        const tau = m.latent('tau', d.gamma(noiseShape, noiseRate), {
          doc: 'the noise precision, 1 / sigma squared',
        })
        m.observe('y', d.normal(X.matmul(gamma), tau.pow(-0.5)), y)
        // endregion
        // region gam:known-scale:end
      })
      // endregion
      return model
    }
    // region gam:log-normal
    const model = defineModel('bayesian-gam', (m) => {
      const row = m.range('row')
      const coef = m.range('coef')
      const X = m.data('X', { dims: [row, coef], role: 'covariate' })
      const penalty = m.data('penalty', {
        dims: [coef],
        doc: 'the eigenvalue of the penalty along each coefficient (0: not smoothed)',
      })
      const flat = m.data('nullPrecision', {
        dims: [coef],
        doc: 'the prior precision of a coefficient the penalty does not touch',
      })
      const y = m.data('y', { dims: [row], role: 'response' })
      const logLam = m.latent('logLambda', d.normal(logLambda.mean, logLambda.sd), {
        doc: 'the log of the smoothing parameter: how heavily wiggliness is penalised',
      })
      const logSig = m.latent('logSigma', d.normal(logSigma.mean, logSigma.sd), {
        doc: 'the log of the noise standard deviation',
      })
      const z = m.latent('z', d.normal(0, 1), {
        dims: [coef],
        doc: 'the coefficients in units of their prior standard deviation',
      })
      const gamma = m.derived('gamma', z.div(logLam.exp().mul(penalty).add(flat).sqrt()), {
        doc: 'the spline coefficients, with the penalty as the precision of their prior',
      })
      m.observe('y', d.normal(X.matmul(gamma), logSig.exp()), y)
    })
    // endregion
    return model
  }
  const definition = definitionOf()
  const standardised = y.map((v) => (v - mean) / sd)
  return {
    definition,
    data:
      hyperKind === 'logNormal' || hyperKind === 'gammaSmooth'
        ? { X: rotate(evaluateRaw(x)), penalty: diagonal, nullPrecision, y: standardised }
        : { X: rotate(evaluateRaw(x)), scale: fixedScale, y: standardised },
    rows: { x, y },
    size: p,
    penalised: diagonal.filter((e) => e > 0).length,
    standardisation: { mean, sd },
    priors:
      hyperKind === 'gammaSmooth'
        ? { logLambda: { mean: Math.log(lambdaShape / lambdaRate), sd: 3 }, logSigma }
        : { logLambda, logSigma },
    prior: 'penalty',
    local,
    slab: { inclusion: 0, slab: 0, spike: 0 },
    ard: { sparsity: 0, shape: 0, rate: 0 },
    hyper: { kind: hyperKind, lambda: lambda0, lambdaShape, lambdaRate, noiseShape, noiseRate, noiseSd },
    design: (at) => rotate(evaluateRaw(at)),
    parts: evaluateRaw,
    // The eigendirections rotated back to the coefficients of the basis functions.
    original: (gamma) =>
      Array.from({ length: p }, (_, j) => {
        let acc = 0
        for (let c = 0; c < p; c++) acc += (eig.v[j * p + c] ?? 0) * (gamma[c] ?? 0)
        return acc
      }),
    precision: (logLam) => diagonal.map((e, j) => Math.exp(logLam) * e + (nullPrecision[j] ?? 0)),
  }
}

/** The rows of a basis at covariate values, as nested numbers. */
function rowsOf(basis: { readonly evaluate: (x: number[]) => Value }): (at: readonly number[]) => number[][] {
  return (at) => {
    const m = toFlat(basis.evaluate([...at]))
    const cols = Math.floor(m.length / Math.max(1, at.length))
    return Array.from({ length: at.length }, (_, i) => Array.from({ length: cols }, (_, j) => m[i * cols + j] ?? 0))
  }
}

/**
 * The versions of the model on a local basis (B-splines without a constraint), where each coefficient can be switched off: spike
 * and slab (a mixture of a slab and a spike) or ARD (a precision of its own).
 */
function localGam(x: readonly number[], y: readonly number[], options: BayesianGamOptions): BayesianGam {
  const logSigma = options.logSigmaPrior ?? { mean: -1, sd: 1 }
  const ardPrior = options.prior === 'ard'
  const sparsity = options.sparsity ?? 1
  const alphaShape = options.alphaShape ?? 1
  const alphaRate = options.alphaRate ?? 0.5
  if (!(sparsity >= 0 && sparsity <= 1 && alphaShape > 0 && alphaRate > 0))
    throw new OmaiError('value', 'bayesianGam: the sparsity is in [0, 1] and the ARD gamma parameters are positive')
  const inclusion = options.inclusion ?? 0.5
  const slab = options.slabScale ?? 1
  const spike = slab * (options.spikeRatio ?? 0.03)
  if (!(inclusion > 0 && inclusion < 1 && slab > 0 && spike > 0 && logSigma.sd > 0))
    throw new OmaiError('value', 'bayesianGam: the inclusion probability is in (0, 1) and the scales are positive')
  const kind = options.basis ?? 'ps'
  if (kind !== 'ps' && kind !== 'bs')
    throw new OmaiError('value', "bayesianGam: the spike and slab and ARD priors need a local basis ('ps' or 'bs')")
  const k = options.k ?? 10
  const degree = options.degree ?? 3
  const basis =
    kind === 'ps' ? ps(x, { k, degree, order: options.order ?? 2 }) : bs(x, { df: k, degree, intercept: true })
  const raw = (at: readonly number[]): number[][] => {
    const m = toFlat(basis.evaluate(at))
    const cols = Math.floor(m.length / Math.max(1, at.length))
    return Array.from({ length: at.length }, (_, i) => Array.from({ length: cols }, (_, j) => m[i * cols + j] ?? 0))
  }
  const rows = raw(x)
  const size = rows[0]?.length ?? 0
  const mean = y.reduce((a, v) => a + v, 0) / y.length
  const sd = Math.sqrt(y.reduce((a, v) => a + (v - mean) ** 2, 0) / y.length) || 1
  // region gam:local
  const definition = defineModel(ardPrior ? 'bayesian-gam-ard' : 'bayesian-gam-spike-slab', (m) => {
    const row = m.range('row')
    const coef = m.range('coef')
    const X = m.data('X', { dims: [row, coef], role: 'covariate' })
    const y = m.data('y', { dims: [row], role: 'response' })
    const logSig = m.latent('logSigma', d.normal(logSigma.mean, logSigma.sd), {
      doc: 'the log of the noise standard deviation',
    })
    // endregion
    if (ardPrior) {
      // region gam:local:ard
      // Non-centred: the coefficient is z over the square root of its precision, a mix of the unit
      // ridge and alpha_j.
      const alpha = m.latent('alpha', d.gamma(alphaShape, alphaRate), {
        dims: [coef],
        doc: 'the precision of each coefficient (relevance): a large one prunes the basis function',
      })
      const z = m.latent('z', d.normal(0, 1), {
        dims: [coef],
        doc: 'the coefficients in units of their prior standard deviation',
      })
      const relevance = alpha.mul(sparsity).add(1 - sparsity)
      const beta = m.derived('beta', z.div(relevance.sqrt()), {
        doc: 'the coefficient of each basis function',
      })
      m.observe('y', d.normal(X.matmul(beta), logSig.exp()), y)
      // endregion
      return
    }
    // region gam:local:spike-slab
    const beta = m.latent('beta', d.normalSpikeSlab(inclusion, slab, spike), {
      dims: [coef],
      doc: 'the coefficient of each basis function: in the slab (used) or the spike (off)',
    })
    m.observe('y', d.normal(X.matmul(beta), logSig.exp()), y)
    // endregion
    // region gam:local:end
  })
  // endregion
  return {
    definition,
    data: { X: rows, y: y.map((v) => (v - mean) / sd) },
    rows: { x, y },
    size,
    penalised: size,
    standardisation: { mean, sd },
    priors: { logLambda: { mean: 0, sd: 1 }, logSigma },
    prior: ardPrior ? 'ard' : 'spikeSlab',
    local: true,
    slab: ardPrior ? { inclusion: 0, slab: 0, spike: 0 } : { inclusion, slab, spike },
    ard: ardPrior ? { sparsity, shape: alphaShape, rate: alphaRate } : { sparsity: 0, shape: 0, rate: 0 },
    hyper: { kind: 'logNormal', lambda: 0, lambdaShape: 0, lambdaRate: 0, noiseShape: 0, noiseRate: 0, noiseSd: 0 },
    design: raw,
    parts: raw,
    original: (gamma) => Array.from({ length: size }, (_, j) => gamma[j] ?? 0),
    precision: () => [],
  }
}

/** What a posterior says of the curve and of the smoothing, as plain numbers (JSON). */
export interface BayesianGamFit {
  /** The covariate values the curves are evaluated at. */
  readonly grid: readonly number[]
  /** The posterior mean of the curve (in the units of the response), and the equal-tailed credible band at `level`. */
  readonly mean: readonly number[]
  readonly lower: readonly number[]
  readonly upper: readonly number[]
  readonly level: number
  /** A few sampled curves. */
  readonly curves: readonly (readonly number[])[]
  /** Draws of the hyperparameters and of the curve at two covariate values, for a pair plot (all of one length). */
  readonly draws: Readonly<Record<string, readonly number[]>>
  /** The posterior mean and standard deviation of the log smoothing parameter (null for the spike and slab prior) and the log noise. */
  readonly hyper: { readonly logLambda: readonly [number, number] | null; readonly logSigma: readonly [number, number] }
  /**
   * The curve as the sum of its basis functions: `mean[j]` is the `j`th weighted basis function `B_j(x)` times the posterior
   * mean of its coefficient on the grid (in the units of the response; the mean of the response is added to the constant, or
   * spread over the bases for a partition of unity), and `curves[c][j]` the same for the `c`th sampled curve. The parts of a
   * curve add up to it.
   */
  readonly bases: {
    readonly mean: readonly (readonly number[])[]
    readonly curves: readonly (readonly (readonly number[])[])[]
    /** Where each basis function peaks on the grid. */
    readonly centres: readonly number[]
    /** Whether the basis is local (each part vanishes away from its knots); the parts of a global basis (`cr`, `tp`) are not humps. */
    readonly local: boolean
  }
  /** For the spike and slab prior: the posterior probability that each coefficient is in the slab. */
  readonly inclusion?: readonly number[]
  /** How many draws the summaries are from. */
  readonly count: number
}

/** Options of {@link bayesianGamFit}. */
export interface BayesianGamFitOptions {
  /** The number of grid points (default 100). */
  readonly gridSize?: number
  /** Draws the summary is made from (default 300). */
  readonly draws?: number
  /** Sampled curves to return (default 8). */
  readonly curves?: number
  /** The credible level of the band (default 0.9). */
  readonly level?: number
  /** The covariate range of the grid (default the range of the rows). */
  readonly range?: readonly [number, number]
}

const linspace = (lo: number, hi: number, n: number): number[] =>
  Array.from({ length: n }, (_, i) => lo + ((hi - lo) * i) / Math.max(1, n - 1))

/**
 * What a posterior of a {@link bayesianGam} says: draws of the coefficients become draws of the curve on a grid (back in the
 * units of the response), summarised as a mean and an equal-tailed band, a few of them kept as sampled curves, and the draws
 * of the log smoothing parameter and the log noise returned with the curve at the first and third quarter of the range.
 *
 * @param model - The model the posterior is of.
 * @param posterior - A posterior of any form (draws, a variational family, a Laplace fit).
 * @param options - The grid, the number of draws and curves, the level of the band.
 * @returns The fit.
 * @concept generalized-additive-models
 * @index function Curve, band, sampled curves and weighted bases of a Bayesian GAM posterior.
 * @example
 * ```ts
 * import { bayesianGam, bayesianGamFit } from 'omai-engine/probprog/models'
 * import { laplaceMethod } from 'omai-engine/probprog/inference'
 * import { session } from 'omai-engine/trace'
 * const x = Array.from({ length: 40 }, (_, i) => i / 39)
 * const g = bayesianGam(x, x.map((v) => Math.sin(6 * v)), { k: 8 })
 * const m = laplaceMethod(g.definition.bind(g.data))
 * const s = session(m, { steps: 200, mode: 'eager' })
 * bayesianGamFit(g, m.posterior!(s.state, s.config), { gridSize: 20 }).mean.length // => 20
 * ```
 */
export function bayesianGamFit(
  model: BayesianGam,
  posterior: Posterior,
  options: BayesianGamFitOptions = {},
): BayesianGamFit {
  const count = options.draws ?? 300
  const level = options.level ?? 0.9
  const xs = model.rows.x
  const [lo, hi] = options.range ?? [Math.min(...xs), Math.max(...xs)]
  const grid = linspace(lo, hi, options.gridSize ?? 100)
  const draws = posterior.sample(makeKey(0), count)
  const size = model.size
  const prior = model.prior
  const spike = prior !== 'penalty'
  // The centred forms of the penalty prior (`gammaNoise`, `fixed`): the coefficients are drawn directly, and the noise is a gamma
  // precision `tau` or a known constant.
  const centred = prior === 'penalty' && model.hyper.kind !== 'logNormal'
  const fixedNoise = centred && model.hyper.kind === 'fixed'
  const smooth = centred && model.hyper.kind === 'gammaSmooth'
  const lambdaDraws = draws['lambda']
  const logLam = draws['logLambda']
  const alphaDraws = draws['alpha']
  const tauDraws = draws['tau']
  // The non-centred forms record the coefficients themselves (`gamma`, `beta`): the draws are read as they are when present (message
  // passing has marginals of each, and the product of independent marginals of `z` and the scale is not their joint); else the
  // standard normal `z` is divided by the square root of the precision of each draw.
  const direct = prior === 'penalty' && !centred ? draws['gamma'] : prior === 'ard' ? draws['beta'] : undefined
  const source = direct ?? draws[centred ? 'gamma' : prior === 'spikeSlab' ? 'beta' : 'z']
  const logSig = fixedNoise ? undefined : centred ? (tauDraws === undefined ? undefined : tauDraws) : draws['logSigma']
  if (
    source === undefined ||
    (logSig === undefined && !fixedNoise) ||
    (prior === 'penalty' && !centred && logLam === undefined && direct === undefined) ||
    (prior === 'ard' && alphaDraws === undefined)
  )
    throw new OmaiError(
      'value',
      `bayesianGamFit: the posterior has no draws of ${centred ? 'gamma and tau' : { penalty: 'z and logLambda', spikeSlab: 'beta', ard: 'z and alpha' }[prior]} and logSigma`,
    )
  const raw = toFlat(source)
  const lams =
    smooth && lambdaDraws !== undefined
      ? Float64Array.from(toFlat(lambdaDraws), Math.log)
      : logLam === undefined
        ? new Float64Array(0)
        : toFlat(logLam)
  const alphas = alphaDraws === undefined ? new Float64Array(0) : toFlat(alphaDraws)
  const n = Math.min(
    Math.floor(raw.length / size),
    prior === 'penalty' && !centred ? lams.length : Number.POSITIVE_INFINITY,
    prior === 'ard' ? Math.floor(alphas.length / size) : Number.POSITIVE_INFINITY,
  )
  const { sparsity } = model.ard
  // The precision of coefficient j in draw i under ARD: the mix of the unit ridge and its own alpha.
  const ardPrecision = (i: number, j: number): number => sparsity * (alphas[i * size + j] ?? 1) + (1 - sparsity)
  // The coefficients of the basis functions in draw i: for the penalty prior the standard normal z over the square root of the
  // precision at that draw's smoothing parameter (rotated back from the eigenbasis); for ARD z over the square root of its own
  // precision; for the spike and slab prior the draw itself.
  const beta: number[][] = Array.from({ length: n }, (_, i) => {
    const row = Array.from({ length: size }, (_, j) => raw[i * size + j] ?? 0)
    if (prior === 'spikeSlab') return row
    if (prior === 'ard') return direct === undefined ? row.map((v, j) => v / Math.sqrt(ardPrecision(i, j))) : row
    if (centred || direct !== undefined) return model.original(row)
    const precision = model.precision(lams[i] ?? 0)
    return model.original(row.map((v, j) => v / Math.sqrt(precision[j] ?? 1)))
  })
  const probe = [lo + 0.25 * (hi - lo), lo + 0.75 * (hi - lo), lo + 0.5 * (hi - lo)]
  const rows = model.parts([...grid, ...probe])
  const { mean: ym, sd: ys } = model.standardisation
  const G = grid.length
  const curveOf = (b: readonly number[]): number[] =>
    rows.map((row) => {
      let acc = 0
      for (let j = 0; j < size; j++) acc += (row[j] ?? 0) * (b[j] ?? 0)
      return ym + ys * acc
    })
  const all = beta.map(curveOf)
  const column = (g: number): number[] => all.map((c) => c[g] ?? 0)
  const tail = (1 - level) / 2
  const at = (f: (v: number[]) => number): number[] => grid.map((_, g) => f(column(g)))
  const lsig = fixedNoise
    ? new Array<number>(n).fill(Math.log(model.hyper.noiseSd))
    : centred
      ? Array.from(toFlat(logSig as never))
          .slice(0, n)
          .map((t) => -0.5 * Math.log(t))
      : Array.from(toFlat(logSig as never)).slice(0, n)
  const stat = (v: readonly number[]): readonly [number, number] => {
    const m = v.reduce((a, b) => a + b, 0) / v.length
    return [m, Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / Math.max(1, v.length - 1))]
  }
  const keep = options.curves ?? 8
  const picks = Array.from({ length: Math.min(keep, n) }, (_, i) => Math.floor((i * n) / keep))
  // The weighted basis functions of a coefficient vector: B_j(x) b_j in the units of the response, with the mean of the
  // response in the constant (the penalty prior's first column) or spread over a partition of unity.
  const partsOf = (b: readonly number[]): number[][] =>
    Array.from({ length: size }, (_, j) =>
      Array.from({ length: G }, (_, g) => {
        const B = rows[g]?.[j] ?? 0
        return ys * B * (b[j] ?? 0) + (model.local ? ym * B : j === 0 ? ym : 0)
      }),
    )
  const meanBeta = Array.from({ length: size }, (_, j) => beta.reduce((a, b) => a + (b[j] ?? 0), 0) / Math.max(1, n))
  const centres = Array.from({ length: size }, (_, j) => {
    let best = 0
    for (let g = 1; g < G; g++) if ((rows[g]?.[j] ?? 0) > (rows[best]?.[j] ?? 0)) best = g
    return grid[best] ?? lo
  })
  // The posterior probability of the slab for each coefficient, averaged over the draws.
  const { inclusion: pi, slab, spike: narrow } = model.slab
  const slabShare = (v: number): number => {
    const a = (pi * Math.exp(-0.5 * (v / slab) ** 2)) / slab
    const c = ((1 - pi) * Math.exp(-0.5 * (v / narrow) ** 2)) / narrow
    return a / (a + c)
  }
  const ll = lams.length === 0 ? [] : Array.from(lams).slice(0, n)
  // How much each basis function is in use in a draw: the posterior probability of the slab, or under ARD the share of the unit
  // ridge's width its precision leaves (1 for a precision of the ridge or less, towards 0 as the basis is pruned).
  const inUse = (i: number, j: number): number =>
    prior === 'ard' ? 1 / Math.max(1, ardPrecision(i, j)) : slabShare(beta[i]?.[j] ?? 0)
  const active = beta.map((_, i) => Array.from({ length: size }, (_, j) => inUse(i, j)).reduce((a, b) => a + b, 0))
  return {
    grid,
    mean: at((v) => v.reduce((a, b) => a + b, 0) / v.length),
    lower: at((v) => quantileOf(v, tail)),
    upper: at((v) => quantileOf(v, 1 - tail)),
    level,
    curves: picks.map((i) => all[i]?.slice(0, G) ?? []),
    draws: spike
      ? { logSigma: lsig, 'f(25%)': column(G), 'f(75%)': column(G + 1), 'active bases': active }
      : centred
        ? {
            ...(smooth && ll.length > 0 ? { logLambda: ll } : {}),
            ...(fixedNoise ? {} : { logSigma: lsig }),
            'f(25%)': column(G),
            'f(75%)': column(G + 1),
            'f(50%)': column((G + 2) as number),
          }
        : { logLambda: ll, logSigma: lsig, 'f(25%)': column(G), 'f(75%)': column(G + 1) },
    hyper: { logLambda: spike || (centred && !(smooth && ll.length > 0)) ? null : stat(ll), logSigma: stat(lsig) },
    bases: { mean: partsOf(meanBeta), curves: picks.map((i) => partsOf(beta[i] ?? [])), centres, local: model.local },
    ...(spike
      ? {
          inclusion: Array.from(
            { length: size },
            (_, j) => beta.reduce((a, _b, i) => a + inUse(i, j), 0) / Math.max(1, n),
          ),
        }
      : {}),
    count: n,
  }
}

/** The exact posterior of a Bayesian GAM on a grid of covariate values: what the methods are checked against. */
export interface BayesianGamReference {
  readonly grid: readonly number[]
  /** The posterior mean and the equal-tailed band of the curve, at `level`. */
  readonly mean: readonly number[]
  readonly lower: readonly number[]
  readonly upper: readonly number[]
  readonly level: number
  /** The posterior mean and standard deviation of the log smoothing parameter and the log noise. */
  readonly hyper: { readonly logLambda: readonly [number, number]; readonly logSigma: readonly [number, number] }
}

interface Cell {
  readonly ll: number
  readonly ls: number
  logWeight: number
  readonly mean: Float64Array
  readonly factor: Float64Array
}

/** The log marginal likelihood of the response at a smoothing parameter and a noise, and the Gaussian of the coefficients given them. */
function conditional(
  model: BayesianGam,
  XtX: Float64Array,
  Xty: Float64Array,
  yty: number,
  n: number,
  ll: number,
  ls: number,
): Cell {
  const p = model.size
  const sigma2 = Math.exp(2 * ls)
  const prec = model.precision(ll)
  const A = Float64Array.from(XtX, (v) => v / sigma2)
  for (let j = 0; j < p; j++) A[j * p + j] = (A[j * p + j] ?? 0) + (prec[j] ?? 0)
  const { factor, info } = dpotrf(A, p)
  if (info !== 0) return { ll, ls, logWeight: Number.NEGATIVE_INFINITY, mean: new Float64Array(p), factor }
  const b = Float64Array.from(Xty, (v) => v / sigma2)
  const mean = dpotrs(factor, b, p)
  let logDetA = 0
  for (let j = 0; j < p; j++) logDetA += 2 * Math.log(factor[j * p + j] ?? 1)
  const logDetD = prec.reduce((a, e) => a + Math.log(e), 0)
  const quad = b.reduce((a, v, j) => a + v * (mean[j] ?? 0), 0)
  const logMarginal =
    -0.5 * n * Math.log(2 * Math.PI * sigma2) - (0.5 * yty) / sigma2 + 0.5 * quad + 0.5 * logDetD - 0.5 * logDetA
  const { logLambda: pl, logSigma: pn } = model.priors
  const h = model.hyper
  // The prior of the pair: normal on both logs; with a known smoothing parameter only the noise (a gamma precision, written
  // for its log standard deviation: log Gamma(tau) + log(2 tau), tau = exp(-2 ls)), or nothing when both are known.
  const gammaLog = (ls2: number): number => {
    const tau = Math.exp(-2 * ls2)
    return (
      h.noiseShape * Math.log(h.noiseRate) -
      item(lgamma(h.noiseShape)) +
      (h.noiseShape - 1) * Math.log(tau) -
      h.noiseRate * tau +
      Math.log(2 * tau)
    )
  }
  // The smoothing parameter's gamma prior written for its log: log Gamma(lambda) + ll (the Jacobian).
  const lambdaLog = (ll2: number): number =>
    h.lambdaShape * Math.log(h.lambdaRate) -
    item(lgamma(h.lambdaShape)) +
    h.lambdaShape * ll2 -
    h.lambdaRate * Math.exp(ll2)
  const logPrior =
    h.kind === 'gammaSmooth'
      ? lambdaLog(ll) + gammaLog(ls)
      : h.kind === 'logNormal'
        ? -0.5 * ((ll - pl.mean) / pl.sd) ** 2 - Math.log(pl.sd) - 0.5 * ((ls - pn.mean) / pn.sd) ** 2 - Math.log(pn.sd)
        : h.kind === 'gammaNoise'
          ? gammaLog(ls)
          : 0
  return { ll, ls, logWeight: logMarginal + logPrior, mean, factor }
}

/**
 * The exact posterior of a {@link bayesianGam}, to the accuracy of a quadrature. Given the log smoothing parameter and the
 * log noise the coefficients are Gaussian (the model is linear-Gaussian in them), with the evidence in closed form, so the
 * posterior of the pair is a two-dimensional density that is evaluated on a grid (a coarse one, then a fine one around where
 * the mass is) and the curve's posterior is the mixture of the Gaussians it implies; the mean and the band are read from the
 * mixture (the band by bisection on its distribution function). Any inference method's answer is checked against this.
 *
 * @param model - The model.
 * @param options - The grid of the curve, the level of the band, and the points of the quadrature per axis (default 40).
 * @returns The mean, the band and the hyperparameter moments.
 * @throws OmaiError with code `value` for the spike and slab prior (the indicators make the posterior a sum over subsets of the bases).
 * @concept generalized-additive-models
 * @index function Exact posterior curve and band of a Bayesian GAM by quadrature.
 * @example
 * ```ts
 * import { bayesianGam, bayesianGamReference } from 'omai-engine/probprog/models'
 * const x = Array.from({ length: 40 }, (_, i) => i / 39)
 * const g = bayesianGam(x, x.map((v) => Math.sin(6 * v) + 0.1 * Math.cos(50 * v)), { k: 8 })
 * const r = bayesianGamReference(g, { gridSize: 10, points: 16 })
 * r.mean.length // => 10
 * ```
 */
export function bayesianGamReference(
  model: BayesianGam,
  options: BayesianGamFitOptions & { readonly points?: number } = {},
): BayesianGamReference {
  if (model.prior !== 'penalty')
    throw new OmaiError('value', 'bayesianGamReference: the exact posterior exists for the penalty prior only')
  const p = model.size
  const X = (model.data['X'] as number[][]) ?? []
  const y = (model.data['y'] as number[]) ?? []
  const n = y.length
  const XtX = new Float64Array(p * p)
  const Xty = new Float64Array(p)
  for (let i = 0; i < n; i++) {
    const row = X[i] ?? []
    for (let a = 0; a < p; a++) {
      Xty[a] = (Xty[a] ?? 0) + (row[a] ?? 0) * (y[i] ?? 0)
      for (let b = 0; b < p; b++) XtX[a * p + b] = (XtX[a * p + b] ?? 0) + (row[a] ?? 0) * (row[b] ?? 0)
    }
  }
  const yty = y.reduce((a, v) => a + v * v, 0)
  const points = options.points ?? 40
  // The axes of the quadrature: both logs when both are inferred; with a known smoothing parameter or noise that axis is one point.
  const h = model.hyper
  const knownLambda = h.kind !== 'logNormal' && h.kind !== 'gammaSmooth'
  const knownNoise = h.kind === 'fixed'
  const run = (llRange: readonly [number, number], lsRange: readonly [number, number]): Cell[] => {
    const cells: Cell[] = []
    const lls = knownLambda ? [Math.log(h.lambda)] : linspace(llRange[0], llRange[1], points)
    const lss = knownNoise ? [Math.log(h.noiseSd)] : linspace(lsRange[0], lsRange[1], points)
    for (const ll of lls) for (const ls of lss) cells.push(conditional(model, XtX, Xty, yty, n, ll, ls))
    const top = Math.max(...cells.map((c) => c.logWeight))
    for (const c of cells) c.logWeight -= top
    return cells
  }
  const moments = (cells: readonly Cell[]): { m: [number, number]; s: [number, number] } => {
    const w = cells.map((c) => Math.exp(c.logWeight))
    const total = w.reduce((a, b) => a + b, 0)
    const m = [0, 1].map((k) => cells.reduce((a, c, i) => a + ((w[i] ?? 0) * (k === 0 ? c.ll : c.ls)) / total, 0))
    const s = [0, 1].map((k) =>
      Math.sqrt(cells.reduce((a, c, i) => a + ((w[i] ?? 0) * ((k === 0 ? c.ll : c.ls) - (m[k] ?? 0)) ** 2) / total, 0)),
    )
    return { m: [m[0] ?? 0, m[1] ?? 0], s: [s[0] ?? 0, s[1] ?? 0] }
  }
  const { logLambda: pl, logSigma: pn } = model.priors
  const coarse = run(
    [pl.mean - 5 * pl.sd, pl.mean + 5 * pl.sd],
    h.kind === 'logNormal' ? [pn.mean - 5 * pn.sd - 1, pn.mean + 5 * pn.sd] : [-6, 2],
  )
  const first = moments(coarse)
  const cells = run(
    [first.m[0] - 5 * first.s[0], first.m[0] + 5 * first.s[0]],
    [first.m[1] - 5 * first.s[1], first.m[1] + 5 * first.s[1]],
  )
  const fine = moments(cells)
  const total = cells.reduce((a, c) => a + Math.exp(c.logWeight), 0)
  // The cells holding all but 1e-5 of the mass, the heaviest first (the tails of the grid carry nothing).
  const ranked = [...cells].sort((a, b) => b.logWeight - a.logWeight)
  const kept: Cell[] = []
  let mass = 0
  for (const c of ranked) {
    if (mass >= 1 - 1e-5 || kept.length >= 300) break
    kept.push(c)
    mass += Math.exp(c.logWeight) / total
  }
  const keptMass = kept.reduce((a, c) => a + Math.exp(c.logWeight), 0)
  const weights = kept.map((c) => Math.exp(c.logWeight) / keptMass)
  const level = options.level ?? 0.9
  const xs = model.rows.x
  const [lo, hi] = options.range ?? [Math.min(...xs), Math.max(...xs)]
  const grid = linspace(lo, hi, options.gridSize ?? 100)
  const rows = model.design(grid)
  const { mean: ym, sd: ys } = model.standardisation
  const centre: number[][] = grid.map(() => [])
  const spread: number[][] = grid.map(() => [])
  for (const c of kept) {
    for (const [g, row] of rows.entries()) {
      let m = 0
      for (let j = 0; j < p; j++) m += (row[j] ?? 0) * (c.mean[j] ?? 0)
      // The variance of row . gamma is |U^-T row|^2 for A = U^T U.
      const v = dpotrs(c.factor, row, p)
      let q = 0
      for (let j = 0; j < p; j++) q += (row[j] ?? 0) * (v[j] ?? 0)
      ;(centre[g] as number[]).push(m)
      ;(spread[g] as number[]).push(Math.sqrt(Math.max(q, 0)))
    }
  }
  const quantileAt = (g: number, q: number): number => {
    const m = centre[g] as number[]
    const s = spread[g] as number[]
    let a = Math.min(...m.map((v, i) => v - 8 * (s[i] ?? 0)))
    let b = Math.max(...m.map((v, i) => v + 8 * (s[i] ?? 0)))
    for (let it = 0; it < 36; it++) {
      const mid = 0.5 * (a + b)
      const z = Float64Array.from(m, (v, i) => (mid - v) / Math.max(s[i] ?? 0, 1e-12))
      const phi = toFlat(ndtr(fromStorage(z, [z.length], 'float64')))
      let cdf = 0
      for (let i = 0; i < m.length; i++) cdf += (weights[i] ?? 0) * (phi[i] ?? 0)
      if (cdf < q) a = mid
      else b = mid
    }
    return 0.5 * (a + b)
  }
  const tail = (1 - level) / 2
  return {
    grid,
    mean: grid.map((_, g) => ym + ys * (centre[g] as number[]).reduce((a, v, i) => a + (weights[i] ?? 0) * v, 0)),
    lower: grid.map((_, g) => ym + ys * quantileAt(g, tail)),
    upper: grid.map((_, g) => ym + ys * quantileAt(g, 1 - tail)),
    level,
    hyper: { logLambda: [fine.m[0], fine.s[0]], logSigma: [fine.m[1], fine.s[1]] },
  }
}

/** How well a fit agrees with the exact reference: three numbers a page can show. */
export interface BayesianGamAgreement {
  /** The root-mean-square difference of the posterior mean curves, in standard deviations of the response. */
  readonly meanError: number
  /** The mean of the ratio of the fit's band width to the reference's (1: the same width; under 1: overconfident). */
  readonly bandRatio: number
  /** The difference of the posterior means of the log smoothing parameter (fit minus reference). */
  readonly logLambdaShift: number
}

/**
 * The agreement of a fit with the exact reference on the same grid.
 *
 * @param fit - A method's fit.
 * @param reference - The reference of the same model and grid.
 * @param scale - The standard deviation of the response (`model.standardisation.sd`), the unit of the mean error.
 * @returns The mean error, the band ratio and the shift of the log smoothing parameter.
 * @throws OmaiError with code `shape` when the grids differ in length.
 * @concept posterior-validation
 * @example
 * ```ts
 * import { bayesianGamAgreement } from 'omai-engine/probprog/models'
 * const f = { grid: [0, 1], mean: [0, 1], lower: [-1, 0], upper: [1, 2], level: 0.9, curves: [], draws: {}, hyper: { logLambda: [0, 1], logSigma: [0, 1] }, bases: { mean: [], curves: [], centres: [], local: true }, count: 1 } as const
 * bayesianGamAgreement(f, { ...f }, 1) // => { meanError: 0, bandRatio: 1, logLambdaShift: 0 }
 * ```
 */
export function bayesianGamAgreement(
  fit: BayesianGamFit,
  reference: BayesianGamReference,
  scale: number,
): BayesianGamAgreement {
  if (fit.mean.length !== reference.mean.length) throw new OmaiError('shape', 'bayesianGamAgreement: the grids differ')
  const n = fit.mean.length
  let sq = 0
  let ratio = 0
  for (let i = 0; i < n; i++) {
    sq += ((fit.mean[i] ?? 0) - (reference.mean[i] ?? 0)) ** 2
    ratio +=
      ((fit.upper[i] ?? 0) - (fit.lower[i] ?? 0)) /
      Math.max((reference.upper[i] ?? 0) - (reference.lower[i] ?? 0), 1e-12)
  }
  return {
    meanError: Math.sqrt(sq / n) / scale,
    bandRatio: ratio / n,
    logLambdaShift: (fit.hyper.logLambda?.[0] ?? Number.NaN) - reference.hyper.logLambda[0],
  }
}

/**
 * An inference method of a kind (`nuts`, `svi`, `laplace`, ... of {@link methodKinds}) on a Bayesian GAM, with one derived path
 * added to its tape, `gam/fit`: the {@link bayesianGamFit} of the posterior at the step viewed (JSON), so a page, or a worker
 * serving it, reads the curve and band of the run as it goes and at any step it is scrubbed to, from the library's own probe.
 *
 * @param kind - A key of {@link methodKinds}.
 * @param model - The model.
 * @param options - The method's name (default the kind) and the options of the fit.
 * @returns The method.
 * @throws OmaiError with code `value` for an unknown kind.
 * @concept generalized-additive-models
 * @index estimator A Bayesian GAM under any inference method, with its fit as a tape probe.
 * @example
 * ```ts
 * import { bayesianGam, bayesianGamMethod } from 'omai-engine/probprog/models'
 * import { session } from 'omai-engine/trace'
 * const x = Array.from({ length: 40 }, (_, i) => i / 39)
 * const g = bayesianGam(x, x.map((v) => Math.sin(6 * v)), { k: 8 })
 * const s = session(bayesianGamMethod('laplace', g, { fit: { gridSize: 12 } }), { steps: 100, mode: 'eager' })
 * (s.probe(s.t, 'gam/fit') as { mean: number[] }).mean.length // => 12
 * ```
 */
export function bayesianGamMethod(
  kind: string,
  model: BayesianGam,
  options: { readonly name?: string; readonly fit?: BayesianGamFitOptions } = {},
): AnyInferenceMethod {
  const make = methodKinds[kind]
  if (make === undefined) throw new OmaiError('value', `bayesianGamMethod: no inference method of kind '${kind}'`)
  const name = options.name ?? kind
  // region gam:bind
  // The chosen method, bound to the model and its data.
  const method = make(model.definition.bind(model.data), { name })
  // endregion
  const contractOf = method.contract
  return {
    ...method,
    contract: (config: Config, key: Key) => ({
      ...(typeof contractOf === 'function' ? contractOf(config, key) : contractOf),
      'gam/fit': derived({
        cost: 'dear',
        description:
          'the curve, credible band, sampled curves and smoothing draws of the posterior at this step, as JSON (null before there are draws)',
        // Before a run has any draws there is nothing to summarise: the probe is null, not an error.
        compute: (state: unknown) => {
          try {
            // The fit is plain numbers, so it is JSON; the interface has no index signature, hence the cast.
            return bayesianGamFit(model, method.posterior?.(state, config) as Posterior, options.fit) as unknown as Json
          } catch {
            return null
          }
        },
      }),
    }),
  }
}

/**
 * The starting knobs of the methods on a Bayesian GAM: what each opens with so that a run is quick on this model (a shallow
 * NUTS tree, a few hundred iterations), and nothing more: every method of the library that runs on the model is offered
 * (`bayesianGamMethods`), these only set where the controls start. The joint mode of the coefficients and the smoothing
 * parameter (Laplace at the MAP) applies and runs, but see {@link bayesianGamNotes}.
 */
export const bayesianGamStarting: Readonly<Record<string, Readonly<Record<string, number>>>> = {
  hmc: { numChains: 2, numWarmup: 150, numDraws: 200 },
  nuts: { numChains: 1, numWarmup: 100, numDraws: 150, maxTreeDepth: 5 },
  svi: { learningRate: 0.05 },
}

/** What is known of how methods behave on a Bayesian GAM, for a page to show beside the list: measured, not a verdict of the method's own check. */
export const bayesianGamNotes: Readonly<Record<string, string>> = {
  laplace:
    'Laplace at the joint mode of the coefficients and the smoothing parameter runs, but the mode sits where the smoothing parameter is far too small (on a sine with 80 rows, log smoothing about -0.9 where the exact posterior has 2.6 +- 0.6) and its band is 25 times too wide: the joint MAP of a hierarchical model is not a point of the marginal posterior.',
  'mean-field and full-rank SVI':
    'A mean-field guide understates the spread of the smoothing parameter and the noise; the full-rank one converges slowly under a two-second budget.',
}

/**
 * Every inference method the library can run on a Bayesian GAM of a prior, as one union space of their knobs and the reason for
 * each that cannot (see `methodSpace`), with the starting knobs of {@link bayesianGamStarting}, opening on HMC.
 *
 * @param prior - The prior of the model (default the smoothness penalty); the verdicts are those of this model.
 * @param hyper - The hyperprior form of the penalty prior (see `BayesianGamOptions.hyper`): expectation propagation applies to every form (the non-centred ones through the `apply-product` factor), `gammaSmooth`, `gammaNoise` and `fixed` open on it, and `fixed` also has exact Gaussian inference.
 * @returns The union (tagged `method`), the applicable kinds and the reasons for the others.
 * @concept inference-engines
 * @example
 * ```ts
 * import { bayesianGamMethods } from 'omai-engine/probprog/models'
 * const m = bayesianGamMethods()
 * [m.applicable.includes('nuts'), m.applicable.includes('ep')] // => [true, true]
 * ```
 */
export function bayesianGamMethods(
  prior: 'penalty' | 'spikeSlab' | 'ard' = 'penalty',
  hyper: 'logNormal' | 'gammaNoise' | 'fixed' | 'gammaSmooth' = 'logNormal',
): MethodSpace {
  const probe = bayesianGam(
    Array.from({ length: 8 }, (_, i) => i),
    Array.from({ length: 8 }, (_, i) => i * i),
    { k: 5, prior, hyper },
  )
  // region gam:offer
  // The forms whose message passing is quick (a second or two) open on expectation propagation, the rest on HMC.
  const quick = prior === 'penalty' && hyper !== 'logNormal'
  // Every method of the library that can run on the model, and the reason for each that cannot.
  const offer = methodSpace(probe.definition.bind(probe.data), {
    starting: bayesianGamStarting,
    omit: ['inferDiscrete'],
    first: quick ? 'ep' : 'hmc',
  })
  // endregion
  return offer
}
