/**
 * Functionals of the fitted smooths with their uncertainty: the partial effect of every term at new rows with standard
 * errors and with chosen terms left out (`exclude`, e.g. predicting without a random effect), the derivative of a
 * smooth with a credible band (central differences, or Richardson extrapolation of them to fourth order), and the
 * difference between two smooths (the levels of a by-variable) with pointwise or simultaneous bands. Each is a linear
 * combination of the coefficients, so its band comes from the posterior covariance exactly as the band of the effect does.
 */

import { OmaiError } from '../base/index.ts'
import { matrixView } from '../learn/index.ts'
import type { TensorLike } from '../tensor/index.ts'
import { type Band, type BandOptions, bandFromDesign, type GamFitContext } from './inference.ts'

/** Partial effects of all terms at new rows, with their standard errors. */
export interface TermPredictions {
  readonly labels: readonly string[]
  /** The effects `[m, terms]`, row-major; a left-out term is zero. */
  readonly effects: Float64Array
  /** Their standard errors `[m, terms]`; zero for a left-out term. */
  readonly se: Float64Array
  /** The band of the sum of the included terms (and the intercept, unless left out). */
  readonly total: Band
}

/**
 * The partial effect of each term at new rows with its standard error, and the band of their sum. Terms in `exclude`
 * contribute nothing (their columns are not even evaluated, so a random-effect level unseen in training is no problem when
 * the term is excluded), as mgcv's `predict(..., exclude = )`.
 *
 * @param ctx - The fitted pieces.
 * @param x - New covariates `[m, d]`.
 * @param options - `exclude` (term indices), `intercept` (default true) and the band options.
 * @returns The effects, standard errors and the band of the total.
 * @concept generalized-additive-models
 * @example
 * ```ts
 * import { GAM_DEFAULTS, fitContext, fitGam, gamProblem, linear, predictTermsSe, s } from 'omai-engine/gam'
 * const x = Array.from({ length: 40 }, (_, i) => [i / 39, (i * 7) % 5])
 * const hyper = { ...GAM_DEFAULTS, terms: [s(0, { k: 5 }), linear(1)] }
 * const problem = gamProblem(x, x.map((r, i) => (r[0] ?? 0) + 0.1 * (r[1] ?? 0) + 0.05 * Math.sin(i)), hyper)
 * const ctx = fitContext(problem, fitGam(problem, hyper).state)
 * predictTermsSe(ctx, [[0.5, 2]], { exclude: [1] }).effects[1] // => 0
 * ```
 */
export function predictTermsSe(
  ctx: GamFitContext,
  x: TensorLike,
  options: BandOptions & { readonly exclude?: readonly number[]; readonly intercept?: boolean } = {},
): TermPredictions {
  const { design } = ctx.problem
  const m = matrixView(x, 'predictTermsSe', { nFeatures: design.nFeatures })
  const out = new Set(options.exclude ?? [])
  const p = design.p
  const k = design.terms.length
  const full = new Float64Array(m.n * p)
  if (options.intercept !== false) for (let i = 0; i < m.n; i++) full[i * p] = 1
  const effects = new Float64Array(m.n * k)
  const se = new Float64Array(m.n * k)
  const included: number[] = options.intercept === false ? [] : [0]
  design.terms.forEach((t, j) => {
    if (out.has(j)) return
    const block = t.evaluate(m)
    const cols = Array.from({ length: t.width }, (_, c) => t.offset + c)
    included.push(...cols)
    const own = new Float64Array(m.n * p)
    for (let i = 0; i < m.n; i++)
      for (let c = 0; c < t.width; c++) {
        full[i * p + t.offset + c] = block[i * t.width + c] ?? 0
        own[i * p + t.offset + c] = block[i * t.width + c] ?? 0
      }
    const band = bandFromDesign(ctx, own, m.n, cols, { ...options, scale: 'link' })
    for (let i = 0; i < m.n; i++) {
      effects[i * k + j] = band.fit[i] ?? 0
      se[i * k + j] = band.se[i] ?? 0
    }
  })
  return {
    labels: design.terms.map((t) => t.label),
    effects,
    se,
    total: bandFromDesign(ctx, full, m.n, included, options),
  }
}

/** Where a smooth is defined: the range of its covariate in the training data. */
function covariateRange(ctx: GamFitContext, feature: number): { lo: number; hi: number } {
  const data = ctx.problem.covariates
  const d = ctx.problem.design.nFeatures
  if (data === undefined)
    throw new OmaiError('value', 'gam: the problem carries no covariates (build it with gamProblem)')
  let lo = Number.POSITIVE_INFINITY
  let hi = Number.NEGATIVE_INFINITY
  for (let i = 0; i < data.length / d; i++) {
    const v = data[i * d + feature] ?? 0
    lo = Math.min(lo, v)
    hi = Math.max(hi, v)
  }
  return { lo, hi }
}

/** A term's block at the points `at` of its covariate, with a `by` covariate (if any) set to one. */
function termBlockAt(ctx: GamFitContext, term: number, at: ArrayLike<number>): Float64Array {
  const { design } = ctx.problem
  const t = design.terms[term]
  if (t === undefined) throw new OmaiError('value', `gam: no term ${term}`)
  if (t.term.kind !== 'smooth' || t.term.features.length !== 1) {
    throw new OmaiError('value', `gam: ${t.label}: derivatives and differences are for smooths of one covariate`)
  }
  const d = design.nFeatures
  const data = new Float64Array(at.length * d)
  for (let i = 0; i < at.length; i++) {
    data[i * d + (t.term.features[0] ?? 0)] = at[i] ?? 0
    if (t.term.by !== undefined) data[i * d + t.term.by] = 1
  }
  return t.evaluate({ n: at.length, d, data })
}

/** Options of {@link smoothDerivative}. */
export interface DerivativeOptions extends BandOptions {
  /** The order, 1 (default) or 2. */
  readonly order?: 1 | 2
  /**
   * `exact` (the default where the basis has a closed-form derivative: `ps`, `bs`, `cr`, `cc`, `tp`, `ts`, `duchon`,
   * `gp`), `richardson` (central differences at `h` and `h / 2` combined to fourth order; the default for other bases) or
   * `central` differences.
   */
  readonly method?: 'exact' | 'central' | 'richardson'
  /** The step as a fraction of the covariate's range (default 0.005). */
  readonly step?: number
}

/**
 * The derivative of a smooth term of one covariate with a credible band. The derivative of the effect is the derivative of
 * the basis times the coefficients, so the band uses the posterior covariance of the term's coefficients. The basis is
 * differentiated exactly where the basis has a closed-form derivative (`SmoothBasis.derivative`), and otherwise numerically:
 * central differences with step `step * range`, or their Richardson extrapolation, whose error is of fourth order in the step. Near the ends of the
 * range the points are shifted inward so the stencil stays inside the training data.
 *
 * @param ctx - The fitted pieces.
 * @param term - The index of a smooth term of one covariate.
 * @param at - The values of the covariate.
 * @param options - Order, method, step and the band options.
 * @returns The band of the derivative (the effect's slope, or curvature for order 2).
 * @throws OmaiError `value` for a term that is not a smooth of one covariate.
 * @concept generalized-additive-models
 * @example
 * ```ts
 * import { GAM_DEFAULTS, fitContext, fitGam, gamProblem, s, smoothDerivative } from 'omai-engine/gam'
 * const x = Array.from({ length: 80 }, (_, i) => [i / 79])
 * const hyper = { ...GAM_DEFAULTS, terms: [s(0, { k: 8 })] }
 * const problem = gamProblem(x, x.map((r) => 3 * (r[0] ?? 0) ** 2), hyper)
 * const slope = smoothDerivative(fitContext(problem, fitGam(problem, hyper).state), 0, [0.5])
 * Math.abs((slope.fit[0] ?? 0) - 3) < 0.05 // => true
 * ```
 */
export function smoothDerivative(
  ctx: GamFitContext,
  term: number,
  at: ArrayLike<number>,
  options: DerivativeOptions = {},
): Band {
  const t = ctx.problem.design.terms[term]
  if (t === undefined) throw new OmaiError('value', `smoothDerivative: no term ${term}`)
  const spec = t.term
  if (spec.kind !== 'smooth' || spec.features.length !== 1) {
    throw new OmaiError('value', `gam: ${t.label}: derivatives and differences are for smooths of one covariate`)
  }
  const feature = spec.features[0] ?? 0
  const { lo, hi } = covariateRange(ctx, feature)
  const order = options.order ?? 1
  const h0 = (options.step ?? 0.005) * (hi - lo)
  const m = at.length
  const p = ctx.problem.design.p
  const exact =
    t.derivative !== undefined && (options.method === 'exact' || options.method === undefined)
      ? (() => {
          const d = ctx.problem.design.nFeatures
          const data = new Float64Array(m * d)
          for (let i = 0; i < m; i++) {
            data[i * d + feature] = at[i] ?? 0
            if (spec.by !== undefined) data[i * d + spec.by] = 1
          }
          return t.derivative?.({ n: m, d, data }, order)
        })()
      : undefined
  if (options.method === 'exact' && exact === undefined) {
    throw new OmaiError('value', `smoothDerivative: the basis of ${t.label} has no closed-form derivative`)
  }
  const stencil = (h: number): Float64Array => {
    // Points x - h, x, x + h per query (shifted inward at the ends), as one block of 3m rows.
    const pts = new Float64Array(3 * m)
    const centre = new Float64Array(m)
    for (let i = 0; i < m; i++) {
      const c = order === 2 ? Math.min(Math.max(at[i] ?? 0, lo + h), hi - h) : (at[i] ?? 0)
      centre[i] = c
      pts[3 * i] = Math.max(c - h, lo)
      pts[3 * i + 1] = c
      pts[3 * i + 2] = Math.min(c + h, hi)
    }
    const block = termBlockAt(ctx, term, pts)
    const w = t.width
    const out = new Float64Array(m * w)
    for (let i = 0; i < m; i++) {
      const a = pts[3 * i] ?? 0
      const c = pts[3 * i + 1] ?? 0
      const b = pts[3 * i + 2] ?? 0
      for (let j = 0; j < w; j++) {
        const fa = block[3 * i * w + j] ?? 0
        const fc = block[(3 * i + 1) * w + j] ?? 0
        const fb = block[(3 * i + 2) * w + j] ?? 0
        out[i * w + j] = order === 1 ? (fb - fa) / (b - a) : (fa - 2 * fc + fb) / ((b - a) / 2) ** 2
        void c
      }
    }
    return out
  }
  let derivative: Float64Array
  if (exact !== undefined) derivative = exact
  else {
    const coarse = stencil(h0)
    derivative = coarse
    if (options.method !== 'central') {
      const fine = stencil(h0 / 2)
      // Error ~ h^2 for both orders, so (4 fine - coarse) / 3 removes it.
      derivative = Float64Array.from(fine, (v, i) => (4 * v - (coarse[i] ?? 0)) / 3)
    }
  }
  const X = new Float64Array(m * p)
  for (let i = 0; i < m; i++)
    for (let j = 0; j < t.width; j++) X[i * p + t.offset + j] = derivative[i * t.width + j] ?? 0
  return bandFromDesign(
    ctx,
    X,
    m,
    Array.from({ length: t.width }, (_, j) => t.offset + j),
    { ...options, scale: 'link' },
  )
}

/**
 * The difference between two smooth terms of the same covariate, `f_a(x) - f_b(x)`, with a pointwise or simultaneous
 * credible band: the comparison of the levels of a by-variable (each level has its own `s(x, { by: indicator })`). A
 * `by` covariate is set to one. The band uses the joint posterior covariance of both terms' coefficients, so it is
 * narrower than the two bands taken apart where the smooths are positively correlated, and the simultaneous version
 * says where on the curve the two differ with the stated overall confidence.
 *
 * @param ctx - The fitted pieces.
 * @param a - The first smooth term.
 * @param b - The second.
 * @param at - The values of the covariate.
 * @param options - The band options.
 * @returns The band of the difference.
 * @concept generalized-additive-models
 * @example
 * ```ts
 * import { GAM_DEFAULTS, fitContext, fitGam, gamProblem, s, smoothDifference } from 'omai-engine/gam'
 * const x = Array.from({ length: 80 }, (_, i) => [i / 79, i % 2, 1 - (i % 2)])
 * const hyper = { ...GAM_DEFAULTS, terms: [s(0, { k: 5, by: 1, basis: 'cr', label: 'a' }), s(0, { k: 5, by: 2, basis: 'cr', label: 'b' })] }
 * const problem = gamProblem(x, x.map((r) => (r[1] ?? 0) * (r[0] ?? 0) + 0.01 * (r[0] ?? 0)), hyper)
 * const diff = smoothDifference(fitContext(problem, fitGam(problem, hyper).state), 0, 1, [0.9])
 * (diff.fit[0] ?? 0) > 0.5 // => true
 * ```
 */
export function smoothDifference(
  ctx: GamFitContext,
  a: number,
  b: number,
  at: ArrayLike<number>,
  options: BandOptions = {},
): Band {
  const { design } = ctx.problem
  const ta = design.terms[a]
  const tb = design.terms[b]
  if (ta === undefined || tb === undefined) throw new OmaiError('value', 'smoothDifference: no such term')
  const m = at.length
  const p = design.p
  const X = new Float64Array(m * p)
  const ba = termBlockAt(ctx, a, at)
  const bb = termBlockAt(ctx, b, at)
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < ta.width; j++) X[i * p + ta.offset + j] = ba[i * ta.width + j] ?? 0
    for (let j = 0; j < tb.width; j++) X[i * p + tb.offset + j] = -(bb[i * tb.width + j] ?? 0)
  }
  const columns = [
    ...Array.from({ length: ta.width }, (_, j) => ta.offset + j),
    ...Array.from({ length: tb.width }, (_, j) => tb.offset + j),
  ]
  return bandFromDesign(ctx, X, m, columns, { ...options, scale: 'link' })
}
