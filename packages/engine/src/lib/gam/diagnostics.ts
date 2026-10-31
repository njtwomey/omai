/**
 * Diagnostics of a fitted generalised additive model, after mgcv's `gam.check`, `concurvity` and `cooks.distance`:
 * deviance, Pearson, working and response residuals; QQ data with a reference band from residuals of datasets simulated
 * from the fitted model; residuals against the linear predictor; the k-index, which asks of each smooth whether the
 * residuals still carry a pattern in its covariate that a larger basis dimension `k` would capture (Wood 2017, section
 * 5.9); three measures of concurvity (how well each term is predicted by the others); and Cook's distance for the
 * penalised fit.
 */

import { MdmlError } from '../base/index.ts'
import { dgemm, dgeqrf, dgesvd } from '../numerics/linalg/index.ts'
import { foldIn, type Key, permutation } from '../random/index.ts'
import { fromStorage, toFlat } from '../tensor/index.ts'
import type { GamFitContext } from './inference.ts'
import { type GamState, influenceDiagonal } from './pirls.ts'

/** The kinds of residual. */
export type ResidualKind = 'deviance' | 'pearson' | 'working' | 'response'

const vec = (a: ArrayLike<number>) => fromStorage(Float64Array.from(a), [a.length], 'float64')

/**
 * The residuals of a fitted model: signed deviance residuals (default), Pearson residuals `sqrt(w) (y - mu) / sqrt(V)`,
 * working residuals `(y - mu) g'(mu)` or the raw response residuals.
 *
 * @param ctx - The fitted pieces.
 * @param kind - The kind (default `deviance`).
 * @returns One residual per observation.
 * @concept generalized-additive-models
 * @example
 * ```ts
 * import { GAM_DEFAULTS, fitContext, fitGam, gamProblem, gamResiduals, s } from 'mdml-engine/gam'
 * const x = Array.from({ length: 30 }, (_, i) => [i / 29])
 * const hyper = { ...GAM_DEFAULTS, terms: [s(0, { k: 5 })] }
 * const problem = gamProblem(x, x.map((r, i) => (r[0] ?? 0) + 0.1 * Math.sin(i * 12.9898)), hyper)
 * gamResiduals(fitContext(problem, fitGam(problem, hyper).state)).length // => 30
 * ```
 */
export function gamResiduals(ctx: GamFitContext, kind: ResidualKind = 'deviance'): Float64Array {
  const { problem } = ctx
  const mu = vec(ctx.mu)
  const y = vec(problem.y)
  if (kind === 'deviance') {
    return Float64Array.from(toFlat(problem.family.devianceResiduals(y, mu, vec(problem.prior))))
  }
  const v = Float64Array.from(toFlat(problem.family.variance(mu)))
  const g = Float64Array.from(toFlat(problem.link.derivative(mu)))
  return Float64Array.from(ctx.mu, (m, i) => {
    const r = (problem.y[i] ?? 0) - m
    if (kind === 'response') return r
    if (kind === 'working') return r * (g[i] ?? 1)
    return (Math.sqrt(problem.prior[i] ?? 1) * r) / Math.sqrt(v[i] ?? 1)
  })
}

/** QQ-plot data with a simulation-based reference band. */
export interface QqData {
  /** The sorted residuals of the data. */
  readonly sample: Float64Array
  /** The mean of the sorted residuals of the simulated datasets: the reference line. */
  readonly theoretical: Float64Array
  readonly lower: Float64Array
  readonly upper: Float64Array
}

/**
 * QQ data for the deviance residuals with a reference band by simulation (mgcv's `qq.gam`): responses are drawn from
 * the fitted model, their deviance residuals against the fitted means sorted, and the band is the pointwise quantiles of
 * the sorted residuals across datasets. Residuals that leave the band are not distributed as the family says they are.
 *
 * @param ctx - The fitted pieces.
 * @param key - The random key.
 * @param options - `reps` simulated datasets (default 50) and the band `level` (default 0.95).
 * @returns The sorted residuals, the reference line and the band.
 * @concept generalized-additive-models
 * @example
 * ```ts
 * import { GAM_DEFAULTS, fitContext, fitGam, gamProblem, qqData, s } from 'mdml-engine/gam'
 * import { key } from 'mdml-engine/random'
 * const x = Array.from({ length: 30 }, (_, i) => [i / 29])
 * const hyper = { ...GAM_DEFAULTS, terms: [s(0, { k: 5 })] }
 * const problem = gamProblem(x, x.map((r, i) => (r[0] ?? 0) + 0.1 * Math.sin(i * 12.9898)), hyper)
 * qqData(fitContext(problem, fitGam(problem, hyper).state), key(0), { reps: 20 }).sample.length // => 30
 * ```
 */
export function qqData(
  ctx: GamFitContext,
  key: Key,
  options: { readonly reps?: number; readonly level?: number } = {},
): QqData {
  const reps = options.reps ?? 50
  const level = options.level ?? 0.95
  const { problem } = ctx
  const n = ctx.mu.length
  const dist = problem.family.distribution(vec(ctx.mu), ctx.scale)
  const sims: Float64Array[] = []
  for (let r = 0; r < reps; r++) {
    const y = toFlat(dist.sample(foldIn(key, r), [])) as number[]
    const dev = Float64Array.from(toFlat(problem.family.devianceResiduals(vec(y), vec(ctx.mu), vec(problem.prior))))
    sims.push(dev.sort())
  }
  const sample = gamResiduals(ctx).sort()
  const at = (i: number, q: number): number => {
    const column = sims.map((s) => s[i] ?? 0).sort((a, b) => a - b)
    return column[Math.min(reps - 1, Math.max(0, Math.round(q * (reps - 1))))] ?? 0
  }
  return {
    sample,
    theoretical: Float64Array.from({ length: n }, (_, i) => sims.reduce((a, s) => a + (s[i] ?? 0), 0) / reps),
    lower: Float64Array.from({ length: n }, (_, i) => at(i, (1 - level) / 2)),
    upper: Float64Array.from({ length: n }, (_, i) => at(i, 1 - (1 - level) / 2)),
  }
}

/**
 * The deviance residuals against the linear predictor, the plot that shows non-constant variance or a wrong link.
 *
 * @param ctx - The fitted pieces.
 * @returns The linear predictor and the residuals.
 * @concept generalized-additive-models
 * @example
 * ```ts
 * import { GAM_DEFAULTS, fitContext, fitGam, gamProblem, residualsVsFitted, s } from 'mdml-engine/gam'
 * const x = Array.from({ length: 30 }, (_, i) => [i / 29])
 * const hyper = { ...GAM_DEFAULTS, terms: [s(0, { k: 5 })] }
 * const problem = gamProblem(x, x.map((r, i) => (r[0] ?? 0) + 0.1 * Math.sin(i * 12.9898)), hyper)
 * residualsVsFitted(fitContext(problem, fitGam(problem, hyper).state)).residuals.length // => 30
 * ```
 */
export function residualsVsFitted(ctx: GamFitContext): {
  readonly fitted: Float64Array
  readonly residuals: Float64Array
} {
  return { fitted: Float64Array.from(ctx.eta), residuals: gamResiduals(ctx) }
}

/** A row of the basis-dimension check. */
export interface KIndex {
  readonly label: string
  /** The basis dimension minus the constraint (the largest effective degrees of freedom the term could have). */
  readonly kPrime: number
  readonly edf: number
  /** Observed over expected roughness of the residuals ordered by the covariate; well below 1 suggests a larger `k`. */
  readonly kIndex: number
  /** The share of residual shuffles with a smaller roughness than the data (small: a pattern remains). */
  readonly pValue: number
}

/**
 * The basis-dimension check of each smooth (mgcv's `k.check`): the residuals are compared with their nearest
 * neighbour in the term's covariates; the mean squared difference over twice the residual variance is near 1 for residuals
 * with no remaining structure and well below 1 where neighbouring residuals are alike, i.e. where a more flexible term
 * would fit more. The reference is the same statistic on random shuffles of the residuals, which also gives the p-value.
 *
 * @param ctx - The fitted pieces.
 * @param key - The random key for the shuffles.
 * @param options - `shuffles` (default 200) and the residual `kind` (default `deviance`).
 * @returns One row per smooth term.
 * @concept generalized-additive-models
 * @example
 * ```ts
 * import { GAM_DEFAULTS, fitContext, fitGam, gamProblem, kIndex, s } from 'mdml-engine/gam'
 * import { key } from 'mdml-engine/random'
 * const x = Array.from({ length: 60 }, (_, i) => [i / 59])
 * const hyper = { ...GAM_DEFAULTS, terms: [s(0, { k: 8 })] }
 * const problem = gamProblem(x, x.map((r, i) => Math.sin(5 * (r[0] ?? 0)) + 0.2 * Math.sin(i * 12.9898)), hyper)
 * kIndex(fitContext(problem, fitGam(problem, hyper).state), key(0))[0]?.label // => 's(x0)'
 * ```
 */
export function kIndex(
  ctx: GamFitContext,
  key: Key,
  options: { readonly shuffles?: number; readonly kind?: ResidualKind } = {},
): KIndex[] {
  const { design } = ctx.problem
  const shuffles = options.shuffles ?? 200
  const e = gamResiduals(ctx, options.kind ?? 'deviance')
  const n = e.length
  const rows: KIndex[] = []
  design.terms.forEach((t, j) => {
    if (t.penalties.length === 0 || t.term.kind === 'linear' || t.term.kind === 'factor') return
    const features = t.term.features
    const point = (i: number): number[] => features.map((f) => covariateOf(ctx, i, f))
    const nn = new Int32Array(n)
    for (let i = 0; i < n; i++) {
      let best = Number.POSITIVE_INFINITY
      const a = point(i)
      for (let k = 0; k < n; k++) {
        if (k === i) continue
        const b = point(k)
        let d = 0
        for (let c = 0; c < a.length; c++) d += ((a[c] ?? 0) - (b[c] ?? 0)) ** 2
        if (d < best) {
          best = d
          nn[i] = k
        }
      }
    }
    const roughness = (r: ArrayLike<number>): number => {
      let s = 0
      for (let i = 0; i < n; i++) s += ((r[nn[i] ?? 0] ?? 0) - (r[i] ?? 0)) ** 2
      return s / n / 2
    }
    const observed = roughness(e)
    const reps: number[] = []
    for (let s = 0; s < shuffles; s++) {
      const order = permutation(foldIn(key, j * 100003 + s), n).data as ArrayLike<number>
      reps.push(roughness(Float64Array.from({ length: n }, (_, i) => e[order[i] ?? 0] ?? 0)))
    }
    let edf = 0
    for (let c = 0; c < t.width; c++) edf += ctx.edf[t.offset + c] ?? 0
    rows.push({
      label: t.label,
      kPrime: t.width,
      edf,
      kIndex: observed / (reps.reduce((a, b) => a + b, 0) / shuffles),
      pValue: reps.filter((r) => r < observed).length / shuffles,
    })
  })
  return rows
}

/** The covariate value of an observation: the design stores only the expanded columns, so it is read from the fit data. */
function covariateOf(ctx: GamFitContext, row: number, feature: number): number {
  const data = ctx.problem.covariates
  if (data === undefined)
    throw new MdmlError('value', 'kIndex: the problem carries no covariates (build it with gamProblem)')
  return data[row * ctx.problem.design.nFeatures + feature] ?? 0
}

/** Concurvity of a term. */
export interface Concurvity {
  readonly label: string
  /** The largest possible share of the term's columns lying in the span of the other terms (worst case). */
  readonly worst: number
  /** The share of the fitted partial effect lying in the span of the other terms. */
  readonly observed: number
  /** The mean share over the term's columns. */
  readonly estimate: number
}

/**
 * Concurvity of each term (mgcv's `concurvity`): the generalisation of collinearity to smooths, how well a term can be
 * approximated by the others. `worst` is the squared cosine of the smallest angle between the term's column space and
 * the others' (the maximum over coefficient vectors of the share of the term in the others' span), `observed` the share
 * for the fitted partial effect, `estimate` the share averaged over the term's columns. Values near 1 mean the terms
 * cannot be told apart and their individual effects are unreliable.
 *
 * @param ctx - The fitted pieces.
 * @returns One row per term with columns besides the intercept.
 * @concept generalized-additive-models
 * @example
 * ```ts
 * import { GAM_DEFAULTS, concurvity, fitContext, fitGam, gamProblem, s } from 'mdml-engine/gam'
 * const x = Array.from({ length: 60 }, (_, i) => [i / 59, i / 59 + 0.001 * Math.sin(i)])
 * const hyper = { ...GAM_DEFAULTS, terms: [s(0, { k: 5 }), s(1, { k: 5 })] }
 * const problem = gamProblem(x, x.map((r) => (r[0] ?? 0) ** 2), hyper)
 * (concurvity(fitContext(problem, fitGam(problem, hyper).state))[0]?.worst ?? 0) > 0.9 // => true
 * ```
 */
export function concurvity(ctx: GamFitContext): Concurvity[] {
  const { design } = ctx.problem
  const { n, p, X } = design
  return design.terms.map((t) => {
    const others: number[] = []
    for (let c = 1; c < p; c++) if (c < t.offset || c >= t.offset + t.width) others.push(c)
    const O = new Float64Array(n * others.length)
    for (let i = 0; i < n; i++) others.forEach((c, k) => (O[i * others.length + k] = X[i * p + c] ?? 0))
    const B = new Float64Array(n * t.width)
    for (let i = 0; i < n; i++) for (let k = 0; k < t.width; k++) B[i * t.width + k] = X[i * p + t.offset + k] ?? 0
    // Project on the span of the others (and the constant): Q from the thin QR of [1, O].
    const q = others.length + 1
    const OC = new Float64Array(n * q)
    for (let i = 0; i < n; i++) {
      OC[i * q] = 1
      for (let k = 0; k < others.length; k++) OC[i * q + k + 1] = O[i * others.length + k] ?? 0
    }
    const Q = dgeqrf(OC, n, q).q
    const kq = Math.min(n, q)
    const project = (M: Float64Array, w: number): Float64Array => {
      const coef = dgemm(Q, M, kq, n, w, { transposeA: true })
      return dgemm(Q, coef, n, kq, w)
    }
    const centre = (M: Float64Array, w: number): Float64Array => {
      const out = Float64Array.from(M)
      for (let k = 0; k < w; k++) {
        let m = 0
        for (let i = 0; i < n; i++) m += M[i * w + k] ?? 0
        m /= n
        for (let i = 0; i < n; i++) out[i * w + k] = (out[i * w + k] ?? 0) - m
      }
      return out
    }
    const Bc = centre(B, t.width)
    const PB = project(Bc, t.width)
    const norm2 = (M: Float64Array): number => M.reduce((a, v) => a + v * v, 0)
    // The worst case: the top singular value of Q' Qb, with Qb an orthonormal basis of the (centred) term.
    const qb = dgeqrf(Bc, n, t.width).q
    const w2 = Math.min(n, t.width)
    const cross = dgemm(Q, qb, kq, n, w2, { transposeA: true })
    const worst = dgesvd(cross, kq, w2).s[0] ?? 0
    const effect = new Float64Array(n)
    for (let i = 0; i < n; i++)
      for (let k = 0; k < t.width; k++)
        effect[i] = (effect[i] ?? 0) + (Bc[i * t.width + k] ?? 0) * (ctx.beta[t.offset + k] ?? 0)
    const pe = project(effect, 1)
    return {
      label: t.label,
      worst: Math.min(1, worst * worst),
      observed: norm2(effect) === 0 ? 0 : norm2(pe) / norm2(effect),
      estimate: norm2(Bc) === 0 ? 0 : norm2(PB) / norm2(Bc),
    }
  })
}

/**
 * Cook's distance of each observation for the penalised fit, `r_i^2 h_i / (scale p_eff (1 - h_i)^2)` with `r` the Pearson
 * residual, `h` the leverage (the diagonal of the influence matrix) and `p_eff` the effective degrees of freedom: the
 * influence of leaving the observation out on the fitted values, to first order.
 *
 * @param ctx - The fitted pieces.
 * @param state - The state the context was made from.
 * @returns One distance per observation.
 * @concept generalized-additive-models
 * @example
 * ```ts
 * import { GAM_DEFAULTS, cooksDistance, fitContext, fitGam, gamProblem, s } from 'mdml-engine/gam'
 * const x = Array.from({ length: 30 }, (_, i) => [i / 29])
 * const y = x.map((r, i) => (r[0] ?? 0) + 0.05 * Math.sin(i * 12.9898))
 * y[15] = 3
 * const hyper = { ...GAM_DEFAULTS, terms: [s(0, { k: 6 })] }
 * const problem = gamProblem(x, y, hyper)
 * const { state } = fitGam(problem, hyper)
 * const d = cooksDistance(fitContext(problem, state), state)
 * d.indexOf(Math.max(...d)) // => 15
 * ```
 */
export function cooksDistance(ctx: GamFitContext, state: GamState): Float64Array {
  const h = influenceDiagonal(ctx.problem, state)
  const r = gamResiduals(ctx, 'pearson')
  return Float64Array.from(h, (hi, i) => ((r[i] ?? 0) ** 2 * hi) / (ctx.scale * ctx.edfTotal * (1 - hi) ** 2))
}
