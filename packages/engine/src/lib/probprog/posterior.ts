/**
 * The posterior interface every inference engine returns, whatever its form (exact distributions, equally weighted or
 * weighted draws, a variational approximation or a point estimate): marginals, joint draws, moments, quantiles, log
 * density and evidence where available, diagnostics and a summary table. Constructors for each form.
 */

import { type Estimate, type Json, OmaiError, quantileOf } from '../base/index.ts'
import { compose, type Distribution } from '../dists/index.ts'
import { weightedPercentile } from '../metrics/index.ts'
import { choice, foldIn, type Key, key as makeKey } from '../random/index.ts'
import { type Chains, ess, rhat } from '../sampling/index.ts'
import {
  asTensor,
  broadcastTo,
  fromStorage,
  primalOf,
  reshape,
  stack,
  type Tensor,
  type TensorLike,
  take,
  toFlat,
  type Value,
} from '../tensor/index.ts'
import type { RunStatus } from '../trace/index.ts'

/**
 * How a posterior is represented: exact distributions, equally weighted draws (Markov chains or exact samplers),
 * weighted draws (importance sampling), a variational approximation, or a point estimate.
 */
export type PosteriorForm = 'exact' | 'samples' | 'weighted' | 'variational' | 'point'

/** One row of {@link Posterior.summary}: one scalar coordinate of one site. */
export interface PosteriorSummaryRow {
  readonly site: string
  /** The coordinate, `site` for scalars or `site[i]` (flat index). */
  readonly name: string
  readonly mean: number
  readonly sd: number
  readonly median: number
  /** The equal-tailed interval at the summary's probability. */
  readonly lower: number
  readonly upper: number
  /** Bulk effective sample size and rank-normalised split R-hat, for Markov chains. */
  readonly ess?: number
  readonly rhat?: number
}

/**
 * A posterior behind one interface, whatever the engine produced (section 3.6 of the design): marginals, joint draws,
 * moments, quantiles, the log density where available, the evidence where the engine estimates it, and a summary.
 *
 * @concept posterior-distributions
 */
export interface Posterior {
  readonly form: PosteriorForm
  /** The engine that produced it. */
  readonly engine: string
  /** The latent (and deterministic) sites it covers. */
  readonly sites: readonly string[]
  readonly status: RunStatus
  /** log p(data), when the engine estimates it. */
  readonly logEvidence?: Estimate
  /** Engine diagnostics (acceptance, divergences, ESS, ELBO, Pareto k, ...). */
  readonly diagnostics: Readonly<Record<string, Json>>
  /** Draws by site `[n, ...shape]`, for sample-based forms. */
  readonly draws?: Readonly<Record<string, Tensor>>
  /** Normalised log weights `[n]` of weighted draws. */
  readonly logWeights?: Tensor
  /** The raw Markov chains on unconstrained coordinates, for MCMC. */
  readonly chains?: Chains
  /** The marginal distribution of a site. */
  marginal(site: string): Distribution
  /** Joint draws `[n, ...shape]` by site. */
  sample(key: Key, n: number): Record<string, Tensor>
  mean(site: string): Tensor
  variance(site: string): Tensor
  stddev(site: string): Tensor
  /** Elementwise quantiles `[probs.length, ...shape]`. */
  quantiles(site: string, probs: readonly number[]): Tensor
  /** log q(values) under the posterior, when it has a density. */
  logProb?(values: Readonly<Record<string, TensorLike>>): number
  /**
   * Means, standard deviations, medians and equal-tailed intervals per coordinate (with ESS and R-hat for chains). A listed
   * site the posterior holds no marginal, draws or samples of (the discrete label a Laplace fit sums out) has no row.
   */
  summary(options?: { readonly prob?: number }): PosteriorSummaryRow[]
}

/** The defining parts of a posterior; {@link definePosterior} fills in the rest. */
export interface PosteriorSpec {
  readonly form: PosteriorForm
  readonly engine: string
  readonly sites: readonly string[]
  readonly status?: RunStatus
  readonly logEvidence?: Estimate
  readonly diagnostics?: Readonly<Record<string, Json>>
  /** Closed-form marginals by site (exact and mean-field forms). */
  readonly marginals?: Readonly<Record<string, Distribution>>
  /** Draws by site `[n, ...shape]` (sample and weighted forms). */
  readonly draws?: Readonly<Record<string, Tensor>>
  /** Unnormalised log weights `[n]` of the draws. */
  readonly logWeights?: TensorLike
  /** `[numChains, numDraws]` when the draws are chains laid end to end. */
  readonly chainShape?: readonly [number, number]
  readonly chains?: Chains
  /** Joint draws by site (otherwise draws are resampled, or marginals sampled independently). */
  readonly sampler?: (key: Key, n: number) => Record<string, Tensor>
  readonly logProb?: (values: Readonly<Record<string, TensorLike>>) => number
  /** How many draws to use for Monte Carlo moments when nothing closed-form is known (default 4000). */
  readonly monteCarloDraws?: number
}

function normalise(logWeights: TensorLike): Float64Array {
  const lw = toFlat(primalOf(asTensor(logWeights)))
  let m = Number.NEGATIVE_INFINITY
  for (const v of lw) if (v > m) m = v
  const w = Float64Array.from(lw, (v) => (Number.isFinite(m) ? Math.exp(v - m) : 1))
  let s = 0
  for (const v of w) s += v
  for (let i = 0; i < w.length; i++) w[i] = (w[i] ?? 0) / s
  return w
}

/** Column statistics of draws `[n, ...shape]`, optionally weighted. */
function columns(draws: Tensor): { readonly shape: number[]; readonly cols: Float64Array[] } {
  const n = draws.shape[0] ?? 0
  const shape = draws.shape.slice(1)
  const size = shape.reduce((a, b) => a * b, 1)
  const flat = toFlat(draws)
  const cols = Array.from({ length: size }, (_, j) =>
    Float64Array.from({ length: n }, (_, i) => flat[i * size + j] ?? 0),
  )
  return { shape, cols }
}

function stat(draws: Tensor, f: (col: Float64Array) => number): Tensor {
  const { shape, cols } = columns(draws)
  return fromStorage(Float64Array.from(cols.map(f)), shape, 'float64')
}

function meanOfCol(col: Float64Array, w?: Float64Array): number {
  let s = 0
  for (let i = 0; i < col.length; i++) s += (col[i] ?? 0) * (w === undefined ? 1 / col.length : (w[i] ?? 0))
  return s
}

function varianceOfCol(col: Float64Array, w?: Float64Array): number {
  const m = meanOfCol(col, w)
  let s = 0
  for (let i = 0; i < col.length; i++) s += ((col[i] ?? 0) - m) ** 2 * (w === undefined ? 1 : (w[i] ?? 0))
  // Unweighted draws use the n - 1 denominator (as ArviZ's sd); weighted ones the self-normalised second moment.
  return w === undefined ? s / Math.max(1, col.length - 1) : s
}

function quantileOfCol(col: Float64Array, q: number, w?: Float64Array): number {
  return w === undefined ? quantileOf(col, q) : weightedPercentile(col, w, q * 100)
}

/**
 * Build a posterior from its parts: moments and quantiles come from closed-form marginals when given, else from the
 * (weighted) draws, else from Monte Carlo draws of the sampler; marginals of draw-based forms are empirical
 * distributions; joint draws resample the draws or call the sampler.
 *
 * @param spec - The parts.
 * @returns The posterior.
 * @throws OmaiError with code `value` for an unknown site, or when nothing can produce draws.
 * @concept posterior-distributions
 * @example
 * ```ts
 * import { definePosterior } from 'omai-engine/probprog'
 * import { tensor, toArray } from 'omai-engine/tensor'
 * const post = definePosterior({ form: 'samples', engine: 'test', sites: ['x'], draws: { x: tensor([1, 2, 3, 6]) } })
 * toArray(post.mean('x')) // => 3
 * post.summary()[0]?.median // => 2.5
 * ```
 */
export function definePosterior(spec: PosteriorSpec): Posterior {
  const weights = spec.logWeights === undefined ? undefined : normalise(spec.logWeights)
  let cache: Record<string, Tensor> | undefined
  const known = new Set(spec.sites)
  const check = (site: string): void => {
    if (!known.has(site)) throw new OmaiError('value', `posterior: no site ${site}`, { site })
  }
  const sample = (key: Key, n: number): Record<string, Tensor> => {
    if (spec.sampler !== undefined) return spec.sampler(key, n)
    const draws = spec.draws
    if (draws !== undefined) {
      const first = Object.values(draws)[0]
      const m = first?.shape[0] ?? 0
      if (m === 0) throw new OmaiError('value', 'posterior: no draws to resample')
      const idx = choice(key, m, n, weights === undefined ? {} : { weights: Array.from(weights) })
      return Object.fromEntries(Object.entries(draws).map(([k, v]) => [k, primalOf(take(v, idx, 0))]))
    }
    const marginals = spec.marginals
    if (marginals !== undefined) {
      return Object.fromEntries(
        Object.entries(marginals).map(([k, d], i) => [k, primalOf(d.sample(foldIn(key, i), [n]))]),
      )
    }
    throw new OmaiError('value', 'posterior: nothing to draw from')
  }
  const monteCarlo = (): Record<string, Tensor> => {
    if (cache === undefined) cache = sample(makeKey(0x5eed), spec.monteCarloDraws ?? 4000)
    return cache
  }
  const drawsOf = (site: string): { readonly x: Tensor; readonly w?: Float64Array } => {
    const d = spec.draws?.[site]
    if (d !== undefined) return weights === undefined ? { x: d } : { x: d, w: weights }
    const mc = monteCarlo()[site]
    if (mc === undefined) throw new OmaiError('value', `posterior: no draws of ${site}`)
    return { x: mc }
  }
  const closed = (site: string): Distribution | undefined => spec.marginals?.[site]
  const tryClosed = (site: string, f: (d: Distribution) => Value): Tensor | undefined => {
    const d = closed(site)
    if (d === undefined) return undefined
    try {
      return primalOf(f(d))
    } catch {
      return undefined
    }
  }
  const mean = (site: string): Tensor => {
    check(site)
    const c = tryClosed(site, (d) => d.mean())
    if (c !== undefined) return c
    const { x, w } = drawsOf(site)
    return stat(x, (col) => meanOfCol(col, w))
  }
  const variance = (site: string): Tensor => {
    check(site)
    const c = tryClosed(site, (d) => d.variance())
    if (c !== undefined) return c
    const { x, w } = drawsOf(site)
    return stat(x, (col) => varianceOfCol(col, w))
  }
  const quantiles = (site: string, probs: readonly number[]): Tensor => {
    check(site)
    const d = closed(site)
    if (d !== undefined) {
      try {
        return primalOf(stack(probs.map((q) => broadcastTo(d.ppf(q), [...d.batchShape, ...d.eventShape]))))
      } catch {
        // No quantile function: fall back to draws.
      }
    }
    const { x, w } = drawsOf(site)
    return primalOf(stack(probs.map((q) => stat(x, (col) => quantileOfCol(col, q, w)))))
  }
  const marginal = (site: string): Distribution => {
    check(site)
    const c = closed(site)
    if (c !== undefined) return c
    const { x, w } = drawsOf(site)
    return w === undefined ? compose.empirical(x) : compose.empirical(x, w)
  }
  // A site the posterior lists but holds nothing of (the discrete label a Laplace fit sums out) has no row: a summary of the
  // rest still reads, where asking for that site's mean fails with 'no draws'.
  const covered = (site: string): boolean => {
    if (spec.marginals?.[site] !== undefined || spec.draws?.[site] !== undefined) return true
    try {
      return monteCarlo()[site] !== undefined
    } catch {
      return false
    }
  }
  const summary = (options: { readonly prob?: number } = {}): PosteriorSummaryRow[] => {
    const prob = options.prob ?? 0.9
    const lo = (1 - prob) / 2
    const rows: PosteriorSummaryRow[] = []
    for (const site of spec.sites) {
      if (!covered(site)) continue
      const m = toFlat(mean(site))
      const v = toFlat(variance(site))
      const q = quantiles(site, [lo, 0.5, 1 - lo])
      const size = m.length
      const qs = toFlat(q)
      const chained = spec.chainShape !== undefined ? spec.draws?.[site] : undefined
      const scalar = size === 1 && mean(site).shape.length === 0
      for (let j = 0; j < size; j++) {
        let diag = {}
        if (chained !== undefined && spec.chainShape !== undefined) {
          const [c, n] = spec.chainShape
          const col = columns(chained).cols[j] ?? new Float64Array(0)
          const grid = reshape(fromStorage(col, [c * n], 'float64'), [c, n])
          diag = { ess: ess(grid), rhat: rhat(grid) }
        }
        rows.push({
          site,
          name: scalar ? site : `${site}[${j}]`,
          mean: m[j] ?? Number.NaN,
          sd: Math.sqrt(v[j] ?? Number.NaN),
          median: qs[size + j] ?? Number.NaN,
          lower: qs[j] ?? Number.NaN,
          upper: qs[2 * size + j] ?? Number.NaN,
          ...diag,
        })
      }
    }
    return rows
  }
  const out: Posterior = {
    form: spec.form,
    engine: spec.engine,
    sites: spec.sites,
    status: spec.status ?? 'completed',
    diagnostics: spec.diagnostics ?? {},
    ...(spec.logEvidence === undefined ? {} : { logEvidence: spec.logEvidence }),
    ...(spec.draws === undefined ? {} : { draws: spec.draws }),
    ...(weights === undefined
      ? {}
      : { logWeights: fromStorage(Float64Array.from(weights, Math.log), [weights.length], 'float64') }),
    ...(spec.chains === undefined ? {} : { chains: spec.chains }),
    ...(spec.logProb === undefined ? {} : { logProb: spec.logProb }),
    marginal,
    sample: (key, n) => {
      const r = sample(key, n)
      return Object.fromEntries(Object.entries(r).filter(([k]) => known.has(k)))
    },
    mean,
    variance,
    stddev: (site) => {
      const v = variance(site)
      return fromStorage(Float64Array.from(toFlat(v), Math.sqrt), v.shape, 'float64')
    },
    quantiles,
    summary,
  }
  return out
}

/**
 * A posterior of equally weighted draws (Markov chains, exact samplers).
 *
 * @param draws - Draws by site `[n, ...shape]`.
 * @param options - The engine, chain shape and chains, evidence, diagnostics and status.
 * @returns The posterior.
 * @concept posterior-distributions
 * @example
 * ```ts
 * import { samplePosterior } from 'omai-engine/probprog'
 * import { tensor, toArray } from 'omai-engine/tensor'
 * toArray(samplePosterior({ x: tensor([[0, 1], [2, 3]]) }, { engine: 'test' }).mean('x')) // => [1, 2]
 * ```
 */
export function samplePosterior(
  draws: Readonly<Record<string, Tensor>>,
  options: Omit<PosteriorSpec, 'form' | 'sites' | 'draws'> & { readonly sites?: readonly string[] },
): Posterior {
  return definePosterior({ ...options, form: 'samples', sites: options.sites ?? Object.keys(draws), draws })
}

/**
 * A posterior of weighted draws (importance sampling, likelihood weighting).
 *
 * @param draws - Draws by site `[n, ...shape]`.
 * @param logWeights - Unnormalised log weights `[n]`.
 * @param options - The engine, evidence, diagnostics and status.
 * @returns The posterior.
 * @concept posterior-distributions
 * @example
 * ```ts
 * import { weightedPosterior } from 'omai-engine/probprog'
 * import { tensor, toArray } from 'omai-engine/tensor'
 * toArray(weightedPosterior({ x: tensor([0, 10]) }, [0, Math.log(3)], { engine: 'test' }).mean('x')) // => 7.5
 * ```
 */
export function weightedPosterior(
  draws: Readonly<Record<string, Tensor>>,
  logWeights: TensorLike,
  options: Omit<PosteriorSpec, 'form' | 'sites' | 'draws' | 'logWeights'> & { readonly sites?: readonly string[] },
): Posterior {
  return definePosterior({
    ...options,
    form: 'weighted',
    sites: options.sites ?? Object.keys(draws),
    draws,
    logWeights,
  })
}

/**
 * A point-estimate posterior (MAP, EM): every marginal is a point mass.
 *
 * @param values - The estimates by site.
 * @param options - The engine, diagnostics and status.
 * @returns The posterior.
 * @concept posterior-distributions
 * @example
 * ```ts
 * import { pointPosterior } from 'omai-engine/probprog'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(pointPosterior({ mu: 1.5 }, { engine: 'map' }).mean('mu')) // => 1.5
 * ```
 */
export function pointPosterior(
  values: Readonly<Record<string, TensorLike>>,
  options: Omit<PosteriorSpec, 'form' | 'sites' | 'marginals'>,
): Posterior {
  const marginals: Record<string, Distribution> = {}
  for (const [k, v] of Object.entries(values)) marginals[k] = compose.deterministic(v)
  return definePosterior({
    ...options,
    form: 'point',
    sites: Object.keys(values),
    marginals,
    sampler: (_key, n) =>
      Object.fromEntries(
        Object.entries(marginals).map(([k, d]) => {
          const m = primalOf(d.mean())
          return [k, primalOf(broadcastTo(reshape(m, [1, ...m.shape]), [n, ...m.shape]))]
        }),
      ),
  })
}
