/**
 * What the regression examples (linear regression with its priors, robust regression with its likelihoods) read off a
 * posterior of coefficients: the draws of the coefficient site `w` and of the scalar sites beside it (the noise, the degrees
 * of freedom) under the names the page shows, the band of a polynomial curve and of a new observation, the exact normal
 * posterior of a Gaussian prior and likelihood, the pair plot, the chain diagnostics, and the check of a method against the
 * exact posterior or against the draws of a reference run. The examples differ in the model they build and in what they
 * add to these paths; this is the part they share.
 */

import { normal as normalDist } from '../dists/index.ts'
import { type ChainDiagnostics, posteriorComparison, posteriorPairs } from '../exposition/index.ts'
import { bayesianLinearFit, designOf } from '../linear/index.ts'
import type { Posterior } from '../probprog/posterior.ts'
import {
  agree,
  distributionOracle,
  fromPosterior,
  type MarginalOracle,
  weightedOracle,
} from '../probprog/validation/index.ts'
import { drawRows, exampleDiagnostics } from './example-method.ts'

/** The powers `1, x, ..., x^degree`. */
export const monomials = (x: number, degree: number): number[] => Array.from({ length: degree + 1 }, (_, j) => x ** j)

/** The polynomial with coefficients `w` (of `1, x, x^2, ...`) at `x`. */
export const polynomialAt = (w: readonly number[], x: number): number => w.reduce((a, c, j) => a + c * x ** j, 0)

/**
 * The quantile of an ascending list at probability `p`, by linear interpolation.
 *
 * @param sorted - The values, ascending.
 * @param p - The probability.
 * @returns The quantile.
 * @example
 * ```ts
 * import { sortedQuantile } from 'omai-engine/bayesian-examples'
 * sortedQuantile([1, 2, 3, 4, 5], 0.5) // => 3
 * ```
 */
export function sortedQuantile(sorted: readonly number[], p: number): number {
  const i = (sorted.length - 1) * p
  const lo = Math.floor(i)
  const hi = Math.ceil(i)
  return (sorted[lo] ?? 0) + ((sorted[hi] ?? 0) - (sorted[lo] ?? 0)) * (i - lo)
}

/** The exact normal posterior of the coefficients of a Gaussian prior and a Gaussian likelihood with a known noise. */
export interface ExactNormal {
  readonly mean: number[]
  readonly sd: number[]
  readonly cov: number[][]
}

/**
 * The exact posterior of the coefficients of `y = X w + e` with a normal prior on `w` (standard deviation `priorScale`) and
 * normal noise of known standard deviation: a normal with precision `I / priorScale^2 + X'X / noiseSd^2`.
 *
 * @param design - The design rows `[n][p]`.
 * @param y - The responses.
 * @param priorScale - The prior standard deviation of each coefficient.
 * @param noiseSd - The noise standard deviation.
 * @returns The mean, the standard deviations and the covariance.
 * @example
 * ```ts
 * import { exactNormalPosterior } from 'omai-engine/bayesian-examples'
 * const e = exactNormalPosterior([[1], [1]], [2, 2], 1000, 1)
 * Math.round(e.mean[0] * 100) / 100 // => 2
 * ```
 */
export function exactNormalPosterior(
  design: readonly (readonly number[])[],
  y: readonly number[],
  priorScale: number,
  noiseSd: number,
): ExactNormal {
  const d = designOf({ x: design as number[][], y: y as number[] }, 'linreg')
  const fit = bayesianLinearFit(d, { alpha: 1 / priorScale ** 2, beta: 1 / noiseSd ** 2 })
  return { mean: fit.mean, sd: fit.covariance.map((r, i) => Math.sqrt(r[i] ?? 0)), cov: fit.covariance }
}

/** How a regression's posterior is read: the coefficients and the scalar sites beside them, with their true values. */
export interface RegressionShape {
  /** The name of each coefficient of the site `w`. */
  readonly names: readonly string[]
  /** The true value of each coefficient. */
  readonly truth: readonly number[]
  /** The scalar sites after `w` (the noise, the degrees of freedom): shown when the posterior has them. */
  readonly scalars: readonly { readonly site: string; readonly name: string; readonly truth: number | null }[]
}

/** The draws of a regression's posterior, named as the page shows them. */
export interface RegressionDraws {
  /** One row per draw: the coefficients, then the scalar sites that are there. */
  readonly rows: number[][]
  /** The name of each column. */
  readonly names: string[]
  readonly chain: number[]
  readonly index: number[]
  /** The number of coefficients. */
  readonly p: number
  /** The true value of each column (null where there is none). */
  readonly truth: (number | null)[]
}

/**
 * At most `n` draws of the coefficients and of the scalar sites of a regression posterior.
 *
 * @param post - The posterior.
 * @param shape - How it is read.
 * @param n - The most draws.
 * @returns The draws, or `undefined` when the posterior has no coefficients or a draw is not finite.
 * @example
 * ```ts
 * import { regressionDraws } from 'omai-engine/bayesian-examples'
 * regressionDraws({ sites: [] } as never, { names: ['a'], truth: [1], scalars: [] }, 5) // => undefined
 * ```
 */
export function regressionDraws(post: Posterior, shape: RegressionShape, n: number): RegressionDraws | undefined {
  const scalars = shape.scalars.filter((s) => post.sites.includes(s.site))
  const got = drawRows(post, ['w', ...scalars.map((s) => s.site)], n)
  if (got === undefined) return undefined
  return {
    rows: got.rows,
    names: [...shape.names, ...scalars.map((s) => s.name)],
    chain: got.chain,
    index: got.index,
    p: shape.names.length,
    truth: [...shape.truth, ...scalars.map((s) => s.truth)],
  }
}

/**
 * The noise standard deviation of a draw: the `sigma` column, or `1 / sqrt(tau)` of the `tau` column (the noise written as
 * a gamma precision), or `known` when the draws have neither.
 *
 * @param names - The names of the columns of the draws.
 * @param row - A draw.
 * @param known - The noise standard deviation when it is not inferred.
 * @returns The standard deviation.
 * @example
 * ```ts
 * import { noiseSdOf } from 'omai-engine/bayesian-examples'
 * noiseSdOf(['a', 'tau'], [1, 4], 9) // => 0.5
 * ```
 */
export function noiseSdOf(names: readonly string[], row: readonly number[], known: number): number {
  const s = names.indexOf('sigma')
  if (s >= 0) return row[s] ?? known
  const t = names.indexOf('tau')
  return t >= 0 ? 1 / Math.sqrt(row[t] ?? 1 / known ** 2) : known
}

/** The curve bands of a polynomial regression at a grid of inputs. */
export interface CurveBands {
  readonly mean: number[]
  readonly lo: number[]
  readonly hi: number[]
  readonly predictiveLo: number[]
  readonly predictiveHi: number[]
  readonly lines: number[][]
}

/**
 * The posterior mean curve of a polynomial, its 95% band, the 95% band of a new observation (the curve plus noise drawn
 * by `noise`) and a few sampled curves, from draws of the coefficients.
 *
 * @param rows - The draws (coefficients first, then whatever `noise` needs).
 * @param p - The number of coefficients.
 * @param grid - The inputs.
 * @param noise - The noise of a new observation for draw `s` (given the draw and a standard normal).
 * @param lines - How many sampled curves to keep.
 * @returns The bands.
 * @example
 * ```ts
 * import { curveBands } from 'omai-engine/bayesian-examples'
 * curveBands([[1, 2], [1, 2]], 2, [0, 1], () => 0, 1).mean // => [1, 3]
 * ```
 */
export function curveBands(
  rows: readonly (readonly number[])[],
  p: number,
  grid: readonly number[],
  noise: (row: readonly number[], s: number, z: number) => number,
  lines: number,
  z: ArrayLike<number> = [],
): CurveBands {
  const at = grid.map((v) => rows.map((r) => r.slice(0, p).reduce((a, c, j) => a + c * v ** j, 0)))
  const curve = at.map((col) => [...col].sort((a, b) => a - b))
  const predictive = at.map((col) => col.map((f, s) => f + noise(rows[s] ?? [], s, z[s] ?? 0)).sort((a, b) => a - b))
  const stride = Math.max(1, Math.floor(rows.length / lines))
  return {
    mean: at.map((col) => col.reduce((a, v) => a + v, 0) / col.length),
    lo: curve.map((c) => sortedQuantile(c, 0.025)),
    hi: curve.map((c) => sortedQuantile(c, 0.975)),
    predictiveLo: predictive.map((c) => sortedQuantile(c, 0.025)),
    predictiveHi: predictive.map((c) => sortedQuantile(c, 0.975)),
    lines: Array.from({ length: Math.min(lines, rows.length) }, (_, s) =>
      at.map((col) => col[s * stride] ?? Number.NaN),
    ),
  }
}

/**
 * The pair plot of a regression's coefficients and scalar sites (a `posteriorPairs` record, the true values marked).
 *
 * @param label - The name of the method on the figure.
 * @param got - The draws.
 * @param coordinates - Which columns to plot (default the first five).
 * @returns The record.
 * @example
 * ```ts
 * import { regressionPairs } from 'omai-engine/bayesian-examples'
 * const got = { rows: [[0, 1], [1, 2], [2, 0]], names: ['a', 'b'], chain: [0, 0, 0], index: [0, 1, 2], p: 2, truth: [1, null] }
 * (regressionPairs('x', got) as { kind: string }).kind // => 'posteriorPairs'
 * ```
 */
export function regressionPairs(label: string, got: RegressionDraws, coordinates?: readonly string[]): unknown {
  const names = coordinates ?? got.names.slice(0, 5)
  const columns = Object.fromEntries(names.map((c) => [c, got.rows.map((r) => r[got.names.indexOf(c)] ?? 0)] as const))
  const truth = Object.fromEntries(
    names.flatMap((c) => {
      const t = got.truth[got.names.indexOf(c)]
      return t === null || t === undefined ? [] : [[c, t] as const]
    }),
  )
  return posteriorPairs([{ name: label, draws: columns, steps: got.index, chain: got.chain }], {
    coordinates: [...names],
    maxPoints: 300,
    frames: 2,
    densityDraws: 300,
    truth,
  })
}

/**
 * The chain diagnostics of a regression posterior (trace, rank, R-hat and ESS of the coefficients and scalar sites).
 *
 * @param post - The posterior.
 * @param shape - How it is read.
 * @param most - The most coordinates to show.
 * @returns The record, or `null` for a posterior that is not made of chains.
 * @example
 * ```ts
 * import { regressionDiagnostics } from 'omai-engine/bayesian-examples'
 * regressionDiagnostics({ chains: undefined } as never, { names: ['a'], truth: [1], scalars: [] }) // => null
 * ```
 */
export function regressionDiagnostics(post: Posterior, shape: RegressionShape, most = 5): ChainDiagnostics | null {
  const rename = (name: string): string => {
    const j = /^w\[(\d+)\]$/.exec(name)?.[1]
    if (j !== undefined) return shape.names[Number(j)] ?? name
    return shape.scalars.find((s) => s.site === name)?.name ?? name
  }
  const scalars = shape.scalars.filter((s) => post.sites.includes(s.site)).map((s) => s.site)
  return exampleDiagnostics(post, ['w', ...scalars], rename, most)
}

/** What a check against a reference needs of the reference: its draws, one column per coefficient and scalar site. */
export interface RegressionReference {
  readonly names: readonly string[]
  readonly columns: readonly (readonly number[])[]
  /** The largest R-hat of the reference's coordinates (a long run that did not mix says so), when it has chains. */
  readonly maxRhat?: number | null
}

/**
 * The reference a run offers to checks of other runs: every draw it has of the coefficients and scalar sites, and the
 * largest R-hat of its coordinates.
 *
 * @param post - The posterior of the reference run.
 * @param shape - How it is read.
 * @returns The reference, or `undefined` without draws.
 * @example
 * ```ts
 * import { regressionReference } from 'omai-engine/bayesian-examples'
 * regressionReference({ sites: [] } as never, { names: ['a'], truth: [1], scalars: [] }) // => undefined
 * ```
 */
export function regressionReference(post: Posterior, shape: RegressionShape): RegressionReference | undefined {
  const got = regressionDraws(post, shape, 400)
  if (got === undefined) return undefined
  const rhats = post
    .summary()
    .filter((r) => r.site === 'w' || shape.scalars.some((s) => s.site === r.site))
    .map((r) => r.rhat)
    .filter((v): v is number => typeof v === 'number' && Number.isFinite(v))
  return {
    names: got.names,
    columns: got.names.map((_, j) => got.rows.map((r) => r[j] ?? 0)),
    maxRhat: rhats.length === 0 ? null : Math.max(...rhats),
  }
}

/**
 * The check of a method's posterior against the exact normal posterior of the coefficients or against the draws of a
 * reference run: a comparison record and the distances (the largest mean error in reference sds, the median spread ratio,
 * the largest Wasserstein distance and KL).
 *
 * @param label - The method's name.
 * @param post - The method's posterior.
 * @param shape - How the posterior is read.
 * @param exact - The exact posterior, when there is one.
 * @param ref - The reference draws, when there are some.
 * @returns The check, or `null` when there is nothing to compare with.
 * @example
 * ```ts
 * import { regressionCheck } from 'omai-engine/bayesian-examples'
 * regressionCheck('x', { sites: [] } as never, { names: ['a'], truth: [1], scalars: [] }, undefined, undefined) // => null
 * ```
 */
export function regressionCheck(
  label: string,
  post: Posterior,
  shape: RegressionShape,
  exact: ExactNormal | undefined,
  ref: RegressionReference | null | undefined,
): unknown {
  if (!post.sites.includes('w')) return null
  if (exact === undefined && (ref === undefined || ref === null)) return null
  const scalars = shape.scalars.filter((s) => post.sites.includes(s.site))
  const approx = fromPosterior(label, post, { sites: ['w', ...scalars.map((s) => s.site)] })
  const rename = (q: string): string => {
    const j = /^w\[(\d+)\]$/.exec(q)?.[1]
    if (j !== undefined) return shape.names[Number(j)] ?? q
    return scalars.find((s) => s.site === q)?.name ?? q
  }
  const draws = Object.fromEntries(Object.entries(approx.draws).map(([q, v]) => [rename(q), v]))
  const named = {
    ...approx,
    draws,
    ...(approx.ess === undefined
      ? {}
      : { ess: Object.fromEntries(Object.entries(approx.ess).map(([q, v]) => [rename(q), v])) }),
  }
  const oracle: Record<string, MarginalOracle> = {}
  if (exact !== undefined)
    shape.names.forEach((name, j) => {
      oracle[name] = distributionOracle(name, normalDist(exact.mean[j] ?? 0, exact.sd[j] ?? 1))
    })
  else if (ref !== undefined && ref !== null)
    ref.names.forEach((name, j) => {
      const column = ref.columns[j]
      if (column !== undefined && name in draws) oracle[name] = weightedOracle(name, column)
    })
  const quantities = Object.keys(oracle)
  if (quantities.length === 0) return null
  const table = agree([named], { oracle, quantities })
  const rows = table.rows
  return {
    comparison: posteriorComparison([named], {
      truth: oracle,
      table,
      quantities,
      gridPoints: 100,
      oracle: exact === undefined ? 'a long NUTS run' : 'the exact normal posterior',
    }),
    rows,
    maxMeanError: Math.max(...rows.map((r) => Math.abs(r.meanError))),
    medianSdRatio: [...rows.map((r) => r.sdRatio)].sort((a, b) => a - b)[Math.floor(rows.length / 2)] ?? Number.NaN,
    maxWasserstein: Math.max(...rows.map((r) => r.wasserstein)),
    maxKl: Math.max(...rows.map((r) => r.kl)),
    against: exact === undefined ? 'reference' : 'exact',
    referenceRhat: exact === undefined ? (ref?.maxRhat ?? null) : null,
    logEvidence: null,
  }
}
