/**
 * What the hierarchical examples (eight schools, radon) read off a posterior, whatever inference method made it: the rows of
 * a summary by coordinate, the estimates of the groups against their raw estimates (the shrinkage), the pair plot of
 * a few coordinates with the divergent transitions of a sampler marked, the diagnostics of its chains, and the check of one
 * posterior's rows against the rows of a reference run, in units of the reference's spread and of the Monte Carlo error.
 * They are plain functions of a `Posterior`, so each example's derived paths are a few lines over them.
 */

import { type ChainDiagnostics, chainDiagnostics, type PosteriorPairs, posteriorPairs } from '../exposition/index.ts'
import { type Posterior, samplePosterior } from '../probprog/posterior.ts'
import { posteriorChains } from '../probprog/posterior-chains.ts'
import { key } from '../random/index.ts'
import { tensor, toFlat } from '../tensor/index.ts'
import { drawRows } from './example-method.ts'

/** One scalar coordinate of a posterior: its mean, spread, interval, and the effective sample size and R-hat of chains. */
export type PosteriorRow = {
  readonly name: string
  readonly mean: number
  readonly sd: number
  readonly lo: number
  readonly hi: number
  readonly ess: number | null
  readonly rhat: number | null
}

/** The rows of some sites of a posterior and the numbers that say how the run went. */
export type PosteriorRows = {
  readonly rows: readonly PosteriorRow[]
  /** The draws the rows are made of (the chains' draws laid end to end; a fitted normal's are drawn). */
  readonly draws: number
  /** Divergent transitions, for a sampler that reports them. */
  readonly divergences: number | null
  /** The largest R-hat and the smallest effective sample size over the rows, for chains. */
  readonly maxRhat: number | null
  readonly minEss: number | null
  /** What kind of posterior it is (`samples`, `variational`, `point`, ...). */
  readonly form: string
}

const finite = (v: number | undefined): number | null => (v !== undefined && Number.isFinite(v) ? v : null)

/**
 * The summary rows of some sites of a posterior (mean, sd, a 90% interval, and R-hat and ESS for chains) with the number of
 * divergences, the worst R-hat and the smallest ESS. `undefined` when the posterior has none of the sites.
 *
 * @param post - The posterior.
 * @param sites - The sites to take (a vector site gives one row per entry).
 * @param prob - The probability of the interval (default 0.9).
 * @returns The rows and the run's numbers.
 * @example
 * ```ts
 * import { posteriorRows } from 'omai-engine/bayesian-examples'
 * posteriorRows({ sites: [], form: 'point', diagnostics: {}, summary: () => [] } as never, ['mu']) // => undefined
 * ```
 */
export function posteriorRows(post: Posterior, sites: readonly string[], prob = 0.9): PosteriorRows | undefined {
  let rows: PosteriorRow[]
  try {
    rows = post
      .summary({ prob })
      .filter((r) => sites.includes(r.site))
      .map((r) => ({
        name: r.name,
        mean: r.mean,
        sd: r.sd,
        lo: r.lower,
        hi: r.upper,
        ess: finite(r.ess),
        rhat: finite(r.rhat),
      }))
  } catch {
    return undefined
  }
  // No rows, a non-finite one, or every spread zero (a sampler stopped before its first draw reports where it started): nothing to read yet.
  if (rows.length === 0 || !rows.every((r) => Number.isFinite(r.mean) && Number.isFinite(r.sd))) return undefined
  if (rows.every((r) => r.sd < 1e-12)) return undefined
  const rhat = rows.flatMap((r) => (r.rhat === null ? [] : [r.rhat]))
  const ess = rows.flatMap((r) => (r.ess === null ? [] : [r.ess]))
  const divergences = post.diagnostics['divergences']
  return {
    rows,
    draws:
      post.chains === undefined
        ? post.draws === undefined
          ? 0
          : drawCount(post)
        : post.chains.numChains * post.chains.numDraws,
    divergences: typeof divergences === 'number' ? divergences : null,
    maxRhat: rhat.length > 0 ? Math.max(...rhat) : null,
    minEss: ess.length > 0 ? Math.min(...ess) : null,
    form: post.form,
  }
}

function drawCount(post: Posterior): number {
  const first = Object.values(post.draws ?? {})[0]
  return first?.shape[0] ?? 0
}

/** A coordinate of the pair plot: a site (or an entry of a vector site, `theta[0]`), optionally on the log scale. */
export interface PairCoordinate {
  /** The name drawn (`log tau`). */
  readonly label: string
  /** The site or coordinate it is read from (`tau`, `theta[0]`). */
  readonly from: string
  readonly log?: boolean
}

/**
 * The pair plot of a few coordinates of a posterior, with the sampler's divergent transitions marked where it has them (the
 * funnel of a hierarchical model is where they sit). `undefined` when fewer than two of the coordinates exist or there are too
 * few draws.
 *
 * @param post - The posterior.
 * @param coordinates - The coordinates, in the order drawn.
 * @param method - The name the draws are listed under.
 * @param truth - The reference value of each coordinate it is known for (by label).
 * @returns The record.
 * @example
 * ```ts
 * import { pairsRecord } from 'omai-engine/bayesian-examples'
 * pairsRecord({ sites: [] } as never, [], 'nuts') // => undefined
 * ```
 */
export function pairsRecord(
  post: Posterior,
  coordinates: readonly PairCoordinate[],
  method: string,
  truth: Readonly<Record<string, number>> = {},
): PosteriorPairs | undefined {
  const sites = [...new Set(coordinates.map((c) => c.from.replace(/\[\d+\]$/, '')))].filter((s) =>
    post.sites.includes(s),
  )
  const draws = drawRows(post, sites, 500)
  if (draws === undefined || draws.rows.length < 3) return undefined
  const present = coordinates.filter((c) => draws.names.includes(c.from))
  if (present.length < 2) return undefined
  const columns = Object.fromEntries(
    present.map((c) => {
      const j = draws.names.indexOf(c.from)
      return [c.label, draws.rows.map((r) => (c.log === true ? Math.log(r[j] ?? Number.NaN) : (r[j] ?? Number.NaN)))]
    }),
  )
  const flags = posteriorChains(post).divergent
  const divergent = draws.chain.map((c, i) => flags?.[c]?.[draws.index[i] as number] ?? 0)
  return posteriorPairs(
    [
      {
        name: method,
        draws: columns,
        steps: draws.index,
        chain: draws.chain,
        ...(flags === undefined ? {} : { divergent }),
      },
    ],
    {
      coordinates: present.map((c) => c.label),
      maxPoints: 300,
      frames: 2,
      densityDraws: 300,
      truth,
    },
  )
}

/**
 * The trace, rank, R-hat and ESS record of a sampler's chains for some coordinates, with its divergences. `undefined` for a
 * posterior that is not made of chains.
 *
 * @param post - The posterior.
 * @param names - The coordinates (`mu`, `tau`, `theta[0]`).
 * @returns The record.
 * @example
 * ```ts
 * import { chainsRecord } from 'omai-engine/bayesian-examples'
 * chainsRecord({ chains: undefined } as never, ['mu']) // => undefined
 * ```
 */
export function chainsRecord(post: Posterior, names: readonly string[]): ChainDiagnostics | undefined {
  if (post.chains === undefined) return undefined
  const chains = posteriorChains(post)
  const columns = Object.fromEntries(Object.entries(chains.draws).filter(([name]) => names.includes(name)))
  if (Object.keys(columns).length === 0) return undefined
  return chainDiagnostics(columns, {
    summary: post.summary(),
    ...(chains.divergent === undefined ? {} : { divergent: chains.divergent }),
  })
}

/** The estimates of the groups of a hierarchical model against their raw (unpooled) estimates, by a precision measure. */
export type GroupEstimates = {
  readonly kind: 'groupEstimates'
  /** What each group's `x` is (`standard error`, `houses in the county`). */
  readonly xLabel: string
  /** What the effect is (`effect`, `county intercept`). */
  readonly yLabel: string
  readonly labels: readonly string[]
  readonly x: readonly number[]
  /** The raw estimate of each group (what no pooling gives). */
  readonly raw: readonly number[]
  /** The posterior mean of each group's effect and its interval. */
  readonly mean: readonly number[]
  readonly lo: readonly number[]
  readonly hi: readonly number[]
  /** The population mean the groups are pulled towards (null when the model has none). */
  readonly population: number | null
  /** The true effect of each group, for generated data (null for real ones). */
  readonly truth: readonly number[] | null
  /** The mean over the groups of the share of the distance from the raw estimate to the population mean that the posterior mean moved (0 none, 1 all of it). */
  readonly meanShrinkage: number
  /** The same for the third of the groups with the least data and the third with the most. */
  readonly shrinkageOfSmallest: number
  readonly shrinkageOfLargest: number
}

/**
 * The records of a posterior's group effects against the raw estimates: for each group the distance the posterior mean moved
 * from the raw estimate towards the population mean, as a share of the distance between them (the shrinkage), averaged over
 * the third of the groups with the least data and the third with the most.
 *
 * @param input - The groups' labels, positions, raw estimates, posterior rows and the population mean.
 * @returns The record.
 * @example
 * ```ts
 * import { groupEstimates } from 'omai-engine/bayesian-examples'
 * const row = (mean: number) => ({ name: 'a', mean, sd: 1, lo: mean - 1, hi: mean + 1, ess: null, rhat: null })
 * const g = groupEstimates({ xLabel: 'n', yLabel: 'effect', labels: ['a', 'b'], x: [1, 9], raw: [4, 4], rows: [row(2), row(3.5)], population: 0, dataIsLargeX: true })
 * [g.shrinkageOfSmallest, g.shrinkageOfLargest] // => [0.5, 0.125]
 * ```
 */
export function groupEstimates(input: {
  readonly xLabel: string
  readonly yLabel: string
  readonly labels: readonly string[]
  readonly x: readonly number[]
  readonly raw: readonly number[]
  readonly rows: readonly PosteriorRow[]
  readonly population: number | null
  /** Whether a larger `x` means more data (house count) rather than less (standard error). */
  readonly dataIsLargeX: boolean
  /** The true effect of each group, where the data were generated. */
  readonly truth?: readonly number[] | null
}): GroupEstimates {
  const { x, raw, rows, population } = input
  const share = raw.map((r, j) => {
    const away = population === null ? 0 : r - population
    return Math.abs(away) < 1e-9 || population === null ? Number.NaN : 1 - ((rows[j]?.mean ?? r) - population) / away
  })
  const order = [...x.keys()].sort((a, b) => (x[a] ?? 0) - (x[b] ?? 0))
  const ordered = input.dataIsLargeX ? order : [...order].reverse()
  const third = Math.max(1, Math.floor(ordered.length / 3))
  const avg = (ids: readonly number[]): number => {
    const v = ids.map((j) => share[j] as number).filter(Number.isFinite)
    return v.length === 0 ? Number.NaN : v.reduce((a, b) => a + b, 0) / v.length
  }
  const all = share.filter(Number.isFinite)
  return {
    kind: 'groupEstimates',
    xLabel: input.xLabel,
    yLabel: input.yLabel,
    labels: input.labels,
    x,
    raw,
    mean: rows.map((r) => r.mean),
    lo: rows.map((r) => r.lo),
    hi: rows.map((r) => r.hi),
    population,
    truth: input.truth ?? null,
    meanShrinkage: all.length === 0 ? Number.NaN : all.reduce((a, b) => a + b, 0) / all.length,
    shrinkageOfSmallest: avg(ordered.slice(0, third)),
    shrinkageOfLargest: avg(ordered.slice(-third)),
  }
}

/** One posterior's rows against a reference's. */
export type RowsCheck = {
  /** How many coordinates were compared, and the draws and ESS of the reference. */
  readonly compared: number
  readonly referenceDraws: number
  readonly referenceMinEss: number | null
  /** The largest difference of means in the reference's sds, and where. */
  readonly maxMeanError: number
  readonly worst: string
  /** The median of the ratio of spreads (this over the reference). */
  readonly medianSdRatio: number
  /** The largest difference of means in units of its Monte Carlo error (both have an ESS), and where; null when not both have chains. */
  readonly maxZ: number | null
  readonly worstZ: string | null
  /** The same for the coordinates named in `focus` (the hyperparameters). */
  readonly focus: readonly { readonly name: string; readonly meanError: number; readonly sdRatio: number }[]
}

/**
 * A posterior's rows against a reference's: for each coordinate both have, the difference of the means in the reference's
 * sds, the ratio of the sds and, when both are chains with an effective sample size, the difference in units of the Monte
 * Carlo error of the two means (`sqrt(sd² / ess + sd_ref² / ess_ref)`), so a sampler that agrees reads as a `z` of a few at most.
 *
 * @param mine - The rows of the run checked.
 * @param reference - The rows of the reference run.
 * @param focus - The coordinates to list on their own (the hyperparameters).
 * @returns The comparison, or `undefined` when no coordinate is in both.
 * @example
 * ```ts
 * import { checkRows } from 'omai-engine/bayesian-examples'
 * const row = (name: string, mean: number) => ({ name, mean, sd: 1, lo: 0, hi: 0, ess: 100, rhat: 1 })
 * const c = checkRows({ rows: [row('mu', 1.1)], draws: 100, divergences: 0, maxRhat: 1, minEss: 100, form: 'samples' }, { rows: [row('mu', 1)], draws: 400, divergences: 0, maxRhat: 1, minEss: 100, form: 'samples' }, ['mu'])
 * [c?.compared, c?.maxMeanError.toFixed(1)] // => [1, '0.1']
 * ```
 */
export function checkRows(
  mine: PosteriorRows,
  reference: PosteriorRows,
  focus: readonly string[] = [],
): RowsCheck | undefined {
  const byName = new Map(reference.rows.map((r) => [r.name, r]))
  const pairs = mine.rows.flatMap((r) => {
    const ref = byName.get(r.name)
    return ref === undefined ? [] : [{ r, ref }]
  })
  if (pairs.length === 0) return undefined
  const error = pairs.map(({ r, ref }) => Math.abs(r.mean - ref.mean) / Math.max(1e-9, ref.sd))
  const ratio = pairs.map(({ r, ref }) => r.sd / Math.max(1e-9, ref.sd))
  const z = pairs.map(({ r, ref }) =>
    r.ess === null || ref.ess === null
      ? Number.NaN
      : Math.abs(r.mean - ref.mean) / Math.sqrt(Math.max(1e-300, r.sd ** 2 / r.ess + ref.sd ** 2 / ref.ess)),
  )
  const argmax = (v: readonly number[]): number =>
    v.reduce(
      (best, x, i) =>
        Number.isFinite(x) && (!Number.isFinite(v[best] ?? Number.NaN) || x > (v[best] as number)) ? i : best,
      0,
    )
  const bestError = argmax(error)
  const bestZ = argmax(z)
  const hasZ = Number.isFinite(z[bestZ] ?? Number.NaN)
  return {
    compared: pairs.length,
    referenceDraws: reference.draws,
    referenceMinEss: reference.minEss,
    maxMeanError: error[bestError] ?? Number.NaN,
    worst: pairs[bestError]?.r.name ?? '',
    medianSdRatio: [...ratio].sort((a, b) => a - b)[Math.floor(ratio.length / 2)] ?? Number.NaN,
    maxZ: hasZ ? (z[bestZ] as number) : null,
    worstZ: hasZ ? (pairs[bestZ]?.r.name ?? null) : null,
    focus: focus.flatMap((name) => {
      const i = pairs.findIndex((p) => p.r.name === name)
      return i < 0 ? [] : [{ name, meanError: error[i] as number, sdRatio: ratio[i] as number }]
    }),
  }
}

/**
 * A check as one flat record of numbers and strings (what a probe hands to a page): the comparison's numbers, the worst
 * coordinates by name, and the listed coordinates as `<name>Error` and `<name>Ratio`.
 *
 * @param check - The comparison.
 * @returns The flat record.
 * @example
 * ```ts
 * import { flattenCheck } from 'omai-engine/bayesian-examples'
 * flattenCheck({ compared: 2, referenceDraws: 9, referenceMinEss: null, maxMeanError: 0.1, worst: 'mu', medianSdRatio: 1, maxZ: null, worstZ: null, focus: [{ name: 'tau', meanError: 0.3, sdRatio: 0.9 }] }).tauError // => 0.3
 * ```
 */
export function flattenCheck(check: RowsCheck): Record<string, number | string | null> {
  const { focus, ...rest } = check
  return {
    ...rest,
    ...Object.fromEntries(
      focus.flatMap((f) => [
        [`${f.name}Error`, f.meanError],
        [`${f.name}Ratio`, f.sdRatio],
      ]),
    ),
  }
}

/**
 * A posterior that has the derived sites a model defines from its latents (`theta = mu + tau z`). A sampler's posterior has
 * them already and is returned as it is; a sampler's that lacks them gets them computed draw by draw on its own draws (its
 * chains kept); a variational or Laplace fit holds only the latents, so `n` draws are taken from it and the posterior of
 * those draws is returned (Monte Carlo moments, no chains).
 *
 * @param post - The posterior.
 * @param names - The derived sites `derive` makes.
 * @param derive - From one draw's latents (by site, flat) to the derived sites (by site, flat).
 * @param n - The draws to take from a posterior that is not made of draws (default 1000).
 * @returns The posterior with the sites.
 * @example
 * ```ts
 * import { withDerivedSites } from 'omai-engine/bayesian-examples'
 * import { pointPosterior } from 'omai-engine/probprog'
 * import { tensor } from 'omai-engine/tensor'
 * const p = withDerivedSites(pointPosterior({ a: tensor(2) }, { engine: 'test' }), ['b'], (d) => ({ b: [(d['a']?.[0] ?? 0) * 3] }), 4)
 * p.sites // => ['a', 'b']
 * ```
 */
export function withDerivedSites(
  post: Posterior,
  names: readonly string[],
  derive: (draw: Readonly<Record<string, number[]>>) => Record<string, number[]>,
  n = 1000,
): Posterior {
  if (names.every((name) => post.sites.includes(name))) return post
  // A sampler's draws are used as they are (every one, in their chains); anything else is sampled.
  const own = post.form === 'samples' && post.draws !== undefined ? post.draws : undefined
  const drawn = own ?? post.sample(key(1), n)
  n = Math.max(1, (Object.values(drawn)[0]?.shape[0] ?? n) as number)
  const flat = Object.fromEntries(Object.entries(drawn).map(([site, t]) => [site, toFlat(t)]))
  const rows = Array.from({ length: n }, (_, i) =>
    derive(
      Object.fromEntries(
        Object.entries(flat).map(([site, v]) => {
          const width = Math.floor(v.length / n)
          return [site, Array.from({ length: width }, (_, j) => v[i * width + j] ?? Number.NaN)]
        }),
      ),
    ),
  )
  const extra = Object.fromEntries(
    names.map((name) => [
      name,
      tensor(rows.map((r) => (r[name]?.length === 1 ? (r[name]?.[0] ?? Number.NaN) : (r[name] ?? [])))),
    ]),
  )
  return samplePosterior(
    { ...drawn, ...extra },
    {
      engine: post.engine,
      diagnostics: post.diagnostics,
      status: post.status,
      ...(own !== undefined && post.chains !== undefined
        ? { chains: post.chains, chainShape: [post.chains.numChains, post.chains.numDraws] as const }
        : {}),
    },
  )
}
