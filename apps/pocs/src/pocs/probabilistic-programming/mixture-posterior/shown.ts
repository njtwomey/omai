/**
 * What the page shows at a moment of the playhead, read from the finished results with no computing beyond slicing: which
 * draws of a job are visible at a time, the posterior predictive density of those draws (its mean, its band and the
 * components of the latest draw), the corner-plot record of the chosen coordinates and the chain diagnostics. The results are
 * plain data from the workers.
 */

import {
  type ChainDiagnostics,
  chainDiagnostics,
  type PairDraws,
  type PosteriorPairs,
  posteriorPairs,
} from 'omai-engine/exposition'
import { ess as essOf, rhat as rhatOf } from 'omai-engine/sampling'
import { tensor } from 'omai-engine/tensor'
import type { MixtureResult } from './compute.ts'

/** The rows of the pool visible at time `t` (milliseconds): a sampler's draws made by then, a fit's draws of its current snapshot. */
export function visibleRows(result: MixtureResult, t: number): number[] {
  const { at, until } = result.pool
  let time = t
  if (result.organisation === 'iterate') {
    time = 0
    for (const f of result.frames) if (f <= t) time = f
  }
  const rows: number[] = []
  for (let i = 0; i < at.length; i++)
    if ((at[i] ?? 0) <= time && (time < (until[i] ?? 0) || until[i] === Number.POSITIVE_INFINITY)) rows.push(i)
  return rows
}

/** The coordinates a view is about: the indices into a row, from the choice on the page. */
export type Coordinates = 'means' | 'weights' | 'both'

/** The indices of the chosen coordinates of a row of `components` means, `components` weights and the spread. */
export function coordinatesOf(choice: Coordinates, components: number): number[] {
  const means = Array.from({ length: components }, (_, k) => k)
  const weights = means.map((k) => components + k)
  return choice === 'means' ? means : choice === 'weights' ? weights : [...means, ...weights]
}

/** The posterior predictive density over `xs` at the playhead: mean and 90% band over the visible draws, and the components of the latest draw. */
export interface Predictive {
  readonly draws: number
  readonly mean: number[]
  readonly lower: number[]
  readonly upper: number[]
  /** `components[k][g]`: the weighted density of component `k` of the most recent draw. */
  readonly components: number[][]
}

const normalPdf = (x: number, mu: number, sd: number): number =>
  Math.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * Math.sqrt(2 * Math.PI))

/**
 * The predictive density of a job's visible draws: for each draw the mixture `sum_k w_k N(x; mu_k, sigma)` on the grid, then
 * the mean and the 5% and 95% points over the draws.
 *
 * @param result - A job's result.
 * @param xs - The grid, in minutes.
 * @param t - The playhead, in milliseconds.
 * @returns The curves, or `undefined` while there is no draw.
 */
export function predictiveAt(result: MixtureResult, xs: readonly number[], t: number): Predictive | undefined {
  const rows = visibleRows(result, t)
  if (rows.length === 0) return undefined
  const { values, p, at } = result.pool
  const K = result.components
  const density = (i: number, k: number, x: number): number =>
    (values[i * p + K + k] ?? 0) * normalPdf(x, values[i * p + k] ?? 0, values[i * p + 2 * K] ?? 1)
  const perDraw = rows.map((i) =>
    xs.map((x) => Array.from({ length: K }, (_, k) => density(i, k, x)).reduce((a, b) => a + b, 0)),
  )
  const at5 = Math.floor(0.05 * (rows.length - 1))
  const at95 = Math.ceil(0.95 * (rows.length - 1))
  const column = xs.map((_, g) => perDraw.map((d) => d[g] as number).sort((a, b) => a - b))
  const latest = rows.reduce((best, i) => ((at[i] ?? 0) >= (at[best] ?? 0) ? i : best), rows[0] as number)
  return {
    draws: rows.length,
    mean: column.map((c) => c.reduce((a, b) => a + b, 0) / c.length),
    lower: column.map((c) => c[at5] as number),
    upper: column.map((c) => c[at95] as number),
    components: Array.from({ length: K }, (_, k) => xs.map((x) => density(latest, k, x))),
  }
}

/** The corner-plot record of the chosen coordinates for the jobs overlaid, the draws stamped with the time they were made. */
export function pairsOf(results: readonly MixtureResult[], coordinates: Coordinates): PosteriorPairs | undefined {
  const runs = results.flatMap((r): PairDraws[] => {
    const rows = visibleRows(r, Number.POSITIVE_INFINITY)
    const picked = coordinatesOf(coordinates, r.components)
    if (rows.length < 2) return []
    const last = r.frames.at(-1) ?? 0
    return [
      {
        name: r.label,
        draws: Object.fromEntries(
          picked.map((j) => [r.names[j] as string, rows.map((i) => r.pool.values[i * r.pool.p + j] ?? Number.NaN)]),
        ),
        steps: rows.map((i) => (r.organisation === 'iterate' ? last : (r.pool.at[i] ?? 0))),
        chain: rows.map((i) => r.pool.chain[i] ?? 0),
      },
    ]
  })
  const first = results[0]
  if (runs.length === 0 || first === undefined) return undefined
  return posteriorPairs(runs, {
    coordinates: coordinatesOf(coordinates, first.components).map((j) => first.names[j] as string),
    frames: 40,
    maxPoints: 300,
  })
}

/**
 * The chain diagnostics of a sampler at time `t` for the chosen coordinates: each chain's draws made by then, with the R-hat
 * and effective sample size of what there is so far (at most 200 draws per chain are used, thinned).
 *
 * @param result - A sampler's result.
 * @param coordinates - Which coordinates.
 * @param t - The playhead, in milliseconds.
 * @returns The record, or `undefined` before there are a few draws per chain.
 */
export function chainsAt(result: MixtureResult, coordinates: Coordinates, t: number): ChainDiagnostics | undefined {
  if (result.organisation !== 'chain') return undefined
  const rows = visibleRows(result, t)
  const byChain = new Map<number, number[]>()
  for (const i of rows) byChain.set(result.pool.chain[i] ?? 0, [...(byChain.get(result.pool.chain[i] ?? 0) ?? []), i])
  const chains = [...byChain.values()]
  const n = Math.min(...chains.map((c) => c.length))
  if (chains.length === 0 || !(n >= 8)) return undefined
  const stride = Math.max(1, Math.ceil(n / 200))
  const picked = [...coordinatesOf(coordinates, result.components), 2 * result.components]
  const draws = Object.fromEntries(
    picked.map((j) => [
      result.names[j] as string,
      chains.map((c) =>
        c
          .slice(0, n)
          .filter((_, r) => r % stride === 0)
          .map((i) => result.pool.values[i * result.pool.p + j] ?? Number.NaN),
      ),
    ]),
  )
  const summary = Object.entries(draws).map(([name, x]) => ({
    name,
    rhat: chains.length >= 2 ? rhatOf(tensor(x)) : undefined,
    ess: chains.length >= 2 ? essOf(tensor(x)) * stride : undefined,
  }))
  return chainDiagnostics(draws, { summary, acfLags: 20, ecdfPoints: 0 })
}

/** The mean of coordinate `j` at each snapshot of a variational fit up to time `t`: the fit's trace. */
export function fitTrace(result: MixtureResult, j: number, t: number): { x: number[]; y: number[] } {
  const { values, p, at } = result.pool
  const x: number[] = []
  const y: number[] = []
  for (const f of result.frames) {
    if (f > t) break
    let sum = 0
    let count = 0
    for (let i = 0; i < at.length; i++)
      if (at[i] === f) {
        sum += values[i * p + j] ?? 0
        count++
      }
    x.push(f / 1000)
    y.push(count === 0 ? Number.NaN : sum / count)
  }
  return { x, y }
}
