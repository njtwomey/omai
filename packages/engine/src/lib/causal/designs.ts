/**
 * Panel designs for effects of a treatment that switches on at different times: the two-by-two difference in
 * differences, the two-way fixed-effects regression (with cluster-robust errors), event studies (leads as the placebo
 * check of parallel trends, lags as the dynamic effect) by the fixed-effects regression or by the group-time estimator
 * of Callaway and Sant'Anna that avoids the forbidden comparisons of staggered adoption, and synthetic control (donor
 * weights on the simplex by quadratic programming, the gap and its placebo-in-space test, one donor per step as an
 * algorithm).
 */

import { minOf, OmaiError } from '../base/index.ts'
import type { LearnData } from '../learn/estimator.ts'
import { matrixView, targetVector } from '../learn/validation.ts'
import { quadprog } from '../optim/index.ts'
import { bootstrapIndices, foldIn, type Key, key as makeKey } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { type Algorithm, channels, runAlgorithm, stateFromJson, stateToJson } from '../trace/index.ts'
import { chi2Sf, covariance, invert, mean, normPpf, wls } from './_dense.ts'
import { type EffectEstimate, effectFromSe } from './data.ts'

/** A long panel: one row per unit and period. */
export interface PanelData {
  readonly unit: ArrayLike<number>
  readonly period: ArrayLike<number>
  readonly outcome: ArrayLike<number>
  /** Whether the unit is treated in that period (0 or 1). */
  readonly treated: ArrayLike<number>
}

/**
 * Read a panel from a dataset whose features hold the unit, the period and the treatment status (the layout of the
 * `panelEffects` generator: columns 0, 1, 2) and whose target is the outcome.
 *
 * @param data - Features `x` and outcome `y`.
 * @param columns - The columns of unit, period and treatment (default 0, 1, 2).
 * @returns The panel.
 * @example
 * ```ts
 * import { splitPanel } from 'omai-engine/causal'
 * splitPanel({ x: [[0, 0, 0], [0, 1, 1]], y: [1, 3] }).treated // => [0, 1]
 * ```
 */
export function splitPanel(
  data: LearnData,
  columns: { readonly unit?: number; readonly period?: number; readonly treated?: number } = {},
): PanelData {
  const m = matrixView(data.x, 'splitPanel')
  if (data.y === undefined) throw new OmaiError('value', 'splitPanel: the data need an outcome y')
  const [cu, cp, ct] = [columns.unit ?? 0, columns.period ?? 1, columns.treated ?? 2]
  const col = (c: number): number[] => Array.from({ length: m.n }, (_, i) => m.data[i * m.d + c] ?? 0)
  return { unit: col(cu), period: col(cp), treated: col(ct), outcome: targetVector(data.y, 'splitPanel', m.n) }
}

interface Indexed {
  readonly n: number
  readonly units: number[]
  readonly periods: number[]
  /** Row-wise indices into `units` and `periods`. */
  readonly u: Int32Array
  readonly t: Int32Array
  readonly y: Float64Array
  readonly d: Float64Array
  /** First treated period index per unit (-1 for never treated). */
  readonly adoption: Int32Array
}

function index(panel: PanelData): Indexed {
  const n = panel.unit.length
  const units = [...new Set(Array.from(panel.unit))].sort((a, b) => a - b)
  const periods = [...new Set(Array.from(panel.period))].sort((a, b) => a - b)
  const um = new Map(units.map((v, i) => [v, i]))
  const pm = new Map(periods.map((v, i) => [v, i]))
  const u = Int32Array.from({ length: n }, (_, i) => um.get(panel.unit[i] ?? 0) ?? 0)
  const t = Int32Array.from({ length: n }, (_, i) => pm.get(panel.period[i] ?? 0) ?? 0)
  const adoption = new Int32Array(units.length).fill(-1)
  for (let i = 0; i < n; i++)
    if (panel.treated[i] === 1) {
      const a = adoption[u[i] ?? 0] ?? -1
      if (a === -1 || (t[i] ?? 0) < a) adoption[u[i] ?? 0] = t[i] ?? 0
    }
  return {
    n,
    units,
    periods,
    u,
    t,
    y: Float64Array.from(panel.outcome),
    d: Float64Array.from(panel.treated),
    adoption,
  }
}

/** Remove unit and period means from the columns of a row-major `n x k` matrix, iterating to convergence (unbalanced panels). */
function demean(ix: Indexed, a: Float64Array, k: number): Float64Array {
  const out = Float64Array.from(a)
  const nu = ix.units.length
  const np = ix.periods.length
  for (let sweep = 0; sweep < 500; sweep++) {
    let change = 0
    for (const [groups, size, key] of [
      [ix.u, nu, 'u'],
      [ix.t, np, 't'],
    ] as const) {
      const sums = new Float64Array(size * k)
      const counts = new Float64Array(size)
      for (let i = 0; i < ix.n; i++) {
        const g = groups[i] ?? 0
        counts[g] = (counts[g] ?? 0) + 1
        for (let j = 0; j < k; j++) sums[g * k + j] = (sums[g * k + j] ?? 0) + (out[i * k + j] ?? 0)
      }
      for (let i = 0; i < ix.n; i++) {
        const g = groups[i] ?? 0
        for (let j = 0; j < k; j++) {
          const m = (sums[g * k + j] ?? 0) / (counts[g] ?? 1)
          out[i * k + j] = (out[i * k + j] ?? 0) - m
          if (key === 't') change = Math.max(change, Math.abs(m))
        }
      }
    }
    if (change < 1e-13) break
  }
  return out
}

/** The result of a two-by-two difference in differences. */
export interface DidResult extends EffectEstimate {
  /** Mean outcome of the never-treated and the ever-treated, before and after the cutoff. */
  readonly means: {
    readonly controlPre: number
    readonly controlPost: number
    readonly treatedPre: number
    readonly treatedPost: number
  }
  /** The first period counted as post. */
  readonly cutoff: number
}

/**
 * The two-by-two difference in differences: the change of the ever-treated units' mean outcome from before to after the
 * cutoff minus the same change of the never-treated units, estimated as the interaction coefficient of a regression on
 * a group indicator, a post indicator and their product, with cluster-robust standard errors by unit.
 *
 * @param panel - The long panel.
 * @param options - The first post period `cutoff` (default the earliest adoption) and `level`.
 * @returns The effect on the treated, with the four cell means.
 * @index function Difference-in-differences with event studies and synthetic control.
 * @example
 * ```ts
 * import { differenceInDifferences } from 'omai-engine/causal'
 * const panel = { unit: [0, 0, 1, 1, 2, 2], period: [0, 1, 0, 1, 0, 1], treated: [0, 1, 0, 1, 0, 0], outcome: [1, 4, 2, 5, 1, 2] }
 * differenceInDifferences(panel).estimate.value // => 2
 * ```
 */
export function differenceInDifferences(
  panel: PanelData,
  options: { readonly cutoff?: number; readonly level?: number } = {},
): DidResult {
  const ix = index(panel)
  const first = minOf(Array.from(ix.adoption).filter((a) => a >= 0))
  if (!Number.isFinite(first)) throw new OmaiError('value', 'differenceInDifferences: no unit is ever treated')
  const cutoff = options.cutoff === undefined ? first : ix.periods.findIndex((p) => p >= (options.cutoff ?? 0))
  const rows = Array.from({ length: ix.n }, (_, i) => i).filter((i) => {
    const a = ix.adoption[ix.u[i] ?? 0] ?? -1
    return a === -1 || a >= cutoff
  })
  const design = new Float64Array(rows.length * 4)
  const y = new Float64Array(rows.length)
  const cluster = new Float64Array(rows.length)
  rows.forEach((r, i) => {
    const g = (ix.adoption[ix.u[r] ?? 0] ?? -1) >= 0 ? 1 : 0
    const post = (ix.t[r] ?? 0) >= cutoff ? 1 : 0
    design.set([1, g, post, g * post], i * 4)
    y[i] = ix.y[r] ?? 0
    cluster[i] = ix.u[r] ?? 0
  })
  const fit = wls(design, rows.length, 4, y)
  const cov = covariance(fit, design, 'cluster', undefined, cluster)
  const cell = (g: number, post: number): number => {
    let s = 0
    let c = 0
    rows.forEach((__, i) => {
      if (design[i * 4 + 1] === g && design[i * 4 + 2] === post) {
        s += y[i] ?? 0
        c++
      }
    })
    return s / c
  }
  const groups = new Set(cluster).size
  const effect = effectFromSe('did', fit.beta[3] ?? Number.NaN, Math.sqrt(cov[15] ?? Number.NaN), rows.length, {
    ...(options.level === undefined ? {} : { level: options.level }),
    df: Math.max(1, groups - 1),
    method: 'difference in differences (cluster-robust by unit)',
  })
  return {
    ...effect,
    means: { controlPre: cell(0, 0), controlPost: cell(0, 1), treatedPre: cell(1, 0), treatedPost: cell(1, 1) },
    cutoff: ix.periods[cutoff] ?? cutoff,
  }
}

/**
 * The two-way fixed-effects regression: the outcome on the treatment indicator with a fixed effect for every unit and
 * every period (by the Frisch-Waugh-Lovell demeaning, iterated for unbalanced panels), cluster-robust standard errors by
 * unit with the small-sample factors of the dummy-variable regression. With staggered adoption and effects that change
 * over time it is a weighted average of comparisons that include already-treated units as controls (Goodman-Bacon).
 *
 * @param panel - The long panel.
 * @param options - `level`.
 * @returns The coefficient of the treatment.
 * @example
 * ```ts
 * import { twoWayFixedEffects } from 'omai-engine/causal'
 * const panel = { unit: [0, 0, 1, 1, 2, 2], period: [0, 1, 0, 1, 0, 1], treated: [0, 1, 0, 1, 0, 0], outcome: [1, 4, 2, 5, 1, 2] }
 * twoWayFixedEffects(panel).estimate.value // => 2
 * ```
 */
export function twoWayFixedEffects(panel: PanelData, options: { readonly level?: number } = {}): EffectEstimate {
  const ix = index(panel)
  const both = new Float64Array(ix.n * 2)
  for (let i = 0; i < ix.n; i++) {
    both[i * 2] = ix.d[i] ?? 0
    both[i * 2 + 1] = ix.y[i] ?? 0
  }
  const dm = demean(ix, both, 2)
  const x = Float64Array.from({ length: ix.n }, (_, i) => dm[i * 2] ?? 0)
  const y = Float64Array.from({ length: ix.n }, (_, i) => dm[i * 2 + 1] ?? 0)
  const fit = wls(x, ix.n, 1, y)
  // the coefficient, a constant and the unit and period dummies of the equivalent dummy-variable regression
  const k = ix.units.length + ix.periods.length
  const cov = covariance(fit, x, 'cluster', undefined, ix.u, k)
  return effectFromSe('did', fit.beta[0] ?? Number.NaN, Math.sqrt(cov[0] ?? Number.NaN), ix.n, {
    ...(options.level === undefined ? {} : { level: options.level }),
    df: Math.max(1, ix.units.length - 1),
    method: 'two-way fixed effects (cluster-robust by unit)',
  })
}

/** Options of {@link eventStudy}. */
export interface EventStudyOptions {
  /** Leads to report (relative times -leads..-1, default 3). */
  readonly leads?: number
  /** Lags to report (relative times 0..lags, default 3). */
  readonly lags?: number
  /** The relative time normalised to zero (default -1). */
  readonly reference?: number
  /** `'twfe'` (default): a fixed-effects regression on relative-time dummies; `'callaway-santanna'`: group-time effects against the never-treated. */
  readonly method?: 'twfe' | 'callaway-santanna'
  /** Bootstrap resamples of units for the Callaway-Sant'Anna errors (default 200). */
  readonly resamples?: number
  readonly key?: Key
  readonly level?: number
}

/** The result of {@link eventStudy}. */
export interface EventStudyResult {
  /** The relative times reported, in order. */
  readonly relativeTimes: readonly number[]
  readonly coefficients: readonly number[]
  readonly standardErrors: readonly number[]
  readonly intervals: readonly (readonly [number, number])[]
  readonly reference: number
  /** Wald test that the leads are jointly zero (parallel pre-trends). */
  readonly pretrend?: { readonly statistic: number; readonly pValue: number; readonly df: number }
  /** The mean of the reported lag coefficients (relative times 0 and after). */
  readonly overall: number
  readonly method: string
}

function wald(
  b: readonly number[],
  cov: Float64Array,
  idx: readonly number[],
  p: number,
): { statistic: number; pValue: number; df: number } | undefined {
  const q = idx.length
  if (q === 0) return undefined
  const sub = Float64Array.from(
    { length: q * q },
    (_, k) => cov[(idx[Math.floor(k / q)] ?? 0) * p + (idx[k % q] ?? 0)] ?? 0,
  )
  const inv = invert(sub, q)
  let stat = 0
  for (let a = 0; a < q; a++)
    for (let c = 0; c < q; c++) stat += (b[idx[a] ?? 0] ?? 0) * (inv[a * q + c] ?? 0) * (b[idx[c] ?? 0] ?? 0)
  return { statistic: stat, pValue: chi2Sf(q, stat), df: q }
}

/** Group-time effects aggregated by event time against the never-treated, from unit indices (with repeats for a bootstrap). */
function groupTimeEffects(
  ix: Indexed,
  unitsInSample: ArrayLike<number>,
  relative: readonly number[],
  reference: number,
): number[] {
  const np = ix.periods.length
  // Outcomes by unit and period (a balanced panel is assumed for the units used; missing cells are skipped).
  const grid = new Float64Array(ix.units.length * np).fill(Number.NaN)
  for (let i = 0; i < ix.n; i++) grid[(ix.u[i] ?? 0) * np + (ix.t[i] ?? 0)] = ix.y[i] ?? 0
  const cohorts = new Map<number, number[]>()
  const never: number[] = []
  for (let k = 0; k < unitsInSample.length; k++) {
    const unit = unitsInSample[k] ?? 0
    const a = ix.adoption[unit] ?? -1
    if (a < 0) never.push(unit)
    else cohorts.set(a, [...(cohorts.get(a) ?? []), unit])
  }
  return relative.map((e) => {
    if (e === reference) return 0
    let num = 0
    let den = 0
    for (const [g, members] of cohorts) {
      const t = g + e
      const base = g + reference
      if (t < 0 || t >= np || base < 0 || base >= np) continue
      const change = (unit: number): number =>
        (grid[unit * np + t] ?? Number.NaN) - (grid[unit * np + base] ?? Number.NaN)
      const treated = members.map(change).filter(Number.isFinite)
      const control = never.map(change).filter(Number.isFinite)
      if (treated.length === 0 || control.length === 0) continue
      num += members.length * (mean(treated) - mean(control))
      den += members.length
    }
    return den === 0 ? Number.NaN : num / den
  })
}

/**
 * An event study: the effect at each period relative to adoption, with the period before adoption (by default)
 * normalised to zero. The coefficients at negative relative times are placebo effects, jointly tested against zero; the
 * coefficients from relative time 0 are the dynamic effect. `twfe` regresses on relative-time dummies with unit and
 * period fixed effects; with staggered adoption and changing effects that regression is contaminated, which
 * `callaway-santanna` avoids by comparing each adoption cohort only with never-treated units (errors by a cluster
 * bootstrap over units).
 *
 * @param panel - The long panel (units never treated serve as controls).
 * @param options - Window, reference period, method and bootstrap size.
 * @returns The coefficients by relative time, their errors, the pre-trend test and the mean post effect.
 * @example
 * ```ts
 * import { eventStudy } from 'omai-engine/causal'
 * const unit = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
 * const period = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
 * const treated = [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0]
 * const outcome = [1, 2, 6, 2, 3, 7, 1, 2, 3, 0, 1, 2]
 * eventStudy({ unit, period, treated, outcome }, { leads: 2, lags: 0, method: 'callaway-santanna', resamples: 5 }).coefficients // => [0, 0, 3]
 * ```
 */
export function eventStudy(panel: PanelData, options: EventStudyOptions = {}): EventStudyResult {
  const ix = index(panel)
  const leads = options.leads ?? 3
  const lags = options.lags ?? 3
  const reference = options.reference ?? -1
  const level = options.level ?? 0.95
  const z = normPpf(0.5 + level / 2)
  const relative = Array.from({ length: leads + lags + 1 }, (_, i) => i - leads)
  const method = options.method ?? 'twfe'
  let coefficients: number[]
  let ses: number[]
  let pre: ReturnType<typeof wald>
  if (method === 'twfe') {
    // Dummies for every relative time present among treated units, except the reference.
    const present = new Set<number>()
    for (let i = 0; i < ix.n; i++) {
      const a = ix.adoption[ix.u[i] ?? 0] ?? -1
      if (a >= 0) present.add((ix.t[i] ?? 0) - a)
    }
    const times = [...present].filter((e) => e !== reference).sort((a, b) => a - b)
    const k = times.length
    const dummies = new Float64Array(ix.n * (k + 1))
    const pos = new Map(times.map((e, j) => [e, j]))
    for (let i = 0; i < ix.n; i++) {
      const a = ix.adoption[ix.u[i] ?? 0] ?? -1
      if (a >= 0) {
        const j = pos.get((ix.t[i] ?? 0) - a)
        if (j !== undefined) dummies[i * (k + 1) + j] = 1
      }
      dummies[i * (k + 1) + k] = ix.y[i] ?? 0
    }
    const dm = demean(ix, dummies, k + 1)
    const x = Float64Array.from({ length: ix.n * k }, (_, q) => dm[Math.floor(q / k) * (k + 1) + (q % k)] ?? 0)
    const y = Float64Array.from({ length: ix.n }, (_, i) => dm[i * (k + 1) + k] ?? 0)
    const fit = wls(x, ix.n, k, y)
    const npar = k + ix.units.length + ix.periods.length - 1
    const cov = covariance(fit, x, 'cluster', undefined, ix.u, npar)
    const coef = (e: number): number => (e === reference ? 0 : (fit.beta[pos.get(e) ?? -1] ?? Number.NaN))
    const se = (e: number): number =>
      e === reference ? 0 : Math.sqrt(cov[(pos.get(e) ?? 0) * k + (pos.get(e) ?? 0)] ?? Number.NaN)
    coefficients = relative.map(coef)
    ses = relative.map(se)
    const leadIdx = times.filter((e) => e < 0 && e >= -leads).map((e) => pos.get(e) ?? 0)
    pre = wald(Array.from(fit.beta), cov, leadIdx, k)
  } else {
    if (!Array.from(ix.adoption).some((a) => a < 0))
      throw new OmaiError('value', "eventStudy: Callaway-Sant'Anna needs never-treated units")
    const all = Int32Array.from({ length: ix.units.length }, (_, i) => i)
    coefficients = groupTimeEffects(ix, all, relative, reference)
    const key = options.key ?? makeKey(0)
    const draws: number[][] = []
    for (let b = 0; b < (options.resamples ?? 200); b++)
      draws.push(groupTimeEffects(ix, toFlat(bootstrapIndices(foldIn(key, b), ix.units.length)), relative, reference))
    const col = (j: number): number[] => draws.map((d) => d[j] ?? Number.NaN).filter(Number.isFinite)
    ses = relative.map((_, j) => {
      const v = col(j)
      const m = mean(v)
      return v.length < 2 ? 0 : Math.sqrt(v.reduce((s, a) => s + (a - m) ** 2, 0) / (v.length - 1))
    })
    const leadJ = relative
      .map((e, j) => [e, j] as const)
      .filter(([e]) => e < 0 && e !== reference)
      .map(([, j]) => j)
    const p = relative.length
    const cov = new Float64Array(p * p)
    const means = relative.map((_, j) => mean(col(j)))
    for (const d of draws)
      for (const a of leadJ)
        for (const c of leadJ)
          cov[a * p + c] =
            (cov[a * p + c] ?? 0) +
            (((d[a] ?? 0) - (means[a] ?? 0)) * ((d[c] ?? 0) - (means[c] ?? 0))) / (draws.length - 1)
    pre = wald(coefficients, cov, leadJ, p)
  }
  const post = coefficients.filter((_, j) => (relative[j] ?? -1) >= 0)
  return {
    relativeTimes: relative,
    coefficients,
    standardErrors: ses,
    intervals: coefficients.map((c, j) => [c - z * (ses[j] ?? 0), c + z * (ses[j] ?? 0)] as const),
    reference,
    ...(pre === undefined ? {} : { pretrend: pre }),
    overall: mean(post),
    method: method === 'twfe' ? 'two-way fixed effects event study' : "Callaway-Sant'Anna group-time event study",
  }
}

/** The units-by-periods layout of a balanced panel's outcomes. */
export interface WideOutcomes {
  readonly units: readonly number[]
  readonly periods: readonly number[]
  /** `[periods, units]` row-major. */
  readonly matrix: Float64Array
}

/**
 * The outcomes of a balanced panel as a periods-by-units matrix (the layout synthetic control uses).
 *
 * @param panel - The long panel.
 * @returns Units, periods and the matrix.
 * @example
 * ```ts
 * import { wideOutcomes } from 'omai-engine/causal'
 * Array.from(wideOutcomes({ unit: [5, 5, 7, 7], period: [0, 1, 0, 1], treated: [0, 0, 0, 0], outcome: [1, 2, 3, 4] }).matrix) // => [1, 3, 2, 4]
 * ```
 */
export function wideOutcomes(panel: PanelData): WideOutcomes {
  const ix = index(panel)
  const nu = ix.units.length
  const matrix = new Float64Array(ix.periods.length * nu).fill(Number.NaN)
  for (let i = 0; i < ix.n; i++) matrix[(ix.t[i] ?? 0) * nu + (ix.u[i] ?? 0)] = ix.y[i] ?? 0
  if (matrix.some(Number.isNaN)) throw new OmaiError('value', 'wideOutcomes: the panel is not balanced')
  return { units: ix.units, periods: ix.periods, matrix }
}

/** What synthetic control is given. */
export interface SyntheticControlData {
  /** The treated unit's outcome in every period `[T]`. */
  readonly treated: ArrayLike<number>
  /** The donors' outcomes, periods by donors, row-major `[T, J]`. */
  readonly donors: ArrayLike<number>
  /** The number of donors `J`. */
  readonly donorCount: number
  /** The index of the first treated period. */
  readonly start: number
}

/** The synthetic control fit. */
export interface SyntheticControlResult {
  /** Donor weights on the simplex. */
  readonly weights: Float64Array
  /** The synthetic unit's outcome in every period. */
  readonly synthetic: Float64Array
  /** Treated minus synthetic in every period. */
  readonly gap: Float64Array
  /** Root mean squared gap before and after the start. */
  readonly preRmspe: number
  readonly postRmspe: number
  /** The post-to-pre ratio of root mean squared gaps (the test statistic of the placebo test). */
  readonly ratio: number
  /** The mean gap after the start: the estimated effect on the treated unit. */
  readonly effect: number
}

function fitSynthetic(
  treated: ArrayLike<number>,
  donors: ArrayLike<number>,
  j: number,
  start: number,
  ridge: number,
): SyntheticControlResult {
  const t = treated.length
  const pre = start
  if (pre < 1 || pre >= t)
    throw new OmaiError('value', 'syntheticControl: the start must leave at least one pre- and one post-period')
  const p = new Float64Array(j * j)
  const q = new Float64Array(j)
  for (let s = 0; s < pre; s++)
    for (let a = 0; a < j; a++) {
      q[a] = (q[a] ?? 0) - 2 * (donors[s * j + a] ?? 0) * (treated[s] ?? 0)
      for (let b = 0; b < j; b++)
        p[a * j + b] = (p[a * j + b] ?? 0) + 2 * (donors[s * j + a] ?? 0) * (donors[s * j + b] ?? 0)
    }
  for (let a = 0; a < j; a++) p[a * j + a] = (p[a * j + a] ?? 0) + 2 * ridge + 1e-10
  const sol = quadprog({
    P: Array.from({ length: j }, (_, a) => Array.from(p.subarray(a * j, (a + 1) * j))),
    q: Array.from(q),
    A: [new Array<number>(j).fill(1)],
    b: [1],
    lb: new Array<number>(j).fill(0),
  })
  const weights = Float64Array.from(toFlat(sol.x), (v) => Math.max(0, v))
  const total = weights.reduce((s, v) => s + v, 0)
  for (let a = 0; a < j; a++) weights[a] = (weights[a] ?? 0) / total
  const synthetic = Float64Array.from({ length: t }, (_, s) => {
    let v = 0
    for (let a = 0; a < j; a++) v += (weights[a] ?? 0) * (donors[s * j + a] ?? 0)
    return v
  })
  const gap = Float64Array.from(synthetic, (v, s) => (treated[s] ?? 0) - v)
  const rm = (from: number, to: number): number =>
    Math.sqrt(gap.slice(from, to).reduce((s, v) => s + v * v, 0) / (to - from))
  const preRmspe = rm(0, pre)
  const postRmspe = rm(pre, t)
  return {
    weights,
    synthetic,
    gap,
    preRmspe,
    postRmspe,
    ratio: postRmspe / Math.max(preRmspe, 1e-12),
    effect: mean(gap.slice(pre)),
  }
}

/**
 * Synthetic control (Abadie, Diamond and Hainmueller): the treated unit's counterfactual is the weighted average of
 * donor units whose non-negative weights, summing to one, minimise the squared distance to the treated unit's outcomes
 * before treatment (a quadratic program solved exactly by the active-set method). The gap after treatment is the effect.
 *
 * @param data - The treated unit's outcomes, the donors' outcomes and the first treated period.
 * @param options - `ridge`, a penalty on the weights (default 0).
 * @returns The weights, the synthetic path, the gap and the pre- and post-treatment fit.
 * @example
 * ```ts
 * import { syntheticControl } from 'omai-engine/causal'
 * const donors = [1, 5, 2, 6, 3, 7, 4, 8, 5, 9]
 * const treated = [3, 4, 5, 6, 7].map((v, s) => (s >= 3 ? v + 10 : v))
 * syntheticControl({ treated, donors, donorCount: 2, start: 3 }).effect // => 10
 * ```
 */
export function syntheticControl(
  data: SyntheticControlData,
  options: { readonly ridge?: number } = {},
): SyntheticControlResult {
  return fitSynthetic(data.treated, data.donors, data.donorCount, data.start, options.ridge ?? 0)
}

/** The placebo-in-space test of a synthetic control. */
export interface PlaceboResult {
  /** The post-to-pre RMSPE ratio of the treated unit then of each donor treated in turn. */
  readonly ratios: readonly number[]
  /** The rank of the treated unit's ratio among all (1 is the most extreme). */
  readonly rank: number
  /** `rank / (donors + 1)`: the share of units with a ratio at least as extreme. */
  readonly pValue: number
  /** The gap path of each placebo donor. */
  readonly gaps: readonly Float64Array[]
}

function donorColumns(donors: ArrayLike<number>, t: number, j: number, leave: number): Float64Array {
  const out = new Float64Array(t * (j - 1))
  for (let s = 0; s < t; s++) {
    let c = 0
    for (let a = 0; a < j; a++) if (a !== leave) out[s * (j - 1) + c++] = donors[s * j + a] ?? 0
  }
  return out
}

/** The state of {@link placeboInSpaceAlgorithm}: one donor treated per step. */
export type PlaceboState = {
  readonly done: number
  readonly ratios: number[]
  readonly gaps: number[][]
}

/**
 * The placebo-in-space test as an algorithm with one donor per step: each donor is treated in turn (the real treated unit
 * left out), a synthetic control built for it from the remaining donors, and the real post-to-pre ratio ranked among
 * the placebos'; the observed `rank` is the current share of placebos at least as extreme.
 *
 * @param data - As {@link syntheticControl}.
 * @returns The algorithm; it observes `donor`, `ratio` and `rank`.
 * @example
 * ```ts
 * import { placeboInSpaceAlgorithm } from 'omai-engine/causal'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const donors = [1, 5, 2, 2, 6, 3, 3, 7, 4, 4, 8, 5, 5, 9, 6]
 * const treated = [2, 3, 4, 5, 6].map((v, s) => (s >= 3 ? v + 9 : v))
 * runAlgorithm(placeboInSpaceAlgorithm({ treated, donors, donorCount: 3, start: 3 }), { maxSteps: 10 }).steps // => 3
 * ```
 */
export function placeboInSpaceAlgorithm(data: SyntheticControlData): Algorithm<PlaceboState> {
  const t = data.treated.length
  const j = data.donorCount
  const real = fitSynthetic(data.treated, data.donors, j, data.start, 0)
  return {
    name: 'placebo-in-space',
    config: { donors: j, periods: t, start: data.start },
    schema: channels({
      metrics: {
        donor: { description: 'the donor treated in this step' },
        ratio: { description: 'its post- to pre-treatment RMSPE ratio' },
        rank: { description: 'the share of placebos at least as extreme as the real ratio' },
      },
    }),
    init: () => ({ done: 0, ratios: [], gaps: [] }),
    step: (s) => {
      if (s.done >= j) return s
      const treated = Float64Array.from({ length: t }, (_, r) => data.donors[r * j + s.done] ?? 0)
      const rest = donorColumns(data.donors, t, j, s.done)
      const placebo = fitSynthetic(treated, rest, j - 1, data.start, 0)
      return { done: s.done + 1, ratios: [...s.ratios, placebo.ratio], gaps: [...s.gaps, Array.from(placebo.gap)] }
    },
    done: (s) => (s.done >= j ? 'converged' : undefined),
    observe: (s) => ({
      metrics: {
        donor: s.done,
        ratio: s.ratios[s.ratios.length - 1] ?? Number.NaN,
        rank: (1 + s.ratios.filter((r) => r >= real.ratio).length) / (1 + s.done),
      },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (json) => stateFromJson(json) as PlaceboState,
  }
}

/**
 * The placebo-in-space test of a synthetic control: treat each donor as if it had been treated, build its synthetic
 * control from the other donors, and compare the real unit's post-to-pre ratio of root mean squared gaps with theirs. The
 * share of units at least as extreme is the exact p-value of the sharp null of no effect.
 *
 * @param data - As {@link syntheticControl}.
 * @returns The ratios (the treated unit's first), its rank, the p-value and the placebo gaps.
 * @example
 * ```ts
 * import { placeboInSpace } from 'omai-engine/causal'
 * const donors = [1, 5, 2, 2, 6, 3, 3, 7, 4, 4, 8, 5, 5, 9, 6]
 * const treated = [2, 3, 4, 5, 6].map((v, s) => (s >= 3 ? v + 9 : v))
 * placeboInSpace({ treated, donors, donorCount: 3, start: 3 }).pValue // => 0.25
 * ```
 */
export function placeboInSpace(data: SyntheticControlData): PlaceboResult {
  const real = fitSynthetic(data.treated, data.donors, data.donorCount, data.start, 0)
  const s = runAlgorithm(placeboInSpaceAlgorithm(data), { maxSteps: data.donorCount, key: makeKey(0) }).state
  const rank = 1 + s.ratios.filter((r) => r >= real.ratio).length
  return {
    ratios: [real.ratio, ...s.ratios],
    rank,
    pValue: rank / (data.donorCount + 1),
    gaps: s.gaps.map((g) => Float64Array.from(g)),
  }
}
