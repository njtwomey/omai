/**
 * Matching estimators of the average effect on the treated, the controls or everyone: each unit of the target is paired
 * with its nearest neighbours in the other arm, on the propensity score (the logit scale), on the standardised
 * covariates or on the Mahalanobis distance, with or without replacement, with a caliper beyond which a unit stays
 * unmatched, and with the Abadie-Imbens standard error that accounts for how often each unit is reused as a match.
 */

import { MdmlError } from '../base/index.ts'
import { neighbourIndex } from '../learn/neighbours.ts'
import { dpotrf } from '../numerics/linalg/index.ts'
import { fromStorage } from '../tensor/index.ts'
import { logit, mean, type Prepared, prepare, variance } from './_dense.ts'
import { type CausalData, type EffectEstimate, effectFromSe } from './data.ts'
import type { AverageEstimand } from './estimation.ts'
import { allRows, fitPropensity, type NuisanceOptions } from './nuisance.ts'

/** Options of {@link matching}. */
export interface MatchingOptions extends NuisanceOptions {
  /** The target: the treated (default), the controls or everyone. */
  readonly estimand?: AverageEstimand
  /** What distance matches units: the logit of the propensity score (default), standardised covariates, or Mahalanobis. */
  readonly on?: 'propensity' | 'covariates' | 'mahalanobis'
  /** Matches per unit (default 1). */
  readonly neighbours?: number
  /** Whether a unit may be matched more than once (default true). Without it units are matched greedily in order of their first coordinate. */
  readonly replacement?: boolean
  /**
   * The largest distance of a match: for the propensity score it is this many standard deviations of the logit
   * score (Rosenbaum and Rubin suggest 0.2), for the other distances it is in the distance's own units. Default none.
   */
  readonly caliper?: number
  /** Known propensity scores for `on: 'propensity'`. */
  readonly scores?: ArrayLike<number>
  readonly level?: number
}

/** The matches of one unit. */
export interface MatchedUnit {
  readonly unit: number
  readonly matches: readonly number[]
  readonly distances: readonly number[]
}

/** A matching estimate with the pairs behind it. */
export interface MatchingResult extends EffectEstimate {
  /** The matched units and their matches. */
  readonly pairs: readonly MatchedUnit[]
  /** The units of the target left without a match (beyond the caliper, or no donor left). */
  readonly unmatched: readonly number[]
  /** How many times each unit served as a match. */
  readonly timesUsed: Int32Array
  /** The caliper in the distance's units, when one applied. */
  readonly caliper?: number
}

/** The coordinates in which Euclidean distance is the requested distance. */
function coordinates(
  p: Prepared,
  on: 'propensity' | 'covariates' | 'mahalanobis',
  scores: Float64Array,
): { z: Float64Array; q: number } {
  if (on === 'propensity') return { z: Float64Array.from(scores, logit), q: 1 }
  const mu = new Float64Array(p.d)
  for (let i = 0; i < p.n; i++) for (let j = 0; j < p.d; j++) mu[j] = (mu[j] ?? 0) + (p.x[i * p.d + j] ?? 0) / p.n
  const cov = new Float64Array(p.d * p.d)
  for (let i = 0; i < p.n; i++)
    for (let a = 0; a < p.d; a++)
      for (let b = 0; b < p.d; b++)
        cov[a * p.d + b] =
          (cov[a * p.d + b] ?? 0) +
          (((p.x[i * p.d + a] ?? 0) - (mu[a] ?? 0)) * ((p.x[i * p.d + b] ?? 0) - (mu[b] ?? 0))) / Math.max(1, p.n - 1)
  const z = new Float64Array(p.n * p.d)
  if (on === 'covariates') {
    for (let i = 0; i < p.n; i++)
      for (let j = 0; j < p.d; j++)
        z[i * p.d + j] = ((p.x[i * p.d + j] ?? 0) - (mu[j] ?? 0)) / Math.sqrt(cov[j * p.d + j] || 1)
    return { z, q: p.d }
  }
  const { factor, info } = dpotrf(cov, p.d)
  if (info !== 0)
    throw new MdmlError('value', 'matching: the covariate covariance is singular, so there is no Mahalanobis distance')
  for (let i = 0; i < p.n; i++)
    for (let j = 0; j < p.d; j++) {
      let s = (p.x[i * p.d + j] ?? 0) - (mu[j] ?? 0)
      for (let k = 0; k < j; k++) s -= (factor[k * p.d + j] ?? 0) * (z[i * p.d + k] ?? 0)
      z[i * p.d + j] = s / (factor[j * p.d + j] ?? 1)
    }
  return { z, q: p.d }
}

const rowsOfZ = (z: Float64Array, q: number, rows: ArrayLike<number>): Float64Array => {
  const out = new Float64Array(rows.length * q)
  for (let i = 0; i < rows.length; i++) for (let j = 0; j < q; j++) out[i * q + j] = z[(rows[i] ?? 0) * q + j] ?? 0
  return out
}

/** Match the `queries` to `donors` (row indices): the donors' nearest neighbours within the caliper. */
function findMatches(
  z: Float64Array,
  q: number,
  queries: readonly number[],
  donors: readonly number[],
  m: number,
  replacement: boolean,
  caliper: number,
): MatchedUnit[] {
  const out: MatchedUnit[] = []
  if (queries.length === 0 || donors.length === 0) return out
  if (replacement) {
    const index = neighbourIndex(fromStorage(rowsOfZ(z, q, donors), [donors.length, q], 'float64'))
    const hits = index.nearest(fromStorage(rowsOfZ(z, q, queries), [queries.length, q], 'float64'), m)
    queries.forEach((unit, k) => {
      const h = hits[k]
      const ok = (h?.indices ?? [])
        .map((idx, r) => [idx, h?.distances[r] ?? Number.POSITIVE_INFINITY] as const)
        .filter(([, dist]) => dist <= caliper)
      if (ok.length > 0)
        out.push({ unit, matches: ok.map(([idx]) => donors[idx] ?? 0), distances: ok.map(([, dist]) => dist) })
    })
    return out
  }
  const used = new Set<number>()
  const order = [...queries].sort((a, b) => (z[b * q] ?? 0) - (z[a * q] ?? 0))
  for (const unit of order) {
    const cand = donors
      .filter((d) => !used.has(d))
      .map((d) => {
        let s = 0
        for (let j = 0; j < q; j++) s += ((z[unit * q + j] ?? 0) - (z[d * q + j] ?? 0)) ** 2
        return [d, Math.sqrt(s)] as const
      })
      .filter(([, dist]) => dist <= caliper)
      .sort((a, b) => a[1] - b[1])
      .slice(0, m)
    if (cand.length === 0) continue
    for (const [d] of cand) used.add(d)
    out.push({ unit, matches: cand.map(([d]) => d), distances: cand.map(([, dist]) => dist) })
  }
  return out.sort((a, b) => a.unit - b.unit)
}

/** The Abadie-Imbens conditional variance of a unit's outcome from its nearest neighbour in its own arm. */
function localVariance(z: Float64Array, q: number, y: ArrayLike<number>, arm: readonly number[]): Map<number, number> {
  const out = new Map<number, number>()
  if (arm.length < 2) return out
  const index = neighbourIndex(fromStorage(rowsOfZ(z, q, arm), [arm.length, q], 'float64'))
  const hits = index.nearest(fromStorage(rowsOfZ(z, q, arm), [arm.length, q], 'float64'), 2)
  arm.forEach((unit, k) => {
    const other = (hits[k]?.indices ?? []).map((idx) => arm[idx] ?? 0).find((u) => u !== unit)
    if (other !== undefined) out.set(unit, 0.5 * ((y[unit] ?? 0) - (y[other] ?? 0)) ** 2)
  })
  return out
}

/**
 * Matching estimator of a treatment effect (Abadie and Imbens): each unit of the estimand's target (the treated, the
 * controls, or everyone) is matched to its `neighbours` nearest units of the opposite arm, its missing potential
 * outcome imputed by their mean outcome, and the effect is the mean of the imputed unit effects. The standard error is
 * the Abadie-Imbens variance, which counts how often each unit is reused; bootstrapping a matching estimator is
 * invalid, so no bootstrap is offered.
 *
 * @param data - Covariates, treatment and outcome.
 * @param options - Distance, neighbours, replacement, caliper and estimand.
 * @returns The effect with the pairs, the unmatched units and how often each unit was used.
 * @index function Matching estimator of a treatment effect.
 * @example
 * ```ts
 * import { matching } from 'mdml-engine/causal'
 * const x = [[0], [1], [2], [3], [0.1], [1.1], [2.1], [3.1]]
 * const r = matching({ x, treatment: [1, 1, 1, 1, 0, 0, 0, 0], outcome: [3, 4, 5, 6, 1, 2, 3, 4] }, { on: 'covariates' })
 * r.estimate.value // => 2
 * ```
 */
export function matching(data: CausalData, options: MatchingOptions = {}): MatchingResult {
  const p = prepare(data.x, data.treatment, data.outcome, 'matching')
  const estimand = options.estimand ?? 'att'
  const on = options.on ?? 'propensity'
  const m = options.neighbours ?? 1
  const replacement = options.replacement ?? true
  const treated = Array.from(allRows(p.n)).filter((i) => p.t[i] === 1)
  const control = Array.from(allRows(p.n)).filter((i) => p.t[i] === 0)
  const scores =
    on === 'propensity'
      ? options.scores === undefined
        ? fitPropensity(p, allRows(p.n), allRows(p.n), options)
        : Float64Array.from(options.scores)
      : new Float64Array(p.n)
  const { z, q } = coordinates(p, on, scores)
  let caliper = Number.POSITIVE_INFINITY
  if (options.caliper !== undefined)
    caliper = on === 'propensity' ? options.caliper * Math.sqrt(variance(z, 1)) : options.caliper
  const queriesTreated = estimand === 'atc' ? [] : treated
  const queriesControl = estimand === 'att' ? [] : control
  const pairsT = findMatches(z, q, queriesTreated, control, m, replacement, caliper)
  const pairsC = findMatches(z, q, queriesControl, treated, m, replacement, caliper)
  const pairs = [...pairsT, ...pairsC]
  if (pairs.length === 0) throw new MdmlError('value', 'matching: no unit found a match (loosen the caliper)')
  const matched = new Set(pairs.map((u) => u.unit))
  const target = estimand === 'att' ? treated : estimand === 'atc' ? control : [...treated, ...control]
  const unmatched = target.filter((u) => !matched.has(u))
  const timesUsed = new Int32Array(p.n)
  const unitEffect = new Map<number, number>()
  for (const u of pairs) {
    const imputed = mean(u.matches.map((j) => p.y[j] ?? 0))
    for (const j of u.matches) timesUsed[j] = (timesUsed[j] ?? 0) + 1
    const own = p.y[u.unit] ?? 0
    unitEffect.set(u.unit, p.t[u.unit] === 1 ? own - imputed : imputed - own)
  }
  const value = mean(pairs.map((u) => unitEffect.get(u.unit) ?? 0))
  const sigma = new Map<number, number>([...localVariance(z, q, p.y, treated), ...localVariance(z, q, p.y, control)])
  let v: number
  if (estimand === 'ate') {
    const n = pairs.length
    let first = 0
    let second = 0
    for (const u of pairs) {
      first += ((unitEffect.get(u.unit) ?? 0) - value) ** 2
    }
    for (let i = 0; i < p.n; i++) {
      const k = (timesUsed[i] ?? 0) / m
      second += (k * k + ((2 * m - 1) / m) * k) * (sigma.get(i) ?? 0)
    }
    v = (first + second) / (n * n)
  } else {
    const donors = estimand === 'att' ? control : treated
    const n = pairs.length
    let first = 0
    for (const u of pairs) first += ((unitEffect.get(u.unit) ?? 0) - value) ** 2
    let second = 0
    for (const j of donors) {
      const k = timesUsed[j] ?? 0
      second += ((k * (k - 1)) / (m * m)) * (sigma.get(j) ?? 0)
    }
    v = (first + second) / (n * n)
  }
  const effect = effectFromSe(estimand, value, Math.sqrt(v), pairs.length, {
    ...(options.level === undefined ? {} : { level: options.level }),
    method: `matching on ${on} (${m} neighbour${m === 1 ? '' : 's'}, ${replacement ? 'with' : 'without'} replacement), Abadie-Imbens standard error`,
  })
  return { ...effect, pairs, unmatched, timesUsed, ...(Number.isFinite(caliper) ? { caliper } : {}) }
}
