/**
 * Metrics of uplift models and targeting rules, which are scored on a randomised experiment (or data with known
 * propensities) because the effect of a unit is never seen: the cumulative gain curves of units ranked by predicted
 * uplift (the Qini curve of Radcliffe and the uplift curve), their random and oracle baselines, the area between the
 * curve and the random line (the Qini coefficient, the AUUC), the uplift in the top fraction or by decile, and the
 * inverse-propensity-weighted value of the policy that treats the top fraction.
 */

import { type MetricResult, OmaiError } from '../base/index.ts'
import { fromStorage, type Tensor, type TensorLike } from '../tensor/index.ts'
import { flat, metricResult } from './metric.ts'

/**
 * A cumulative gain curve over the number of units targeted, from the highest score down. The vectors are `[n + 1]`
 * tensors (migration, D-009: they were `Float64Array` and `Int32Array`).
 */
export interface UpliftCurve {
  /** Units targeted at each point, `0..n`. */
  readonly targeted: Tensor<'float64'>
  /** The same as a fraction of the population. */
  readonly fraction: Tensor<'float64'>
  /** The curve's value at each point. */
  readonly value: Tensor<'float64'>
  /** The straight line from the origin to the curve's end: the value of targeting at random. */
  readonly random: Tensor<'float64'>
  /** Treated and control units among those targeted (int32). */
  readonly treated: Tensor<'int32'>
  readonly control: Tensor<'int32'>
}

/** The points of an {@link UpliftCurve} as typed arrays, the form the area metrics read. */
interface UpliftRows {
  readonly targeted: Float64Array
  readonly fraction: Float64Array
  readonly value: Float64Array
  readonly random: Float64Array
  readonly treated: Int32Array
  readonly control: Int32Array
}

/** Which gain curve. */
export type CurveKind = 'qini' | 'uplift'

interface Ranked {
  readonly order: Int32Array
  readonly t: Float64Array
  readonly y: Float64Array
}

function rank(score: TensorLike, treatment: TensorLike, outcome: TensorLike, who: string): Ranked {
  const s = flat(score)
  const t = flat(treatment)
  const y = flat(outcome)
  if (t.length !== s.length || y.length !== s.length)
    throw new OmaiError('shape', `${who}: scores, treatment and outcome must have the same length`)
  for (const v of t) if (v !== 0 && v !== 1) throw new OmaiError('value', `${who}: the treatment must be 0 or 1`)
  const order = Int32Array.from(s.keys()).sort((a, b) => (s[b] ?? 0) - (s[a] ?? 0) || a - b)
  return { order, t, y }
}

/**
 * The cumulative gain curve of a targeting rule on experimental data: rank the units by score, highest first, and at
 * each count of units targeted compare the treated and control among them. The Qini curve is
 * `R_t - R_c N_t / N_c` (responders among the treated, less the controls' responders scaled to the treated count), the
 * uplift curve `(R_t / N_t - R_c / N_c)` times the number targeted; both are zero at the origin and end at the same
 * overall effect (times n for the uplift curve).
 *
 * @param score - The uplift score of each unit (higher is treated first).
 * @param treatment - The 0/1 treatment received.
 * @param outcome - The outcome (a 0/1 response, or any quantity whose sum is the gain).
 * @param kind - `'qini'` (default) or `'uplift'`.
 * @returns The curve and its random baseline, as `[n + 1]` tensors (was typed arrays).
 * @example
 * ```ts
 * import { upliftCurve } from 'omai-engine/metrics'
 * const c = upliftCurve([4, 3, 2, 1], [1, 0, 1, 0], [1, 0, 0, 0], 'qini')
 * c.value // => [0, 0, 1, 1, 1]
 * ```
 */
export function upliftCurve(
  score: TensorLike,
  treatment: TensorLike,
  outcome: TensorLike,
  kind: CurveKind = 'qini',
): UpliftCurve {
  const c = upliftRows(score, treatment, outcome, kind)
  const shape: [number] = [c.targeted.length]
  return {
    targeted: fromStorage(c.targeted, shape, 'float64'),
    fraction: fromStorage(c.fraction, shape, 'float64'),
    value: fromStorage(c.value, shape, 'float64'),
    random: fromStorage(c.random, shape, 'float64'),
    treated: fromStorage(c.treated, shape, 'int32'),
    control: fromStorage(c.control, shape, 'int32'),
  }
}

function upliftRows(score: TensorLike, treatment: TensorLike, outcome: TensorLike, kind: CurveKind): UpliftRows {
  const { order, t, y } = rank(score, treatment, outcome, 'upliftCurve')
  const n = order.length
  const targeted = new Float64Array(n + 1)
  const value = new Float64Array(n + 1)
  const treated = new Int32Array(n + 1)
  const control = new Int32Array(n + 1)
  let rt = 0
  let rc = 0
  for (let k = 1; k <= n; k++) {
    const i = order[k - 1] ?? 0
    targeted[k] = k
    treated[k] = (treated[k - 1] ?? 0) + (t[i] === 1 ? 1 : 0)
    control[k] = (control[k - 1] ?? 0) + (t[i] === 1 ? 0 : 1)
    if (t[i] === 1) rt += y[i] ?? 0
    else rc += y[i] ?? 0
    const nt = treated[k] ?? 0
    const nc = control[k] ?? 0
    if (kind === 'qini') value[k] = nc === 0 ? 0 : rt - (rc * nt) / nc
    else value[k] = nt === 0 || nc === 0 ? 0 : (rt / nt - rc / nc) * k
  }
  const last = value[n] ?? 0
  return {
    targeted,
    fraction: targeted.map((k) => k / n),
    value,
    random: targeted.map((k) => (last * k) / n),
    treated,
    control,
  }
}

const area = (x: ArrayLike<number>, y: ArrayLike<number>): number => {
  let a = 0
  for (let i = 1; i < x.length; i++) a += (((y[i] ?? 0) + (y[i - 1] ?? 0)) / 2) * ((x[i] ?? 0) - (x[i - 1] ?? 0))
  return a
}

/** Options of the area metrics. */
export interface AreaOptions {
  /** Subtract the area under the random line (default true): the gain over targeting at random. */
  readonly subtractRandom?: boolean
  /** Which curve (default `'qini'` for {@link qiniCoefficient}, `'uplift'` for {@link auuc}). */
  readonly kind?: CurveKind
}

/**
 * The Qini coefficient: the area between the Qini curve of a ranking and the random line, with the population as a
 * fraction on the x axis (so the units are responders). Zero for a random ranking in expectation, positive when the
 * ranking puts the persuadable first.
 *
 * @param score - The uplift score of each unit.
 * @param treatment - The 0/1 treatment.
 * @param outcome - The outcome.
 * @param options - `subtractRandom` (default true).
 * @returns The coefficient.
 * @example
 * ```ts
 * import { qiniCoefficient } from 'omai-engine/metrics'
 * qiniCoefficient([4, 3, 2, 1], [1, 0, 1, 0], [1, 0, 0, 0]).estimate.value // => 0.125
 * ```
 */
export function qiniCoefficient(
  score: TensorLike,
  treatment: TensorLike,
  outcome: TensorLike,
  options: AreaOptions = {},
): MetricResult {
  const c = upliftRows(score, treatment, outcome, options.kind ?? 'qini')
  const total = area(c.fraction, c.value)
  const v = options.subtractRandom === false ? total : total - area(c.fraction, c.random)
  return metricResult('qini-coefficient', v, c.value.length - 1)
}

/**
 * The area under the uplift curve (AUUC), the gain over targeting at random by default: the uplift curve's value at a
 * fraction of the population is the uplift among those targeted times their number, so the area summarises how fast a
 * ranking finds the units the treatment helps.
 *
 * @param score - The uplift score of each unit.
 * @param treatment - The 0/1 treatment.
 * @param outcome - The outcome.
 * @param options - `subtractRandom` (default true).
 * @returns The area.
 * @example
 * ```ts
 * import { auuc } from 'omai-engine/metrics'
 * auuc([4, 3, 2, 1], [1, 0, 1, 0], [1, 0, 0, 0]).estimate.value > 0 // => true
 * ```
 */
export function auuc(
  score: TensorLike,
  treatment: TensorLike,
  outcome: TensorLike,
  options: AreaOptions = {},
): MetricResult {
  const c = upliftRows(score, treatment, outcome, options.kind ?? 'uplift')
  const total = area(c.fraction, c.value)
  const v = options.subtractRandom === false ? total : total - area(c.fraction, c.random)
  return metricResult('auuc', v, c.value.length - 1)
}

/**
 * The uplift among the units targeted: the treated's mean outcome minus the controls' among the top `fraction` by score.
 *
 * @param score - The uplift score of each unit.
 * @param treatment - The 0/1 treatment.
 * @param outcome - The outcome.
 * @param fraction - The share of units targeted (default 0.3).
 * @returns The difference of mean outcomes (NaN if an arm is empty).
 * @example
 * ```ts
 * import { upliftAtK } from 'omai-engine/metrics'
 * upliftAtK([4, 3, 2, 1], [1, 0, 1, 0], [1, 0, 0, 0], 0.5).estimate.value // => 1
 * ```
 */
export function upliftAtK(score: TensorLike, treatment: TensorLike, outcome: TensorLike, fraction = 0.3): MetricResult {
  const { order, t, y } = rank(score, treatment, outcome, 'upliftAtK')
  const k = Math.max(1, Math.round(fraction * order.length))
  let [st, sc, nt, nc] = [0, 0, 0, 0]
  for (let r = 0; r < k; r++) {
    const i = order[r] ?? 0
    if (t[i] === 1) {
      st += y[i] ?? 0
      nt++
    } else {
      sc += y[i] ?? 0
      nc++
    }
  }
  return metricResult('uplift-at-k', nt === 0 || nc === 0 ? Number.NaN : st / nt - sc / nc, k, {
    treated: nt,
    control: nc,
  })
}

/** One decile of {@link upliftDeciles}. */
export interface UpliftBin {
  readonly bin: number
  readonly n: number
  readonly treated: number
  /** The mean score of the units in the bin. */
  readonly meanScore: number
  /** The observed uplift in the bin (treated mean minus control mean). */
  readonly uplift: number
}

/**
 * The observed uplift in equal-sized bins of units ranked by score, highest first (deciles by default): for a good
 * model the uplift falls from the first bin to the last.
 *
 * @param score - The uplift score of each unit.
 * @param treatment - The 0/1 treatment.
 * @param outcome - The outcome.
 * @param bins - The number of bins (default 10).
 * @returns One row per bin.
 * @example
 * ```ts
 * import { upliftDeciles } from 'omai-engine/metrics'
 * upliftDeciles([4, 3, 2, 1], [1, 0, 1, 0], [1, 0, 0, 0], 2).map((b) => b.uplift) // => [1, 0]
 * ```
 */
export function upliftDeciles(score: TensorLike, treatment: TensorLike, outcome: TensorLike, bins = 10): UpliftBin[] {
  const { order, t, y } = rank(score, treatment, outcome, 'upliftDeciles')
  const s = flat(score)
  const n = order.length
  return Array.from({ length: bins }, (_, b) => {
    const rows = Array.from(order.subarray(Math.floor((b * n) / bins), Math.floor(((b + 1) * n) / bins)))
    const arm = (a: number): number[] => rows.filter((i) => t[i] === a).map((i) => y[i] ?? 0)
    const mean = (v: number[]): number => (v.length === 0 ? Number.NaN : v.reduce((p, q) => p + q, 0) / v.length)
    return {
      bin: b,
      n: rows.length,
      treated: arm(1).length,
      meanScore: mean(rows.map((i) => s[i] ?? 0)),
      uplift: mean(arm(1)) - mean(arm(0)),
    }
  })
}

/** Options of {@link policyValue}. */
export interface PolicyOptions {
  /** The share of units the policy may treat (default 0.3). */
  readonly budget?: number
  /** The probability of treatment of each unit, or one value (default the treated share, as in an experiment). */
  readonly propensity?: number | TensorLike
}

/**
 * The value of the policy that treats the top `budget` of units by score, estimated by inverse propensity weighting on
 * experimental data: the mean over units of the outcome that was observed when the policy's action matches the action
 * taken, divided by the probability of that action. The breakdown holds the values of treating everyone and no one, so
 * the policy's gain is read against them (it crosses treat-everyone at the budget where treating more stops helping).
 *
 * @param score - The uplift score of each unit.
 * @param treatment - The 0/1 treatment received.
 * @param outcome - The outcome.
 * @param options - Budget and propensity.
 * @returns The value, with `treatAll` and `treatNone` in the breakdown.
 * @example
 * ```ts
 * import { policyValue } from 'omai-engine/metrics'
 * policyValue([4, 3, 2, 1], [1, 0, 1, 0], [1, 0, 0, 0], { budget: 0.5, propensity: 0.5 }).estimate.value // => 0.5
 * ```
 */
export function policyValue(
  score: TensorLike,
  treatment: TensorLike,
  outcome: TensorLike,
  options: PolicyOptions = {},
): MetricResult {
  const { order, t, y } = rank(score, treatment, outcome, 'policyValue')
  const n = order.length
  const treatedShare = t.reduce((a, b) => a + b, 0) / n
  const p =
    options.propensity === undefined
      ? treatedShare
      : typeof options.propensity === 'number'
        ? options.propensity
        : flat(options.propensity)
  const e = (i: number): number => (typeof p === 'number' ? p : (p[i] ?? treatedShare))
  const k = Math.round((options.budget ?? 0.3) * n)
  const action = new Float64Array(n)
  for (let r = 0; r < k; r++) action[order[r] ?? 0] = 1
  const value = (a: (i: number) => number): number => {
    let s = 0
    for (let i = 0; i < n; i++) {
      const ai = a(i)
      if (ai === t[i]) s += (y[i] ?? 0) / (ai === 1 ? e(i) : 1 - e(i))
    }
    return s / n
  }
  return metricResult(
    'policy-value',
    value((i) => action[i] ?? 0),
    n,
    {
      budget: k / n,
      treatAll: value(() => 1),
      treatNone: value(() => 0),
    },
  )
}
