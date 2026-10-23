import type { Alternative } from './options.ts'

/**
 * An estimate with its uncertainty: the shape shared by every metric, test effect size and summary statistic.
 *
 * @concept estimates
 */
export interface Estimate {
  readonly value: number
  readonly standardError?: number
  readonly interval?: readonly [number, number]
  readonly level?: number
  readonly method?: string
}

/**
 * The result of every statistical test in whitebox-js.
 *
 * @concept hypothesis-testing
 */
export interface TestResult {
  readonly test: string
  readonly statistic: number
  readonly pValue: number
  readonly alternative: Alternative
  readonly df?: number | readonly number[]
  readonly effectSize?: Estimate & { readonly name: string }
  readonly estimate?: Estimate
  readonly assumptions?: Readonly<Record<string, boolean | number | string>>
  readonly method?: string
}

/** Which direction of a metric is better. */
export type Direction = 'higher' | 'lower'

/**
 * The declaration of a metric: what it consumes and which way is better, so tuners and reports can use any metric.
 *
 * @concept evaluation-metrics
 */
export interface MetricSpec {
  readonly name: string
  readonly direction: Direction
  readonly range: readonly [number, number]
  readonly inputs: 'labels' | 'scores' | 'probabilities' | 'distributions' | 'values' | 'rankings' | 'sets'
  readonly category: string
  readonly concept?: string
}

/**
 * The result of a metric: an estimate plus optional breakdowns (per class, per sample, per group).
 *
 * @concept evaluation-metrics
 */
export interface MetricResult {
  readonly metric: string
  readonly estimate: Estimate
  readonly n: number
  readonly breakdown?: Readonly<Record<string, number>>
}

/**
 * Build an {@link Estimate} from a value and an optional standard error, adding a normal-approximation interval.
 *
 * @param value - The point estimate.
 * @param standardError - Its standard error, if known.
 * @param level - The interval's coverage (default 0.95).
 * @returns The estimate.
 * @example
 * ```ts
 * import { estimate } from 'whitebox-engine/base'
 * estimate(1, 0.5).interval // => [0.020018, 1.979982]
 * ```
 */
export function estimate(value: number, standardError?: number, level = 0.95): Estimate {
  if (standardError === undefined) return { value }
  const z = normalQuantile(0.5 + level / 2)
  return {
    value,
    standardError,
    interval: [value - z * standardError, value + z * standardError],
    level,
    method: 'normal approximation',
  }
}

/**
 * Format any result for display: an estimate with its interval, or a test with its statistic and p-value.
 *
 * @param result - An estimate, a test result or a metric result.
 * @param digits - Significant digits (default 4).
 * @returns A one-line description.
 * @example
 * ```ts
 * import { formatResult } from 'whitebox-engine/base'
 * formatResult({ value: 0.25, interval: [0.2, 0.3], level: 0.95 }) // => '0.25 (95% CI 0.2 to 0.3)'
 * ```
 */
export function formatResult(result: Estimate | TestResult | MetricResult, digits = 4): string {
  const f = (x: number): string => String(Number(x.toPrecision(digits)))
  if ('pValue' in result) return `${result.test}: statistic ${f(result.statistic)}, p = ${f(result.pValue)}`
  const e = 'estimate' in result ? result.estimate : result
  const head = 'metric' in result ? `${result.metric} ` : ''
  if (e.interval === undefined) return `${head}${f(e.value)}`
  const pct = Math.round((e.level ?? 0.95) * 100)
  return `${head}${f(e.value)} (${pct}% CI ${f(e.interval[0])} to ${f(e.interval[1])})`
}

// Acklam's rational approximation (relative error below 1.2e-9). Used only for interval widths here; the full
// special-function family lives in numerics/special.
const A = [
  -39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239,
] as const
const B = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572] as const
const C = [
  -0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968,
  2.938163982698783,
] as const
const D = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416] as const

function tail(q: number): number {
  const [c0, c1, c2, c3, c4, c5] = C
  const [d0, d1, d2, d3] = D
  return (((((c0 * q + c1) * q + c2) * q + c3) * q + c4) * q + c5) / ((((d0 * q + d1) * q + d2) * q + d3) * q + 1)
}

function normalQuantile(p: number): number {
  const low = 0.02425
  if (p < low) return tail(Math.sqrt(-2 * Math.log(p)))
  if (p > 1 - low) return -tail(Math.sqrt(-2 * Math.log(1 - p)))
  const [a0, a1, a2, a3, a4, a5] = A
  const [b0, b1, b2, b3, b4] = B
  const q = p - 0.5
  const r = q * q
  return (
    ((((((a0 * r + a1) * r + a2) * r + a3) * r + a4) * r + a5) * q) /
    (((((b0 * r + b1) * r + b2) * r + b3) * r + b4) * r + 1)
  )
}
