/**
 * The segment costs of offline changepoint detection as an option table: the cost of one stretch of a series under a
 * model of it (a constant mean by squared error, a constant median by absolute error, a Gaussian with its own variance,
 * a Poisson rate), with prefix sums for the closed-form ones. The table is the cost slot of PELT and binary
 * segmentation; each option carries the shortest segment it can score.
 */

import { dict, type Option, type OptionTable, optionTable } from '../../space/index.ts'

/** A segment cost for offline changepoint detection. */
export type SegmentCost = 'l2' | 'l1' | 'normal' | 'poisson'

/** The cost of y[a..b) and the shortest segment the cost can score. */
export interface CostFunction {
  readonly error: (a: number, b: number) => number
  readonly minSize: number
}

/** What a segment cost looks for, for validity rules and selectors. */
export interface SegmentCostProperties {
  /** What changes between segments. */
  readonly shifts: 'mean' | 'median' | 'mean and variance' | 'rate'
  /** Whether the observations must be counts. */
  readonly counts: boolean
  /** Whether it is evaluated in constant time from prefix sums (else it reads the segment). */
  readonly constantTime: boolean
}

function cost(
  description: string,
  properties: SegmentCostProperties,
  name: SegmentCost,
): Option<(y: readonly number[]) => CostFunction, SegmentCostProperties> {
  return { description, properties, space: dict({}), make: () => (y) => costOf(y, name) }
}

/**
 * The cost of y[start..end) under a cost model, with prefix sums for the closed-form ones.
 *
 * @param y - The series.
 * @param name - The cost.
 * @returns The cost function and its shortest scorable segment.
 * @example
 * ```ts
 * import { filtering } from 'mdml-engine/dynamics'
 * filtering.costOf([0, 0, 4, 4], 'l2').error(0, 4) // => 16
 * ```
 */
export function costOf(y: readonly number[], name: SegmentCost): CostFunction {
  const n = y.length
  const s1 = new Float64Array(n + 1)
  const s2 = new Float64Array(n + 1)
  for (let i = 0; i < n; i++) {
    s1[i + 1] = (s1[i] ?? 0) + (y[i] ?? 0)
    s2[i + 1] = (s2[i] ?? 0) + (y[i] ?? 0) ** 2
  }
  const sse = (a: number, b: number): number => {
    const m = b - a
    const sum = (s1[b] ?? 0) - (s1[a] ?? 0)
    return Math.max(0, (s2[b] ?? 0) - (s2[a] ?? 0) - (sum * sum) / m)
  }
  switch (name) {
    case 'l2':
      return { error: sse, minSize: 1 }
    case 'normal':
      // ruptures' CostNormal in one dimension: n log(var + 1e-6).
      return { error: (a, b) => (b - a) * Math.log(sse(a, b) / (b - a) + 1e-6), minSize: 2 }
    case 'l1':
      return {
        error: (a, b) => {
          const seg = y.slice(a, b).sort((p, q) => p - q)
          const k = seg.length
          const med = k % 2 === 1 ? (seg[(k - 1) / 2] ?? 0) : 0.5 * ((seg[k / 2 - 1] ?? 0) + (seg[k / 2] ?? 0))
          return seg.reduce((acc, v) => acc + Math.abs(v - med), 0)
        },
        minSize: 2,
      }
    case 'poisson':
      // Twice the negative Poisson log-likelihood at the segment's rate, without the log y! terms.
      return {
        error: (a, b) => {
          const m = b - a
          const sum = (s1[b] ?? 0) - (s1[a] ?? 0)
          const rate = sum / m
          return rate === 0 ? 0 : 2 * (m * rate - sum * Math.log(rate))
        },
        minSize: 1,
      }
  }
}

/**
 * The segment costs as one slot: `l2` (squared error around the segment mean: a shift in the mean), `l1` (absolute
 * error around the median: robust to outliers), `normal` (a Gaussian with its own variance: a shift in the mean or the
 * variance) and `poisson` (a rate of counts). Each makes the cost function of a series.
 *
 * @concept changepoint-detection
 * @example
 * ```ts
 * import { filtering } from 'mdml-engine/dynamics'
 * filtering.segmentCosts.make({ kind: 'l1' })([0, 1, 9]).error(0, 3) // => 9
 * ```
 */
export const segmentCosts: OptionTable<(y: readonly number[]) => CostFunction, SegmentCostProperties> = optionTable(
  {
    l2: cost('squared error around the segment mean', { shifts: 'mean', counts: false, constantTime: true }, 'l2'),
    l1: cost(
      'absolute error around the segment median',
      { shifts: 'median', counts: false, constantTime: false },
      'l1',
    ),
    normal: cost(
      'twice the negative Gaussian log-likelihood with the segment’s own mean and variance',
      { shifts: 'mean and variance', counts: false, constantTime: true },
      'normal',
    ),
    poisson: cost(
      'twice the negative Poisson log-likelihood at the segment’s rate',
      { shifts: 'rate', counts: true, constantTime: true },
      'poisson',
    ),
  },
  { default: 'l2', label: 'cost', description: 'What a segment’s homogeneity is measured by.' },
)
