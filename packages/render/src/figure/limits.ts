/**
 * Axis limit policies for reactive figures: `auto` fits the data, `fixed` holds a stated range (so a moving mean
 * visibly moves), `expand` grows but never shrinks and `follow` tracks a window, with the range helpers behind them
 * (extents, padding, nice rounding, and widening a range too narrow to tick).
 */

import { WhiteboxError } from 'whitebox-engine/base'
import { niceTicks } from '../theme/format.ts'
import type { Range } from './aspect.ts'

/**
 * How an axis chooses its range as data change (for reactive figures):
 * - `auto`: fit the data, padded and optionally rounded to nice ticks;
 * - `fixed`: a stated range, so moving a Gaussian's mean visibly moves it;
 * - `expand`: fit the data but never shrink below the previous range;
 * - `follow`: a window of fixed width tracking the latest data (a streaming metric).
 */
export type LimitPolicy =
  | { readonly kind: 'auto'; readonly pad?: number; readonly nice?: boolean; readonly includeZero?: boolean }
  | { readonly kind: 'fixed'; readonly range: Range }
  | { readonly kind: 'expand'; readonly pad?: number; readonly nice?: boolean }
  | { readonly kind: 'follow'; readonly window: number; readonly lead?: number }

/** A limit policy or a fixed range written as a pair. */
export type Limits = LimitPolicy | Range

/**
 * The least and greatest finite values, or `undefined` when there are none.
 *
 * @param values - Any numbers (NaN and infinities are skipped).
 * @returns The extent.
 * @example
 * ```ts
 * import { extentOf } from 'whitebox-render/figure'
 * extentOf([3, Number.NaN, -1, 2]) // => [-1, 3]
 * extentOf([]) // => undefined
 * ```
 */
export function extentOf(values: Iterable<number>): Range | undefined {
  let lo = Number.POSITIVE_INFINITY
  let hi = Number.NEGATIVE_INFINITY
  for (const v of values) {
    if (!Number.isFinite(v)) continue
    if (v < lo) lo = v
    if (v > hi) hi = v
  }
  return lo <= hi ? [lo, hi] : undefined
}

/**
 * The smallest range containing every given range (skipping missing ones).
 *
 * @param ranges - Ranges or `undefined`.
 * @returns The union, or `undefined` when none is given.
 * @example
 * ```ts
 * import { unionRange } from 'whitebox-render/figure'
 * unionRange([[0, 1], undefined, [-2, 0.5]]) // => [-2, 1]
 * ```
 */
export function unionRange(ranges: Iterable<Range | undefined>): Range | undefined {
  let out: Range | undefined
  for (const r of ranges) {
    if (r === undefined) continue
    out = out === undefined ? r : [Math.min(out[0], r[0]), Math.max(out[1], r[1])]
  }
  return out
}

/**
 * A range widened by a fraction of its span on each side; a degenerate range gets a unit-scaled width.
 *
 * @param range - The range.
 * @param fraction - The padding on each side as a fraction of the span (default 0.05).
 * @returns The padded range.
 * @example
 * ```ts
 * import { padRange } from 'whitebox-render/figure'
 * padRange([0, 10]) // => [-0.5, 10.5]
 * padRange([2, 2]) // => [1.8, 2.2]
 * ```
 */
export function padRange(range: Range, fraction = 0.05): Range {
  const [lo, hi] = range
  const span = hi - lo
  if (span > 0) return [lo - span * fraction, hi + span * fraction]
  const d = lo === 0 ? 1 : Math.abs(lo) * 0.1
  return [lo - d, hi + d]
}

/**
 * A range rounded outwards to nice tick positions.
 *
 * @param range - The range.
 * @param count - About how many ticks (default 5).
 * @returns The rounded range.
 * @example
 * ```ts
 * import { niceRange } from 'whitebox-render/figure'
 * niceRange([0.13, 0.92]) // => [0, 1]
 * ```
 */
export function niceRange(range: Range, count = 5): Range {
  return niceTicks(range[0], range[1], count).range
}

/**
 * A range an axis can draw: one too narrow to tick (equal ends, or a span below a billionth of its magnitude or below
 * 1e-12, as an objective converged to ~1e-26 gives) widened about its centre, non-finite ends replaced by [0, 1].
 *
 * @param range - The range.
 * @returns The range, wide enough to tick.
 * @example
 * ```ts
 * import { legibleRange } from 'whitebox-render/figure'
 * legibleRange([2, 4]) // => [2, 4]
 * legibleRange([-1e-26, 1e-26]) // => [-5e-13, 5e-13]
 * legibleRange([3, 3]) // => [2.7, 3.3]
 * ```
 */
export function legibleRange(range: Range): Range {
  const [lo, hi] = range
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return [0, 1]
  if (hi === lo) return padRange(range)
  const least = Math.max(1e-9 * Math.max(Math.abs(lo), Math.abs(hi)), 1e-12)
  if (Math.abs(hi - lo) >= least) return range
  const centre = (lo + hi) / 2
  const half = least / 2
  return hi > lo ? [centre - half, centre + half] : [centre + half, centre - half]
}

/**
 * Resolve an axis range under a limit policy.
 *
 * @param limits - The policy, or a fixed range.
 * @param extent - The current data extent (undefined when there is no data).
 * @param previous - The range shown before (used by `expand`).
 * @returns The range to show.
 * @throws WhiteboxError with code `value` for an empty fixed range or a non-positive follow window.
 * @example
 * ```ts
 * import { resolveLimits } from 'whitebox-render/figure'
 * resolveLimits({ kind: 'auto', pad: 0 }, [1, 3]) // => [1, 3]
 * resolveLimits([-5, 5], [1, 3]) // => [-5, 5]
 * resolveLimits({ kind: 'expand', pad: 0 }, [1, 3], [0, 2]) // => [0, 3]
 * resolveLimits({ kind: 'follow', window: 10 }, [0, 42]) // => [32, 42]
 * ```
 */
export function resolveLimits(limits: Limits, extent: Range | undefined, previous?: Range): Range {
  const policy: LimitPolicy = isRange(limits) ? { kind: 'fixed', range: limits } : limits
  switch (policy.kind) {
    case 'fixed': {
      if (!(policy.range[1] > policy.range[0])) throw new WhiteboxError('value', 'limits: a fixed range must increase')
      return policy.range
    }
    case 'auto': {
      if (extent === undefined) return previous ?? [0, 1]
      const base: Range = policy.includeZero === true ? [Math.min(0, extent[0]), Math.max(0, extent[1])] : extent
      const p = padRange(base, policy.pad ?? 0.05)
      // A baseline at zero stays at zero: bars and densities start on the axis rather than below it.
      const padded: Range = policy.includeZero === true ? [base[0] === 0 ? 0 : p[0], base[1] === 0 ? 0 : p[1]] : p
      return policy.nice === true ? niceRange(padded) : padded
    }
    case 'expand': {
      if (extent === undefined) return previous ?? [0, 1]
      const padded = padRange(extent, policy.pad ?? 0.05)
      const fit = policy.nice === true ? niceRange(padded) : padded
      return previous === undefined ? fit : [Math.min(fit[0], previous[0]), Math.max(fit[1], previous[1])]
    }
    case 'follow': {
      if (!(policy.window > 0)) throw new WhiteboxError('value', 'limits: the follow window must be positive')
      const end = (extent?.[1] ?? policy.window) + (policy.lead ?? 0)
      return [end - policy.window, end]
    }
  }
}

function isRange(limits: Limits): limits is Range {
  return Array.isArray(limits)
}
