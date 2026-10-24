import type { SlotContext } from './entry.ts'

/**
 * Slot props that colour a view's rows by the data: class labels as they are, and regression targets (a Swiss roll's
 * position along the roll) cut into equal-width bands, so an embedding shows which rows were neighbours on the
 * manifold.
 *
 * @param context - The slot's context.
 * @param bands - How many bands to cut targets into (default 8).
 * @returns `{ labels }` for the view, or nothing when the data has neither labels nor targets.
 * @example
 * ```ts
 * import { rowColours } from 'whitebox-hub'
 * rowColours({ targets: [0, 0.5, 1], params: {} }, 2).labels // => [0, 1, 1]
 * rowColours({ labels: [2, 0], params: {} }).labels // => [2, 0]
 * ```
 */
export function rowColours(context: SlotContext, bands = 8): { readonly labels?: readonly number[] } {
  if (context.labels !== undefined) return { labels: context.labels }
  const t = context.targets
  if (t === undefined || t.length === 0) return {}
  let lo = Number.POSITIVE_INFINITY
  let hi = Number.NEGATIVE_INFINITY
  for (const v of t) {
    if (v < lo) lo = v
    if (v > hi) hi = v
  }
  const width = hi > lo ? (hi - lo) / bands : 1
  return { labels: t.map((v) => Math.min(bands - 1, Math.floor((v - lo) / width))) }
}
