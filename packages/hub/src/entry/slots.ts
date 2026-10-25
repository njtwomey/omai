/**
 * Shared slot props for hub views: `rowColours` colours a view's points by the data (class labels as categories,
 * regression targets on a colour ramp with a colour bar), so an embedding or projection shows where each row came from.
 */

import type { SlotContext } from './entry.ts'

/**
 * Slot props that colour a view's rows by the data: class labels as they are, and regression targets (a Swiss roll's
 * position along the roll) as values on a colour ramp with a colour bar, so an embedding shows which rows were
 * neighbours on the manifold.
 *
 * @param context - The slot's context.
 * @returns `{ labels }` or `{ values }` for the view, or nothing when the data has neither labels nor targets.
 * @example
 * ```ts
 * import { rowColours } from 'whitebox-hub'
 * rowColours({ targets: [0, 0.5, 1], params: {} }).values // => [0, 0.5, 1]
 * rowColours({ labels: [2, 0], params: {} }).labels // => [2, 0]
 * ```
 */
export function rowColours(context: SlotContext): {
  readonly labels?: readonly number[]
  readonly values?: readonly number[]
} {
  if (context.labels !== undefined) return { labels: context.labels }
  const t = context.targets
  if (t === undefined || t.length === 0) return {}
  return { values: t }
}
