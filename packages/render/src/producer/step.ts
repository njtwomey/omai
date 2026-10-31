/**
 * Keeping several sessions in step: the head a shared cursor may reach (the slowest unfinished variant's, or the
 * furthest once every variant has ended), and advancing several sessions within one budget so that they stay in step.
 * Pure, and shared by the main-thread producer and the worker's.
 */

import type { RunStatus, SliceBudget } from 'mdml-engine/trace'

/**
 * The head of a comparison: the furthest step every variant has reached, so the shared cursor never shows a step a
 * variant has not computed. While some variants are unfinished it is the slowest of those (a variant that ended early
 * stands at its end and holds nothing back); once all have ended it is the furthest head.
 *
 * @param variants - Each variant's head and whether it has ended.
 * @returns The shared head (0 with no variants).
 */
export function sharedHead(variants: readonly { readonly head: number; readonly done: boolean }[]): number {
  if (variants.length === 0) return 0
  const open = variants.filter((v) => !v.done)
  return open.length > 0 ? Math.min(...open.map((v) => v.head)) : Math.max(...variants.map((v) => v.head))
}

/** What advancing in step needs of a session: the step reached, the status once ended, and `advance`. */
export interface Advancing {
  readonly t: number
  readonly status: RunStatus | undefined
  advance(budget: SliceBudget): unknown
}

/** Steps one variant takes before the next furthest behind has its turn, while computing ahead within a time budget. */
const STRIDE = 8

/**
 * Advance several sessions within one budget, keeping them in step. A step budget brings every unfinished session up
 * to the slowest one's head plus the steps (those ahead wait). A time budget goes round: the unfinished session
 * furthest behind takes a short slice, then whichever is furthest behind next, until the time is spent, so the heads
 * never differ by more than a slice and a dearer variant gets the time its steps cost. One session takes the whole
 * budget at once, as it would on its own.
 *
 * @param sessions - The sessions.
 * @param budget - Steps (default the rest of every run), or milliseconds with a clock.
 */
export function advanceInStep(sessions: readonly Advancing[], budget: SliceBudget): void {
  const open = sessions.filter((s) => s.status === undefined)
  if (open.length === 0) return
  const clock = budget.clock
  if (budget.ms === undefined || clock === undefined) {
    const steps = budget.steps ?? Number.POSITIVE_INFINITY
    const target = Math.min(...open.map((s) => s.t)) + steps
    for (const s of open) if (s.t < target) s.advance({ steps: target - s.t })
    return
  }
  const deadline = clock() + budget.ms
  for (;;) {
    const running = open.filter((s) => s.status === undefined)
    const [first] = running
    if (first === undefined) return
    const left = deadline - clock()
    if (left <= 0) return
    let behind = first
    for (const s of running) if (s.t < behind.t) behind = s
    const steps = running.length === 1 ? (budget.steps ?? Number.POSITIVE_INFINITY) : STRIDE
    behind.advance({ steps, ms: left, clock })
  }
}
