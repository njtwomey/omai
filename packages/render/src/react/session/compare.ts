/**
 * The pure part of comparing several runs of one method: the head a shared cursor may reach (the slowest unfinished
 * variant's, or the furthest once every variant has ended), advancing several sessions within one budget so that they
 * stay in step, the curves of several runs overlaid with one colour per run, and the trials of a sweep (a metric
 * against the knob logged at the same step, or against the step) that the sweep panel draws.
 */

import { metricSeries, type Run, type RunStatus, type SliceBudget } from 'mdml-engine/trace'
import { type Curve, lines } from '../../charts/lines.ts'
import type { Layer } from '../../charts/plot.ts'
import { categoricalColour } from '../../theme/palettes.ts'
import type { ColourMode } from '../../theme/tokens.ts'
import { metricCurves } from '../../trace/metrics.ts'

/** The last segment of a path (`model/loss` is `loss`). */
export const leaf = (path: string): string => path.split('/').at(-1) ?? path

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

const DASHES = ['solid', 'dashed', 'dotted'] as const

/**
 * The curves of several runs on one chart, one colour per run: each series is named by its run's label (and, with
 * several paths, the path's leaf, one dash per path), so the legend reads as the variants. One run is drawn as its
 * own metric curves, named by path.
 *
 * @param runs - The labelled runs.
 * @param paths - The scalar paths.
 * @param options - The colour mode.
 * @returns The layer.
 */
export function overlayCurves(
  runs: readonly { readonly label: string; readonly run: Run }[],
  paths: readonly string[],
  options: { readonly mode?: ColourMode } = {},
): Layer {
  const mode = options.mode ?? 'light'
  const [only] = runs
  if (only !== undefined && runs.length === 1) return metricCurves(only.run, paths, { mode })
  const curves: Curve[] = runs.flatMap((r, i) =>
    paths.map((path, j): Curve => {
      const s = metricSeries(r.run, path)
      return {
        x: s.steps,
        y: s.values,
        name: paths.length === 1 ? r.label : `${r.label} · ${leaf(path)}`,
        colour: categoricalColour(i),
        dash: DASHES[j % DASHES.length] ?? 'solid',
      }
    }),
  )
  return lines(curves, { mode })
}

/** The trials of a sweep: parallel arrays of the step each was logged at, its x (the knob or the step) and the metric. */
export interface SweepTrials {
  readonly steps: readonly number[]
  readonly x: readonly number[]
  readonly y: readonly number[]
}

/**
 * The trials of a sweep from a tape: each recorded value of a metric is a trial, placed at the value of the knob
 * logged at the same step (`over`), or at the step itself when no knob is named. A trial whose knob was not logged at
 * its step is left out.
 *
 * @param run - The tape.
 * @param path - The metric logged once per trial.
 * @param over - The scalar path of the knob each trial set.
 * @returns The trials.
 */
export function sweepTrials(run: Run, path: string, over?: string): SweepTrials {
  const metric = metricSeries(run, path)
  if (over === undefined) return { steps: metric.steps, x: metric.steps, y: metric.values }
  const knob = metricSeries(run, over)
  const at = new Map(knob.steps.map((s, i) => [s, knob.values[i]]))
  const steps: number[] = []
  const x: number[] = []
  const y: number[] = []
  metric.steps.forEach((s, i) => {
    const v = at.get(s)
    if (v === undefined) return
    steps.push(s)
    x.push(v)
    y.push(metric.values[i] ?? Number.NaN)
  })
  return { steps, x, y }
}
