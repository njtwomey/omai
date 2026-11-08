/**
 * The pure part of comparing several runs of one method: the head a shared cursor may reach (the slowest unfinished
 * variant's, or the furthest once every variant has ended), advancing several sessions within one budget so that they
 * stay in step, the curves of several runs overlaid with one colour per run, and the trials of a sweep (a metric
 * against the knob logged at the same step, or against the step) that the sweep panel draws.
 */

import { metricSeries, type Run } from 'omai-engine/trace'
import { type Curve, lines } from '../../charts/lines.ts'
import type { Layer } from '../../charts/plot.ts'
import { categoricalColour } from '../../theme/palettes.ts'
import type { ColourMode } from '../../theme/tokens.ts'
import { metricCurves } from '../../trace/metrics.ts'

/** The last segment of a path (`model/loss` is `loss`). */
export const leaf = (path: string): string => path.split('/').at(-1) ?? path

export { type Advancing, advanceInStep, sharedHead } from '../../producer/step.ts'

const DASHES = ['solid', 'dashed', 'dotted'] as const

/**
 * The curves of several runs on one chart, one colour per run: each series is named by its run's label (and, with
 * several paths, the path's leaf, one dash per path), so the legend reads as the variants. With one path the dash
 * follows the run instead, so colour is not the only thing that tells the runs apart (the default palette is not safe
 * for every colour-vision type past four colours). One run is drawn as its own metric curves, named by path.
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
        dash: DASHES[(paths.length === 1 ? i : j) % DASHES.length] ?? 'solid',
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
