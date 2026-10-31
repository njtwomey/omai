/**
 * The curve families of the curves and sweep panels as pure layer builders: many runs faint with their mean in bold,
 * bands and fans between a lower and an upper path, a truth line, reference slopes on log-log axes, vertical marks for
 * the events of the tape, and the interval strip (many intervals as horizontal bars against a truth, the misses
 * marked). The panels pick which apply from the layout's options.
 */

import type { EChartsOption, SeriesOption } from 'echarts'
import { metricSeries, type Run } from 'mdml-engine/trace'
import { band, type Curve, lines } from '../../charts/lines.ts'
import { cartesian, type Layer, layer } from '../../charts/plot.ts'
import type { Figure } from '../../figure/index.ts'
import { unionRange } from '../../figure/limits.ts'
import { categoricalColour } from '../../theme/palettes.ts'
import type { ColourMode } from '../../theme/tokens.ts'
import { leaf } from './compare.ts'
import type { BandSpec, EventMarks, ReferenceSlope, TruthLine } from './layout.ts'
import { stepMarker } from './scrub.ts'

/** The curve-family options of a panel: the fields the layout's `curves` and `sweep` panels share, whichever apply. */
export interface FamilyOptions {
  readonly runs?: boolean
  readonly bands?: readonly BandSpec[]
  readonly truth?: TruthLine
  readonly log?: 'x' | 'y' | 'xy'
  readonly slopes?: readonly ReferenceSlope[]
  readonly events?: readonly EventMarks[]
}

/** Whether a panel asks for any curve family beyond plain overlaid curves. */
export function hasFamily(options: FamilyOptions): boolean {
  return (
    options.runs === true ||
    (options.bands?.length ?? 0) > 0 ||
    options.truth !== undefined ||
    options.log !== undefined ||
    (options.slopes?.length ?? 0) > 0 ||
    (options.events?.length ?? 0) > 0
  )
}

/** The axes a `log` option asks to be logarithmic. */
export function logAxes(log: FamilyOptions['log']): { readonly x: boolean; readonly y: boolean } {
  return { x: log === 'x' || log === 'xy', y: log === 'y' || log === 'xy' }
}

/** A run with its label (the variant's), as the family builders take it. */
export interface LabelledRun {
  readonly label: string
  readonly run: Run
}

// The points of a series that a log axis can show.
function usable(
  steps: readonly number[],
  values: readonly number[],
  log: { readonly x: boolean; readonly y: boolean },
): { x: number[]; y: number[] } {
  const x: number[] = []
  const y: number[] = []
  steps.forEach((s, i) => {
    const v = values[i] ?? Number.NaN
    if ((log.x && !(s > 0)) || (log.y && !(v > 0))) return
    x.push(s)
    y.push(v)
  })
  return { x, y }
}

// The mean over runs at each step any run recorded (runs that lack a step are left out of that step's mean).
function meanSeries(series: readonly { readonly steps: readonly number[]; readonly values: readonly number[] }[]): {
  steps: number[]
  values: number[]
} {
  const sums = new Map<number, [number, number]>()
  for (const s of series)
    s.steps.forEach((step, i) => {
      const v = s.values[i]
      if (v === undefined || !Number.isFinite(v)) return
      const at = sums.get(step) ?? [0, 0]
      sums.set(step, [at[0] + v, at[1] + 1])
    })
  const steps = [...sums.keys()].sort((a, b) => a - b)
  return { steps, values: steps.map((s) => (sums.get(s)?.[0] ?? 0) / (sums.get(s)?.[1] ?? 1)) }
}

const byStep = (s: { readonly steps: readonly number[]; readonly values: readonly number[] }): Map<number, number> =>
  new Map(s.steps.map((step, i) => [step, s.values[i] ?? Number.NaN]))

const DASHES = ['solid', 'dashed', 'dotted'] as const

/**
 * The layers of a curves panel with family options: the curves (every run faint and the mean bold under `runs`; one
 * colour per run otherwise, as plain overlaid curves), bands beneath, then the truth line, reference slopes and event
 * marks. The first layer is the curves; the others follow in drawing order.
 *
 * @param runs - The labelled runs (the variants of a comparison).
 * @param paths - The scalar paths.
 * @param options - The family options and the colour mode.
 * @returns The layers.
 */
export function curveFamily(
  runs: readonly LabelledRun[],
  paths: readonly string[],
  options: FamilyOptions & { readonly mode: ColourMode },
): Layer[] {
  const { mode } = options
  const log = logAxes(options.log)
  const compared = runs.length > 1
  const curves: Curve[] = []
  if (options.runs === true && compared) {
    paths.forEach((path, j) => {
      const colour = categoricalColour(j)
      const all = runs.map((r) => metricSeries(r.run, path))
      all.forEach((s, i) => {
        const { x, y } = usable(s.steps, s.values, log)
        curves.push({ x, y, name: `__run${j}-${i}`, colour, width: 1, opacity: 0.3 })
      })
      const mean = meanSeries(all)
      const { x, y } = usable(mean.steps, mean.values, log)
      curves.push({ x, y, name: paths.length === 1 ? 'mean' : `mean · ${leaf(path)}`, colour, width: 3 })
    })
  } else {
    for (const [i, r] of runs.entries())
      for (const [j, path] of paths.entries()) {
        const s = metricSeries(r.run, path)
        const { x, y } = usable(s.steps, s.values, log)
        curves.push({
          x,
          y,
          name: !compared ? leaf(path) : paths.length === 1 ? r.label : `${r.label} · ${leaf(path)}`,
          colour: categoricalColour(compared ? i : j),
          dash: compared ? (DASHES[j % DASHES.length] ?? 'solid') : 'solid',
        })
      }
  }
  const main = lines(curves, { mode })
  const under: Layer[] = []
  for (const [i, r] of runs.entries())
    for (const [k, b] of (options.bands ?? []).entries()) {
      const at = paths.indexOf(b.path)
      const centre = metricSeries(r.run, b.path)
      const lower = byStep(metricSeries(r.run, b.lower))
      const upper = byStep(metricSeries(r.run, b.upper))
      const x: number[] = []
      const lo: number[] = []
      const hi: number[] = []
      for (const s of centre.steps) {
        const l = lower.get(s)
        const u = upper.get(s)
        if (l === undefined || u === undefined || (log.x && !(s > 0)) || (log.y && !(l > 0))) continue
        x.push(s)
        lo.push(l)
        hi.push(u)
      }
      const colour = categoricalColour(options.runs === true || !compared ? (at < 0 ? k : at) : i)
      under.push(band([{ x, lower: lo, upper: hi, colour, opacity: 0.18, name: `__band${i}-${k}` }], { mode }))
    }
  const layers: Layer[] = [main, ...under]
  const xr = unionRange(layers.map((l) => l.extent.x))
  const yr = unionRange(layers.map((l) => l.extent.y))
  const over: Layer[] = []
  const truth = options.truth
  if (truth !== undefined) {
    const label = truth.label ?? 'truth'
    if (truth.value !== undefined && xr !== undefined)
      over.push(
        lines([{ x: xr, y: [truth.value, truth.value], name: label, colour: 'fg', width: 1.5, dash: 'dashed' }], {
          mode,
        }),
      )
    else if (truth.path !== undefined) {
      const s = metricSeries(runs[0]?.run ?? { schema: 'mdml.run/1', events: [] }, truth.path)
      const { x, y } = usable(s.steps, s.values, log)
      over.push(lines([{ x, y, name: label, colour: 'fg', width: 1.5, dash: 'dashed' }], { mode }))
    }
  }
  const first = curves.find((c) => (c.name ?? '').startsWith('__run') === false) ?? curves[0]
  over.push(...slopeLayers(options.slopes, first, xr, mode))
  for (const e of options.events ?? []) {
    const stepsAt = runs[0] === undefined ? [] : eventSteps(runs[0].run, e.path)
    if (stepsAt.length === 0 || yr === undefined) continue
    const [lo, hi] = yr
    over.push(
      lines(
        [
          {
            x: stepsAt.flatMap((s) => [s, s, s]),
            y: stepsAt.flatMap(() => [lo, hi, Number.NaN]),
            name: e.label ?? leaf(e.path),
            colour: 'danger',
            width: 1.2,
            dash: 'dashed',
            opacity: 0.8,
          },
        ],
        { mode },
      ),
    )
  }
  return [...layers, ...over]
}

/** The steps at which a scalar path was logged with a finite non-zero value (an event that fired). */
export function eventSteps(run: Run, path: string): number[] {
  const s = metricSeries(run, path)
  return s.steps.filter((_, i) => {
    const v = s.values[i]
    return v !== undefined && Number.isFinite(v) && v !== 0
  })
}

/**
 * Reference slopes for log-log axes: for each, the line `y = y0 (x / x0)^slope` through the first point of the first
 * curve (or the given `through`), across the x range, dotted and muted.
 *
 * @param slopes - The slopes.
 * @param curve - The curve whose first point anchors them.
 * @param xr - The x range to cover.
 * @param mode - The colour mode.
 * @returns One layer per slope.
 */
export function slopeLayers(
  slopes: readonly ReferenceSlope[] | undefined,
  curve: { readonly x?: unknown; readonly y?: unknown } | undefined,
  xr: readonly [number, number] | undefined,
  mode: ColourMode,
): Layer[] {
  if (slopes === undefined || xr === undefined) return []
  const xs = (curve?.x as readonly number[] | undefined) ?? []
  const ys = (curve?.y as readonly number[] | undefined) ?? []
  const k = xs.findIndex((x, i) => x > 0 && (ys[i] ?? 0) > 0)
  return slopes.flatMap((s) => {
    const [x0, y0] = s.through ?? (k < 0 ? [undefined, undefined] : [xs[k], ys[k]])
    if (x0 === undefined || y0 === undefined || !(xr[0] > 0)) return []
    const x = [xr[0], xr[1]]
    return [
      lines(
        [
          {
            x,
            y: x.map((v) => y0 * (v / x0) ** s.slope),
            name: s.label ?? `slope ${s.slope}`,
            colour: 'muted',
            width: 1.5,
            dash: 'dotted',
          },
        ],
        { mode },
      ),
    ]
  })
}

/**
 * The layers of a sweep with a truth line and reference slopes: the line across the x range of the trials.
 *
 * @param trials - The trials' x and y (as drawn).
 * @param options - The truth, the slopes, the truth path's series if any, and the colour mode.
 * @returns The layers.
 */
export function sweepFamily(
  trials: { readonly x: readonly number[]; readonly y: readonly number[] },
  options: {
    readonly truth?: TruthLine
    readonly truthSeries?: { readonly steps: readonly number[]; readonly values: readonly number[] }
    readonly slopes?: readonly ReferenceSlope[]
    readonly mode: ColourMode
  },
): Layer[] {
  const finite = trials.x.filter(Number.isFinite)
  if (finite.length === 0) return []
  const xr: [number, number] = [Math.min(...finite), Math.max(...finite)]
  const out: Layer[] = []
  const { truth, mode } = options
  if (truth?.value !== undefined)
    out.push(
      lines(
        [
          {
            x: xr,
            y: [truth.value, truth.value],
            name: truth.label ?? 'truth',
            colour: 'fg',
            width: 1.5,
            dash: 'dashed',
          },
        ],
        {
          mode,
        },
      ),
    )
  out.push(...slopeLayers(options.slopes, trials, xr, mode))
  return out
}

/**
 * The layers of an interval strip: trial k is row k (the first at the top), its interval a horizontal bar and its
 * estimate a dot, the truth a vertical dashed line; bars that miss the truth are red, and trials after the viewed step
 * are faint. The covered and the missed are counted in their legend names.
 *
 * @param input - The trials' steps, estimates and bounds, the truth (a constant or one per trial), the viewed step.
 * @param mode - The colour mode.
 * @returns The layers.
 */
export function intervalStrip(
  input: {
    readonly steps: readonly number[]
    readonly centre: readonly number[]
    readonly lower: readonly number[]
    readonly upper: readonly number[]
    readonly truth?: number | readonly number[]
    readonly t: number
  },
  mode: ColourMode,
): Layer[] {
  const n = input.steps.length
  const truthAt = (k: number): number | undefined => (typeof input.truth === 'number' ? input.truth : input.truth?.[k])
  const missed = (k: number): boolean => {
    const v = truthAt(k)
    return v !== undefined && ((input.lower[k] ?? 0) > v || (input.upper[k] ?? 0) < v)
  }
  const group = (keep: (k: number) => boolean): number[] => Array.from({ length: n }, (_, k) => k).filter(keep)
  const seen = (k: number): boolean => (input.steps[k] ?? 0) <= input.t
  const bars = (ks: number[], name: string, colour: string, opacity: number): Curve => ({
    x: ks.flatMap((k) => [input.lower[k] ?? 0, input.upper[k] ?? 0, input.upper[k] ?? 0]),
    y: ks.flatMap((k) => [k, k, Number.NaN]),
    name,
    colour,
    width: 2.5,
    opacity,
  })
  const covered = group((k) => seen(k) && !missed(k))
  const misses = group((k) => seen(k) && missed(k))
  const ahead = group((k) => !seen(k))
  const curves: Curve[] = [
    bars(covered, `covered ${covered.length}`, categoricalColour(0), 0.9),
    bars(misses, `missed ${misses.length}`, categoricalColour(3), 0.95),
    bars(ahead, '__ahead', categoricalColour(0), 0.2),
  ]
  const out: Layer[] = [lines(curves, { mode })]
  const truths = Array.from({ length: n }, (_, k) => truthAt(k))
  if (typeof input.truth === 'number')
    out.push(
      lines([{ x: [input.truth, input.truth], y: [-1, n], name: 'truth', colour: 'fg', width: 1.5, dash: 'dashed' }], {
        mode,
      }),
    )
  else if (truths.some((v) => v !== undefined))
    out.push(
      layer(
        [
          {
            type: 'scatter',
            name: 'truth',
            data: truths.flatMap((v, k) => (v === undefined ? [] : [[v, k]])),
            symbol: 'diamond',
            symbolSize: 7,
            color: categoricalColour(1),
            clip: true,
          } as SeriesOption,
        ],
        {},
      ),
    )
  out.push(
    layer(
      [
        {
          type: 'scatter',
          name: '__centres',
          data: group(seen).map((k) => [input.centre[k] ?? 0, k]),
          symbolSize: 4,
          color: categoricalColour(0),
          itemStyle: { color: categoricalColour(0) },
          clip: true,
        } as SeriesOption,
      ],
      { y: [-1, n] },
    ),
  )
  return out
}

/**
 * The chart of a curves panel with family options at a viewed step: the family's layers and the step marker spanning
 * all of them, on log axes where asked.
 *
 * @param runs - The labelled runs.
 * @param paths - The scalar paths.
 * @param options - The family options, the viewed step, the colour mode and the figure.
 * @returns The chart option.
 */
export function familyOption(
  runs: readonly LabelledRun[],
  paths: readonly string[],
  options: FamilyOptions & { readonly t: number; readonly mode: ColourMode; readonly figure: Figure },
): EChartsOption {
  const { mode, t } = options
  const log = logAxes(options.log)
  const layers = curveFamily(runs, paths, options)
  const head = layers[0]
  const y = unionRange(layers.map((l) => l.extent.y))
  const marker =
    head === undefined || (log.x && !(t > 0))
      ? []
      : stepMarker({ ...head, extent: { ...head.extent, ...(y === undefined ? {} : { y }) } }, t, mode)
  return cartesian([...layers, ...marker], {
    mode,
    figure: options.figure,
    x: { label: 'step', ...(log.x ? { scale: 'log' as const } : {}) },
    ...(log.y ? { y: { scale: 'log' as const } } : {}),
    legend: true,
  })
}

/**
 * The chart of an interval strip at a viewed step.
 *
 * @param input - The trials and the viewed step (see {@link intervalStrip}).
 * @param options - The colour mode, the figure and the axis label.
 * @returns The chart option.
 */
export function intervalOption(
  input: Parameters<typeof intervalStrip>[0],
  options: { readonly mode: ColourMode; readonly figure: Figure; readonly label?: string },
): EChartsOption {
  return {
    ...cartesian(intervalStrip(input, options.mode), {
      mode: options.mode,
      figure: options.figure,
      x: { label: options.label ?? 'estimate' },
      y: { label: 'trial', inverse: true },
      legend: true,
    }),
    animation: false,
  }
}
