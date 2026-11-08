/**
 * Views of how far predictions can be trusted: reliability (calibration) diagrams with each curve's expected
 * calibration error and the forecasts per bin, and conformal prediction (intervals with the truth, or the sizes of
 * prediction sets) with its calibration scores and the coverage as data arrive.
 */

import type { EChartsOption } from 'echarts'
import { maxOf, minOf } from 'omai-engine/base'
import { type CalibrationRecord, type ConformalRecord, setSizes } from 'omai-engine/exposition'
import { bars, histogram } from '../charts/bars.ts'
import { type Panel, subplots } from '../charts/compose.ts'
import { referenceLines } from '../charts/functions.ts'
import { band, lines } from '../charts/lines.ts'
import { cartesian, type Layer } from '../charts/plot.ts'
import { points } from '../charts/points.ts'
import { gridSpec } from '../figure/grid.ts'
import type { ColourMode } from '../theme/tokens.ts'
import { short, type ViewFrames, type ViewOptions, viewFigure } from './_view.ts'

/** The colour of each reliability curve: by its group when it has one (groups share a colour), else by its index. */
function curveColours(record: CalibrationRecord): string[] {
  const groups = [...new Set(record.curves.flatMap((c) => (c.group === undefined ? [] : [c.group])))]
  return record.curves.map((c, i) => `tab${(c.group === undefined ? i : groups.indexOf(c.group)) % 10}`)
}

/**
 * A reliability diagram: each curve's observed frequency against its mean forecast per bin, with the diagonal of
 * perfect calibration and each curve's expected calibration error in its name; beneath, how many forecasts fell in
 * each bin.
 *
 * @param record - The record.
 * @param options - Mode and figure.
 * @returns The ECharts option.
 * @example
 * ```ts
 * import { calibrationRecord } from 'omai-engine/exposition'
 * import { reliabilityView } from 'omai-render/views'
 * const r = calibrationRecord({ nBins: 2, strategy: 'uniform', curves: [{ name: 'model', meanPredicted: [0.2, 0.8], observed: [0.1, 0.9], counts: [5, 5], ece: 0.1, mce: 0.1 }] })
 * reliabilityView(r).grid.length // => 2
 * ```
 */
export function reliabilityView(record: CalibrationRecord, options: ViewOptions = {}): EChartsOption {
  const fig = viewFigure(options, { aspect: 1.1 })
  const mode = fig.mode
  const colours = curveColours(record)
  const dashes = new Map<string, number>()
  const dashOf = (group: string | undefined): 'solid' | 'dashed' | 'dotted' => {
    if (group === undefined) return 'solid'
    const seen = dashes.get(group) ?? 0
    dashes.set(group, seen + 1)
    return seen === 0 ? 'solid' : seen === 1 ? 'dashed' : 'dotted'
  }
  const named = record.curves.map((c) => `${c.name} (ECE ${short(c.ece)})`)
  const styles = record.curves.map((c) => dashOf(c.group))
  const diagram: Panel = {
    layers: [
      lines([{ x: [0, 1], y: [0, 1], name: '__perfect', colour: 'muted', dash: 'dotted' }], { mode }),
      lines(
        record.curves.map((c, i) => ({
          x: c.meanPredicted,
          y: c.observed,
          name: named[i] ?? c.name,
          colour: colours[i] ?? 'tab0',
          dash: styles[i] ?? 'solid',
          markers: true,
        })),
        { mode },
      ),
    ],
    options: {
      x: { label: 'mean forecast', limits: [0, 1] },
      y: { label: 'observed frequency', limits: [0, 1] },
    },
  }
  const counts: Panel = {
    layers: [
      lines(
        record.curves.map((c, i) => ({
          x: c.meanPredicted,
          y: c.counts,
          name: `__counts${i}`,
          colour: colours[i] ?? 'tab0',
          dash: styles[i] ?? 'solid',
          markers: true,
        })),
        { mode },
      ),
    ],
    options: { x: { label: 'mean forecast', limits: [0, 1] }, y: { label: 'forecasts' } },
  }
  const grid = gridSpec(2, 1, {
    heightRatios: [3, 1.1],
    hspace: 64,
    cells: [
      { id: 'diagram', row: 0, col: 0 },
      { id: 'counts', row: 1, col: 0 },
    ],
  })
  return subplots(grid, { diagram, counts }, { figure: fig, tooltip: 'item' })
}

/** Options of {@link conformalView}. */
export interface ConformalViewOptions extends ViewOptions {
  /** Show the stream only up to this time (a frame of the player). */
  readonly upTo?: number
}

function intervalPanel(r: ConformalRecord, mode: ColourMode, upTo: number): Panel {
  const x = r.x ?? []
  const keep = x.flatMap((v, i) => (v <= upTo ? [i] : []))
  const pick = (v: readonly number[] | undefined): number[] => keep.map((i) => v?.[i] ?? Number.NaN)
  const xs = pick(x)
  const y = pick(r.y)
  const lo = pick(r.lower)
  const hi = pick(r.upper)
  const outside = keep.filter((i) => {
    const v = r.y?.[i] ?? 0
    return v < (r.lower?.[i] ?? 0) || v > (r.upper?.[i] ?? 0)
  })
  const layers: Layer[] = [
    band([{ x: xs, lower: lo, upper: hi, name: '__interval', colour: 'tab0' }], {
      mode,
    }),
  ]
  if (r.prediction !== undefined)
    layers.push(lines([{ x: xs, y: pick(r.prediction), name: '__prediction', colour: 'tab0', width: 1.6 }], { mode }))
  if (r.y !== undefined) {
    layers.push(points([{ x: xs, y, name: '__truth', colour: 'fgSecondary', size: 4, opacity: 0.7 }], { mode }))
    layers.push(
      points(
        [
          {
            x: outside.map((i) => x[i] ?? 0),
            y: outside.map((i) => r.y?.[i] ?? 0),
            name: '__outside',
            colour: 'danger',
            size: 6,
          },
        ],
        { mode },
      ),
    )
  }
  const covered = keep.length - outside.length
  return {
    layers,
    options: {
      title: `${covered}/${keep.length} inside, nominal ${short(100 * (1 - r.alpha))}%`,
      x: { label: r.xLabel ?? 'x', limits: [minOf(x), maxOf(x)] },
      y: { label: 'y' },
    },
  }
}

function setPanel(r: ConformalRecord, mode: ColourMode): Panel {
  const s = setSizes(r)
  return {
    layers: [
      bars(
        s.counts.map((_, k) => String(k)),
        [{ name: 'rows', values: s.counts, colour: 'tab0' }],
        { mode },
      ),
    ],
    options: {
      title: `${r.method}: coverage ${short(r.coverage)} (nominal ${short(1 - r.alpha)})`,
      x: { label: 'set size' },
      y: { label: 'rows' },
    },
  }
}

function scoresPanel(r: ConformalRecord, mode: ColourMode): Panel | undefined {
  if (r.scores === undefined || r.scores.length === 0) return undefined
  const layers: Layer[] = [histogram(r.scores, { bins: 30, name: '__scores', colour: 'tab1', mode })]
  if (r.quantile !== undefined && Number.isFinite(r.quantile))
    layers.push(referenceLines({ x: [r.quantile] }, { mode, colour: 'danger' }))
  return {
    layers,
    options: { title: 'scores and quantile', x: { label: 'conformity score' }, y: { label: 'rows' } },
  }
}

function streamPanel(r: ConformalRecord, mode: ColourMode, upTo: number): Panel | undefined {
  const s = r.stream
  if (s === undefined) return undefined
  const keep = s.t.flatMap((v, i) => (v <= upTo ? [i] : []))
  const t = keep.map((i) => s.t[i] ?? 0)
  const curves = [{ x: t, y: keep.map((i) => s.coverage[i] ?? 0), name: '__coverage', colour: 'tab0' }]
  if (s.baseline !== undefined)
    curves.push({ x: t, y: keep.map((i) => s.baseline?.[i] ?? 0), name: '__baseline', colour: 'tab1' })
  return {
    layers: [
      lines(curves, { mode }),
      referenceLines(
        { y: [{ value: 1 - r.alpha, label: 'nominal' }] },
        { mode, colour: 'fgSecondary', dash: 'dashed' },
      ),
    ],
    options: {
      title: s.baseline === undefined ? 'running coverage' : 'running coverage (orange: fixed)',
      x: { label: r.xLabel ?? 'time', limits: [minOf(s.t), maxOf(s.t)] },
      y: { label: 'coverage', limits: [Math.max(0, 1 - 4 * r.alpha - 0.05), 1] },
    },
  }
}

/**
 * Conformal prediction as one figure: for regression the intervals along the input with the truth (rows outside their
 * interval marked), for classification the distribution of set sizes; the calibration scores with their quantile; and
 * with a stream, the running coverage against the nominal level (and a predictor that does not adapt).
 *
 * @param record - The record.
 * @param options - Mode, figure, and the time to show the stream up to.
 * @returns The ECharts option.
 * @example
 * ```ts
 * import { conformalRecord } from 'omai-engine/exposition'
 * import { conformalView } from 'omai-render/views'
 * const r = conformalRecord({ task: 'regression', method: 'split', alpha: 0.1, coverage: 0.5, x: [0, 1], y: [1, 5], lower: [0, 1], upper: [2, 3], scores: [0.5, 1] })
 * conformalView(r).grid.length // => 2
 * ```
 */
export function conformalView(record: ConformalRecord, options: ConformalViewOptions = {}): EChartsOption {
  const upTo = options.upTo ?? Number.POSITIVE_INFINITY
  const fig = viewFigure(options, { aspect: 1.75 })
  const mode = fig.mode
  const panels: Record<string, Panel> = {
    main: record.task === 'regression' ? intervalPanel(record, mode, upTo) : setPanel(record, mode),
  }
  const scores = scoresPanel(record, mode)
  const stream = streamPanel(record, mode, upTo)
  if (scores !== undefined) panels['scores'] = scores
  if (stream !== undefined) panels['stream'] = stream
  const side = ['stream', 'scores'].filter((id) => panels[id] !== undefined)
  if (side.length === 0) {
    const p = panels['main'] as Panel
    return cartesian(p.layers, { ...p.options, figure: fig, mode, title: p.options?.title ?? '' })
  }
  const grid = gridSpec(side.length, 2, {
    widthRatios: [1.5, 1],
    wspace: 64,
    hspace: 64,
    cells: [{ id: 'main', row: 0, col: 0, rowSpan: side.length }, ...side.map((id, r) => ({ id, row: r, col: 1 }))],
  })
  return subplots(grid, panels, { figure: fig, tooltip: 'item', legend: false })
}

/**
 * A conformal record with a stream as frames: the intervals and the running coverage revealed as data arrive.
 *
 * @param record - The record (frames only when it has a stream).
 * @param options - Mode and figure.
 * @returns The frames.
 */
export function conformalFrames(record: ConformalRecord, options: ViewOptions = {}): ViewFrames {
  const t = record.stream?.t ?? []
  const count = Math.min(80, t.length)
  const at = (f: number): number =>
    count <= 1 ? Number.POSITIVE_INFINITY : (t[Math.round((f * (t.length - 1)) / (count - 1))] ?? 0)
  return {
    count: Math.max(1, count),
    label: (f) => (count <= 1 ? '' : `${record.xLabel ?? 't'} ${short(at(f))}`),
    option: (f) => conformalView(record, { ...options, upTo: at(f) }),
  }
}
