/**
 * Player frames for `Epicycles`: a closed curve drawn by a chain of rotating arrows, frame by frame round one cycle.
 * The main panel shows the target, the circles the largest arrows ride on, the chain and the traced partial sum; an
 * optional panel follows the pen at a magnification, so the small fast arrows near the tip show; the spectrum beneath
 * marks the kept and dropped coefficients on a log scale.
 */

import type { EChartsOption } from 'echarts'
import { maxOf, minOf } from 'omai-engine/base'
import type { Epicycles } from 'omai-engine/exposition'
import { type Panel, subplots } from '../charts/compose.ts'
import { lines } from '../charts/lines.ts'
import { points } from '../charts/points.ts'
import { gridSpec } from '../figure/grid.ts'
import { formatNumber } from '../theme/format.ts'
import { type ViewFrames, type ViewOptions, viewFigure } from './_view.ts'

/** Options of {@link epicycleFrames}. */
export interface EpicycleViewOptions extends ViewOptions {
  /** Draw the target curve (default true). */
  readonly target?: boolean
  /** Draw the circles (default true). */
  readonly circles?: boolean
}

// The half-width of the main view: outlines are centred with unit radius.
const RANGE = 1.25
// Circles drawn in the main view (the largest), and the smallest radius worth a circle.
const MAX_CIRCLES = 80
const MIN_RADIUS = 0.004
// Points per circle, per arc in the zoomed view, and on the exact stretch of trace near the pen.
const CIRCLE_POINTS = 64
const ARC_POINTS = 48
const ZOOM_TRACE = 200

/**
 * The figure aspect (width over height) that fits an epicycles record: square panels over a spectrum strip.
 *
 * @param record - The record.
 * @returns The aspect.
 */
export function epicycleAspect(record: Epicycles): number {
  return record.zoom === undefined ? 1 / 1.4 : 2 / 1.5
}

// The joints of the chain at time t: the centroid, then the running sums of the arrows.
function joints(r: Epicycles, t: number): { x: number[]; y: number[] } {
  let x = r.centroid[0]
  let y = r.centroid[1]
  const xs = [x]
  const ys = [y]
  for (let i = 0; i < r.chain.k.length; i++) {
    const a = 2 * Math.PI * (r.chain.k[i] ?? 0) * t
    const c = Math.cos(a)
    const s = Math.sin(a)
    const re = r.chain.re[i] ?? 0
    const im = r.chain.im[i] ?? 0
    x += re * c - im * s
    y += re * s + im * c
    xs.push(x)
    ys.push(y)
  }
  return { x: xs, y: ys }
}

// The pen's position at time s, from every arrow.
function penAt(r: Epicycles, s: number): [number, number] {
  let x = r.centroid[0]
  let y = r.centroid[1]
  for (let i = 0; i < r.chain.k.length; i++) {
    const a = 2 * Math.PI * (r.chain.k[i] ?? 0) * s
    const re = r.chain.re[i] ?? 0
    const im = r.chain.im[i] ?? 0
    x += re * Math.cos(a) - im * Math.sin(a)
    y += re * Math.sin(a) + im * Math.cos(a)
  }
  return [x, y]
}

/**
 * An epicycles record as frames for a player: frame j shows the chain at t = j / steps, the curve traced so far, the
 * zoom on the pen when the record has one, and the spectrum.
 *
 * @param record - The record.
 * @param options - Mode, figure, and whether to draw the target and the circles.
 * @returns The frames.
 */
export function epicycleFrames(record: Epicycles, options: EpicycleViewOptions = {}): ViewFrames {
  const fig = viewFigure(options, { aspect: epicycleAspect(record) })
  const mode = fig.mode
  const steps = record.steps
  const radius = record.chain.re.map((re, i) => Math.hypot(re, record.chain.im[i] ?? 0))
  const circled = radius
    .map((_, i) => i)
    .sort((a, b) => (radius[b] ?? 0) - (radius[a] ?? 0))
    .slice(0, MAX_CIRCLES)
    .filter((i) => (radius[i] ?? 0) > MIN_RADIUS)
  const fine = record.trace.x.length - 1
  const closedTarget = {
    x: [...record.target.x, record.target.x[0] ?? 0],
    y: [...record.target.y, record.target.y[0] ?? 0],
  }
  const showTarget = options.target !== false
  const showCircles = options.circles !== false
  const zoom = record.zoom
  const half = zoom === undefined ? RANGE : RANGE / zoom
  const angular = record.chain.k.map((k) => 2 * Math.PI * k)
  // The spectrum: kept and dropped coefficients, on a log scale from five decades below the largest.
  const mags = record.spectrum.magnitude
  const topMag = 10 ** Math.ceil(Math.log10(Math.max(1e-12, maxOf(mags))))
  const kmax = Math.min(
    2 ** Math.ceil(Math.log2(Math.max(32, maxOf(record.spectrum.k.map(Math.abs))))),
    Math.max(32, Math.floor(record.target.x.length / 2)),
  )
  const pick = (kept: boolean): { x: number[]; y: number[] } => {
    const x: number[] = []
    const y: number[] = []
    for (const [i, k] of record.spectrum.k.entries())
      if ((record.spectrum.kept[i] ?? false) === kept) {
        x.push(k)
        y.push(mags[i] ?? 0)
      }
    return { x, y }
  }
  const on = pick(true)
  const off = pick(false)
  const cutoff = minOf(on.y)
  const spectrum: Panel = {
    layers: [
      points([{ x: off.x, y: off.y, name: 'dropped', colour: 'fgSecondary', size: 3, opacity: 0.5 }], { mode }),
      points([{ x: on.x, y: on.y, name: 'kept', colour: 'accent', size: 5 }], { mode }),
      lines(
        [{ x: [-kmax, kmax], y: [cutoff, cutoff], name: 'cut-off', colour: 'fgSecondary', dash: 'dashed', width: 1 }],
        {
          mode,
        },
      ),
    ],
    options: {
      x: { label: 'frequency k (turns per cycle)', limits: [-kmax, kmax] },
      y: { label: '|c_k|', scale: 'log', limits: [topMag * 1e-5, topMag], format: 'power' },
    },
  }
  const option = (frame: number): EChartsOption => {
    const t = Math.min(Math.max(frame, 0), steps) / steps
    const j = joints(record, t)
    const pen: [number, number] = [j.x.at(-1) ?? 0, j.y.at(-1) ?? 0]
    const ring = { x: [] as number[], y: [] as number[] }
    if (showCircles)
      for (const i of circled) {
        const r = radius[i] ?? 0
        for (let a = 0; a <= CIRCLE_POINTS; a++) {
          ring.x.push((j.x[i] ?? 0) + r * Math.cos((2 * Math.PI * a) / CIRCLE_POINTS))
          ring.y.push((j.y[i] ?? 0) + r * Math.sin((2 * Math.PI * a) / CIRCLE_POINTS))
        }
        ring.x.push(Number.NaN)
        ring.y.push(Number.NaN)
      }
    const upto = Math.round(t * fine) + 1
    const traced = { x: record.trace.x.slice(0, upto), y: record.trace.y.slice(0, upto) }
    const mainLayers = [
      ...(showTarget
        ? [
            lines(
              [{ ...closedTarget, name: 'target', colour: 'fgSecondary', dash: 'dashed', width: 1.5, opacity: 0.6 }],
              { mode },
            ),
          ]
        : []),
      ...(showCircles
        ? [lines([{ ...ring, name: 'circles', colour: 'accent', width: 0.75, opacity: 0.45 }], { mode })]
        : []),
      lines([{ ...j, name: 'arrows', colour: 'fg', width: 1 }], { mode }),
      lines([{ ...traced, name: 'traced', colour: 'accent', width: 2 }], { mode }),
      points([{ x: [pen[0]], y: [pen[1]], name: 'pen', colour: 'accent', size: 8 }], { mode }),
      ...(zoom === undefined
        ? []
        : [
            lines(
              [
                {
                  x: [pen[0] - half, pen[0] + half, pen[0] + half, pen[0] - half, pen[0] - half],
                  y: [pen[1] - half, pen[1] - half, pen[1] + half, pen[1] + half, pen[1] - half],
                  name: 'zoom window',
                  colour: 'fg',
                  dash: 'dashed',
                  width: 1,
                },
              ],
              { mode },
            ),
          ]),
    ]
    const main: Panel = {
      layers: mainLayers,
      options: {
        x: { label: 'Re z', limits: [-RANGE, RANGE] },
        y: { label: 'Im z', limits: [-RANGE, RANGE] },
        aspect: 'equal',
      },
    }
    if (zoom === undefined) {
      const grid = gridSpec(2, 1, {
        heightRatios: [1, 0.4],
        hspace: 56,
        cells: [
          { id: 'main', row: 0, col: 0 },
          { id: 'spectrum', row: 1, col: 0 },
        ],
      })
      return subplots(grid, { main, spectrum }, { figure: fig, legend: false, tooltip: false })
    }
    // The zoomed view, relative to the pen: every arrow whose circle crosses the window, as an arc fine enough to
    // stay round, the arrows themselves, and the trace near the pen evaluated exactly.
    const arcs = { x: [] as number[], y: [] as number[] }
    for (const [i, r] of radius.entries()) {
      if (r < half / 200) continue
      const ox = (j.x[i] ?? 0) - pen[0]
      const oy = (j.y[i] ?? 0) - pen[1]
      if (Math.abs(Math.hypot(ox, oy) - r) > 1.5 * half) continue
      const span = r < 1.5 * half ? Math.PI : Math.min(Math.PI, (3 * half) / r)
      const mid = Math.atan2(-oy, -ox)
      for (let a = 0; a <= ARC_POINTS; a++) {
        const th = mid - span + (2 * span * a) / ARC_POINTS
        arcs.x.push(ox + r * Math.cos(th))
        arcs.y.push(oy + r * Math.sin(th))
      }
      arcs.x.push(Number.NaN)
      arcs.y.push(Number.NaN)
    }
    let vx = 0
    let vy = 0
    for (const [i, w] of angular.entries()) {
      const a = w * t
      const re = record.chain.re[i] ?? 0
      const im = record.chain.im[i] ?? 0
      vx += -w * (re * Math.sin(a) + im * Math.cos(a))
      vy += w * (re * Math.cos(a) - im * Math.sin(a))
    }
    const speed = Math.hypot(vx, vy)
    const stretch = Math.min(t, speed > 0 ? (4 * half) / speed : t)
    const near = { x: [] as number[], y: [] as number[] }
    for (let q = 0; q <= ZOOM_TRACE; q++) {
      const [x, y] = penAt(record, t - stretch + (stretch * q) / ZOOM_TRACE)
      near.x.push(x - pen[0])
      near.y.push(y - pen[1])
    }
    const before = Math.max(0, traced.x.length - Math.ceil(stretch * fine) - 1)
    const zoomed: Panel = {
      layers: [
        ...(showTarget
          ? [
              lines(
                [
                  {
                    x: closedTarget.x.map((v) => v - pen[0]),
                    y: closedTarget.y.map((v) => v - pen[1]),
                    name: 'target',
                    colour: 'fgSecondary',
                    dash: 'dashed',
                    width: 1.5,
                    opacity: 0.6,
                  },
                ],
                { mode },
              ),
            ]
          : []),
        ...(showCircles
          ? [lines([{ ...arcs, name: 'circles', colour: 'accent', width: 0.75, opacity: 0.45 }], { mode })]
          : []),
        lines(
          [{ x: j.x.map((v) => v - pen[0]), y: j.y.map((v) => v - pen[1]), name: 'arrows', colour: 'fg', width: 1 }],
          {
            mode,
          },
        ),
        lines(
          [
            {
              x: [...traced.x.slice(0, before).map((v) => v - pen[0]), ...near.x],
              y: [...traced.y.slice(0, before).map((v) => v - pen[1]), ...near.y],
              name: 'traced',
              colour: 'accent',
              width: 2,
            },
          ],
          { mode },
        ),
        points([{ x: [0], y: [0], name: 'pen', colour: 'accent', size: 8 }], { mode }),
      ],
      options: {
        title: `${formatNumber(zoom)}× on the pen`,
        x: { label: 'Re z − pen', limits: [-half, half] },
        y: { label: 'Im z − pen', limits: [-half, half] },
        aspect: 'equal',
      },
    }
    const grid = gridSpec(2, 2, {
      heightRatios: [1, 0.5],
      wspace: 56,
      hspace: 56,
      cells: [
        { id: 'main', row: 0, col: 0 },
        { id: 'zoom', row: 0, col: 1 },
        { id: 'spectrum', row: 1, col: 0, colSpan: 2 },
      ],
    })
    return subplots(grid, { main, zoom: zoomed, spectrum }, { figure: fig, legend: false, tooltip: false })
  }
  return {
    count: steps + 1,
    label: (f) =>
      `t ${(Math.min(Math.max(f, 0), steps) / steps).toFixed(3)} · ${record.chain.k.length} terms · ${(100 * record.energyKept).toFixed(2)}% of the energy · ${record.source}`,
    option,
  }
}
