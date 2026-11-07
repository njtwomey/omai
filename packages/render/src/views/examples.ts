/**
 * Views of example-based and contrastive records: a counterfactual table (the original beside each counterfactual,
 * changed cells coloured by the direction and size of the change, with predictions, distances and validity
 * beneath), and prototypes and criticisms (marked on a scatter of the data, or drawn as images).
 */

import type { EChartsOption, SeriesOption } from 'echarts'
import { maxOf } from 'mdml-engine/base'
import type { CounterfactualSet, PrototypeSet } from 'mdml-engine/exposition'
import { type Panel, subplots } from '../charts/compose.ts'
import { heatmap } from '../charts/matrices.ts'
import { cartesian, layer } from '../charts/plot.ts'
import type { Figure } from '../figure/figure.ts'
import { gridSpec } from '../figure/grid.ts'
import { type ColourMode, tokens } from '../theme/tokens.ts'
import { short, spanOf, type ViewOptions, viewFigure } from './_view.ts'
import { ATTRIBUTION_DOWN, ATTRIBUTION_UP } from './attributions.ts'

/** Options of {@link counterfactualTableView}. */
export interface CounterfactualTableOptions extends ViewOptions {
  /** Show every feature (default: only those some counterfactual changes, and the immutable ones). */
  readonly all?: boolean
}

/**
 * The feature rows a counterfactual table shows.
 *
 * @param record - The counterfactual set.
 * @param all - Whether to list every feature row rather than only the changed (default false).
 * @returns The feature row indices.
 */
export function counterfactualRows(record: CounterfactualSet, all = false): number[] {
  if (all) return record.features.map((_, j) => j)
  const shown = new Set<number>([...record.counterfactuals.flatMap((c) => c.changed), ...(record.immutable ?? [])])
  const rows = record.features.map((_, j) => j).filter((j) => shown.has(j))
  return rows.length === 0 ? record.features.map((_, j) => j) : rows
}

/**
 * The height a counterfactual table needs at a figure's font size: a header, one line per shown feature and three
 * summary lines.
 *
 * @param record - The record.
 * @param fig - The figure.
 * @param all - Whether every feature is shown.
 * @returns The height in pixels.
 * @example
 * ```ts
 * import { counterfactualSet } from 'mdml-engine/exposition'
 * import { figure } from 'mdml-render/figure'
 * import { counterfactualTableHeight } from 'mdml-render/views'
 * const r = counterfactualSet({ method: 'm', features: ['a', 'b'], output: 'y', original: [0, 0], originalPrediction: 0, counterfactuals: [{ x: [1, 0], prediction: 1, distance: 1, valid: true, changed: [0] }] })
 * counterfactualTableHeight(r, figure()) > 0 // => true
 * ```
 */
export function counterfactualTableHeight(record: CounterfactualSet, fig: Figure, all = false): number {
  const line = Math.round(fig.fontSize * 2.1)
  return fig.padding.top + line * (counterfactualRows(record, all).length + 5) + 12
}

function mixHex(a: string, b: string, t: number): string {
  const pa = [1, 3, 5].map((i) => Number.parseInt(a.slice(i, i + 2), 16))
  const pb = [1, 3, 5].map((i) => Number.parseInt(b.slice(i, i + 2), 16))
  return `#${pa
    .map((v, i) =>
      Math.round(v + ((pb[i] ?? 0) - v) * t)
        .toString(16)
        .padStart(2, '0'),
    )
    .join('')}`
}

function changeColour(delta: number, scale: number, mode: ColourMode): string {
  const bg = tokens(mode).bg
  const base = /^#[0-9a-f]{6}$/i.test(bg) ? bg : mode === 'dark' ? '#121212' : '#ffffff'
  const t = Math.min(1, Math.abs(delta) / (2 * (scale || 1)))
  return mixHex(base, delta >= 0 ? ATTRIBUTION_UP : ATTRIBUTION_DOWN, 0.25 + 0.55 * t)
}

/**
 * Counterfactuals as a table (DiCE's and alibi's displays): a row per feature (those changed, by default), the
 * original's values in the first column and each counterfactual's beside it, a changed cell coloured red for an
 * increase and blue for a decrease (deeper for larger changes, in units of the feature's scale), immutable features
 * marked, and each column's prediction, distance and validity at the foot.
 *
 * @param record - The record.
 * @param options - all, mode and figure.
 * @returns The ECharts option (drawn with graphic elements).
 * @example
 * ```ts
 * import { counterfactualSet } from 'mdml-engine/exposition'
 * import { counterfactualTableView } from 'mdml-render/views'
 * const r = counterfactualSet({ method: 'm', features: ['a', 'b'], output: 'y', original: [0, 0], originalPrediction: 0, counterfactuals: [{ x: [1, 0], prediction: 1, distance: 1, valid: true, changed: [0] }] })
 * counterfactualTableView(r).graphic.length > 4 // => true
 * ```
 */
export function counterfactualTableView(
  record: CounterfactualSet,
  options: CounterfactualTableOptions = {},
): EChartsOption {
  const all = options.all ?? false
  const base = viewFigure(options, {})
  const fig = viewFigure(options, { height: counterfactualTableHeight(record, base, all) })
  const t = tokens(fig.mode)
  const fs = fig.fontSize
  const line = Math.round(fs * 2.1)
  const rows = counterfactualRows(record, all)
  const labelWidth = Math.min(
    200,
    Math.max(80, 12 + 0.62 * fs * Math.max(maxOf(record.features.map((f) => f.length)), 7)),
  )
  const cols = record.counterfactuals.length + 1
  const left = fig.padding.left > 40 ? 16 : fig.padding.left
  const colWidth = Math.max(54, Math.floor((fig.width - left - 16 - labelWidth) / cols))
  const top = fig.padding.top
  const fixed = new Set(record.immutable ?? [])
  const graphic: Record<string, unknown>[] = []
  const text = (x: number, y: number, s: string, opts: Record<string, unknown> = {}): void => {
    graphic.push({
      type: 'text',
      x,
      y,
      style: { text: s, fill: t.fg, font: `${fs}px ${t.font}`, align: 'center', verticalAlign: 'middle', ...opts },
    })
  }
  const colX = (c: number): number => left + labelWidth + c * colWidth
  // Header.
  text(colX(0) + colWidth / 2, top + line / 2, 'original', { font: `600 ${fs}px ${t.font}` })
  record.counterfactuals.forEach((c, i) => {
    text(colX(i + 1) + colWidth / 2, top + line / 2, c.label ?? `#${i + 1}`, { font: `600 ${fs}px ${t.font}` })
  })
  rows.forEach((j, r) => {
    const y = top + line * (r + 1)
    const name = `${record.features[j] ?? `x${j}`}${fixed.has(j) ? ' (fixed)' : ''}`
    text(left + labelWidth - 8, y + line / 2, name, { align: 'right', fill: fixed.has(j) ? t.muted : t.fg })
    text(colX(0) + colWidth / 2, y + line / 2, short(record.original[j] ?? 0), { fill: t.fgSecondary })
    record.counterfactuals.forEach((c, i) => {
      const delta = (c.x[j] ?? 0) - (record.original[j] ?? 0)
      const changed = c.changed.includes(j)
      if (changed)
        graphic.push({
          type: 'rect',
          shape: { x: colX(i + 1) + 2, y: y + 2, width: colWidth - 4, height: line - 4, r: 3 },
          style: { fill: changeColour(delta, record.scales?.[j] ?? Math.abs(record.original[j] ?? 1), fig.mode) },
        })
      text(colX(i + 1) + colWidth / 2, y + line / 2, changed ? short(c.x[j] ?? 0) : '·', {
        fill: changed ? t.fg : t.muted,
        ...(changed ? { font: `600 ${fs}px ${t.font}` } : {}),
      })
    })
  })
  // Foot: prediction, distance, validity.
  const foot = top + line * (rows.length + 1) + 6
  graphic.push({
    type: 'line',
    shape: { x1: left, y1: foot - 3, x2: colX(cols), y2: foot - 3 },
    style: { stroke: t.muted, lineWidth: 1 },
  })
  const footRow = (r: number, label: string, cells: string[], colours?: string[]): void => {
    const y = foot + line * r
    text(left + labelWidth - 8, y + line / 2, label, { align: 'right', fill: t.fgSecondary })
    cells.forEach((s, c) => {
      text(colX(c) + colWidth / 2, y + line / 2, s, colours?.[c] === undefined ? {} : { fill: colours[c] })
    })
  }
  footRow(0, record.output, [
    short(record.originalPrediction),
    ...record.counterfactuals.map((c) => short(c.prediction)),
  ])
  footRow(1, 'distance', ['', ...record.counterfactuals.map((c) => short(c.distance))])
  footRow(
    2,
    'reaches outcome',
    ['', ...record.counterfactuals.map((c) => (c.valid ? 'yes' : 'no'))],
    [t.fg, ...record.counterfactuals.map((c) => (c.valid ? t.fg : ATTRIBUTION_UP))],
  )
  return { backgroundColor: t.bg, animation: false, graphic, series: [] } as EChartsOption
}

/**
 * Prototypes and criticisms (MMD-critic): with an image shape, the prototype images in a row and the criticisms in
 * the next; otherwise every row on a scatter (coloured by its label when given), the prototypes as large diamonds
 * and the criticisms as crosses, the reader seeing which parts of the data the prototypes cover.
 *
 * @param record - The record.
 * @param options - Mode and figure.
 * @returns The ECharts option.
 * @example
 * ```ts
 * import { prototypeSet } from 'mdml-engine/exposition'
 * import { prototypeView } from 'mdml-render/views'
 * const r = prototypeSet({ method: 'MMD-critic', features: ['a', 'b'], prototypes: [0], criticisms: [1], prototypeData: [[0, 0]], criticismData: [[3, 3]], witness: [0, 0.2], points: [[0, 0], [3, 3]] })
 * prototypeView(r).series.length // => 3
 * ```
 */
export function prototypeView(record: PrototypeSet, options: ViewOptions = {}): EChartsOption {
  const shape = record.imageShape
  if (shape !== undefined) {
    const [h, w] = shape
    const np = record.prototypeData.length
    const nc = record.criticismData.length
    const cols = Math.max(np, nc, 1)
    const fig = viewFigure(options, { aspect: Math.min(3, Math.max(1, cols / 2)) })
    const panels: Record<string, Panel> = {}
    const hidden = { show: false, splitLine: false }
    const image = (v: readonly number[]): number[][] =>
      Array.from({ length: h }, (_, r) => Array.from({ length: w }, (_, c) => v[(h - 1 - r) * w + c] ?? 0))
    record.prototypeData.forEach((v, i) => {
      panels[`p${i}`] = {
        layers: [heatmap(image(v), { smooth: 'cells', ramp: 'greys', colourBar: false, mode: fig.mode })],
        options: { title: `prototype ${i + 1}`, x: hidden, y: hidden, aspect: 'equal' },
      }
    })
    record.criticismData.forEach((v, i) => {
      panels[`c${i}`] = {
        layers: [heatmap(image(v), { smooth: 'cells', ramp: 'greys', colourBar: false, mode: fig.mode })],
        options: { title: `criticism ${i + 1}`, x: hidden, y: hidden, aspect: 'equal' },
      }
    })
    const grid = gridSpec(nc > 0 ? 2 : 1, cols, {
      wspace: 8,
      hspace: 30,
      cells: [
        ...record.prototypeData.map((_, i) => ({ id: `p${i}`, row: 0, col: i })),
        ...record.criticismData.map((_, i) => ({ id: `c${i}`, row: 1, col: i })),
      ],
    })
    return subplots(grid, panels, { figure: fig, legend: false, tooltip: false })
  }
  const fig = viewFigure(options, { aspect: 1.35 })
  const t = tokens(fig.mode)
  const pts = record.points ?? record.witness.map((_, i) => [i, record.witness[i] ?? 0] as [number, number])
  const axes = record.points === undefined ? (['row', 'witness'] as const) : (record.axes ?? ['x', 'y'])
  const labels = record.labels
  const names = labels === undefined ? [] : Array.from(new Set(labels))
  const palette = [
    '#4e79a7',
    '#f28e2b',
    '#59a14f',
    '#76b7b2',
    '#edc948',
    '#b07aa1',
    '#ff9da7',
    '#9c755f',
    '#bab0ac',
    '#e15759',
  ]
  const all = {
    type: 'scatter',
    name: '__data',
    data: pts.map((p, i) => ({
      value: [p[0], p[1]],
      itemStyle: {
        color: labels === undefined ? t.muted : (palette[names.indexOf(labels[i] ?? '') % palette.length] ?? t.muted),
        opacity: 0.55,
      },
    })),
    symbolSize: 5,
    z: 2,
  } as SeriesOption
  const marked = (idx: readonly number[], name: string, symbol: string, size: number, colour: string): SeriesOption =>
    ({
      type: 'scatter',
      name,
      data: idx.map((i) => pts[i] ?? [0, 0]),
      symbol,
      symbolSize: size,
      itemStyle: { color: colour, borderColor: t.bg, borderWidth: 1.5 },
      z: 4,
    }) as SeriesOption
  const xs = pts.map((p) => p[0])
  const ys = pts.map((p) => p[1])
  return cartesian(
    [
      layer(
        [
          all,
          marked(record.prototypes, 'prototypes', 'diamond', 16, t.fg),
          marked(
            record.criticisms,
            'criticisms',
            'path://M2,0 L10,8 L18,0 L20,2 L12,10 L20,18 L18,20 L10,12 L2,20 L0,18 L8,10 L0,2 Z',
            14,
            ATTRIBUTION_UP,
          ),
        ],
        { x: spanOf(xs), y: spanOf(ys) },
      ),
    ],
    { figure: fig, mode: fig.mode, x: { label: axes[0] }, y: { label: axes[1] }, legend: true, tooltip: 'item' },
  )
}
