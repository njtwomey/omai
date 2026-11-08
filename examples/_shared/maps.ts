/**
 * Field maps for the examples, so that their figures agree: a classifier's score or probability as a heat map on the
 * two class colours (class 0 blue, class 1 orange, white where the two are equal), the decision boundary and margins as
 * contours of that same field, the rows over it in the same two colours; and a density as a heat map on one pale ramp
 * with its own contours. The map covers exactly the data's window (the data's extent and a small margin), and the axes
 * are that window, so nothing is drawn outside them.
 */

import { cartesian, contours, heatmap, type Layer, points } from 'omai-render/charts'
import { figure } from 'omai-render/figure'
import { categoricalColour } from 'omai-render/theme'
import { INK, inkOf, type Mode, mix, PAPER, paperOf, tab } from './colours.ts'

/** The two class colours (tab10's blue and orange), used for the rows and the ends of the class ramp. */
export const CLASS_COLOURS: readonly [string, string] = [categoricalColour(0), categoricalColour(1)]

/** The ramp of a two-class field: class 0, white, class 1. */
export const CLASS_RAMP: readonly string[] = [CLASS_COLOURS[0], mix(PAPER, INK, 0.04), CLASS_COLOURS[1]]
/** The class ramp of a colour mode. */
export const classRamp = (mode: Mode): readonly string[] => [
  CLASS_COLOURS[0],
  mix(paperOf(mode), inkOf(mode), 0.04),
  CLASS_COLOURS[1],
]

/** The ramp of a density or a likelihood: white to a muted blue-grey, so coloured rows stay readable over it. */
export const DENSITY_RAMP: readonly string[] = [PAPER, mix(PAPER, tab(0), 0.15), mix(PAPER, tab(0), 0.45)]
/** The density ramp of a colour mode: the page's colour towards the first category's, so it is blues in either. */
export const densityRamp = (mode: Mode): readonly string[] => [
  paperOf(mode),
  mix(paperOf(mode), tab(0), mode === 'dark' ? 0.3 : 0.15),
  mix(paperOf(mode), tab(0), mode === 'dark' ? 0.65 : 0.45),
]

/** The window a map covers: the data's extent in each coordinate and a small margin. */
export interface Window {
  readonly x: readonly [number, number]
  readonly y: readonly [number, number]
}

/**
 * The data's window.
 *
 * @param rows - Rows of two coordinates.
 * @param margin - The margin, as a share of the extent (default 0.06).
 * @returns The window.
 */
export function windowOf(rows: readonly (readonly number[])[], margin = 0.06): Window {
  const span = (c: 0 | 1): [number, number] => {
    const v = rows.map((r) => r[c] ?? 0)
    const lo = Math.min(...v)
    const hi = Math.max(...v)
    const pad = (hi - lo) * margin
    return [lo - pad, hi + pad]
  }
  return { x: span(0), y: span(1) }
}

/** Grid coordinates over a window: n columns and n rows, the end points on the window's edges. */
export function axesOf(window: Window, n: number): { readonly x: number[]; readonly y: number[] } {
  const line = ([lo, hi]: readonly [number, number]): number[] =>
    Array.from({ length: n }, (_, i) => lo + ((hi - lo) * i) / (n - 1))
  return { x: line(window.x), y: line(window.y) }
}

/** The points of the grid, row by row (y outer, x inner), as the rows a model is asked about. */
export function gridRows(axes: { readonly x: readonly number[]; readonly y: readonly number[] }): number[][] {
  return axes.y.flatMap((y) => axes.x.map((x) => [x, y]))
}

/** Values in the order of {@link gridRows} cut into the rows of a grid. */
export function gridOfValues(values: ArrayLike<number>, n: number): number[][] {
  return Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => values[i * n + j] ?? 0))
}

/** What {@link classMap} draws. */
export interface ClassMapOptions {
  readonly window: Window
  /** The field on the grid of `axesOf(window, n)`: rows are y, columns x. */
  readonly field: readonly (readonly number[])[]
  /** The value that is white (0.5 for a probability, 0 for a score). */
  readonly centre: number
  /** The values at the two ends of the colour ramp. */
  readonly range: readonly [number, number]
  /** What the field is, for the colour bar. */
  readonly label: string
  /** Contours of the field: the boundary (solid) and further levels (dashed, e.g. the margins). */
  readonly boundary: number
  readonly dashed?: readonly number[]
  /** Blocky cells (a piecewise constant field) rather than a smooth image. */
  readonly cells?: boolean
  readonly rows: readonly (readonly number[])[]
  /** Class index of each row (0 or 1). */
  readonly labels: readonly number[]
  readonly classNames: readonly [string, string]
  /** Further layers drawn over the map and under the rows (for example a second contour, the true boundary). */
  readonly extra?: readonly Layer[]
  /** Rows to ring (the support vectors), with the legend name. */
  readonly ringed?: readonly number[]
  readonly ringName?: string
  readonly width: number
  readonly height: number
  readonly xLabel: string
  readonly yLabel: string
  /** The colour mode to draw in (default light). */
  readonly mode?: Mode
}

/** A two-class field as a figure: the heat map, its contours, the rows and any ringed rows. */
export function classMap(o: ClassMapOptions): unknown {
  const mode = o.mode ?? 'light'
  const n = o.field.length
  const axes = axesOf(o.window, n)
  const layers: Layer[] = [
    heatmap(o.field, {
      x: o.window.x,
      y: o.window.y,
      ramp: classRamp(mode),
      mode,
      range: o.range,
      centre: o.centre,
      smooth: o.cells === true ? 'cells' : 'bicubic',
      resolution: 96,
      label: o.label,
      opacity: 0.8,
    }),
    contours(o.field, { x: axes.x, y: axes.y, levels: [o.boundary], colour: 'fg', width: 2, labels: false, mode }),
  ]
  if ((o.dashed ?? []).length > 0)
    layers.push(
      contours(o.field, {
        x: axes.x,
        y: axes.y,
        levels: o.dashed ?? [],
        colour: 'fg',
        width: 1,
        dash: 'dashed',
        labels: false,
        mode,
      }),
    )
  layers.push(...(o.extra ?? []))
  for (const c of [0, 1] as const) {
    const idx = o.labels.flatMap((l, i) => (l === c ? [i] : []))
    layers.push(
      points(
        [
          {
            x: idx.map((i) => o.rows[i]?.[0] ?? 0),
            y: idx.map((i) => o.rows[i]?.[1] ?? 0),
            name: o.classNames[c],
            colour: `tab${c}`,
            size: 6,
            opacity: 1,
          },
        ],
        { colourOffset: 0, mode },
      ),
    )
  }
  const ring = o.ringed ?? []
  if (ring.length > 0)
    layers.push(
      points(
        [
          {
            x: ring.map((i) => o.rows[i]?.[0] ?? 0),
            y: ring.map((i) => o.rows[i]?.[1] ?? 0),
            name: o.ringName ?? 'ringed',
            colour: 'fg',
            size: 13,
            symbol: 'emptyCircle',
            opacity: 1,
            outline: false,
          },
        ],
        { mode },
      ),
    )
  return cartesian(layers, {
    mode,
    figure: figure({ width: o.width, height: o.height, padding: { top: 30, right: 12, bottom: 38, left: 42 } }),
    x: { label: o.xLabel, limits: [...o.window.x] },
    y: { label: o.yLabel, limits: [...o.window.y] },
    legend: true,
  })
}

/** A density on a grid over a window as a figure: the heat map and its contours, with optional rows over it. */
export function densityMap(o: {
  readonly window: Window
  readonly field: readonly (readonly number[])[]
  readonly label: string
  readonly levels?: number
  readonly rows?: readonly (readonly number[])[]
  readonly groups?: readonly number[]
  readonly names?: readonly string[]
  readonly width: number
  readonly height: number
  readonly xLabel: string
  readonly yLabel: string
  readonly mode?: Mode
}): unknown {
  const mode = o.mode ?? 'light'
  const n = o.field.length
  const axes = axesOf(o.window, n)
  const layers: Layer[] = [
    heatmap(o.field, {
      x: o.window.x,
      y: o.window.y,
      ramp: densityRamp(mode),
      mode,
      range: [0, Math.max(...o.field.flat())],
      resolution: 96,
      label: o.label,
    }),
    contours(o.field, {
      x: axes.x,
      y: axes.y,
      levels: o.levels ?? 6,
      colour: 'fg',
      width: 1,
      opacity: 0.6,
      labels: false,
      mode,
    }),
  ]
  const rows = o.rows ?? []
  const groups = o.groups ?? rows.map(() => 0)
  const k = Math.max(0, ...groups) + 1
  for (let c = 0; c < k; c++) {
    const idx = groups.flatMap((g, i) => (g === c ? [i] : []))
    if (idx.length === 0) continue
    layers.push(
      points(
        [
          {
            x: idx.map((i) => rows[i]?.[0] ?? 0),
            y: idx.map((i) => rows[i]?.[1] ?? 0),
            name: o.names?.[c] ?? `component ${c}`,
            colour: `tab${c}`,
            size: 6,
            opacity: 1,
          },
        ],
        { mode },
      ),
    )
  }
  return cartesian(layers, {
    mode,
    figure: figure({ width: o.width, height: o.height, padding: { top: 30, right: 12, bottom: 38, left: 42 } }),
    x: { label: o.xLabel, limits: [...o.window.x] },
    y: { label: o.yLabel, limits: [...o.window.y] },
    legend: rows.length > 0 && o.names !== undefined,
  })
}

/**
 * The layers of a field on one pale ramp with its own contours, to draw paths or rows over: the heat map covers exactly
 * `window` (set the axes to the same window) and the contours follow the same field. {@link densityMap} is this plus
 * rows; use this when the layers over it are lines.
 */
export function fieldLayers(o: {
  readonly window: Window
  readonly field: readonly (readonly number[])[]
  readonly label: string
  readonly levels?: number | readonly number[]
  readonly colourBar?: boolean
  /** The values at the two ends of the ramp (default the field's own extent; a density starts at 0). */
  readonly range?: readonly [number, number]
  readonly mode?: Mode
}): Layer[] {
  const mode = o.mode ?? 'light'
  const axes = axesOf(o.window, o.field.length)
  return [
    heatmap(o.field, {
      x: o.window.x,
      y: o.window.y,
      ramp: densityRamp(mode),
      mode,
      range: o.range ?? [Math.min(...o.field.flat()), Math.max(...o.field.flat())],
      resolution: 96,
      label: o.label,
      colourBar: o.colourBar ?? true,
    }),
    contours(o.field, {
      x: axes.x,
      y: axes.y,
      levels: o.levels ?? 8,
      colour: 'fg',
      width: 1,
      opacity: 0.45,
      labels: false,
      mode,
    }),
  ]
}
