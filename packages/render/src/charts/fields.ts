/**
 * Figures of two-dimensional fields: contour lines by marching squares, a classifier's decision regions coloured by the
 * winning class and faded where it is unsure (with `argmaxMargins` giving the boundaries), vector fields as arrows
 * coloured by magnitude, and streamlines with arrow heads showing the direction of flow.
 */

import type { SeriesOption } from 'echarts'
import { maxOf, minOf, OmaiError } from 'omai-engine/base'
import { contourLevels, contourLines } from 'omai-engine/numerics/geometry'
import type { Pixels } from 'omai-engine/scene'
import { type TensorLike, toFlat } from 'omai-engine/tensor'
import type { Range } from '../figure/aspect.ts'
import { extentOf } from '../figure/limits.ts'
import { formatNumber } from '../theme/format.ts'
import { categoricalColour, colourScale, mixColours, parseColour, sequentialRamp } from '../theme/palettes.ts'
import { type ColourMode, tokens } from '../theme/tokens.ts'
import { axisCoords, colourFor, type Grid2D, grid2d, gridRows, numbers } from './_data.ts'
import { resample } from './_resample.ts'
import { imageSeries } from './matrices.ts'
import { type Layer, layer } from './plot.ts'

/** Options of {@link contours}. */
export interface ContourOptions {
  /** Column coordinates (one per column, or a `[low, high]` range). */
  readonly x?: readonly number[]
  /** Row coordinates. */
  readonly y?: readonly number[]
  /** A count of nice levels (default 7) or the levels. */
  readonly levels?: number | readonly number[]
  /** A colour for every line (default `fg`), or `ramp` to colour each level along viridis. */
  readonly colour?: string | 'ramp'
  readonly width?: number
  readonly dash?: 'solid' | 'dashed' | 'dotted'
  readonly opacity?: number
  /** Label each level on its longest line (default true). */
  readonly labels?: boolean
  readonly mode?: ColourMode
  readonly name?: string
}

/**
 * Contour lines of a gridded field (marching squares from omai-engine's `numerics.geometry.contourLines`), one
 * line series per level with gaps between separate lines, and a label per level.
 *
 * @param z - Values `[rows, cols]`.
 * @param options - Coordinates, levels, colour, width, dash and labels.
 * @returns The layer.
 * @index function Contour lines of a 2-D array: `contours(z, { levels: 7 })`.
 * @example
 * ```ts
 * import { contours } from 'omai-render/charts'
 * const l = contours([[0, 0, 0], [0, 1, 0], [0, 0, 0]], { levels: [0.5], labels: false })
 * l.series.length // => 1
 * l.series[0].data.length // => 5
 * ```
 */
export function contours(z: TensorLike, options: ContourOptions = {}): Layer {
  const mode = options.mode ?? 'light'
  const g = grid2d(z, 'contours')
  const x = axisCoords(options.x, g.cols)
  const y = axisCoords(options.y, g.rows)
  const levels =
    typeof options.levels === 'object' ? [...options.levels] : toFlat(contourLevels(gridRows(g), options.levels ?? 7))
  const ramp =
    options.colour === 'ramp'
      ? colourScale(sequentialRamp('viridis'), [levels[0] ?? 0, levels[levels.length - 1] ?? 1])
      : undefined
  const fixed = options.colour === 'ramp' ? undefined : colourFor(options.colour ?? 'fg', mode)
  const series: SeriesOption[] = []
  const labels: { value: number[]; colour: string }[] = []
  levels.forEach((level, k) => {
    const paths = contourLines(gridRows(g), level, { x, y })
    if (paths.length === 0) return
    const colour = fixed ?? ramp?.(level) ?? '#888888'
    const data: (number | null)[][] = []
    let longest: [number, number][] = []
    for (const p of paths) {
      for (const pt of p) data.push([pt[0], pt[1]])
      data.push([null, null])
      if (p.length > longest.length) longest = p
    }
    data.pop()
    const mid = longest[Math.floor(longest.length / 2)]
    if (mid !== undefined) labels.push({ value: [mid[0], mid[1], level], colour })
    series.push({
      type: 'line',
      name: options.name ?? `__contour${k}`,
      data,
      connectNulls: false,
      showSymbol: false,
      silent: true,
      color: colour,
      lineStyle: {
        color: colour,
        width: options.width ?? 1,
        type: options.dash ?? 'solid',
        opacity: options.opacity ?? 1,
      },
      tooltip: { show: false },
      clip: true,
      z: 3,
    } as SeriesOption)
  })
  if (options.labels ?? true) {
    const t = tokens(mode)
    series.push({
      type: 'scatter',
      name: '__contour-labels',
      data: labels.map((l) => ({ value: l.value, label: { color: l.colour } })),
      symbolSize: 0,
      silent: true,
      label: {
        show: true,
        formatter: (p: { value: number[] }) => formatNumber(p.value[2] ?? 0, { digits: 3 }),
        fontSize: 10,
        backgroundColor: t.bg,
        padding: [1, 3],
        borderRadius: 2,
      },
      tooltip: { show: false },
      clip: true,
      z: 4,
    } as SeriesOption)
  }
  return layer(series, {
    x: [minOf(x), maxOf(x)],
    y: [minOf(y), maxOf(y)],
    tight: true,
  })
}

/**
 * The argmax margins of K score fields on one grid: m_k = s_k - max over j != k of s_j, positive exactly where k wins,
 * so the zero contours of the margins are the decision boundaries.
 *
 * @param fields - K grids of the same shape.
 * @returns K margin grids.
 * @example
 * ```ts
 * import { argmaxMargins } from 'omai-render/charts'
 * argmaxMargins([[[0.2, 0.7]], [[0.8, 0.3]]]) // => [[[-0.6000000000000001, 0.39999999999999997]], [[0.6000000000000001, -0.39999999999999997]]]
 * ```
 */
export function argmaxMargins(fields: readonly TensorLike[]): number[][][] {
  const grids = fields.map((f) => grid2d(f, 'argmaxMargins'))
  const first = grids[0]
  if (first === undefined) return []
  if (grids.some((g) => g.rows !== first.rows || g.cols !== first.cols))
    throw new OmaiError('shape', 'argmaxMargins: the fields must have the same shape')
  return grids.map((g, k) =>
    Array.from({ length: g.rows }, (_, r) =>
      Array.from({ length: g.cols }, (_, c) => {
        let other = Number.NEGATIVE_INFINITY
        grids.forEach((h, j) => {
          if (j !== k) other = Math.max(other, h.values[r * g.cols + c] ?? Number.NEGATIVE_INFINITY)
        })
        return (g.values[r * g.cols + c] ?? 0) - other
      }),
    ),
  )
}

/** Options of {@link decisionRegions}. */
export interface DecisionRegionOptions {
  readonly x?: readonly number[]
  readonly y?: readonly number[]
  /** Class colours (default tab10 in order). */
  readonly colours?: readonly string[]
  /** Shade by confidence: a winning probability near 1/K fades to the background (default true). */
  readonly confidence?: boolean
  /** Draw the boundaries between regions (default true). */
  readonly boundaries?: boolean
  /** Image resolution (largest side, default 200). */
  readonly resolution?: number
  /** Region opacity (default 0.85). */
  readonly opacity?: number
  readonly mode?: ColourMode
}

/**
 * Decision regions of a classifier from per-class probability (or score) grids: each pixel takes the colour of the
 * winning class, faded towards the background where the classifier is unsure, drawn smoothly as one image (the grids
 * are upsampled bicubically before the argmax), with the boundaries as the zero contours of the argmax margins.
 *
 * @param probabilities - K grids `[rows, cols]`, one per class.
 * @param options - Coordinates, colours, confidence shading, boundaries, resolution and opacity.
 * @returns The layer.
 * @index function A classifier's decision regions from class probability grids: `decisionRegions(probs, { x, y })`.
 * @example
 * ```ts
 * import { decisionRegions } from 'omai-render/charts'
 * const l = decisionRegions([[[0.9, 0.1], [0.8, 0.3]], [[0.1, 0.9], [0.2, 0.7]]], { x: [-1, 1], y: [-1, 1] })
 * l.extent // => { x: [-1, 1], y: [-1, 1], tight: true }
 * ```
 */
export function decisionRegions(probabilities: readonly TensorLike[], options: DecisionRegionOptions = {}): Layer {
  const mode = options.mode ?? 'light'
  const grids: Grid2D[] = probabilities.map((p) => grid2d(p, 'decisionRegions'))
  const first = grids[0]
  if (first === undefined) throw new OmaiError('value', 'decisionRegions: give at least one class')
  const K = grids.length
  const x = axisCoords(options.x, first.cols)
  const y = axisCoords(options.y, first.rows)
  const target = options.resolution ?? 200
  const scale = target / Math.max(first.rows, first.cols)
  const rows = Math.max(first.rows, Math.round(first.rows * scale))
  const cols = Math.max(first.cols, Math.round(first.cols * scale))
  const up = grids.map((g) => resample(g, rows, cols, 'bicubic'))
  const bg = tokens(mode).bg
  const shades = Array.from({ length: K }, (_, k) => {
    const c = options.colours?.[k] ?? categoricalColour(k)
    // 64 confidence steps per class, precomputed: fine enough that the fade shows no bands.
    return Array.from({ length: 65 }, (_, s) => parseColour(mixColours(bg, c, 0.25 + 0.75 * (s / 64))))
  })
  const rgba = new Uint8ClampedArray(rows * cols * 4)
  const alpha = Math.round((options.opacity ?? 0.85) * 255)
  const sum = K > 1 && grids.every((g) => g.values.every((v) => v >= 0 && v <= 1))
  for (let r = 0; r < rows; r++)
    for (let c = 0; c < cols; c++) {
      const i = (rows - 1 - r) * cols + c
      let best = 0
      let bestV = Number.NEGATIVE_INFINITY
      for (let k = 0; k < K; k++) {
        const v = up[k]?.[i] ?? Number.NEGATIVE_INFINITY
        if (v > bestV) {
          bestV = v
          best = k
        }
      }
      if (!Number.isFinite(bestV)) continue
      const conf = options.confidence === false || !sum ? 1 : Math.min(1, Math.max(0, (bestV - 1 / K) / (1 - 1 / K)))
      const [cr = 0, cg = 0, cb = 0] = shades[best]?.[Math.round(conf * 64)] ?? [0, 0, 0, 1]
      const p = 4 * (r * cols + c)
      rgba[p] = cr
      rgba[p + 1] = cg
      rgba[p + 2] = cb
      rgba[p + 3] = alpha
    }
  const pixels: Pixels = { width: cols, height: rows, rgba }
  const xr: Range = [x[0] ?? 0, x[x.length - 1] ?? 1]
  const yr: Range = [y[0] ?? 0, y[y.length - 1] ?? 1]
  const series: SeriesOption[] = [imageSeries(pixels, xr, yr, { name: '__regions' })]
  if ((options.boundaries ?? true) && K > 1) {
    const margins = K === 2 ? argmaxMargins(probabilities).slice(0, 1) : argmaxMargins(probabilities)
    for (const m of margins)
      series.push(
        ...contours(m, { x, y, levels: [0], labels: false, colour: 'fg', width: 1.2, mode, opacity: 0.8 }).series,
      )
  }
  return layer(series, { x: xr, y: yr, tight: true })
}

/** One arrow of a sampled vector field: its two ends and the field's magnitude at its centre. */
export interface FieldArrow {
  readonly from: readonly [number, number]
  readonly to: readonly [number, number]
  readonly magnitude: number
}

/** Options of {@link fieldArrows}. */
export interface FieldArrowOptions {
  readonly x: Range
  readonly y: Range
  /** Grid points along each axis (default 15), or `[nx, ny]`. */
  readonly n?: number | readonly [number, number]
  /** `unit`: equal lengths, direction only (default); `magnitude`: lengths proportional to the field. */
  readonly length?: 'unit' | 'magnitude'
  /** The longest arrow as a fraction of the grid spacing (default 0.8). */
  readonly scale?: number
}

/**
 * Sample a vector field as arrows centred on an nx by ny grid. Lengths are in grid-spacing units, so equal arrows look
 * equal whatever the axes' aspect; points where the field vanishes or is not finite get no arrow.
 *
 * @param field - The field `(x, y) => [u, v]`.
 * @param options - Ranges, grid size, length mode and scale.
 * @returns The arrows.
 * @throws OmaiError with code `value` when the grid has fewer than two points along an axis.
 * @example
 * ```ts
 * import { fieldArrows } from 'omai-render/charts'
 * fieldArrows(() => [1, 0], { x: [0, 1], y: [0, 1], n: 2 })[0] // => { from: [-0.4, 0], to: [0.4, 0], magnitude: 1 }
 * ```
 */
export function fieldArrows(
  field: (x: number, y: number) => readonly [number, number],
  options: FieldArrowOptions,
): FieldArrow[] {
  const { x, y, n = 15, length = 'unit', scale = 0.8 } = options
  const [nx, ny] = typeof n === 'number' ? [n, n] : n
  if (!(nx >= 2 && ny >= 2)) throw new OmaiError('value', 'fieldArrows: n must be at least 2 along each axis')
  const dx = (x[1] - x[0]) / (nx - 1)
  const dy = (y[1] - y[0]) / (ny - 1)
  const samples: { cx: number; cy: number; gu: number; gv: number; cell: number; magnitude: number }[] = []
  for (let i = 0; i < ny; i++)
    for (let j = 0; j < nx; j++) {
      const cx = x[0] + j * dx
      const cy = y[0] + i * dy
      const [u, v] = field(cx, cy)
      const magnitude = Math.hypot(u, v)
      if (!(magnitude > 1e-12) || !Number.isFinite(magnitude)) continue
      samples.push({ cx, cy, gu: u / dx, gv: v / dy, cell: Math.hypot(u / dx, v / dy), magnitude })
    }
  const longest = Math.max(0, maxOf(samples.map((s) => s.cell)))
  return samples.map(({ cx, cy, gu, gv, cell, magnitude }) => {
    const unit = (scale / 2) * (length === 'unit' ? 1 / cell : 1 / longest)
    const hx = gu * unit * dx
    const hy = gv * unit * dy
    return { from: [cx - hx, cy - hy], to: [cx + hx, cy + hy], magnitude }
  })
}

/**
 * Arrows as a layer (an ECharts lines series with arrow heads), coloured by magnitude along viridis or in one colour.
 *
 * @param arrows - The arrows (from {@link fieldArrows}, or any segments).
 * @param options - A single colour (default by magnitude), head size, width, colour mode.
 * @returns The layer.
 * @index function Arrows of a vector field: `vectorField(fieldArrows(f, { x, y }))`.
 * @example
 * ```ts
 * import { fieldArrows, vectorField } from 'omai-render/charts'
 * vectorField(fieldArrows((x, y) => [-y, x], { x: [-1, 1], y: [-1, 1], n: 3 })).series[0].data.length // => 8
 * ```
 */
export function vectorField(
  arrows: readonly FieldArrow[],
  options: {
    readonly colour?: string
    readonly head?: number
    readonly width?: number
    readonly mode?: ColourMode
  } = {},
): Layer {
  const mode = options.mode ?? 'light'
  const mags = arrows.map((a) => a.magnitude)
  const [lo, hi] = extentOf(mags) ?? [0, 1]
  const scaleOf = colourScale(sequentialRamp('viridis'), [lo, hi])
  const fixed = options.colour === undefined ? undefined : colourFor(options.colour, mode)
  const series = {
    type: 'lines',
    coordinateSystem: 'cartesian2d',
    name: '__arrows',
    data: arrows.map((a) => ({
      coords: [
        [a.from[0], a.from[1]],
        [a.to[0], a.to[1]],
      ],
      lineStyle: { color: fixed ?? scaleOf(a.magnitude) },
    })),
    symbol: ['none', 'arrow'],
    symbolSize: options.head ?? 7,
    lineStyle: { width: options.width ?? 1.4, opacity: 0.9 },
    silent: true,
    z: 3,
  } as unknown as SeriesOption
  const ends = arrows.flatMap((a) => [a.from, a.to])
  const xe = extentOf(ends.map((p) => p[0]))
  const ye = extentOf(ends.map((p) => p[1]))
  return layer([series], { ...(xe === undefined ? {} : { x: xe }), ...(ye === undefined ? {} : { y: ye }) })
}

/**
 * Streamlines (trajectories of a field, e.g. from omai-engine's `dynamics.streamline`) as lines, with an arrow head
 * at the middle of each to show the direction of flow.
 *
 * @param paths - Each path as `[x, y]` points (or an `[m, 2]` tensor) in time order.
 * @param options - Colour (default `muted`), width and colour mode.
 * @returns The layer.
 * @example
 * ```ts
 * import { streamlines } from 'omai-render/charts'
 * streamlines([[[0, 0], [1, 0], [2, 1]]]).extent // => { x: [0, 2], y: [0, 1] }
 * ```
 */
export function streamlines(
  paths: readonly (TensorLike | readonly (readonly [number, number])[])[],
  options: { readonly colour?: string; readonly width?: number; readonly mode?: ColourMode } = {},
): Layer {
  const mode = options.mode ?? 'light'
  const colour = colourFor(options.colour ?? 'muted', mode)
  const pts = paths.map((p) => {
    const flat = numbers(p as TensorLike)
    return Array.from({ length: Math.floor(flat.length / 2) }, (_, i): [number, number] => [
      flat[2 * i] ?? 0,
      flat[2 * i + 1] ?? 0,
    ])
  })
  const data: (number | null)[][] = []
  for (const p of pts) {
    for (const q of p) data.push([q[0], q[1]])
    data.push([null, null])
  }
  const heads = pts
    .filter((p) => p.length >= 2)
    .map((p) => {
      const k = Math.floor((p.length - 1) / 2)
      const a = p[k] ?? [0, 0]
      const b = p[k + 1] ?? a
      return { coords: [a, b] }
    })
  const series: SeriesOption[] = [
    {
      type: 'line',
      name: '__streamlines',
      data,
      connectNulls: false,
      showSymbol: false,
      silent: true,
      color: colour,
      lineStyle: { color: colour, width: options.width ?? 1.2 },
      tooltip: { show: false },
      z: 2,
    } as SeriesOption,
    {
      type: 'lines',
      coordinateSystem: 'cartesian2d',
      name: '__stream-heads',
      data: heads,
      symbol: ['none', 'arrow'],
      symbolSize: 7,
      lineStyle: { color: colour, width: 0 },
      silent: true,
      z: 2,
    } as unknown as SeriesOption,
  ]
  const all = pts.flat()
  const xe = extentOf(all.map((p) => p[0]))
  const ye = extentOf(all.map((p) => p[1]))
  return layer(series, { ...(xe === undefined ? {} : { x: xe }), ...(ye === undefined ? {} : { y: ye }) })
}
