/**
 * Gridded values as pictures: heatmaps of fields drawn as one smoothly upsampled image (bicubic or bilinear), filled
 * contour bands or blocky cells, images from pixels or tensors (imshow style), and labelled matrices for confusion,
 * correlation, attention and kernel matrices, each cell coloured along a ramp with its value written in it.
 */

import type { EChartsOption, SeriesOption } from 'echarts'
import { OmaiError } from 'omai-engine/base'
import { contourLevels } from 'omai-engine/numerics/geometry'
import { type ImageSource, type Pixels, toPixels } from 'omai-engine/scene'
import { type TensorLike, toFlat } from 'omai-engine/tensor'
import type { Range } from '../figure/aspect.ts'
import { extentOf } from '../figure/limits.ts'
import { pngDataUrl } from '../svg/png.ts'
import { formatNumber } from '../theme/format.ts'
import {
  type DivergingName,
  divergingRamp,
  interpolateRgba,
  mixColours,
  parseColour,
  relativeLuminance,
  type SequentialName,
  sampleRamp,
  scaleFraction,
  sequentialRamp,
  toHex,
} from '../theme/palettes.ts'
import { type ColourMode, tokens } from '../theme/tokens.ts'
import { axisCoords, type Grid2D, grid2d, modeOf } from './_data.ts'
import { type Resampling, repeatCells, resample } from './_resample.ts'
import { cartesian, type Layer, layer, type PlotOptions } from './plot.ts'
import { DIMMED_OPACITY, dimColour, type ImageEncoder, type ValueFilter } from './selection.ts'

/** A colour ramp: a named sequential or diverging ramp, or explicit stops from low to high. */
export type RampSpec = SequentialName | DivergingName | readonly string[]

/**
 * How a gridded field is drawn: interpolated (`bilinear`, or `bicubic`, the default), as filled contour bands
 * (`contours`), or as blocky cells (`cells`).
 */
export type Smoothing = 'bicubic' | 'bilinear' | 'contours' | 'cells'

/** Options of {@link heatmap}. */
export interface HeatmapOptions {
  /** Column coordinates: one per column, or a `[low, high]` range (default 0 to cols - 1). */
  readonly x?: readonly number[]
  /** Row coordinates (row 0 at the bottom, as a field sampled on a mesh). */
  readonly y?: readonly number[]
  readonly smooth?: Smoothing
  /** The colour ramp (default viridis, or rdbu when `centre` is given). */
  readonly ramp?: RampSpec
  /** The value range mapped to the ramp (default the data's). */
  readonly range?: Range
  /** A value mapped to the ramp's midpoint (a diverging field centred on zero, say). */
  readonly centre?: number
  /** Contour levels for `smooth: 'contours'`: a count (default 8 nice levels) or the levels. */
  readonly levels?: number | readonly number[]
  /** Largest side of the upsampled image in pixels (default four times the grid, between 64 and 256). */
  readonly resolution?: number
  /** Show a colour bar (default true) with this label. */
  readonly colourBar?: boolean
  readonly label?: string
  readonly opacity?: number
  readonly mode?: ColourMode
  readonly name?: string
}

/**
 * The stops of a ramp specification: named ramps for the colour mode, or the given stops.
 *
 * @param ramp - The ramp (default viridis, or rdbu when `diverging`).
 * @param mode - The colour mode.
 * @param diverging - Whether the default should be diverging.
 * @returns The stops.
 * @example
 * ```ts
 * import { rampStops } from 'omai-render/charts'
 * rampStops(undefined, 'light', true)[0] // => '#2166ac'
 * rampStops(['#000', '#fff'], 'dark') // => ['#000', '#fff']
 * ```
 */
export function rampStops(ramp: RampSpec | undefined, mode: ColourMode, diverging = false): readonly string[] {
  if (ramp === undefined) return diverging ? divergingRamp('rdbu') : sequentialRamp('viridis', mode)
  if (typeof ramp !== 'string') return ramp
  return ramp === 'rdbu' || ramp === 'coolwarm' ? divergingRamp(ramp) : sequentialRamp(ramp as SequentialName, mode)
}

interface ImageApi {
  coord(p: readonly number[]): number[]
}

/**
 * One image covering a data rectangle, as an ECharts custom series (drawn once, scaled by the renderer, so a 30 by 30
 * field costs one image rather than 900 cells).
 *
 * @param pixels - The image.
 * @param x - The x range it covers.
 * @param y - The y range it covers.
 * @param options - Name, opacity and z order.
 * @returns The series.
 * @example
 * ```ts
 * import { imageSeries } from 'omai-render/charts'
 * imageSeries({ width: 1, height: 1, rgba: new Uint8ClampedArray(4) }, [0, 1], [0, 1]).type // => 'custom'
 * ```
 */
export function imageSeries(
  pixels: Pixels,
  x: Range,
  y: Range,
  options: { readonly name?: string; readonly opacity?: number; readonly z?: number } = {},
): SeriesOption {
  const opacity = options.opacity ?? 1
  return {
    type: 'custom',
    name: options.name ?? '__image',
    data: [[x[0], y[0], x[1], y[1]]],
    // A PNG data URL draws anywhere (SVG, export, server rendering) but a browser decodes it before it can paint, so a
    // chart that updates often shows blank frames; `reencodeImages` swaps the pixels in for a canvas, which paints at once.
    // The PNG is encoded when the item is first drawn, so a chart drawn from canvases never pays for it.
    renderItem: imageRenderer(() => pngDataUrl(pixels), x, y, opacity),
    omaiImage: { pixels, x, y, opacity },
    clip: true,
    silent: true,
    z: options.z ?? 1,
    tooltip: { show: false },
  } as unknown as SeriesOption
}

/** What an image series keeps of its pixels, so its encoding can be chosen when the chart is drawn. */
interface ImageSeriesSource {
  readonly pixels: Pixels
  readonly x: Range
  readonly y: Range
  readonly opacity: number
}

/**
 * The series of a chart option with every image series (see {@link imageSeries}) drawn through `encode`: a browser's
 * canvas element instead of a data URL, so an image that changes at every step of a live chart paints in the frame that
 * asks for it rather than after its decode (which, at a step every few milliseconds, is never). The other series are
 * returned as they are.
 *
 * @param series - The `series` of an ECharts option.
 * @param encode - Turns pixels into what an image element draws.
 * @returns The series, with the images re-encoded.
 * @example
 * ```ts
 * import { heatmap, reencodeImages } from 'omai-render/charts'
 * const layer = heatmap([[0, 1], [1, 0]], { x: [0, 1], y: [0, 1] })
 * reencodeImages(layer.series, () => 'canvas').length // => layer.series.length
 * ```
 */
export function reencodeImages(series: readonly SeriesOption[], encode: ImageEncoder): SeriesOption[] {
  return series.map((s) => {
    const source = (s as { omaiImage?: ImageSeriesSource }).omaiImage
    if (source === undefined) return s
    return {
      ...s,
      renderItem: imageRenderer(() => encode(source.pixels), source.x, source.y, source.opacity),
    } as unknown as SeriesOption
  })
}

// The renderItem of an image series: the image (a data URL, or a canvas) stretched over a data rectangle. The image is
// made on the first draw and kept, so a series that is never drawn never encodes its pixels.
function imageRenderer(
  make: () => unknown,
  x: Range,
  y: Range,
  opacity: number,
): (params: unknown, api: ImageApi) => Record<string, unknown> {
  let made = false
  let image: unknown
  return (_params, api) => {
    if (!made) {
      image = make()
      made = true
    }
    const [x0 = 0, y1 = 0] = api.coord([x[0], y[0]])
    const [x1 = 0, y0 = 0] = api.coord([x[1], y[1]])
    return {
      type: 'image',
      style: { image, x: x0, y: y0, width: x1 - x0, height: y1 - y0, opacity },
      silent: true,
    }
  }
}

/** A selection to paint: its weights, the background dimmed values blend into, and whether its edges are soft. */
interface Dimming {
  readonly keep: ValueFilter
  readonly background: readonly number[]
  /** Ramp the weights over a pixel's change in value, so a smooth field's dimmed region has anti-aliased edges. */
  readonly soft: boolean
}

// The largest change in value from a grid point to its four neighbours: how much the value moves across a pixel.
function spreadAt(values: Float64Array, rows: number, cols: number, i: number, j: number): number {
  const v = values[i * cols + j] ?? Number.NaN
  let s = 0
  for (const [di, dj] of [
    [1, 0],
    [-1, 0],
    [0, 1],
    [0, -1],
  ] as const) {
    const a = i + di
    const b = j + dj
    if (a < 0 || a >= rows || b < 0 || b >= cols) continue
    const d = Math.abs((values[a * cols + b] ?? Number.NaN) - v)
    if (d > s) s = d
  }
  return s
}

// Values outside a selection are blended towards the background (see dimColour).
function paintGrid(
  values: Float64Array,
  rows: number,
  cols: number,
  colour: (v: number) => readonly number[],
  dimming?: Dimming,
): Pixels {
  // Image rows run top to bottom, grid rows bottom to top: flip.
  const rgba = new Uint8ClampedArray(rows * cols * 4)
  for (let r = 0; r < rows; r++)
    for (let c = 0; c < cols; c++) {
      const i = rows - 1 - r
      const v = values[i * cols + c] ?? Number.NaN
      const p = 4 * (r * cols + c)
      if (!Number.isFinite(v)) continue
      const [cr = 0, cg = 0, cb = 0, ca = 1] = colour(v)
      const w = dimming === undefined ? 1 : dimming.keep(v, dimming.soft ? spreadAt(values, rows, cols, i, c) : 0)
      const [dr = 0, dg = 0, db = 0] =
        w >= 1 || dimming === undefined ? [cr, cg, cb] : dimColour([cr, cg, cb], dimming.background, w)
      rgba[p] = dr
      rgba[p + 1] = dg
      rgba[p + 2] = db
      rgba[p + 3] = Math.round(ca * 255)
    }
  return { width: cols, height: rows, rgba }
}

/**
 * A gridded field as an image, smooth by default: the grid is upsampled (bicubic or bilinear, corner-aligned so grid
 * nodes keep their values) to one image, or posterised into filled contour bands, or drawn as blocky cells. Missing
 * (NaN) values are transparent.
 *
 * @param z - Values `[rows, cols]`; `z[i][j]` is the field at `(x[j], y[i])`.
 * @param options - Coordinates, smoothing, ramp, range, centre, levels, resolution and colour bar.
 * @returns The layer: the image and a colour bar.
 * @example
 * ```ts
 * import { heatmap } from 'omai-render/charts'
 * const l = heatmap([[0, 1], [1, 2]], { x: [0, 1], y: [0, 1] })
 * l.extent // => { x: [0, 1], y: [0, 1], tight: true }
 * l.colourBar.min // => 0
 * ```
 */
export function heatmap(z: TensorLike, options: HeatmapOptions = {}): Layer {
  const mode = options.mode ?? 'light'
  const g = grid2d(z, 'heatmap')
  if (g.rows < 1 || g.cols < 1) throw new OmaiError('shape', 'heatmap: the grid is empty')
  const smooth = options.smooth ?? 'bicubic'
  const xs = axisCoords(options.x, g.cols)
  const ys = axisCoords(options.y, g.rows)
  const [lo, hi] = options.range ?? extentOf(g.values) ?? [0, 1]
  const centre = options.centre
  const stops = rampStops(options.ramp, mode, centre !== undefined)
  const parsed = stops.map(parseColour)
  const domain: readonly [number, number] | readonly [number, number, number] =
    centre === undefined ? [lo, hi] : [Math.min(lo, centre), centre, Math.max(hi, centre)]
  // The painted grid and its colouring, kept so a colour bar selection can repaint it.
  let paint: (keep?: ValueFilter) => Pixels
  let levels: number[] | undefined
  let bandColours: string[] | undefined
  const ramp = (v: number): readonly number[] => interpolateRgba(parsed, scaleFraction(v, domain))
  const background = parseColour(tokens(mode).bg)
  const dimming = (keep: ValueFilter | undefined, soft: boolean): Dimming | undefined =>
    keep === undefined ? undefined : { keep, background, soft }
  const cells = smooth === 'cells' || g.rows < 2 || g.cols < 2
  if (cells) {
    const k = Math.max(1, Math.min(16, Math.floor(256 / Math.max(g.rows, g.cols))))
    const values = repeatCells(g, k)
    paint = (keep) => paintGrid(values, g.rows * k, g.cols * k, ramp, dimming(keep, false))
  } else {
    const target = options.resolution ?? Math.min(256, Math.max(64, 4 * Math.max(g.rows, g.cols)))
    const scale = target / Math.max(g.rows, g.cols)
    const rows = Math.max(g.rows, Math.round(g.rows * scale))
    const cols = Math.max(g.cols, Math.round(g.cols * scale))
    const method: Resampling = smooth === 'bilinear' ? 'bilinear' : 'bicubic'
    const values = resample(g, rows, cols, method)
    if (smooth === 'contours') {
      const finite = Array.from(g.values).filter(Number.isFinite)
      const inner =
        typeof options.levels === 'object'
          ? [...options.levels]
          : toFlat(contourLevels([finite.length > 0 ? finite : [0, 1]], options.levels ?? 8))
      levels = [lo, ...inner.filter((l) => l > lo && l < hi), hi]
      const mids = levels.slice(0, -1).map((l, i) => (l + (levels?.[i + 1] ?? l)) / 2)
      bandColours = mids.map((m) => toHex(interpolateRgba(parsed, scaleFraction(m, domain))))
      const bandRgba = bandColours.map(parseColour)
      const edges = levels
      const band = (v: number): readonly number[] => {
        let b = 0
        while (b < edges.length - 2 && v >= (edges[b + 1] ?? Number.POSITIVE_INFINITY)) b++
        return bandRgba[b] ?? [0, 0, 0, 0]
      }
      paint = (keep) => paintGrid(values, rows, cols, band, dimming(keep, true))
    } else paint = (keep) => paintGrid(values, rows, cols, ramp, dimming(keep, true))
  }
  const dx = g.cols > 1 ? ((xs[xs.length - 1] ?? 1) - (xs[0] ?? 0)) / (g.cols - 1) : 1
  const dy = g.rows > 1 ? ((ys[ys.length - 1] ?? 1) - (ys[0] ?? 0)) / (g.rows - 1) : 1
  const x0 = xs[0] ?? 0
  const x1 = xs[xs.length - 1] ?? 1
  const y0 = ys[0] ?? 0
  const y1 = ys[ys.length - 1] ?? 1
  const xr: Range = cells ? [x0 - dx / 2, x1 + dx / 2] : [x0, x1]
  const yr: Range = cells ? [y0 - dy / 2, y1 + dy / 2] : [y0, y1]
  const opacity = options.opacity ?? 1
  const series: SeriesOption[] = [
    imageSeries(paint(), xr, yr, {
      ...(options.name === undefined ? {} : { name: options.name }),
      opacity,
    }),
  ]
  // A selection on the colour bar repaints the image with the values outside it dimmed.
  const recolour = (keep: ValueFilter | undefined, encode: ImageEncoder): Record<string, unknown> => ({
    renderItem: imageRenderer(() => encode(paint(keep)), xr, yr, opacity),
  })
  const showBar = options.colourBar ?? true
  if (showBar) series.push({ type: 'scatter', name: '__colourbar', data: [], silent: true } as SeriesOption)
  return layer(
    series,
    { x: xr, y: yr, tight: true },
    showBar
      ? {
          colourBar: {
            min: domain[0],
            max: domain[domain.length - 1] ?? hi,
            colours: bandColours ?? sampleRamp(stops, 9),
            target: 1,
            select: { series: 0, recolour },
            ...(levels === undefined ? {} : { levels }),
            ...(options.label === undefined ? {} : { label: options.label }),
          },
        }
      : {},
  )
}

/**
 * An image from pixels, or from a 2-D (grey) or RGB(A) tensor as omai-engine's `toPixels` reads it (imshow-style),
 * covering a data rectangle (default one unit per pixel, row 0 at the top).
 *
 * @param source - Pixels or image data.
 * @param options - The x and y ranges it covers, the value range for grey images, opacity.
 * @returns The layer.
 * @example
 * ```ts
 * import { image } from 'omai-render/charts'
 * image([[0, 1], [1, 0]]).extent // => { x: [0, 2], y: [0, 2], tight: true }
 * ```
 */
export function image(
  source: ImageSource,
  options: { readonly x?: Range; readonly y?: Range; readonly range?: Range; readonly opacity?: number } = {},
): Layer {
  const pixels = toPixels(source, options.range === undefined ? {} : { range: options.range })
  const xr = options.x ?? [0, pixels.width]
  const yr = options.y ?? [0, pixels.height]
  return layer([imageSeries(pixels, xr, yr, options.opacity === undefined ? {} : { opacity: options.opacity })], {
    x: xr,
    y: yr,
    tight: true,
  })
}

/** Options of {@link matrix}. */
export interface MatrixOptions {
  readonly rowLabels?: readonly string[]
  readonly colLabels?: readonly string[]
  readonly ramp?: RampSpec
  readonly range?: Range
  readonly centre?: number
  /** Write each cell's value in it (default true for up to 400 cells). */
  readonly annotate?: boolean
  readonly format?: (value: number) => string
  readonly colourBar?: boolean
  readonly label?: string
  readonly mode?: ColourMode
}

interface CellApi {
  coord(p: readonly (number | string)[]): number[]
  size(p: readonly number[]): number[]
  value(d: number): number
}

/**
 * A matrix as labelled cells (confusion, correlation, attention, kernel matrices): row 0 at the top, columns left to
 * right, each cell coloured along a ramp with its value written in a contrasting colour.
 *
 * @param z - The matrix `[rows, cols]`.
 * @param options - Labels, ramp, range, centre, annotation, number format and colour bar.
 * @returns The layer, with categories on both axes.
 * @example
 * ```ts
 * import { matrix } from 'omai-render/charts'
 * const l = matrix([[1, 2], [3, 4]], { rowLabels: ['a', 'b'] })
 * l.yCategories // => ['b', 'a']
 * l.series[0].data.length // => 4
 * ```
 */
export function matrix(z: TensorLike, options: MatrixOptions = {}): Layer {
  const mode = options.mode ?? 'light'
  const g: Grid2D = grid2d(z, 'matrix')
  const rows = options.rowLabels ?? Array.from({ length: g.rows }, (_, i) => String(i))
  const cols = options.colLabels ?? Array.from({ length: g.cols }, (_, j) => String(j))
  if (rows.length !== g.rows || cols.length !== g.cols)
    throw new OmaiError('shape', 'matrix: give one label per row and per column')
  const [lo, hi] = options.range ?? extentOf(g.values) ?? [0, 1]
  const centre = options.centre
  const stops = rampStops(options.ramp, mode, centre !== undefined)
  const parsed = stops.map(parseColour)
  const domain: readonly [number, number] | readonly [number, number, number] =
    centre === undefined ? [lo, hi] : [Math.min(lo, centre), centre, Math.max(hi, centre)]
  const fmt = options.format ?? ((v: number): string => formatNumber(v, { digits: 3 }))
  const annotate = options.annotate ?? g.rows * g.cols <= 400
  const t = tokens(mode)
  const data: { value: number[]; colour: string; dimmed: string; text: string; ink: string }[] = []
  for (let r = 0; r < g.rows; r++)
    for (let c = 0; c < g.cols; c++) {
      const v = g.values[r * g.cols + c] ?? Number.NaN
      const colour = Number.isFinite(v) ? toHex(interpolateRgba(parsed, scaleFraction(v, domain))) : t.surface
      const ink = relativeLuminance(colour) > 0.4 ? '#1f2328' : '#ffffff'
      // Category index of the row on the y axis, which lists rows bottom-up so row 0 sits at the top.
      // Its colour outside a colour bar selection: blended towards the background.
      const dimmed = mixColours(t.bg, colour, DIMMED_OPACITY)
      data.push({ value: [c, g.rows - 1 - r, v], colour, dimmed, text: Number.isFinite(v) ? fmt(v) : '', ink })
    }
  // Cells outside a colour bar selection are drawn dimmed, values and all.
  const renderer =
    (keep?: ValueFilter) =>
    (params: { dataIndex: number }, api: CellApi): Record<string, unknown> => {
      const d = data[params.dataIndex]
      if (d === undefined) return { type: 'group', children: [] }
      const [cx = 0, cy = 0] = api.coord([api.value(0), api.value(1)])
      const [w = 0, h = 0] = api.size([1, 1])
      const v = d.value[2] ?? Number.NaN
      const dim = keep !== undefined && Number.isFinite(v) && keep(v) < 0.5
      return {
        type: 'group',
        children: [
          {
            type: 'rect',
            shape: { x: cx - w / 2, y: cy - h / 2, width: w, height: h },
            style: { fill: dim ? d.dimmed : d.colour },
          },
          ...(annotate
            ? [
                {
                  type: 'text',
                  style: {
                    text: d.text,
                    x: cx,
                    y: cy,
                    fill: d.ink,
                    opacity: dim ? DIMMED_OPACITY : 1,
                    align: 'center',
                    verticalAlign: 'middle',
                    fontSize: Math.max(8, Math.min(12, h / 3)),
                  },
                },
              ]
            : []),
        ],
      }
    }
  const series: SeriesOption[] = [
    {
      type: 'custom',
      name: '__matrix',
      data: data.map((d) => d.value),
      encode: { x: 0, y: 1, tooltip: 2 },
      renderItem: renderer(),
    } as unknown as SeriesOption,
  ]
  const showBar = options.colourBar ?? true
  if (showBar) series.push({ type: 'scatter', name: '__colourbar', data: [], silent: true } as SeriesOption)
  return layer(
    series,
    {},
    {
      xCategories: cols,
      yCategories: [...rows].reverse(),
      ...(showBar
        ? {
            colourBar: {
              min: domain[0],
              max: domain[domain.length - 1] ?? hi,
              colours: sampleRamp(stops, 9),
              target: 1,
              select: { series: 0, recolour: (keep: ValueFilter | undefined) => ({ renderItem: renderer(keep) }) },
              ...(options.label === undefined ? {} : { label: options.label }),
            },
          }
        : {}),
    },
  )
}

/**
 * A confusion matrix: true classes down, predicted classes across, optionally normalised by row (recall per class).
 *
 * @param counts - The counts `[k, k]`.
 * @param labels - The class names.
 * @param options - Row normalisation (default false), colour mode, and plot options.
 * @returns The ECharts option.
 * @index function A labelled confusion matrix: `confusionMatrixChart(counts, ['cat', 'dog'])`.
 * @example
 * ```ts
 * import { confusionMatrixChart } from 'omai-render/charts'
 * confusionMatrixChart([[5, 1], [2, 7]], ['cat', 'dog']).xAxis[0].name // => 'predicted'
 * ```
 */
export function confusionMatrixChart(
  counts: TensorLike,
  labels: readonly string[],
  options: PlotOptions & { readonly normalise?: boolean } = {},
): EChartsOption {
  const mode = modeOf(options)
  const g = grid2d(counts, 'confusionMatrixChart')
  let values: number[][] = Array.from({ length: g.rows }, (_, r) =>
    Array.from(g.values.subarray(r * g.cols, (r + 1) * g.cols)),
  )
  if (options.normalise === true)
    values = values.map((row) => {
      const s = row.reduce((a, b) => a + b, 0)
      return row.map((v) => (s > 0 ? v / s : 0))
    })
  const { normalise, ...plot } = options
  return cartesian(
    [
      matrix(values, {
        rowLabels: labels,
        colLabels: labels,
        ramp: 'blues',
        mode,
        ...(normalise === true ? { range: [0, 1], format: (v: number) => formatNumber(v, { digits: 2 }) } : {}),
      }),
    ],
    { x: { label: 'predicted' }, y: { label: 'true' }, ...plot },
  )
}

/**
 * A correlation matrix on a diverging ramp fixed to [-1, 1].
 *
 * @param corr - The correlations `[k, k]`.
 * @param labels - The variable names.
 * @param options - Plot options.
 * @returns The ECharts option.
 * @example
 * ```ts
 * import { correlationChart } from 'omai-render/charts'
 * correlationChart([[1, 0.5], [0.5, 1]], ['a', 'b']).visualMap[0].min // => -1
 * ```
 */
export function correlationChart(
  corr: TensorLike,
  labels: readonly string[],
  options: PlotOptions = {},
): EChartsOption {
  return cartesian(
    [matrix(corr, { rowLabels: labels, colLabels: labels, range: [-1, 1], centre: 0, mode: modeOf(options) })],
    options,
  )
}

/**
 * A heatmap chart of a gridded field (smooth by default) with its colour bar.
 *
 * @param z - Values `[rows, cols]`.
 * @param heat - Heatmap options (coordinates, smoothing, ramp, range, colour bar).
 * @param options - Plot options.
 * @returns The ECharts option.
 * @index function A heatmap of a 2-D array: `heatmapChart(z)`.
 * @example
 * ```ts
 * import { heatmapChart } from 'omai-render/charts'
 * heatmapChart([[0, 1], [2, 3]], { smooth: 'cells' }).xAxis[0].min // => -0.5
 * ```
 */
export function heatmapChart(z: TensorLike, heat: HeatmapOptions = {}, options: PlotOptions = {}): EChartsOption {
  return cartesian([heatmap(z, { mode: modeOf(options), ...heat })], options)
}
