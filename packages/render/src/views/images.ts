/**
 * Player frames for `ImageSteps`: at each recorded step a mosaic with a row of small images per group (the data, the
 * model's reconstructions, its samples from fixed noise) beside the training objective, for autoencoders and generative
 * models.
 */

import type { EChartsOption, SeriesOption } from 'echarts'
import { maxOf } from 'mdml-engine/base'
import type { ImageSteps } from 'mdml-engine/exposition'
import type { Pixels } from 'mdml-engine/scene'
import { type Panel, subplots } from '../charts/compose.ts'
import { lines } from '../charts/lines.ts'
import { imageSeries } from '../charts/matrices.ts'
import { cartesian, layer } from '../charts/plot.ts'
import { points } from '../charts/points.ts'
import { type Figure, figure } from '../figure/figure.ts'
import { gridSpec } from '../figure/grid.ts'
import { parseColour } from '../theme/palettes.ts'
import { type ColourMode, tokens } from '../theme/tokens.ts'
import { ariaOf, short, type ViewFrames, type ViewOptions, viewFigure } from './_view.ts'

/** Options of {@link imageFrames}. */
export interface ImageViewOptions extends ViewOptions {
  /** The most images drawn per group (default 12). */
  readonly columns?: number
  /** Draw the objective beneath the mosaic when the steps carry one (default true). */
  readonly objective?: boolean
}

// The width of the group names' column, in tiles.
const LABEL_TILES = 2.2
// The height of a caption strip under each image, in pixels of the mosaic.
const CAPTION_PX = 14

const hasCaptions = (record: ImageSteps): boolean => record.steps.some((s) => s.labels !== undefined)

// The scale of an image's pixels: k by k mosaic pixels each, and the share of a row the image takes (the rest holds
// its caption).
function tileGeometry(record: ImageSteps): { k: number; share: number } {
  const k = Math.max(2, Math.min(8, Math.floor(64 / Math.max(record.height, record.width))))
  const image = record.height * k + 1
  return { k, share: hasCaptions(record) ? image / (image + CAPTION_PX) : 1 }
}

/**
 * The mosaic of one step's images: a row of tiles per group, grey levels mapped from the background to the ink, a
 * pixel-wide gap between tiles.
 */
function mosaic(
  record: ImageSteps,
  images: ImageSteps['steps'][number]['images'],
  columns: number,
  mode: ColourMode,
): Pixels {
  const t = tokens(mode)
  const bg = parseColour(t.bg)
  const ink = parseColour(t.fg)
  const gap = parseColour(t.grid)
  const { height: h, width: w } = record
  const channels = record.channels ?? 1
  const [lo, hi] = record.range ?? [0, 1]
  const span = hi > lo ? hi - lo : 1
  const unit = (x: number | undefined): number => Math.min(1, Math.max(0, ((x ?? lo) - lo) / span))
  // Grey levels run from the background to the ink; colour values are drawn as they are.
  // Each pixel of an image as k by k screen pixels, a one-pixel line between tiles.
  const k = tileGeometry(record).k
  const tileW = w * k + 1
  const tileH = h * k + 1 + (hasCaptions(record) ? CAPTION_PX : 0)
  const width = columns * tileW + 1
  const height = record.groups.length * tileH + 1
  const rgba = new Uint8ClampedArray(width * height * 4)
  for (let i = 0; i < width * height; i++) {
    rgba[4 * i] = gap[0] ?? 0
    rgba[4 * i + 1] = gap[1] ?? 0
    rgba[4 * i + 2] = gap[2] ?? 0
    rgba[4 * i + 3] = 255
  }
  for (const [g, group] of images.entries())
    for (const [c, img] of group.slice(0, columns).entries())
      for (let r = 0; r < h; r++)
        for (let q = 0; q < w; q++) {
          const at = (r * w + q) * channels
          const v = unit(img[at])
          const colour =
            channels === 3
              ? [0, 1, 2].map((ch) => 255 * unit(img[at + ch]))
              : [0, 1, 2].map((ch) => (bg[ch] ?? 0) + v * ((ink[ch] ?? 0) - (bg[ch] ?? 0)))
          for (let a = 0; a < k; a++)
            for (let b = 0; b < k; b++) {
              const y = 1 + g * tileH + r * k + a
              const x = 1 + c * tileW + q * k + b
              const p = 4 * (y * width + x)
              rgba[p] = colour[0] ?? 0
              rgba[p + 1] = colour[1] ?? 0
              rgba[p + 2] = colour[2] ?? 0
            }
        }
  return { width, height, rgba }
}

// The images drawn per group.
function columnsOf(record: ImageSteps, columns: number | undefined): number {
  const most = Math.max(1, maxOf(record.steps.flatMap((s) => s.images.map((g) => g.length))))
  return Math.max(1, Math.min(columns ?? 12, most))
}

// The mosaic's own shape, the names' column included.
function mosaicAspectOf(record: ImageSteps, columns: number): number {
  return (
    ((columns + LABEL_TILES) * ((record.width / record.height) * tileGeometry(record).share)) /
    Math.max(1, record.groups.length)
  )
}

/**
 * The figure aspect (width over height) that fits an image record's mosaic, with room for its objective beneath.
 *
 * @param record - The record.
 * @param columns - The most images drawn per group (default 12).
 * @param objective - Whether the objective is drawn beneath when the steps carry one (default true).
 * @returns The aspect.
 */
export function imageAspect(record: ImageSteps, columns?: number, objective = true): number {
  const a = mosaicAspectOf(record, columnsOf(record, columns))
  const withObjective = objective && record.steps.filter((s) => s.objective !== undefined).length > 1
  return withObjective ? Math.max(1.2, 1 / (1 / a + 0.45)) : Math.max(1.2, a)
}

/**
 * Image steps as frames for a player: at each recorded step a mosaic with a row of images per group (the data, the
 * model's reconstructions, its samples from fixed noise), the groups named on the left, beside the objective with the
 * step marked when the steps carry one.
 *
 * @param record - The record.
 * @param options - Mode, figure and the images per group.
 * @returns The frames.
 */
export function imageFrames(record: ImageSteps, options: ImageViewOptions = {}): ViewFrames {
  const count = Math.max(1, record.steps.length)
  const columns = columnsOf(record, options.columns)
  const rows = record.groups.length
  const objective = record.steps.filter((s) => s.objective !== undefined)
  const withObjective = options.objective !== false && objective.length > 1
  const mosaicAspect = mosaicAspectOf(record, columns)
  const fig = viewFigure(options, { aspect: imageAspect(record, options.columns, options.objective !== false) })
  const mode = fig.mode
  const t = tokens(mode)
  const stepName = record.stepName ?? 'step'
  // Tiles are as tall as they are wide in data units: an image h by w spans w/h by 1.
  const share = tileGeometry(record).share
  const aspect = (record.width / record.height) * share
  const option = (frame: number): EChartsOption => {
    const at = Math.min(Math.max(frame, 0), count - 1)
    const step = record.steps[at]
    const pixels = mosaic(record, step?.images ?? [], columns, mode)
    const x1 = columns * aspect
    const series: SeriesOption[] = [imageSeries(pixels, [0, x1], [0, rows], { name: '__images' })]
    series.push({
      type: 'scatter',
      name: '__groups',
      data: record.groups.map((g, i) => ({ value: [-0.15, rows - i - share / 2], name: g })),
      symbolSize: 0,
      silent: true,
      label: {
        show: true,
        position: 'left',
        formatter: '{b}',
        color: t.fgSecondary,
        fontFamily: t.font,
        fontSize: 11,
      },
    } as SeriesOption)
    // A caption under each image, in the strip its row keeps for it.
    if (step?.labels !== undefined)
      series.push({
        type: 'scatter',
        name: '__captions',
        data: step.labels.flatMap((group, g) =>
          group.slice(0, columns).map((text, c) => ({
            value: [(c + 0.5) * aspect, rows - g - 1 + (1 - share) / 2],
            name: text,
          })),
        ),
        symbolSize: 0,
        silent: true,
        label: {
          show: true,
          position: 'inside',
          formatter: '{b}',
          color: t.fgSecondary,
          fontFamily: t.font,
          fontSize: 10,
        },
      } as SeriesOption)
    const main: Panel = {
      layers: [layer(series, { x: [-LABEL_TILES * aspect, x1], y: [0, rows], tight: true })],
      options: { x: { show: false, splitLine: false }, y: { show: false, splitLine: false }, aspect: 'equal' },
    }
    if (!withObjective || step === undefined)
      return cartesian(main.layers, { ...main.options, figure: fig, mode, legend: false, tooltip: false })
    const xs = objective.map((s) => s.iteration)
    const ys = objective.map((s) => s.objective ?? 0)
    const k = Math.max(
      0,
      xs.findLastIndex((x) => x <= step.iteration),
    )
    const progress: Panel = {
      layers: [
        lines([{ x: xs, y: ys, name: record.objectiveName ?? 'objective', colour: 'fgSecondary', width: 1.5 }], {
          mode,
        }),
        points([{ x: [xs[k] ?? 0], y: [ys[k] ?? 0], name: '__now', colour: 'accent', size: 9 }], { mode }),
      ],
      options: { x: { label: stepName }, y: { label: record.objectiveName ?? 'objective' } },
    }
    // The objective beneath the mosaic, so the images get the full width.
    const grid = gridSpec(2, 1, {
      heightRatios: [1 / mosaicAspect, 0.35],
      hspace: 48,
      cells: [
        { id: 'main', row: 0, col: 0 },
        { id: 'objective', row: 1, col: 0 },
      ],
    })
    return subplots(grid, { main, objective: progress }, { figure: fig, legend: false, tooltip: false })
  }
  return {
    count,
    label: (f) => {
      const s = record.steps[Math.min(Math.max(f, 0), count - 1)]
      if (s === undefined) return 'no images'
      const last = record.steps[record.steps.length - 1]?.iteration ?? 0
      return `${stepName} ${s.iteration}/${last}${s.objective === undefined ? '' : ` · ${short(s.objective)}`}`
    },
    option,
  }
}

/**
 * The index of the step of an image record to show at a tape step: the latest recorded at or before it (the first when
 * none is yet).
 *
 * @param record - The record.
 * @param step - The tape step.
 * @returns The index into `record.steps`.
 * @example
 * ```ts
 * import { imageSteps } from 'mdml-engine/exposition'
 * import { imageStepIndex } from 'mdml-render/views'
 * const r = imageSteps({ method: 'm', height: 1, width: 1, groups: ['g'], steps: [{ iteration: 0, images: [[[0]]] }, { iteration: 10, images: [[[1]]] }] })
 * imageStepIndex(r, 12) // => 1
 * ```
 */
export function imageStepIndex(record: ImageSteps, step: number): number {
  return Math.max(
    0,
    record.steps.findLastIndex((s) => s.iteration <= step),
  )
}

// The padding of a grid of images: the axes are hidden, so the figure is nearly all picture.
const GRID_PADDING = 8

/**
 * The figure that fits a grid of images: the given width, a height that makes the tiles square (the captions' strip
 * and the names' column counted) and almost no padding.
 *
 * @param record - The record.
 * @param width - The figure's width in pixels (default 440).
 * @param columns - The most images drawn per group (default 12).
 * @returns The figure.
 * @example
 * ```ts
 * import { imageSteps } from 'mdml-engine/exposition'
 * import { imageGridFigure } from 'mdml-render/views'
 * const r = imageSteps({ method: 'm', height: 2, width: 2, groups: ['g'], steps: [{ iteration: 0, images: [[[0, 0, 0, 0], [1, 1, 1, 1]]] }] })
 * imageGridFigure(r, 440).width // => 440
 * ```
 */
export function imageGridFigure(record: ImageSteps, width = 440, columns?: number): Figure {
  const inner = width - 2 * GRID_PADDING
  const height = Math.round(inner / mosaicAspectOf(record, columnsOf(record, columns))) + 2 * GRID_PADDING
  return figure({
    width,
    height,
    padding: { top: GRID_PADDING, right: GRID_PADDING, bottom: GRID_PADDING, left: GRID_PADDING },
  })
}

/**
 * The images of a record at a tape step as a grid: a row of small images per group with the group's name, and the
 * captions under the images when the step has them; no objective, no player.
 *
 * @param record - The record.
 * @param step - The tape step (default the last recorded one).
 * @param options - Mode, figure and the images per group.
 * @returns The ECharts option.
 * @example
 * ```ts
 * import { imageSteps } from 'mdml-engine/exposition'
 * import { imageGridView } from 'mdml-render/views'
 * const r = imageSteps({ method: 'm', height: 1, width: 2, groups: ['g'], steps: [{ iteration: 0, images: [[[0, 1]]], labels: [['a']] }] })
 * imageGridView(r).series.length // => 3
 * ```
 */
export function imageGridView(record: ImageSteps, step?: number, options: ImageViewOptions = {}): EChartsOption {
  const frames = imageFrames(record, {
    ...options,
    objective: false,
    figure: options.figure ?? imageGridFigure(record, 440, options.columns),
  })
  return {
    ...frames.option(step === undefined ? frames.count - 1 : imageStepIndex(record, step)),
    ...ariaOf(`Grid of images in ${record.groups.length} rows: ${record.groups.join(', ')}`),
  }
}
