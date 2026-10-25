/**
 * Player frames for `ImageSteps`: at each recorded step a mosaic with a row of small images per group (the data, the
 * model's reconstructions, its samples from fixed noise) beside the training objective, for autoencoders and generative
 * models.
 */

import type { EChartsOption, SeriesOption } from 'echarts'
import type { ImageSteps } from 'mdml-engine/exposition'
import type { Pixels } from 'mdml-engine/scene'
import { type Panel, subplots } from '../charts/compose.ts'
import { lines } from '../charts/lines.ts'
import { imageSeries } from '../charts/matrices.ts'
import { cartesian, layer } from '../charts/plot.ts'
import { points } from '../charts/points.ts'
import { gridSpec } from '../figure/grid.ts'
import { parseColour } from '../theme/palettes.ts'
import { type ColourMode, tokens } from '../theme/tokens.ts'
import { short, type ViewFrames, type ViewOptions, viewFigure } from './_view.ts'

/** Options of {@link imageFrames}. */
export interface ImageViewOptions extends ViewOptions {
  /** The most images drawn per group (default 12). */
  readonly columns?: number
}

// The width of the group names' column, in tiles.
const LABEL_TILES = 2.2

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
  const k = Math.max(2, Math.min(8, Math.floor(64 / Math.max(h, w))))
  const tileW = w * k + 1
  const tileH = h * k + 1
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
  const most = Math.max(1, ...record.steps.flatMap((s) => s.images.map((g) => g.length)))
  return Math.max(1, Math.min(columns ?? 12, most))
}

// The mosaic's own shape, the names' column included.
function mosaicAspectOf(record: ImageSteps, columns: number): number {
  return ((columns + LABEL_TILES) * (record.width / record.height)) / Math.max(1, record.groups.length)
}

/**
 * The figure aspect (width over height) that fits an image record's mosaic, with room for its objective beneath.
 *
 * @param record - The record.
 * @param columns - The most images drawn per group (default 12).
 * @returns The aspect.
 */
export function imageAspect(record: ImageSteps, columns?: number): number {
  const a = mosaicAspectOf(record, columnsOf(record, columns))
  const withObjective = record.steps.filter((s) => s.objective !== undefined).length > 1
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
  const withObjective = objective.length > 1
  const mosaicAspect = mosaicAspectOf(record, columns)
  const fig = viewFigure(options, { aspect: imageAspect(record, options.columns) })
  const mode = fig.mode
  const t = tokens(mode)
  const stepName = record.stepName ?? 'step'
  // Tiles are as tall as they are wide in data units: an image h by w spans w/h by 1.
  const aspect = record.width / record.height
  const option = (frame: number): EChartsOption => {
    const at = Math.min(Math.max(frame, 0), count - 1)
    const step = record.steps[at]
    const pixels = mosaic(record, step?.images ?? [], columns, mode)
    const x1 = columns * aspect
    const series: SeriesOption[] = [imageSeries(pixels, [0, x1], [0, rows], { name: '__images' })]
    series.push({
      type: 'scatter',
      name: '__groups',
      data: record.groups.map((g, i) => ({ value: [-0.15, rows - i - 0.5], name: g })),
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
