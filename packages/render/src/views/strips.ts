/**
 * The view of a `Strip` record: tokens along x, a row of cells beneath them for each track (probabilities, an attention
 * row, gates) coloured by value, rows of tags written under the tokens, the chosen path as a line through its cells, and
 * the marked positions outlined.
 */

import type { EChartsOption, SeriesOption } from 'echarts'
import type { Strip } from 'omai-engine/exposition'
import { lines } from '../charts/lines.ts'
import { matrix } from '../charts/matrices.ts'
import { cartesian, type Layer, layer } from '../charts/plot.ts'
import { categoricalColour } from '../theme/palettes.ts'
import { tokens } from '../theme/tokens.ts'
import { ariaOf, short, spanOf, type ViewOptions, viewFigure } from './_view.ts'
import { cellShapes } from './matrix-marks.ts'

/** Options of {@link stripView}. */
export interface StripViewOptions extends ViewOptions {
  /** Write each cell's value in it (default when there are at most 400 cells). */
  readonly annotate?: boolean
}

/**
 * The figure aspect (width over height) that fits a strip: wide for many tokens, more rows taller.
 *
 * @param record - The record.
 * @returns The aspect.
 * @example
 * ```ts
 * import { stripRecord } from 'omai-engine/exposition'
 * import { stripAspect } from 'omai-render/views'
 * stripAspect(stripRecord({ tokens: ['a', 'b'], tracks: [{ name: 'p', values: [0, 1] }] })) // => 1.6
 * ```
 */
export function stripAspect(record: Strip): number {
  const rows = record.tracks.length + (record.tags?.length ?? 0)
  return Math.min(4, Math.max(1.6, (0.55 * (record.tokens.length + 2)) / (rows + 1.5)))
}

/**
 * A strip as a figure: tokens on x, a row of cells per track, rows of tags, the chosen path and the marked positions.
 *
 * @param record - The record.
 * @param options - Annotation, mode and figure.
 * @returns The ECharts option.
 * @example
 * ```ts
 * import { stripRecord } from 'omai-engine/exposition'
 * import { stripView } from 'omai-render/views'
 * const r = stripRecord({
 *   tokens: ['the', 'cat'],
 *   tracks: [{ name: 'noun', values: [0.1, 0.9] }, { name: 'det', values: [0.8, 0.1] }],
 *   path: [1, 0],
 *   marks: [1],
 * })
 * stripView(r).series.length // => 4
 * ```
 */
export function stripView(record: Strip, options: StripViewOptions = {}): EChartsOption {
  const fig = viewFigure(options, { aspect: stripAspect(record) })
  const mode = fig.mode
  const t = tokens(mode)
  const tags = record.tags ?? []
  const names = [...record.tracks.map((k) => k.name), ...tags.map((k) => k.name)]
  const n = record.tokens.length
  const rows = names.length
  const all = record.tracks.flatMap((k) => k.values)
  const [lo, hi] = spanOf(all)
  const range = record.range ?? (lo >= 0 && hi <= 1 ? [0, 1] : [lo, hi])
  // Tag rows carry no value: their cells are blank and the labels are drawn over them.
  const z = [...record.tracks.map((k) => [...k.values]), ...tags.map(() => new Array<number>(n).fill(Number.NaN))]
  const layers: Layer[] = [
    matrix(z, {
      rowLabels: names,
      colLabels: [...record.tokens],
      ramp: 'blues',
      range: [range[0], range[1]],
      colourBar: false,
      annotate: options.annotate ?? record.tracks.length * n <= 400,
      format: short,
      mode,
    }),
  ]
  // A cell of row r (from the top) sits at category index rows - 1 - r on the y axis, which lists rows bottom-up.
  const yOf = (r: number): number => rows - 1 - r
  if (tags.length > 0) {
    const labelled: SeriesOption = {
      type: 'scatter',
      name: '__tags',
      symbolSize: 0,
      silent: true,
      data: tags.flatMap((tag, k) =>
        tag.labels.map((text, i) => ({
          value: [i, yOf(record.tracks.length + k)],
          label: { show: true, formatter: text, color: categoricalColour(Math.abs(hash(text)) % 10) },
        })),
      ),
      label: { show: true, position: 'inside', fontFamily: t.font, fontSize: 11, fontWeight: 600 },
    } as SeriesOption
    layers.push(layer([labelled]))
  }
  if (record.path !== undefined) {
    const x = record.path.map((_, i) => i)
    const y = record.path.map((k) => yOf(k))
    layers.push(lines([{ x, y, name: '__path', colour: 'fg', width: 2, opacity: 0.85, markers: true }], { mode }))
  }
  if ((record.marks ?? []).length > 0)
    layers.push(
      cellShapes(
        (record.marks ?? []).map((m) => ({
          shape: 'box' as const,
          x0: m - 0.5,
          y0: rows - 0.5,
          x1: m + 0.5,
          y1: -0.5,
          colour: 'danger',
          width: 2,
        })),
        rows,
        n,
        mode,
      ),
    )
  return {
    ...cartesian(layers, { mode, figure: fig, legend: false, tooltip: false }),
    animation: false,
    ...ariaOf(
      `Strip of ${n} tokens (${record.tokens.join(' ')}) with ${record.tracks.length} tracks of ${record.valueName ?? 'values'}: ${record.tracks.map((k) => k.name).join(', ')}${(record.marks ?? []).length === 0 ? '' : `; positions ${(record.marks ?? []).join(', ')} are marked`}`,
    ),
  }
}

// A small stable hash of a tag, so equal tags share a colour.
function hash(s: string): number {
  let h = 0
  for (let i = 0; i < s.length; i++) h = (Math.imul(h, 31) + s.charCodeAt(i)) | 0
  return h
}
