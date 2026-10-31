/**
 * The view of a `StabilityRegion` record: the region of absolute stability filled on equal axes with its boundary
 * (`|R| = 1`) drawn as a line, the imaginary axis for reference, and the eigenvalues times the step size marked: a
 * circle where the method is stable, a cross where it is not (and a diamond where the point is off the plot), so the
 * verdict does not rest on colour. `stabilityVerdict` counts them for the caption.
 */

import type { EChartsOption, SeriesOption } from 'echarts'
import { markVerdict, type StabilityRegion } from 'mdml-engine/exposition'
import type { Pixels } from 'mdml-engine/scene'
import { contours } from '../charts/fields.ts'
import { lines } from '../charts/lines.ts'
import { imageSeries } from '../charts/matrices.ts'
import { cartesian, type Layer, layer } from '../charts/plot.ts'
import { points } from '../charts/points.ts'
import { parseColour } from '../theme/palettes.ts'
import { tokens } from '../theme/tokens.ts'
import { ariaOf, short, type ViewOptions, viewFigure } from './_view.ts'

// A filled cross, the marker of an unstable point.
const CROSS = 'path://M-3,-2L-2,-3L0,-1L2,-3L3,-2L1,0L3,2L2,3L0,1L-2,3L-3,2L-1,0Z'

/** How many of the marks lie in the stable region, outside it, and off the plot. */
export interface StabilityVerdict {
  readonly stable: number
  readonly unstable: number
  readonly off: number
}

/**
 * Count the marks of a record by where they lie.
 *
 * @param record - The record.
 * @returns The counts.
 * @example
 * ```ts
 * import { sampleStabilityRegion } from 'mdml-engine/exposition'
 * import { stabilityVerdict } from 'mdml-render/views'
 * const r = sampleStabilityRegion('euler', { numerator: [1, 1] }, { real: [-3, 1], imaginary: [-2, 2] }, {
 *   marks: [{ at: [-1, 0] }, { at: [-2.5, 0] }],
 * })
 * stabilityVerdict(r) // => { stable: 1, unstable: 1, off: 0 }
 * ```
 */
export function stabilityVerdict(record: StabilityRegion): StabilityVerdict {
  const v = { stable: 0, unstable: 0, off: 0 }
  for (const m of record.marks ?? []) v[markVerdict(record, m.at)]++
  return v
}

// The region as pixels, row 0 at the top: the stable cells tinted with the accent, the rest clear.
function regionPixels(record: StabilityRegion, accent: string): Pixels {
  const rows = record.y.length
  const cols = record.x.length
  const [r, g, b] = parseColour(accent)
  const rgba = new Uint8ClampedArray(rows * cols * 4)
  for (let row = 0; row < rows; row++) {
    const grid = record.modulus[rows - 1 - row] ?? []
    for (let c = 0; c < cols; c++)
      if ((grid[c] ?? Number.POSITIVE_INFINITY) <= 1) rgba.set([r, g, b, 110], 4 * (row * cols + c))
  }
  return { width: cols, height: rows, rgba }
}

/**
 * The layers of a stability region: the filled region, its boundary, the imaginary axis and the marks.
 *
 * @param record - The record.
 * @param options - The colour mode.
 * @returns The layers.
 * @example
 * ```ts
 * import { sampleStabilityRegion } from 'mdml-engine/exposition'
 * import { stabilityRegionLayers } from 'mdml-render/views'
 * const r = sampleStabilityRegion('euler', { numerator: [1, 1] }, { real: [-3, 1], imaginary: [-2, 2] })
 * stabilityRegionLayers(r).length // => 3
 * ```
 */
export function stabilityRegionLayers(record: StabilityRegion, options: ViewOptions = {}): Layer[] {
  const mode = options.mode ?? options.figure?.mode ?? 'light'
  const t = tokens(mode)
  const x: [number, number] = [record.x[0] ?? 0, record.x[record.x.length - 1] ?? 1]
  const y: [number, number] = [record.y[0] ?? 0, record.y[record.y.length - 1] ?? 1]
  const fill = [imageSeries(regionPixels(record, t.accent), x, y, { name: '__region', z: 1 })] as SeriesOption[]
  const layers: Layer[] = [
    layer(fill, { x, y }),
    contours(record.modulus, {
      x: [...record.x],
      y: [...record.y],
      levels: [1],
      colour: 'fg',
      width: 1.8,
      labels: false,
      mode,
      name: 'boundary |R| = 1',
    }),
    lines([{ x: [0, 0], y, name: '__imaginary axis', colour: 'muted', width: 1, dash: 'dotted' }], { mode }),
  ]
  const sets: {
    x: number[]
    y: number[]
    labels?: string[]
    name: string
    colour: string
    symbol: string
    size: number
  }[] = []
  const by = { stable: [] as number[], unstable: [] as number[], off: [] as number[] }
  for (const [k, m] of (record.marks ?? []).entries()) by[markVerdict(record, m.at)].push(k)
  const marks = record.marks ?? []
  const set = (kind: keyof typeof by, name: string, colour: string, symbol: string, size: number): void => {
    const index = by[kind]
    if (index.length === 0) return
    sets.push({
      x: index.map((k) => marks[k]?.at[0] ?? 0),
      y: index.map((k) => marks[k]?.at[1] ?? 0),
      name,
      colour,
      symbol,
      size,
      ...(marks.some((m) => m.label !== undefined) ? { labels: index.map((k) => marks[k]?.label ?? '') } : {}),
    })
  }
  set('stable', 'hλ stable', 'success', 'circle', 12)
  set('unstable', 'hλ unstable', 'danger', CROSS, 17)
  set('off', 'hλ off the plot', 'warning', 'diamond', 11)
  if (sets.length > 0) layers.push(points(sets, { mode }))
  return layers
}

/**
 * A stability region as a figure on equal axes.
 *
 * @param record - The record.
 * @param options - Mode and figure.
 * @returns The ECharts option.
 * @example
 * ```ts
 * import { sampleStabilityRegion } from 'mdml-engine/exposition'
 * import { stabilityRegionView } from 'mdml-render/views'
 * const r = sampleStabilityRegion('euler', { numerator: [1, 1] }, { real: [-3, 1], imaginary: [-2, 2] })
 * stabilityRegionView(r).series.length // => 3
 * ```
 */
export function stabilityRegionView(record: StabilityRegion, options: ViewOptions = {}): EChartsOption {
  const fig = viewFigure(options, { aspect: 1.3 })
  const v = stabilityVerdict(record)
  const x: [number, number] = [record.x[0] ?? 0, record.x[record.x.length - 1] ?? 1]
  const y: [number, number] = [record.y[0] ?? 0, record.y[record.y.length - 1] ?? 1]
  return {
    ...cartesian(stabilityRegionLayers(record, { ...options, figure: fig }), {
      mode: fig.mode,
      figure: fig,
      aspect: 'equal',
      x: { label: 'real part of hλ', limits: x },
      y: { label: 'imaginary part of hλ', limits: y },
      legend: true,
    }),
    animation: false,
    ...ariaOf(
      `Region of absolute stability of ${record.method}${record.order === undefined ? '' : `, order ${record.order}`}: ${v.stable} of the eigenvalues times the step size are inside it, ${v.unstable} outside${v.off > 0 ? `, ${v.off} off the plot` : ''}${record.stepSize === undefined ? '' : `, step size ${short(record.stepSize)}`}`,
    ),
  }
}
