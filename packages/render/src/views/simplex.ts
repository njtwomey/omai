/**
 * The view of a `Simplex` record: the triangle of three strategies (or the unit square of two populations) as an
 * outline with named vertices, a field of arrows over it, the paths of the state ending in their current point and the
 * fixed points marked by their stability. `simplexPoint` is the map from a state to the plane that the figure uses.
 */

import type { EChartsOption } from 'echarts'
import type { FixedPointClass, Simplex } from 'mdml-engine/exposition'
import { vectorField as arrows, type FieldArrow } from '../charts/fields.ts'
import { lines } from '../charts/lines.ts'
import { cartesian, type Layer } from '../charts/plot.ts'
import { points } from '../charts/points.ts'
import { type ViewOptions, viewFigure } from './_view.ts'
import { FIXED_POINT_LOOK } from './fields.ts'

const HEIGHT = Math.sqrt(3) / 2
const VERTICES: readonly (readonly [number, number])[] = [
  [0, 0],
  [1, 0],
  [0.5, HEIGHT],
]

/**
 * Where a state is drawn: barycentric coordinates on the triangle with its corners at `(0, 0)`, `(1, 0)` and
 * `(1/2, √3/2)`, the pair of probabilities itself on the square.
 *
 * @param shape - `triangle` or `square`.
 * @param p - The state (three probabilities, or two).
 * @returns The point in the plane.
 * @example
 * ```ts
 * import { simplexPoint } from 'mdml-render/views'
 * simplexPoint('triangle', [0, 0, 1]) // => [0.5, 0.866]
 * simplexPoint('square', [0.25, 0.75]) // => [0.25, 0.75]
 * ```
 */
export function simplexPoint(shape: Simplex['shape'], p: readonly number[]): [number, number] {
  if (shape === 'square') return [p[0] ?? 0, p[1] ?? 0]
  const [b, c] = [p[1] ?? 0, p[2] ?? 0]
  return [b + c / 2, c * HEIGHT]
}

// The direction of a change of state in the plane (the same linear map, without the offset).
function simplexDirection(shape: Simplex['shape'], d: readonly number[]): [number, number] {
  if (shape === 'square') return [d[0] ?? 0, d[1] ?? 0]
  return [(d[1] ?? 0) + (d[2] ?? 0) * 0.5, (d[2] ?? 0) * HEIGHT]
}

/**
 * The layers of a simplex: its outline, vertex names, the field's arrows, the paths with a marker at each current
 * point, and the fixed points by class.
 *
 * @param record - The record.
 * @param options - The colour mode.
 * @returns The layers.
 * @example
 * ```ts
 * import { simplexRecord } from 'mdml-engine/exposition'
 * import { simplexLayers } from 'mdml-render/views'
 * const r = simplexRecord({ shape: 'triangle', labels: ['a', 'b', 'c'], paths: [{ name: 'run', points: [[1, 0, 0], [0.5, 0.5, 0]] }] })
 * simplexLayers(r).length // => 4
 * ```
 */
export function simplexLayers(record: Simplex, options: ViewOptions = {}): Layer[] {
  const mode = options.mode ?? options.figure?.mode ?? 'light'
  const shape = record.shape
  const corners: readonly (readonly [number, number])[] =
    shape === 'triangle'
      ? VERTICES
      : [
          [0, 0],
          [1, 0],
          [1, 1],
          [0, 1],
        ]
  const outline = [...corners, corners[0] as readonly [number, number]]
  const layers: Layer[] = [
    lines([{ x: outline.map((c) => c[0]), y: outline.map((c) => c[1]), name: '__outline', colour: 'fg', width: 1.6 }], {
      mode,
    }),
  ]
  // The names sit outside the corners (for a square: beside the two axes it has, at the foot and the left).
  const names =
    shape === 'triangle'
      ? record.labels.map((text, i) => {
          const c = VERTICES[i] as readonly [number, number]
          const away: [number, number] = [c[0] - 0.5, c[1] - HEIGHT / 3]
          const n = Math.hypot(away[0], away[1])
          return { at: [c[0] + (0.1 * away[0]) / n, c[1] + (0.1 * away[1]) / n] as [number, number], text }
        })
      : [
          { at: [0.5, -0.07] as [number, number], text: `${record.labels[0] ?? ''} →` },
          { at: [-0.07, 0.5] as [number, number], text: `${record.labels[1] ?? ''} ↑` },
        ]
  layers.push(
    points(
      [
        {
          x: names.map((n) => n.at[0]),
          y: names.map((n) => n.at[1]),
          labels: names.map((n) => n.text),
          name: '__names',
          size: 0.1,
          symbol: 'none',
        },
      ],
      { mode },
    ),
  )
  if (record.field !== undefined) {
    const arrowList: FieldArrow[] = []
    const lengths = record.field.by.map((b) => Math.hypot(...simplexDirection(shape, b)))
    for (const [i, at] of record.field.at.entries()) {
      const d = simplexDirection(shape, record.field.by[i] ?? [])
      const m = lengths[i] ?? 0
      if (!(m > 1e-12)) continue
      const c = simplexPoint(shape, at)
      const h = 0.025
      arrowList.push({
        from: [c[0] - (h * d[0]) / m, c[1] - (h * d[1]) / m],
        to: [c[0] + (h * d[0]) / m, c[1] + (h * d[1]) / m],
        magnitude: m,
      })
    }
    layers.push(arrows(arrowList, { mode, colour: 'grid' }))
  }
  const heads: { x: number; y: number }[] = []
  ;(record.paths ?? []).forEach((path, k) => {
    const xy = path.points.map((p) => simplexPoint(shape, p))
    layers.push(
      lines([{ x: xy.map((p) => p[0]), y: xy.map((p) => p[1]), name: path.name, width: 1.8 }], {
        mode,
        colourOffset: k,
      }),
    )
    const last = xy[xy.length - 1]
    if (last !== undefined) heads.push({ x: last[0], y: last[1] })
  })
  if (heads.length > 0)
    layers.push(
      points(
        heads.map((h, k) => ({ x: [h.x], y: [h.y], name: `__head${k}`, size: 11, symbol: 'circle' })),
        { mode },
      ),
    )
  const fixed = (Object.keys(FIXED_POINT_LOOK) as FixedPointClass[]).flatMap((c) => {
    const at = (record.fixedPoints ?? []).filter((p) => p.stability === c)
    if (at.length === 0) return []
    const look = FIXED_POINT_LOOK[c]
    const xy = at.map((p) => simplexPoint(shape, p.at))
    return [
      {
        x: xy.map((p) => p[0]),
        y: xy.map((p) => p[1]),
        name: c,
        colour: look.colour,
        symbol: look.symbol,
        size: 13,
        outline: c !== 'unstable',
        ...(at.some((p) => p.label !== undefined) ? { labels: at.map((p) => p.label ?? '') } : {}),
      },
    ]
  })
  if (fixed.length > 0) layers.push(points(fixed, { mode }))
  return layers
}

/**
 * A simplex as a figure: the triangle (or square) with its field, paths and fixed points, on equal axes with no ticks.
 *
 * @param record - The record.
 * @param options - Mode and figure.
 * @returns The ECharts option.
 * @example
 * ```ts
 * import { simplexRecord } from 'mdml-engine/exposition'
 * import { simplexView } from 'mdml-render/views'
 * const r = simplexRecord({ shape: 'square', labels: ['row plays heads', 'column plays heads'], paths: [{ name: 'run', points: [[0.2, 0.9], [0.4, 0.7]] }] })
 * simplexView(r).series.length // => 4
 * ```
 */
export function simplexView(record: Simplex, options: ViewOptions = {}): EChartsOption {
  const fig = viewFigure(options, { aspect: record.shape === 'triangle' ? 1.15 : 1.1 })
  const triangle = record.shape === 'triangle'
  return {
    ...cartesian(simplexLayers(record, { ...options, figure: fig }), {
      mode: fig.mode,
      figure: fig,
      aspect: 'equal',
      x: { show: false, splitLine: false, limits: triangle ? [-0.2, 1.2] : [-0.2, 1.2] },
      y: { show: false, splitLine: false, limits: triangle ? [-0.15, HEIGHT + 0.15] : [-0.2, 1.2] },
      legend: (record.paths?.length ?? 0) + (record.fixedPoints?.length ?? 0) > 0,
    }),
    animation: false,
  }
}
