/**
 * The view of a `LandscapeSlice` record: the loss along one direction as a curve with the optimiser's iterates on it
 * and the lines that decide a step (Armijo, Wolfe), or over a plane as contours with the path of the iterates drawn
 * over them, or as a surface to orbit. The iterates are numbered by their order in the legend names and the first and
 * last are marked by shape as well as colour.
 */

import type { EChartsOption } from 'echarts'
import type { LandscapeSlice } from 'mdml-engine/exposition'
import { axes3d, type Paint, point3d, polyline3d, type Scene3D, scene3d, surface3d } from 'mdml-engine/scene'
import { contours } from '../charts/fields.ts'
import { lines } from '../charts/lines.ts'
import { cartesian, type Layer } from '../charts/plot.ts'
import { points } from '../charts/points.ts'
import { colourScale, sequentialRamp } from '../theme/palettes.ts'
import type { ColourMode } from '../theme/tokens.ts'
import { ariaOf, short, spanOf, type ViewOptions, viewFigure, wantsLog } from './_view.ts'

/**
 * The loss of a slice at a point of its first direction, by linear interpolation along the curve (clamped to the ends).
 *
 * @param record - A one-dimensional slice.
 * @param alpha - The coordinate along the direction.
 * @returns The loss.
 * @example
 * ```ts
 * import { landscapeSlice } from 'mdml-engine/exposition'
 * import { curveLossAt } from 'mdml-render/views'
 * const r = landscapeSlice({ dims: 1, axes: ['a'], x: [0, 2], curve: [1, 5] })
 * curveLossAt(r, 1) // => 3
 * ```
 */
export function curveLossAt(record: LandscapeSlice, alpha: number): number {
  const { x, curve } = record
  if (curve === undefined || x.length === 0) return Number.NaN
  if (alpha <= (x[0] as number)) return curve[0] as number
  for (let j = 1; j < x.length; j++) {
    const a = x[j - 1] as number
    const b = x[j] as number
    if (alpha <= b)
      return (curve[j - 1] as number) + ((alpha - a) / (b - a)) * ((curve[j] as number) - (curve[j - 1] as number))
  }
  return curve[curve.length - 1] as number
}

/**
 * The lowest loss on a slice and where it is.
 *
 * @param record - The slice.
 * @returns The coordinates (`[alpha]` or `[alpha, beta]`) and the loss.
 * @example
 * ```ts
 * import { landscapeSlice } from 'mdml-engine/exposition'
 * import { sliceMinimum } from 'mdml-render/views'
 * sliceMinimum(landscapeSlice({ dims: 1, axes: ['a'], x: [0, 1, 2], curve: [3, 1, 2] })) // => { at: [1], loss: 1 }
 * ```
 */
export function sliceMinimum(record: LandscapeSlice): { at: number[]; loss: number } {
  if (record.grid !== undefined && record.y !== undefined) {
    let best = { at: [record.x[0] ?? 0, record.y[0] ?? 0], loss: Number.POSITIVE_INFINITY }
    for (const [i, row] of record.grid.entries())
      for (const [j, v] of row.entries())
        if (v < best.loss) best = { at: [record.x[j] ?? 0, record.y[i] ?? 0], loss: v }
    return best
  }
  let best = { at: [record.x[0] ?? 0], loss: Number.POSITIVE_INFINITY }
  for (const [j, v] of (record.curve ?? []).entries()) if (v < best.loss) best = { at: [record.x[j] ?? 0], loss: v }
  return best
}

/**
 * Contour levels of a loss grid: evenly spaced in the loss, or in its logarithm when it spans decades (a landscape
 * is mostly flat floor and steep walls, and equal steps in the loss would draw only the walls).
 *
 * @param values - The loss values of the grid.
 * @param count - How many levels (default 12).
 * @returns The levels, ascending.
 * @example
 * ```ts
 * import { contourLevels } from 'mdml-render/views'
 * contourLevels([1, 100], 3) // => [1, 10, 100]
 * contourLevels([0, 4], 3) // => [0, 2, 4]
 * ```
 */
export function contourLevels(values: readonly number[], count = 12): number[] {
  const [lo, hi] = spanOf(values)
  const log = wantsLog(values.filter((v) => Number.isFinite(v)))
  const at = (k: number): number => {
    const f = count > 1 ? k / (count - 1) : 0
    return log ? Math.exp(Math.log(lo) + f * (Math.log(hi) - Math.log(lo))) : lo + f * (hi - lo)
  }
  return Array.from({ length: count }, (_, k) => {
    const v = at(k)
    // Round off the float noise of the logarithm at the ends.
    return k === 0 ? lo : k === count - 1 ? hi : Number(v.toPrecision(12))
  })
}

/**
 * The layers of a slice: a curve with its guides and iterates, or contours with the path over them.
 *
 * @param record - The slice.
 * @param options - The colour mode.
 * @returns The layers.
 * @example
 * ```ts
 * import { landscapeSlice } from 'mdml-engine/exposition'
 * import { landscapeSliceLayers } from 'mdml-render/views'
 * const r = landscapeSlice({ dims: 1, axes: ['a'], x: [0, 1, 2], curve: [3, 1, 2], paths: [{ coords: [[0], [1]] }] })
 * landscapeSliceLayers(r).length // => 3
 * ```
 */
export function landscapeSliceLayers(record: LandscapeSlice, options: ViewOptions = {}): Layer[] {
  const mode = options.mode ?? options.figure?.mode ?? 'light'
  const layers: Layer[] = []
  const lossName = record.lossName ?? 'loss'
  if (record.dims === 1) {
    layers.push(
      lines([{ x: [...record.x], y: [...(record.curve ?? [])], name: lossName, colour: 'tab0', width: 2 }], { mode }),
    )
    for (const [k, g] of (record.guides ?? []).entries())
      layers.push(
        lines(
          [
            {
              x: [g.from[0], g.to[0]],
              y: [g.from[1], g.to[1]],
              name: g.name,
              colour: `tab${(k + 2) % 10}`,
              width: 1.4,
              dash: 'dashed',
            },
          ],
          { mode },
        ),
      )
  } else if (record.grid !== undefined && record.y !== undefined) {
    layers.push(
      contours(record.grid, {
        x: [...record.x],
        y: [...record.y],
        colour: 'ramp',
        levels: contourLevels(record.grid.flat()),
        labels: false,
        opacity: 0.85,
        mode,
        name: '__contours',
      }),
    )
    const min = sliceMinimum(record)
    layers.push(
      points(
        [
          {
            x: [min.at[0] ?? 0],
            y: [min.at[1] ?? 0],
            name: 'lowest on the slice',
            colour: 'success',
            symbol: 'diamond',
            size: 12,
          },
        ],
        { mode },
      ),
    )
  }
  for (const [k, p] of (record.paths ?? []).entries()) {
    if (p.coords.length === 0) continue
    const name = p.name ?? `iterates ${k + 1}`
    const colour = `tab${(k + 1) % 10}`
    const xs = p.coords.map((c) => c[0] ?? 0)
    const ys =
      record.dims === 1
        ? p.coords.map((c, i) => p.loss?.[i] ?? curveLossAt(record, c[0] ?? 0))
        : p.coords.map((c) => c[1] ?? 0)
    layers.push(lines([{ x: xs, y: ys, name, colour, width: 1.6, opacity: 0.9 }], { mode }))
    const last = p.coords.length - 1
    layers.push(
      points(
        [
          { x: xs.slice(0, last), y: ys.slice(0, last), name: `__${name} iterates`, colour, size: 6, symbol: 'circle' },
          { x: [xs[0] ?? 0], y: [ys[0] ?? 0], name: `${name}: start`, colour, size: 10, symbol: 'rect' },
          { x: [xs[last] ?? 0], y: [ys[last] ?? 0], name: `${name}: now`, colour, size: 13, symbol: 'triangle' },
        ],
        { mode },
      ),
    )
  }
  return layers
}

/**
 * A landscape slice as a figure: the curve, or the contours, with the iterates and the guides over them.
 *
 * @param record - The slice.
 * @param options - Mode and figure.
 * @returns The ECharts option.
 * @example
 * ```ts
 * import { sampleLandscape } from 'mdml-engine/exposition'
 * import { landscapeSliceView } from 'mdml-render/views'
 * landscapeSliceView(sampleLandscape((a) => a * a, [-1, 0, 1])).series.length // => 1
 * ```
 */
export function landscapeSliceView(record: LandscapeSlice, options: ViewOptions = {}): EChartsOption {
  const fig = viewFigure(options, { aspect: 1.3 })
  const min = sliceMinimum(record)
  const where = record.dims === 1 ? `${short(min.at[0] ?? 0)}` : `(${short(min.at[0] ?? 0)}, ${short(min.at[1] ?? 0)})`
  const moves = (record.paths ?? []).reduce((n, p) => n + p.coords.length, 0)
  return {
    ...cartesian(landscapeSliceLayers(record, { ...options, figure: fig }), {
      mode: fig.mode,
      figure: fig,
      x: { label: record.axes[0] ?? 'direction', limits: [record.x[0] ?? 0, record.x[record.x.length - 1] ?? 1] },
      y:
        record.dims === 1
          ? { label: record.lossName ?? 'loss' }
          : {
              label: record.axes[1] ?? 'direction 2',
              limits: [record.y?.[0] ?? 0, record.y?.[(record.y?.length ?? 1) - 1] ?? 1],
            },
      legend: true,
    }),
    animation: false,
    ...ariaOf(
      `Slice of the ${record.lossName ?? 'loss'} landscape in ${record.axes.join(' and ')}, lowest at ${where} (${short(min.loss)}), with ${moves} optimiser iterates${record.caption === undefined ? '' : `: ${record.caption}`}`,
    ),
  }
}

/**
 * A two-dimensional slice as a surface to orbit: the loss as height over the plane (on a log scale when it spans
 * decades), coloured along a ramp, with the iterates as a path lying on it and the current one as a larger point.
 *
 * @param record - A two-dimensional slice.
 * @param options - The colour mode.
 * @returns The scene.
 * @example
 * ```ts
 * import { sampleLandscape } from 'mdml-engine/exposition'
 * import { landscapeSliceScene3d } from 'mdml-render/views'
 * const r = sampleLandscape((a, b) => a * a + b * b, [-1, 0, 1], [-1, 0, 1])
 * landscapeSliceScene3d(r).items.length // => 4 + 8
 * ```
 */
export function landscapeSliceScene3d(record: LandscapeSlice, options: { readonly mode?: ColourMode } = {}): Scene3D {
  const grid = record.grid ?? []
  const y = record.y ?? []
  const flat = grid.flat()
  const log = wantsLog(flat)
  const height = (v: number): number => (log ? Math.log(Math.max(v, 1e-300)) : v)
  const [lo, hi] = spanOf(flat.map(height))
  const span = hi > lo ? hi - lo : 1
  const unitX = (v: number): number =>
    ((v - (record.x[0] ?? 0)) / ((record.x[record.x.length - 1] ?? 1) - (record.x[0] ?? 0) || 1)) * 2 - 1
  const unitY = (v: number): number => ((v - (y[0] ?? 0)) / ((y[y.length - 1] ?? 1) - (y[0] ?? 0) || 1)) * 2 - 1
  const unitZ = (v: number): number => (height(v) - lo) / span
  const scale = colourScale(sequentialRamp('viridis', options.mode ?? 'light'), [0, 1])
  const faces = surface3d(
    record.x.map(unitX),
    y.map(unitY),
    grid.map((row) => row.map(unitZ)),
    { fill: (z) => scale(z) as Paint, stroke: 'grid', strokeWidth: 0.3 },
  )
  const lossAt = (c: readonly number[]): number => {
    // The surface height under an iterate: the nearest grid cell.
    const j = nearest(record.x, c[0] ?? 0)
    const i = nearest(y, c[1] ?? 0)
    return grid[i]?.[j] ?? 0
  }
  const items = (record.paths ?? []).flatMap((p) => {
    const pts = p.coords.map((c, k): [number, number, number] => [
      unitX(c[0] ?? 0),
      unitY(c[1] ?? 0),
      unitZ(p.loss?.[k] ?? lossAt(c)) + 0.02,
    ])
    const last = pts[pts.length - 1]
    return [
      ...(pts.length > 1 ? [polyline3d(pts, { stroke: 'tab3', strokeWidth: 2 })] : []),
      ...(last === undefined ? [] : [point3d(last, { fill: 'tab3', r: 4 })]),
    ]
  })
  return scene3d([...faces, ...items, ...axes3d(1.1)], {
    title: `${record.lossName ?? 'loss'} over ${record.axes.join(' and ')}${log ? ' (log height)' : ''}`,
  })
}

function nearest(axis: readonly number[], v: number): number {
  let best = 0
  let gap = Number.POSITIVE_INFINITY
  for (const [k, a] of axis.entries()) {
    const d = Math.abs(a - v)
    if (d < gap) {
      gap = d
      best = k
    }
  }
  return best
}
