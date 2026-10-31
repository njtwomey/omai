/**
 * The view of a `VectorField` record: arrows or streamlines over the plane (the grid's components interpolated), the
 * curves on which a component vanishes as dashed lines, and the fixed points marked by their stability class. The
 * layers are exposed so that a panel can draw more over the field (a trajectory, samples).
 */

import type { EChartsOption } from 'echarts'
import type { FixedPointClass, VectorField } from 'mdml-engine/exposition'
import { vectorField as arrows, fieldArrows, streamlines } from '../charts/fields.ts'
import { lines } from '../charts/lines.ts'
import { cartesian, type Layer } from '../charts/plot.ts'
import { points } from '../charts/points.ts'
import { type ViewOptions, viewFigure } from './_view.ts'

/** Options of {@link vectorFieldView} and {@link vectorFieldLayers}. */
export interface VectorFieldViewOptions extends ViewOptions {
  /** `arrows` (default), `streamlines`, or `both`. */
  readonly style?: 'arrows' | 'streamlines' | 'both'
  /** Arrows of equal length showing direction (default), or lengths proportional to the field. */
  readonly length?: 'unit' | 'magnitude'
  /** Arrows along each axis (default 17) and streamline seeds along each axis (default 8). */
  readonly density?: number
  /** One colour for every arrow (default: by magnitude). */
  readonly colour?: string
  /** Draw the nullclines (default true). */
  readonly nullclines?: boolean
  /** Draw the fixed points (default true). */
  readonly fixedPoints?: boolean
}

/** How a fixed point of each stability class is drawn: a colour token and a marker, shared by the views that classify them. */
export const FIXED_POINT_LOOK: Readonly<Record<FixedPointClass, { colour: string; symbol: string }>> = {
  stable: { colour: 'success', symbol: 'circle' },
  unstable: { colour: 'danger', symbol: 'emptyCircle' },
  saddle: { colour: 'warning', symbol: 'diamond' },
  centre: { colour: 'info', symbol: 'triangle' },
}

// The index i with xs[i] <= v < xs[i + 1], clamped to a cell, and the fraction across it.
function cell(xs: readonly number[], v: number): [number, number] {
  const last = xs.length - 2
  if (last < 0) return [0, 0]
  let i = 0
  while (i < last && (xs[i + 1] ?? Number.POSITIVE_INFINITY) <= v) i++
  const a = xs[i] ?? 0
  const b = xs[i + 1] ?? a + 1
  return [i, Math.min(1, Math.max(0, (v - a) / (b - a || 1)))]
}

/**
 * The record's field as a function: the grid's components bilinearly interpolated, held at the edge outside it.
 *
 * @param record - The record.
 * @returns The function `(x, y) => [u, v]`.
 * @example
 * ```ts
 * import { vectorFieldRecord } from 'mdml-engine/exposition'
 * import { vectorFieldAt } from 'mdml-render/views'
 * const r = vectorFieldRecord({ x: [0, 1], y: [0, 1], u: [[0, 2], [0, 2]], v: [[1, 1], [1, 1]] })
 * vectorFieldAt(r)(0.5, 0.5) // => [1, 1]
 * ```
 */
export function vectorFieldAt(record: VectorField): (x: number, y: number) => readonly [number, number] {
  const sample = (grid: readonly (readonly number[])[], i: number, j: number, fx: number, fy: number): number => {
    const at = (r: number, c: number): number =>
      grid[Math.min(r, grid.length - 1)]?.[Math.min(c, (grid[0]?.length ?? 1) - 1)] ?? 0
    const lo = at(j, i) * (1 - fx) + at(j, i + 1) * fx
    const hi = at(j + 1, i) * (1 - fx) + at(j + 1, i + 1) * fx
    return lo * (1 - fy) + hi * fy
  }
  return (x, y) => {
    const [i, fx] = cell(record.x, x)
    const [j, fy] = cell(record.y, y)
    return [sample(record.u, i, j, fx, fy), sample(record.v, i, j, fx, fy)]
  }
}

// A streamline through a seed: steps of a fixed arc length (so a slow field draws as far as a fast one) both ways.
function trace(
  f: (x: number, y: number) => readonly [number, number],
  seed: readonly [number, number],
  box: { readonly x: readonly [number, number]; readonly y: readonly [number, number] },
  h: number,
  steps: number,
): [number, number][] {
  const inside = (p: readonly [number, number]): boolean =>
    p[0] >= box.x[0] && p[0] <= box.x[1] && p[1] >= box.y[0] && p[1] <= box.y[1]
  const sx = box.x[1] - box.x[0] || 1
  const sy = box.y[1] - box.y[0] || 1
  // The direction in unit-box coordinates, so that h is a fraction of the box and the aspect does not skew it.
  const dir = (x: number, y: number): [number, number] => {
    const [u, v] = f(x, y)
    const nu = u / sx
    const nv = v / sy
    const n = Math.hypot(nu, nv)
    return n < 1e-12 ? [0, 0] : [(nu / n) * sx, (nv / n) * sy]
  }
  const walk = (sign: 1 | -1): [number, number][] => {
    const out: [number, number][] = []
    let p: [number, number] = [seed[0], seed[1]]
    for (let k = 0; k < steps; k++) {
      const [a, b] = dir(p[0], p[1])
      if (a === 0 && b === 0) break
      const [c, d] = dir(p[0] + (sign * h * a) / 2, p[1] + (sign * h * b) / 2)
      const next: [number, number] = [p[0] + sign * h * c, p[1] + sign * h * d]
      if (!inside(next)) break
      out.push(next)
      p = next
    }
    return out
  }
  return [...walk(-1).reverse(), [seed[0], seed[1]], ...walk(1)]
}

/**
 * The layers of a vector field: arrows (coloured by magnitude) or streamlines, its nullclines and its fixed points.
 *
 * @param record - The record.
 * @param options - Style, arrow lengths, density, colour and what to draw besides the field.
 * @returns The layers, to combine in a plot with others.
 * @example
 * ```ts
 * import { vectorFieldRecord } from 'mdml-engine/exposition'
 * import { vectorFieldLayers } from 'mdml-render/views'
 * const r = vectorFieldRecord({ x: [0, 1], y: [0, 1], u: [[1, 1], [1, 1]], v: [[0, 0], [0, 0]], fixedPoints: [{ at: [0.5, 0.5], stability: 'stable' }] })
 * vectorFieldLayers(r).length // => 2
 * ```
 */
export function vectorFieldLayers(record: VectorField, options: VectorFieldViewOptions = {}): Layer[] {
  const mode = options.mode ?? options.figure?.mode ?? 'light'
  const f = vectorFieldAt(record)
  const x: [number, number] = [record.x[0] ?? 0, record.x[record.x.length - 1] ?? 1]
  const y: [number, number] = [record.y[0] ?? 0, record.y[record.y.length - 1] ?? 1]
  const style = options.style ?? 'arrows'
  const layers: Layer[] = []
  if (style !== 'streamlines')
    layers.push(
      arrows(fieldArrows(f, { x, y, n: options.density ?? 17, length: options.length ?? 'unit' }), {
        mode,
        ...(options.colour === undefined ? (style === 'both' ? { colour: 'grid' } : {}) : { colour: options.colour }),
      }),
    )
  if (style !== 'arrows') {
    const n = Math.max(2, Math.min(options.density ?? 8, style === 'both' ? 8 : 12))
    const seeds: [number, number][] = []
    for (let j = 0; j < n; j++)
      for (let i = 0; i < n; i++)
        seeds.push([x[0] + ((i + 0.5) / n) * (x[1] - x[0]), y[0] + ((j + 0.5) / n) * (y[1] - y[0])])
    layers.push(
      streamlines(
        seeds.map((s) => trace(f, s, { x, y }, 0.02, 60)),
        { mode, colour: options.colour ?? 'muted', width: 1.1 },
      ),
    )
  }
  if (options.nullclines !== false)
    (record.nullclines ?? []).forEach((n, k) => {
      const xs: number[] = []
      const ys: number[] = []
      for (const p of n.paths) {
        for (const q of p) {
          xs.push(q[0])
          ys.push(q[1])
        }
        xs.push(Number.NaN)
        ys.push(Number.NaN)
      }
      layers.push(
        lines([{ x: xs, y: ys, name: n.name, colour: `tab${(k + 1) % 10}`, width: 1.6, dash: 'dashed' }], { mode }),
      )
    })
  if (options.fixedPoints !== false) {
    const sets = (Object.keys(FIXED_POINT_LOOK) as FixedPointClass[]).flatMap((c) => {
      const at = (record.fixedPoints ?? []).filter((p) => p.stability === c)
      if (at.length === 0) return []
      const look = FIXED_POINT_LOOK[c]
      return [
        {
          x: at.map((p) => p.at[0]),
          y: at.map((p) => p.at[1]),
          name: c,
          colour: look.colour,
          symbol: look.symbol,
          size: 13,
          outline: c !== 'unstable',
          ...(at.some((p) => p.label !== undefined) ? { labels: at.map((p) => p.label ?? '') } : {}),
        },
      ]
    })
    if (sets.length > 0) layers.push(points(sets, { mode }))
  }
  return layers
}

/**
 * A vector field as a figure: arrows (or streamlines) over the plane, with nullclines and classified fixed points.
 *
 * @param record - The record.
 * @param options - Style, arrow lengths, density, colour, mode and figure.
 * @returns The ECharts option.
 * @example
 * ```ts
 * import { sampleVectorField } from 'mdml-engine/exposition'
 * import { vectorFieldView } from 'mdml-render/views'
 * const r = sampleVectorField((x, y) => [y, -x], [-1, 0, 1], [-1, 0, 1])
 * vectorFieldView(r).series.length // => 1
 * ```
 */
export function vectorFieldView(record: VectorField, options: VectorFieldViewOptions = {}): EChartsOption {
  const fig = viewFigure(options, { aspect: 1.25 })
  const names = record.features ?? ['x', 'y']
  return {
    ...cartesian(vectorFieldLayers(record, { ...options, figure: fig }), {
      mode: fig.mode,
      figure: fig,
      x: { label: names[0] },
      y: { label: names[1] },
      legend: (record.nullclines?.length ?? 0) + (record.fixedPoints?.length ?? 0) > 0,
    }),
    animation: false,
  }
}
