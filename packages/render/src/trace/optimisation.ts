/**
 * Optimisation paths over an objective: a parameter channel's first two elements traced over a run, drawn as polylines
 * with start and end markers over the objective's filled contours.
 */

import type { SeriesOption } from 'echarts'
import { MdmlError } from 'mdml-engine/base'
import { type Run, tensorSeries } from 'mdml-engine/trace'
import { colourFor } from '../charts/_data.ts'
import { contours } from '../charts/fields.ts'
import { heatmap, type RampSpec } from '../charts/matrices.ts'
import { type Layer, layer } from '../charts/plot.ts'
import type { Range } from '../figure/aspect.ts'
import { extentOf } from '../figure/limits.ts'
import type { ColourMode } from '../theme/tokens.ts'

/**
 * The 2-D path of a parameter channel over a run (its first two elements at each recorded step), for drawing an
 * optimiser's trajectory over its objective.
 *
 * @param run - The trace.
 * @param name - The tensor channel holding the parameters.
 * @param dims - Which two flat elements to use (default 0 and 1).
 * @returns The points in step order.
 * @throws MdmlError with code `value` when the channel recorded no values.
 * @example
 * ```ts
 * import { parameterPath } from 'mdml-render/trace'
 * const ev = (step: number, values: number[]) => ({ type: 'tensor' as const, runId: 'r', step, name: 'x', role: 'param' as const, shape: [2], values })
 * parameterPath({ schema: 'mdml.run/1', events: [ev(0, [1, 2]), ev(1, [0, 1])] }, 'x') // => [[1, 2], [0, 1]]
 * ```
 */
export function parameterPath(run: Run, name: string, dims: readonly [number, number] = [0, 1]): [number, number][] {
  const out: [number, number][] = []
  for (const e of tensorSeries(run, name)) {
    if (e.values === undefined) continue
    const at = (d: number): number | undefined =>
      e.indices === undefined ? e.values?.[d] : e.values?.[e.indices.indexOf(d)]
    const a = at(dims[0])
    const b = at(dims[1])
    if (a !== undefined && b !== undefined) out.push([a, b])
  }
  if (out.length === 0) throw new MdmlError('value', `parameterPath: no recorded values for '${name}'`)
  return out
}

/** Options of {@link optimisationPaths}. */
export interface OptimisationPathOptions {
  readonly x: Range
  readonly y: Range
  /** Grid points per side for the objective (default 80). */
  readonly n?: number
  /** Contour levels (a count or the levels; default 12 on a log-spaced objective). */
  readonly levels?: number | readonly number[]
  /** Colour the background by the objective (default true) and draw contour lines (default true). */
  readonly fill?: boolean
  readonly lines?: boolean
  /** Use log(f - min + 1) for the colours and levels, which separates the valleys (default true). */
  readonly logScale?: boolean
  readonly ramp?: RampSpec
  readonly mode?: ColourMode
}

/**
 * Optimisation paths over an objective's contours: the objective sampled on a grid (filled and with contour lines),
 * and each path as a polyline with markers, its start hollow and its end filled.
 *
 * @param objective - The objective `f(x, y)`.
 * @param paths - Named paths (points in order).
 * @param options - Ranges, grid size, levels, fill, log scale, ramp and colour mode.
 * @returns The layers: the objective, then the paths.
 * @example
 * ```ts
 * import { optimisationPaths } from 'mdml-render/trace'
 * const ls = optimisationPaths((x, y) => x * x + y * y, [{ name: 'gd', points: [[1, 1], [0.5, 0.5], [0, 0]] }], { x: [-1, 1], y: [-1, 1], n: 20 })
 * ls.length // => 3
 * ```
 */
export function optimisationPaths(
  objective: (x: number, y: number) => number,
  paths: readonly {
    readonly name: string
    readonly points: readonly (readonly [number, number])[]
    readonly colour?: string
  }[],
  options: OptimisationPathOptions,
): Layer[] {
  const mode = options.mode ?? 'light'
  const n = Math.max(4, options.n ?? 80)
  const xs = Array.from({ length: n }, (_, j) => options.x[0] + ((options.x[1] - options.x[0]) * j) / (n - 1))
  const ys = Array.from({ length: n }, (_, i) => options.y[0] + ((options.y[1] - options.y[0]) * i) / (n - 1))
  let z = ys.map((y) => xs.map((x) => objective(x, y)))
  if (options.logScale ?? true) {
    const lo = extentOf(z.flat())?.[0] ?? 0
    z = z.map((row) => row.map((v) => Math.log(v - lo + 1)))
  }
  const layers: Layer[] = []
  if (options.fill ?? true)
    layers.push(heatmap(z, { x: xs, y: ys, ramp: options.ramp ?? 'blues', mode, colourBar: false, opacity: 0.85 }))
  if (options.lines ?? true)
    layers.push(
      contours(z, { x: xs, y: ys, levels: options.levels ?? 12, labels: false, colour: 'muted', mode, opacity: 0.7 }),
    )
  paths.forEach((p, i) => {
    const colour = colourFor(p.colour, mode, i + 1)
    const pts = p.points.map((q) => [q[0], q[1]])
    const last = pts[pts.length - 1]
    const series: SeriesOption[] = [
      {
        type: 'line',
        name: p.name,
        data: pts,
        showSymbol: true,
        symbolSize: 4,
        color: colour,
        lineStyle: { color: colour, width: 1.8 },
        itemStyle: { color: colour },
        z: 5,
      } as SeriesOption,
      {
        type: 'scatter',
        name: `__ends-${p.name}`,
        data: [
          { value: pts[0], itemStyle: { color: 'transparent', borderColor: colour, borderWidth: 2 } },
          ...(last === undefined ? [] : [{ value: last, itemStyle: { color: colour } }]),
        ],
        symbolSize: 10,
        z: 6,
      } as SeriesOption,
    ]
    layers.push(layer(series, { x: options.x, y: options.y, tight: true }))
  })
  return layers
}
