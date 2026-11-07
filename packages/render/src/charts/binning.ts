/**
 * Binning many points in the plane: hexagonal bins (hexbin) coloured by how many points fall in each, for scatter
 * plots too dense to read point by point.
 */

import type { SeriesOption } from 'echarts'
import { maxOf } from 'mdml-engine/base'
import { extentOf } from '../figure/limits.ts'
import { colourScale, sequentialRamp } from '../theme/palettes.ts'
import type { ColourMode } from '../theme/tokens.ts'
import { type Layer, layer } from './plot.ts'

/**
 * Hexagonal bins over points: a pointy-topped hexagonal grid of the given radius (in data units), each hexagon with at
 * least one point drawn and coloured by its count on a sequential ramp (log counts when asked).
 *
 * @param x - The points' x.
 * @param y - The points' y.
 * @param options - The hexagons' radius (default a 30th of the x range), log colouring and colour mode.
 * @returns The layer, with the bins' counts in its series' data.
 * @example
 * ```ts
 * import { hexbin } from 'mdml-render/charts'
 * hexbin([0, 0.01, 5], [0, 0.01, 5], { radius: 1 }).series[0].data.length // => 2
 * ```
 */
export function hexbin(
  x: readonly number[],
  y: readonly number[],
  options: { readonly radius?: number; readonly log?: boolean; readonly mode?: ColourMode } = {},
): Layer {
  const xr = extentOf(x) ?? [0, 1]
  const yr = extentOf(y) ?? [0, 1]
  const r = options.radius ?? ((xr[1] - xr[0]) / 30 || 1)
  const w = Math.sqrt(3) * r
  const h = 1.5 * r
  const bins = new Map<string, { cx: number; cy: number; count: number }>()
  for (let i = 0; i < x.length; i++) {
    const px = x[i] ?? 0
    const py = y[i] ?? 0
    // Nearest centre of the two offset rectangular lattices that make up the hexagonal grid.
    const row = Math.round(py / h)
    const col = Math.round(px / w - (row % 2 === 0 ? 0 : 0.5))
    let best = { row, col, d: Number.POSITIVE_INFINITY }
    for (const dr of [-1, 0, 1])
      for (const dc of [-1, 0, 1]) {
        const rr = row + dr
        const cc = col + dc
        const cx = (cc + (Math.abs(rr) % 2 === 0 ? 0 : 0.5)) * w
        const cy = rr * h
        const d = (px - cx) ** 2 + (py - cy) ** 2
        if (d < best.d) best = { row: rr, col: cc, d }
      }
    const k = `${best.row},${best.col}`
    const bin = bins.get(k)
    if (bin === undefined)
      bins.set(k, { cx: (best.col + (Math.abs(best.row) % 2 === 0 ? 0 : 0.5)) * w, cy: best.row * h, count: 1 })
    else bin.count++
  }
  const cells = [...bins.values()]
  const shade = (c: number): number => (options.log === true ? Math.log10(c) : c)
  const colour = colourScale(sequentialRamp('viridis', options.mode ?? 'light'), [
    shade(1),
    Math.max(shade(1) + 1e-9, maxOf(cells.map((c) => shade(c.count)))),
  ])
  const series = {
    type: 'custom',
    name: '__hexbin',
    renderItem: (_params: unknown, api: { value: (i: number) => number; coord: (p: number[]) => number[] }) => {
      const cx = api.value(0)
      const cy = api.value(1)
      const pts = Array.from({ length: 6 }, (_, k) => {
        const a = (Math.PI / 3) * k + Math.PI / 6
        return api.coord([cx + r * Math.cos(a), cy + r * Math.sin(a)])
      })
      return { type: 'polygon', shape: { points: pts }, style: { fill: colour(shade(api.value(2))) } }
    },
    data: cells.map((c) => [c.cx, c.cy, c.count]),
    silent: true,
  } as unknown as SeriesOption
  return layer([series], { x: [xr[0] - r, xr[1] + r], y: [yr[0] - r, yr[1] + r] })
}
