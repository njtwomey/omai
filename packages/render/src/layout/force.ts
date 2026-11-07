/**
 * Graph layouts without a hierarchy: a seeded, deterministic force-directed layout (Fruchterman and Reingold: edges
 * pull like springs, all nodes repel, and a cooling temperature runs until the energy settles), nodes evenly around a
 * circle, and a spectral layout that places nodes by the Laplacian eigenvectors with the smallest nonzero eigenvalues,
 * so well-connected nodes sit together.
 */

import { maxOf, minOf } from 'mdml-engine/base'
import { eigh } from 'mdml-engine/numerics/linalg'
import { key as makeKey, uniform } from 'mdml-engine/random'
import { toArray } from 'mdml-engine/tensor'
import { type Box, type GraphLayout, type LayoutGraph, normaliseLayout, straightRoutes } from './graph.ts'

/** Options of {@link forceLayout}. */
export interface ForceOptions {
  /** The seed of the starting positions (default 0): the same seed always gives the same layout. */
  readonly seed?: number
  /** The ideal edge length (default 80). */
  readonly length?: number
  /** The most iterations (default 300). */
  readonly maxIter?: number
  /** Stop when the energy changes by less than this fraction between iterations (default 1e-4). */
  readonly tol?: number
  /** Space around the drawing (default 16). */
  readonly margin?: number
}

/** What {@link forceLayout} reports besides the layout: how many iterations ran and the final energy. */
export interface ForceLayout extends GraphLayout {
  readonly iterations: number
  readonly energy: number
}

function boxes(graph: LayoutGraph, xy: readonly (readonly [number, number])[]): Record<string, Box> {
  const out: Record<string, Box> = {}
  graph.nodes.forEach((n, i) => {
    const [x, y] = xy[i] ?? [0, 0]
    out[n.id] = { x, y, width: n.width, height: n.height }
  })
  return out
}

/**
 * A force-directed layout (Fruchterman and Reingold): edges pull like springs, every pair of nodes pushes apart, and
 * the temperature cools until the energy settles. Seeded and deterministic: the starting positions come from a key, and
 * the run stops on an energy-based rule rather than a clock.
 *
 * @param graph - Nodes with sizes and undirected edges.
 * @param options - Seed, edge length, iteration cap, tolerance and margin.
 * @returns The layout, the iterations it took and its final energy.
 * @index function A seeded force-directed layout: `forceLayout(graph, { seed: 1 })`.
 * @example
 * ```ts
 * import { forceLayout } from 'mdml-render/layout'
 * const g = { nodes: ['a', 'b', 'c'].map((id) => ({ id, width: 10, height: 10 })), edges: [{ from: 'a', to: 'b' }, { from: 'b', to: 'c' }] }
 * forceLayout(g, { seed: 1 }).nodes['a'].x === forceLayout(g, { seed: 1 }).nodes['a'].x // => true
 * ```
 */
export function forceLayout(graph: LayoutGraph, options: ForceOptions = {}): ForceLayout {
  const { seed = 0, length = 80, maxIter = 300, tol = 1e-4, margin = 16 } = options
  const n = graph.nodes.length
  const index = new Map(graph.nodes.map((v, i) => [v.id, i]))
  const edges = graph.edges.flatMap((e) => {
    const a = index.get(e.from)
    const b = index.get(e.to)
    return a === undefined || b === undefined || a === b ? [] : [[a, b] as const]
  })
  const side = length * Math.sqrt(Math.max(n, 1))
  const start = n === 0 ? [] : (toArray(uniform(makeKey(seed), [n, 2], { low: 0, high: side })) as number[][])
  const xy: [number, number][] = start.map(([x = 0, y = 0]) => [x, y])
  const k = length
  let temperature = side / 10
  let energy = Number.POSITIVE_INFINITY
  let iterations = 0
  for (; iterations < maxIter; iterations++) {
    const disp: [number, number][] = xy.map(() => [0, 0])
    let e = 0
    for (let i = 0; i < n; i++)
      for (let j = i + 1; j < n; j++) {
        const pi = xy[i] as [number, number]
        const pj = xy[j] as [number, number]
        let dx = pi[0] - pj[0]
        let dy = pi[1] - pj[1]
        let d = Math.hypot(dx, dy)
        if (d < 1e-9) {
          // Coincident points separate along a fixed direction, so the run stays deterministic.
          dx = 1e-3 * (i - j)
          dy = 1e-3
          d = Math.hypot(dx, dy)
        }
        const f = (k * k) / d
        const di = disp[i] as [number, number]
        const dj = disp[j] as [number, number]
        di[0] += (dx / d) * f
        di[1] += (dy / d) * f
        dj[0] -= (dx / d) * f
        dj[1] -= (dy / d) * f
        e += k * k * Math.log(d)
      }
    for (const [a, b] of edges) {
      const pa = xy[a] as [number, number]
      const pb = xy[b] as [number, number]
      const dx = pa[0] - pb[0]
      const dy = pa[1] - pb[1]
      const d = Math.max(Math.hypot(dx, dy), 1e-9)
      const f = (d * d) / k
      const da = disp[a] as [number, number]
      const db = disp[b] as [number, number]
      da[0] -= (dx / d) * f
      da[1] -= (dy / d) * f
      db[0] += (dx / d) * f
      db[1] += (dy / d) * f
      e += (d * d * d) / (3 * k)
    }
    xy.forEach((p, i) => {
      const [dx, dy] = disp[i] as [number, number]
      const d = Math.hypot(dx, dy)
      if (d > 0) {
        const step = Math.min(d, temperature)
        p[0] += (dx / d) * step
        p[1] += (dy / d) * step
      }
    })
    temperature *= 0.97
    const settled = Math.abs(energy - e) <= tol * Math.abs(e)
    energy = e
    if (settled) {
      iterations++
      break
    }
  }
  const placed = boxes(graph, xy)
  return {
    ...normaliseLayout(placed, straightRoutes(placed, graph.edges), {}, margin),
    iterations,
    energy: Number.isFinite(energy) ? energy : 0,
  }
}

/**
 * Nodes evenly around a circle, in order.
 *
 * @param graph - The graph.
 * @param options - The radius (default enough for the nodes to sit apart) and margin.
 * @returns The layout.
 * @example
 * ```ts
 * import { circularLayout } from 'mdml-render/layout'
 * const g = { nodes: ['a', 'b', 'c', 'd'].map((id) => ({ id, width: 10, height: 10 })), edges: [] }
 * const l = circularLayout(g, { radius: 50 })
 * Math.round(l.nodes['c'].y - l.nodes['a'].y) // => 100
 * ```
 */
export function circularLayout(
  graph: LayoutGraph,
  options: { readonly radius?: number; readonly margin?: number } = {},
): GraphLayout {
  const n = graph.nodes.length
  const widest = Math.max(10, maxOf(graph.nodes.map((v) => Math.max(v.width, v.height))))
  const radius = options.radius ?? Math.max(widest, (n * (widest + 12)) / (2 * Math.PI))
  const xy = graph.nodes.map((_, i): [number, number] => {
    const a = (2 * Math.PI * i) / Math.max(n, 1) - Math.PI / 2
    return [radius * Math.cos(a), radius * Math.sin(a)]
  })
  const placed = boxes(graph, xy)
  return normaliseLayout(placed, straightRoutes(placed, graph.edges), {}, options.margin ?? 16)
}

/**
 * A spectral layout: each node at its entries in the two eigenvectors of the graph Laplacian with the smallest nonzero
 * eigenvalues, scaled to `size`, so well-connected nodes sit together.
 *
 * @param graph - The graph (undirected).
 * @param options - The drawing's size (default 400) and margin.
 * @returns The layout.
 * @example
 * ```ts
 * import { spectralLayout } from 'mdml-render/layout'
 * const g = { nodes: ['a', 'b', 'c', 'd'].map((id) => ({ id, width: 10, height: 10 })), edges: [{ from: 'a', to: 'b' }, { from: 'b', to: 'c' }, { from: 'c', to: 'd' }] }
 * Object.keys(spectralLayout(g).nodes).length // => 4
 * ```
 */
export function spectralLayout(
  graph: LayoutGraph,
  options: { readonly size?: number; readonly margin?: number } = {},
): GraphLayout {
  const { size = 400, margin = 16 } = options
  const n = graph.nodes.length
  const index = new Map(graph.nodes.map((v, i) => [v.id, i]))
  const lap = Array.from({ length: n }, () => new Array<number>(n).fill(0))
  for (const e of graph.edges) {
    const a = index.get(e.from)
    const b = index.get(e.to)
    if (a === undefined || b === undefined || a === b) continue
    const ra = lap[a] as number[]
    const rb = lap[b] as number[]
    ra[b] = (ra[b] ?? 0) - 1
    rb[a] = (rb[a] ?? 0) - 1
    ra[a] = (ra[a] ?? 0) + 1
    rb[b] = (rb[b] ?? 0) + 1
  }
  let xy: [number, number][]
  if (n < 3) xy = graph.nodes.map((_, i): [number, number] => [i * size, 0])
  else {
    const { eigenvectors: vectors } = eigh(lap)
    const v = toArray(vectors) as number[][]
    // Eigenvalues ascend; column 0 is the constant vector.
    const col = (j: number): number[] => v.map((row) => row[j] ?? 0)
    const [u, w] = [col(1), col(2)]
    const scale = (c: number[]): number[] => {
      const lo = minOf(c)
      const hi = maxOf(c)
      return c.map((x) => (hi === lo ? 0 : ((x - lo) / (hi - lo)) * size))
    }
    const [su, sw] = [scale(u), scale(w)]
    xy = graph.nodes.map((_, i): [number, number] => [su[i] ?? 0, sw[i] ?? 0])
  }
  const placed = boxes(graph, xy)
  return normaliseLayout(placed, straightRoutes(placed, graph.edges), {}, margin)
}
