/**
 * What a view of a Kolmogorov-Arnold network needs, as plain data: every edge's function on a grid over its input's
 * range (with its mean magnitude and whether it is switched on), and the grids themselves (each layer's input ranges and,
 * for spline edges, the interior knots, which move when they are trained or the grid is updated). The trainers log these
 * at chosen steps so the edges can be played as they evolve.
 */

import { type TensorLike, toFlat, type Value } from '../../tensor/index.ts'
import { SplineLayer } from '../additive/spline-layer.ts'
import type { Params } from '../module/module.ts'
import type { KAN, KanState } from './kan.ts'

/** One edge's function on a grid. */
export interface KanEdgeCurve {
  readonly layer: number
  /** The source node in layer `layer` and the target node in layer `layer + 1`. */
  readonly i: number
  readonly j: number
  readonly grid: number[]
  readonly values: number[]
  /** The mean magnitude `E|phi|` on the reference rows. */
  readonly scale: number
  /** Whether the edge is switched on (not masked). */
  readonly active: boolean
}

/** Every edge of a network on its grid. */
export interface KanCurves {
  readonly widths: number[]
  readonly edges: KanEdgeCurve[]
}

/**
 * Every edge of a network as a curve over its input's range, with its magnitude on reference rows.
 *
 * @param kan - The network.
 * @param state - Its parameters and buffers.
 * @param x - The reference rows `[n, n_in]` (they give the magnitudes).
 * @param points - Points per curve (default 41).
 * @returns The widths and the curves, layer by layer, source node by target node.
 * @concept splines
 * @example
 * ```ts
 * import { KAN, kanCurves } from 'mdml-engine/nn'
 * import { key } from 'mdml-engine/random'
 * const net = new KAN([2, 3, 1], { key: key(0) })
 * const c = kanCurves(net, net.state, [[0.1, 0.2], [0.5, -0.3]], 11)
 * c.edges.length // => 9
 * c.edges[0]?.values.length // => 11
 * ```
 */
export function kanCurves(kan: KAN, state: KanState, x: TensorLike, points = 41): KanCurves {
  const scales = kan.edgeScales(state.params, x, state.buffers).map((s) => toFlat(s))
  const edges: KanEdgeCurve[] = []
  kan.layers.forEach((layer, l) => {
    const ranges = layer.inputRange((state.buffers[String(l)] ?? {}) as Params)
    const mask = toFlat(((state.buffers[String(l)] ?? {}) as Params)['mask'] as Value)
    for (let i = 0; i < layer.inFeatures; i++) {
      const [lo, hi] = ranges[i] as [number, number]
      const grid = Array.from({ length: points }, (_, k) => lo + ((hi - lo) * k) / Math.max(1, points - 1))
      for (let j = 0; j < layer.outFeatures; j++) {
        const at = j * layer.inFeatures + i
        edges.push({
          layer: l,
          i,
          j,
          grid,
          values: Array.from(toFlat(kan.edgeCurve(l, i, j, grid, state))),
          scale: scales[l]?.[at] as number,
          active: (mask[at] as number) !== 0,
        })
      }
    }
  })
  return { widths: [...kan.widths], edges }
}

/** The grid of one layer. */
export interface KanGrid {
  readonly layer: number
  /** The range of each input node. */
  readonly ranges: [number, number][]
  /** The grid size of the layer's edges. */
  readonly grid: number
  /** For spline edges, each input's interior knots; `null` for the other families. */
  readonly knots: number[][] | null
}

/**
 * The grids of a network: each layer's input ranges and, for spline edges, the interior knots as they stand (fixed,
 * placed at quantiles, or trained).
 *
 * @param kan - The network.
 * @param state - Its parameters and buffers.
 * @returns One entry per layer.
 * @concept splines
 * @example
 * ```ts
 * import { KAN, kanGrids } from 'mdml-engine/nn'
 * import { key } from 'mdml-engine/random'
 * const net = new KAN([1, 1], { grid: 4, key: key(0) })
 * kanGrids(net, net.state)[0]?.knots // => [[-0.5, 0, 0.5]]
 * ```
 */
export function kanGrids(kan: KAN, state: KanState): KanGrid[] {
  return kan.layers.map((layer, l) => {
    const p = (state.params[String(l)] ?? {}) as Params
    const b = (state.buffers[String(l)] ?? {}) as Params
    const edge = layer.edge
    let knots: number[][] | null = null
    if (edge instanceof SplineLayer) {
      const flat = toFlat(edge.interior(p['edge'] as Params, b['edge'] as Params))
      const m = edge.interiorKnots
      knots = Array.from({ length: layer.inFeatures }, (_, i) => Array.from(flat.slice(i * m, (i + 1) * m)))
    }
    return { layer: l, ranges: layer.inputRange(b), grid: layer.grid, knots }
  })
}
