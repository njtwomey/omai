// A grid of nodes over a few real coordinates, zoomed onto where the weight is: the numeric core of the sites whose tilted
// distribution is a mixture over the nodes of a coordinate vector (the latent coordinates of a point of a latent variable model).
// `tiltGrid` (`_windows.ts`) does this for one coordinate and keeps no nodes; here the nodes are kept, because the rule needs the
// weight of each one (the mixture of the conditional Gaussians of the weights over the nodes), for one to three coordinates. The
// first grid lays `points` nodes per coordinate over the cavity's window; while the nodes that carry weight (more than `LIVE` of the
// largest) fill less than `SHRINK` of the window on every coordinate, the same number of nodes is laid over the box they span
// (a node of margin either side), up to `GRIDS` grids in all, so a likelihood that is narrow against its cavity is resolved.

import { ImproperMessageError } from '../core/errors.ts'
import { zeroFloat64 } from './_arrays.ts'
import { shiftedSum } from './_moments.ts'

/** The weights, relative to the largest, below which a node is outside the mass: the zoom cuts there. */
const LIVE = 1e-13
/** A box that shrinks to less than this fraction of the window on some coordinate is worth another grid. */
const SHRINK = 0.6
/** The most grids one tilt lays: the first over the cavity's window, then up to two over the box the mass was found in. */
const GRIDS = 3

/** The window of one coordinate. */
export interface NodeWindow {
  readonly lo: number
  readonly hi: number
}

/** The nodes of a grid over `q` coordinates, with their log weights shifted and exponentiated. */
export interface NodeGrid {
  /** The number of coordinates. */
  readonly q: number
  /** The number of nodes (`points ** q`). */
  readonly count: number
  /** The nodes, `q` numbers each: the coordinate `i` of node `k` is `nodes[k * q + i]`. The first coordinate varies fastest. */
  readonly nodes: Float64Array
  /** `exp(logw - top)`: at most one, the largest exactly one. */
  readonly weights: Float64Array
  /** The sum of the weights. */
  readonly total: number
  /** The largest log weight. */
  readonly top: number
  /** The volume of the cell of one node (the product of the spacings): the integral is `exp(top) * total * volume`. */
  readonly volume: number
}

/**
 * A grid over the box of `windows` (one per coordinate), zoomed onto the nodes that carry the weight of `logw`.
 *
 * @param windows - The window of each coordinate (one to three), from the cavity.
 * @param points - The nodes per coordinate of each grid (at least 3).
 * @param logw - The log of the integrand (the cavity's density included) at a node.
 * @returns The nodes of the last grid with their shifted weights, the total, the largest log weight and the cell volume.
 * @throws ImproperMessageError when no node of the first grid has a finite weight.
 */
export function zoomNodeGrid(
  windows: readonly NodeWindow[],
  points: number,
  logw: (x: Float64Array) => number,
): NodeGrid {
  const q = windows.length
  let box = windows
  const total = points ** q
  const x = zeroFloat64(q)
  for (let grid = 1; ; grid++) {
    const steps = box.map((w) => (w.hi - w.lo) / (points - 1))
    const nodes = zeroFloat64(total * q)
    const logs = zeroFloat64(total)
    for (let k = 0; k < total; k++) {
      let rest = k
      for (let i = 0; i < q; i++) {
        const v = (box[i] as NodeWindow).lo + (rest % points) * (steps[i] as number)
        rest = Math.floor(rest / points)
        nodes[k * q + i] = v
        x[i] = v
      }
      logs[k] = logw(x)
    }
    const { top, weights, total: sum } = shiftedSum(logs)
    const volume = steps.reduce((a, b) => a * b, 1)
    const result: NodeGrid = { q, count: total, nodes, weights, total: sum, top, volume }
    if (!Number.isFinite(top)) {
      if (grid === 1)
        throw new ImproperMessageError('the factor has no mass on any node of the grid over the coordinates')
      return result
    }
    if (grid === GRIDS) return result
    // The box the live nodes span, a node of margin either side, clipped to the window the grid was laid on.
    const lo = box.map(() => Number.POSITIVE_INFINITY)
    const hi = box.map(() => Number.NEGATIVE_INFINITY)
    for (let k = 0; k < total; k++) {
      if (!((weights[k] as number) >= LIVE)) continue
      for (let i = 0; i < q; i++) {
        const v = nodes[k * q + i] as number
        if (v < (lo[i] as number)) lo[i] = v
        if (v > (hi[i] as number)) hi[i] = v
      }
    }
    const next = box.map((w, i) => ({
      lo: Math.max(w.lo, (lo[i] as number) - (steps[i] as number)),
      hi: Math.min(w.hi, (hi[i] as number) + (steps[i] as number)),
    }))
    const zoomed = next.some((w, i) => w.hi - w.lo < SHRINK * ((box[i] as NodeWindow).hi - (box[i] as NodeWindow).lo))
    if (!zoomed) return result
    box = next
  }
}
