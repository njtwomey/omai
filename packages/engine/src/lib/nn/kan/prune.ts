/**
 * Pruning a Kolmogorov-Arnold network to the part that matters, as in the paper. A node is scored by its incoming
 * importance (the largest mean magnitude among the edges that feed it) and its outgoing importance (the largest among
 * the edges it feeds); a hidden node whose two scores are not both above the threshold computes nothing that reaches the
 * output and is removed. Edges whose own magnitude is below a (separate) threshold are switched off with the mask. The
 * scores are recomputed on the edges that survive until nothing more goes, so a node whose only feeders were pruned is
 * pruned too. The result is a smaller `KAN` with the surviving parameters copied over, so it computes what the survivors
 * did and can be trained or refined further; the inputs can be pruned too, in which case the new network takes only
 * the columns that were kept.
 */

import { OmaiError } from '../../base/index.ts'
import { key as makeKey } from '../../random/index.ts'
import { asTensor, fromStorage, type TensorLike, take, toFlat, type Value } from '../../tensor/index.ts'
import type { Params } from '../module/module.ts'
import { KAN, type KanState } from './kan.ts'

/** Options of {@link pruneKan}. */
export interface PruneOptions {
  /** A hidden node stays when both its incoming and its outgoing importance exceed this (default 0.01, the paper's). */
  readonly nodeThreshold?: number
  /** An edge is switched off when its mean magnitude is below this (default 0.03, pykan's `prune_edge`). */
  readonly edgeThreshold?: number
  /** Also remove inputs that feed nothing important (default false: the network keeps its input columns). */
  readonly pruneInputs?: boolean
}

/** The importance of the nodes of one layer. */
export interface NodeScores {
  /** The largest magnitude among the edges that feed each node (1 for an input node). */
  readonly incoming: readonly number[]
  /** The largest magnitude among the edges each node feeds (1 for an output node). */
  readonly outgoing: readonly number[]
}

/** A pruned network. */
export interface Pruned {
  readonly kan: KAN
  readonly state: KanState
  /** For each layer of nodes, the indices (in the old network) of the nodes that survived. */
  readonly kept: readonly (readonly number[])[]
  /** The scores that were measured before anything was removed, for each layer of nodes. */
  readonly scores: readonly NodeScores[]
  /** The edges switched off (below the threshold or attached to a removed node) among the surviving nodes. */
  readonly edgesOff: number
  /** The parameter count before and after. */
  readonly parameters: { readonly before: number; readonly after: number }
}

const layerOf = (tree: Params, l: number): Params => (tree[String(l)] ?? {}) as Params

/** The leaves of a layer's edge tree that carry a node axis, with the axes (out, in) they are indexed by. */
const EDGE_AXES: Readonly<Record<string, readonly ('out' | 'in')[]>> = {
  coef: ['out', 'in'],
  knotLogits: ['in'],
  range: ['in'],
  interior: ['in'],
}

const pick = (v: Value, axis: number, indices: readonly number[]): Value => take(v, asTensor(indices, 'int32'), axis)

/** A layer's tree restricted to the surviving output and input nodes. */
function restrict(tree: Params, keepOut: readonly number[], keepIn: readonly number[]): Params {
  const out: Record<string, unknown> = {}
  for (const [name, v] of Object.entries(tree)) {
    if (name === 'edge') {
      const e: Record<string, unknown> = {}
      for (const [k, leaf] of Object.entries(v as Params)) {
        const axes = EDGE_AXES[k]
        if (axes === undefined) throw new OmaiError('value', `pruneKan: do not know how to prune edge leaf ${k}`)
        let t = leaf as Value
        axes.forEach((a, axis) => {
          t = pick(t, axis, a === 'out' ? keepOut : keepIn)
        })
        e[k] = t
      }
      out[name] = e
    } else if (name === 'bias') out[name] = pick(v as Value, 0, keepOut)
    else out[name] = pick(pick(v as Value, 0, keepOut), 1, keepIn)
  }
  return out
}

/**
 * Prune a trained network on a sample: score the nodes by the paper's incoming and outgoing importance, switch off weak
 * edges, remove the nodes that no longer matter, and return the smaller network with the surviving parameters.
 *
 * @param kan - The network.
 * @param state - Its parameters and buffers.
 * @param x - Inputs `[n, n_in]` on which the edge magnitudes are measured.
 * @param options - The node and edge thresholds and whether to prune inputs.
 * @returns The pruned network and state, the surviving node indices per layer, the scores and the counts.
 * @concept regularisation
 * @example
 * ```ts
 * import { KAN, pruneKan } from 'omai-engine/nn'
 * import { key } from 'omai-engine/random'
 * const net = new KAN([2, 4, 1], { key: key(0) })
 * const x = [[0.1, 0.2], [0.5, -0.3], [-0.4, 0.6]]
 * const small = pruneKan(net, net.state, x, { nodeThreshold: 1e9 })
 * small.kan.widths // => [2, 1, 1]
 * ```
 */
export function pruneKan(kan: KAN, state: KanState, x: TensorLike, options: PruneOptions = {}): Pruned {
  const nodeThreshold = options.nodeThreshold ?? 0.01
  const edgeThreshold = options.edgeThreshold ?? 0.03
  const widths = kan.widths
  const depth = kan.layers.length
  const scales = kan.edgeScales(state.params, x, state.buffers).map((s) => toFlat(s))
  const masks = kan.layers.map((_, l) => toFlat(layerOf(state.buffers, l)['mask'] as Value))
  // alive[l][i]: node i of layer l survives; on[l][o * in + i]: the edge is switched on.
  const alive = widths.map((w) => new Array<boolean>(w).fill(true))
  const on = scales.map((s, l) => Array.from(s, (v, k) => (masks[l]?.[k] as number) !== 0 && v >= edgeThreshold))
  const measure = (): NodeScores[] =>
    widths.map((w, l) => ({
      incoming: Array.from({ length: w }, (_, i) => {
        if (l === 0) return 1
        let best = 0
        const din = widths[l - 1] as number
        for (let j = 0; j < din; j++)
          if (alive[l - 1]?.[j] === true && on[l - 1]?.[i * din + j] === true)
            best = Math.max(best, (scales[l - 1]?.[i * din + j] as number) ?? 0)
        return best
      }),
      outgoing: Array.from({ length: w }, (_, i) => {
        if (l === depth) return 1
        let best = 0
        const din = w
        const dout = widths[l + 1] as number
        for (let o = 0; o < dout; o++)
          if (alive[l + 1]?.[o] === true && on[l]?.[o * din + i] === true)
            best = Math.max(best, (scales[l]?.[o * din + i] as number) ?? 0)
        return best
      }),
    }))
  const first = measure()
  for (let sweep = 0; sweep < widths.length * 4; sweep++) {
    const now = measure()
    let changed = false
    for (let l = 0; l <= depth; l++) {
      const hidden = l > 0 && l < depth
      if (!hidden && !(l === 0 && options.pruneInputs === true)) continue
      const score = now[l] as NodeScores
      // Keep the best node of a layer if every node would go: the network must still compute something.
      const ok = (alive[l] as boolean[]).map(
        (a, i) =>
          a &&
          (l === 0 || (score.incoming[i] as number) > nodeThreshold) &&
          (score.outgoing[i] as number) > nodeThreshold,
      )
      if (!ok.some(Boolean)) {
        const best = (alive[l] as boolean[]).reduce(
          (b, a, i) =>
            a && (score.incoming[i] as number) * (score.outgoing[i] as number) > (b.v as number)
              ? { i, v: (score.incoming[i] as number) * (score.outgoing[i] as number) }
              : b,
          { i: 0, v: -1 },
        )
        ok[best.i] = true
      }
      ok.forEach((v, i) => {
        if (v !== (alive[l] as boolean[])[i]) {
          ;(alive[l] as boolean[])[i] = v
          changed = true
        }
      })
    }
    if (!changed) break
  }
  const kept = alive.map((a) => a.flatMap((v, i) => (v ? [i] : [])))
  const { widths: _w, ...structure } = kan.spec
  const next = new KAN(
    kept.map((k) => k.length),
    { ...structure, key: makeKey(0) },
  )
  const params: Record<string, unknown> = {}
  const buffers: Record<string, unknown> = {}
  let edgesOff = 0
  kan.layers.forEach((_, l) => {
    const keepIn = kept[l] as number[]
    const keepOut = kept[l + 1] as number[]
    const p = restrict(layerOf(state.params, l), keepOut, keepIn)
    const b = restrict(layerOf(state.buffers, l), keepOut, keepIn)
    const mask = Float64Array.from(
      keepOut.flatMap((o) =>
        keepIn.map((i) => {
          const din = widths[l] as number
          const live = on[l]?.[o * din + i] === true
          if (!live) edgesOff++
          return live ? 1 : 0
        }),
      ),
    )
    params[String(l)] = p
    buffers[String(l)] = { ...b, mask: fromStorage(mask, [keepOut.length, keepIn.length], 'float64') }
  })
  return {
    kan: next,
    state: { params: params as Params, buffers: buffers as Params },
    kept,
    scores: first,
    edgesOff,
    parameters: { before: kan.parameterCount(), after: next.parameterCount() },
  }
}
