/**
 * The sparsification of a Kolmogorov-Arnold network, as in the paper. To make a network small enough to read, its loss
 * is given two penalties on the average magnitudes `s_{o,i} = E_x |phi_{o,i}(x_i)|` of the edges (their L1 norms on the
 * data): the sum of the magnitudes, which pushes edges to zero, and the entropy of their distribution within each layer,
 * `-sum p log p` with `p = s / sum s`, which pushes a layer to put its weight on few edges. A third penalty on the
 * coefficients themselves is pykan's `lamb_coef`. The magnitudes are measured on a fixed reference sample, so the
 * penalty is a function of the parameters alone, as the trainer base's `penalty` hook wants. `edgeSparsity` reports how
 * much of the network is negligible.
 */

import {
  abs,
  add,
  asTensor,
  div,
  log,
  mean,
  mul,
  sum,
  type TensorLike,
  toFlat,
  type Value,
} from '../../tensor/index.ts'
import type { Params } from '../module/module.ts'
import type { KAN, KanState } from './kan.ts'

/** The weights of the three penalties. */
export interface KanRegularisation {
  /** On the sum of the edges' mean magnitudes (the paper's `lambda mu_1`, default 0). */
  readonly l1?: number
  /** On the entropy of the edges' magnitudes within each layer (the paper's `lambda mu_2`, default 0). */
  readonly entropy?: number
  /** On the mean absolute value of the coefficients of each edge (pykan's `lamb_coef`, default 0). */
  readonly coef?: number
}

/** The penalty terms, before their weights, and their weighted sum. */
export interface KanPenalties {
  /** The sum over layers of the sum of edge magnitudes. */
  readonly l1: Value
  /** The sum over layers of the entropy of the edge magnitudes. */
  readonly entropy: Value
  /** The sum over edges of the mean absolute coefficient. */
  readonly coef: Value
  /** The weighted sum. */
  readonly total: Value
}

/**
 * The paper's sparsification penalties of a network on reference rows: `l1` (the sum of edge magnitudes), `entropy` (the
 * per-layer entropy of the normalised magnitudes) and `coef`, weighted into `total`. Differentiable in the parameters.
 *
 * @param kan - The network.
 * @param params - The parameters.
 * @param x - The reference rows `[n, n_in]`.
 * @param weights - The weights of the three penalties.
 * @param buffers - The buffers (default the network's).
 * @returns The terms and their weighted sum.
 * @concept regularisation
 * @example
 * ```ts
 * import { KAN, kanPenalties } from 'mdml-engine/nn'
 * import { key } from 'mdml-engine/random'
 * import { toFlat } from 'mdml-engine/tensor'
 * const net = new KAN([2, 3, 1], { key: key(0) })
 * const p = kanPenalties(net, net.params, [[0.1, 0.2], [0.5, -0.3]], { l1: 1 })
 * toFlat(p.total)[0] === toFlat(p.l1)[0] // => true
 * ```
 */
export function kanPenalties(
  kan: KAN,
  params: Params,
  x: TensorLike,
  weights: KanRegularisation,
  buffers?: Params,
): KanPenalties {
  const scales = kan.edgeScales(params, x, buffers)
  let l1: Value = asTensor(0)
  let entropy: Value = asTensor(0)
  for (const s of scales) {
    const total = add(sum(s), 1e-12)
    const p = div(s, total)
    l1 = add(l1, sum(s))
    entropy = add(entropy, mul(sum(mul(p, log(add(p, 1e-12)))), -1))
  }
  let coef: Value = asTensor(0)
  if ((weights.coef ?? 0) > 0)
    kan.layers.forEach((layer, l) => {
      const c = layer.edge.coefficients(
        ((params[String(l)] ?? {}) as Params)['edge'] as Params,
        ((buffers ?? kan.buffers)[String(l)] as Params | undefined)?.['edge'] as Params | undefined,
      )
      coef = add(coef, sum(mean(abs(c), 2)))
    })
  const total = add(add(mul(l1, weights.l1 ?? 0), mul(entropy, weights.entropy ?? 0)), mul(coef, weights.coef ?? 0))
  return { l1, entropy, coef, total }
}

/** How sparse a network is on a sample. */
export interface KanSparsity {
  /** The edges that are switched on (not masked). */
  readonly active: number
  /** The edges in all. */
  readonly edges: number
  /** Active edges whose mean magnitude is below the threshold. */
  readonly negligible: number
  /** The share of the active edges that are negligible (0 when none are active). */
  readonly fraction: number
}

/**
 * The share of a network's edges that have negligible activation on a sample: an edge counts when it is not masked and
 * its mean magnitude `E|phi|` is below `threshold`.
 *
 * @param kan - The network.
 * @param state - Its parameters and buffers.
 * @param x - Inputs `[n, n_in]`.
 * @param threshold - The magnitude below which an edge is negligible (default 0.01, the paper's).
 * @returns The counts and the fraction.
 * @concept regularisation
 * @example
 * ```ts
 * import { KAN, edgeSparsity } from 'mdml-engine/nn'
 * import { key } from 'mdml-engine/random'
 * const net = new KAN([2, 3, 1], { key: key(0) })
 * edgeSparsity(net, net.state, [[0.1, 0.2], [0.5, -0.3]]).edges // => 9
 * ```
 */
export function edgeSparsity(kan: KAN, state: KanState, x: TensorLike, threshold = 0.01): KanSparsity {
  const scales = kan.edgeScales(state.params, x, state.buffers).map((s) => toFlat(s))
  let active = 0
  let edges = 0
  let negligible = 0
  scales.forEach((s, l) => {
    const mask = toFlat(((state.buffers[String(l)] ?? {}) as Params)['mask'] as Value)
    s.forEach((v, k) => {
      edges++
      if ((mask[k] as number) === 0) return
      active++
      if (v < threshold) negligible++
    })
  })
  return { active, edges, negligible, fraction: active === 0 ? 0 : negligible / active }
}
