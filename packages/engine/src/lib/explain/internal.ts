/**
 * Attributions inside a network (captum's layer and neuron methods): how much a hidden layer's units carry the
 * target (activation times gradient, internal influence, layer conductance), how much each input reaches the target
 * through one neuron (neuron conductance, neuron integrated gradients), and how attention mixes tokens across a
 * transformer's layers (attention rollout and attention flow). A network is split into a head (inputs to the layer)
 * and a tail (the layer to the outputs).
 */
import { vjp } from '../autodiff/index.ts'
import { MdmlError } from '../base/index.ts'
import { digraph, maxFlow } from '../graph/index.ts'
import {
  add,
  asTensor,
  fromStorage,
  isTensor,
  mul,
  sub,
  type Tensor,
  type TensorLike,
  toFlat,
  type Value,
} from '../tensor/index.ts'
import {
  type DifferentiableModel,
  integratedGradients,
  type PathMethod,
  pathRule,
  type Target,
  targetGradient,
} from './gradients.ts'

function concrete(x: TensorLike | Value, who: string): Tensor {
  const t = asTensor(x as TensorLike)
  if (!isTensor(t)) throw new MdmlError('value', `${who}: expected concrete values`)
  return t.dtype === 'float64' ? t : fromStorage(Float64Array.from(toFlat(t)), t.shape, 'float64')
}

function baselineLike(xt: Tensor, baseline: TensorLike | undefined, who: string): Tensor {
  if (baseline === undefined) return fromStorage(new Float64Array(xt.data.length), xt.shape, 'float64')
  const b = toFlat(asTensor(baseline))
  if (b.length === xt.data.length) return fromStorage(Float64Array.from(b), xt.shape, 'float64')
  const d = xt.data.length / (xt.shape[0] ?? 1)
  if (b.length !== d) throw new MdmlError('shape', `${who}: the baseline must be one input or one per input`)
  return fromStorage(
    Float64Array.from({ length: xt.data.length }, (_, i) => b[i % d] ?? 0),
    xt.shape,
    'float64',
  )
}

/** Options of the path-integrated layer methods. */
export interface LayerPathOptions {
  readonly target?: Target
  /** The input baseline (default zeros). */
  readonly baseline?: TensorLike
  /** Path points (default 50). */
  readonly nSteps?: number
  /** The quadrature (default Gauss-Legendre, captum's). */
  readonly method?: PathMethod
}

/**
 * Layer activation times gradient (captum LayerGradientXActivation): each unit of a hidden layer's activation times
 * the target output's gradient with respect to it, the layer's analogue of input times gradient.
 *
 * @param head - Inputs to the layer's activations.
 * @param tail - The layer's activations to the outputs.
 * @param x - Inputs.
 * @param options - target (default 0).
 * @returns Attributions shaped like the layer.
 * @example
 * ```ts
 * import * as explain from 'mdml-engine/explain'
 * import { matmul, toArray } from 'mdml-engine/tensor'
 * const head = (x) => matmul(x, [[1, 0], [0, 2]])
 * const tail = (h) => matmul(h, [[3], [1]])
 * toArray(explain.layerGradientXActivation(head, tail, [[1, 1]])) // => [[3, 2]]
 * ```
 */
export function layerGradientXActivation(
  head: DifferentiableModel,
  tail: DifferentiableModel,
  x: TensorLike,
  options: { readonly target?: Target } = {},
): Tensor {
  const a = concrete(head(concrete(x, 'layerGradientXActivation')), 'layerGradientXActivation')
  return mul(a, targetGradient(tail, a, options.target ?? 0)) as Tensor
}

/**
 * Internal influence (Leino et al. 2018; captum InternalInfluence): the target output's gradient with respect to a
 * hidden layer, averaged along the straight input path from a baseline to the input (integrated gradients' path,
 * read at the layer).
 *
 * @param head - Inputs to the layer.
 * @param tail - The layer to the outputs.
 * @param x - Inputs.
 * @param options - target, baseline, nSteps and method.
 * @returns Influences shaped like the layer.
 * @example
 * ```ts
 * import * as explain from 'mdml-engine/explain'
 * import { matmul, toArray } from 'mdml-engine/tensor'
 * // A linear tail's gradient is its weights wherever the path goes.
 * toArray(explain.internalInfluence((x) => x, (h) => matmul(h, [[3], [1]]), [[1, 1]], { nSteps: 4, method: 'riemann_left' })) // => [[3, 1]]
 * ```
 */
export function internalInfluence(
  head: DifferentiableModel,
  tail: DifferentiableModel,
  x: TensorLike,
  options: LayerPathOptions = {},
): Tensor {
  const xt = concrete(x, 'internalInfluence')
  const base = baselineLike(xt, options.baseline, 'internalInfluence')
  const diff = sub(xt, base) as Tensor
  const { alphas, weights } = pathRule(options.method ?? 'gausslegendre', options.nSteps ?? 50)
  let total: Float64Array | undefined
  let shape: readonly number[] = []
  alphas.forEach((alpha, s) => {
    const h = concrete(head(add(base, mul(diff, alpha)) as Tensor), 'internalInfluence')
    const g = toFlat(targetGradient(tail, h, options.target ?? 0))
    total ??= new Float64Array(g.length)
    shape = h.shape
    for (let i = 0; i < g.length; i++) total[i] = (total[i] ?? 0) + (weights[s] ?? 0) * (g[i] ?? 0)
  })
  return fromStorage(total ?? new Float64Array(0), shape, 'float64')
}

/**
 * Layer conductance (Dhamdhere, Sundararajan and Yan 2018; captum LayerConductance): the flow of integrated gradients
 * through each unit of a hidden layer, sum over path steps of (h(alpha_{k+1}) - h(alpha_k)) times the target output's
 * gradient at h(alpha_k), with nSteps + 1 path points. The units' conductances add up to the output's change from
 * baseline to input, as integrated gradients' do.
 *
 * @param head - Inputs to the layer.
 * @param tail - The layer to the outputs.
 * @param x - Inputs.
 * @param options - target, baseline, nSteps and method.
 * @returns Conductances shaped like the layer.
 * @example
 * ```ts
 * import * as explain from 'mdml-engine/explain'
 * import { matmul, toArray } from 'mdml-engine/tensor'
 * // Linear head and tail: each unit's change times its weight.
 * const r = explain.layerConductance((x) => matmul(x, [[1, 0], [0, 2]]), (h) => matmul(h, [[3], [1]]), [[1, 1]], { nSteps: 5, method: 'riemann_trapezoid' })
 * toArray(r).map((row) => row.map((v) => Math.round(v * 1e9) / 1e9)) // => [[3, 2]]
 * ```
 */
export function layerConductance(
  head: DifferentiableModel,
  tail: DifferentiableModel,
  x: TensorLike,
  options: LayerPathOptions = {},
): Tensor {
  const xt = concrete(x, 'layerConductance')
  const base = baselineLike(xt, options.baseline, 'layerConductance')
  const diff = sub(xt, base) as Tensor
  const n = options.nSteps ?? 50
  const { alphas } = pathRule(options.method ?? 'gausslegendre', n + 1)
  const evals = alphas.map((alpha) => concrete(head(add(base, mul(diff, alpha)) as Tensor), 'layerConductance'))
  const first = evals[0] as Tensor
  const total = new Float64Array(first.data.length)
  for (let s = 0; s < n; s++) {
    const h0 = evals[s] as Tensor
    const h1 = toFlat(evals[s + 1] as Tensor)
    const g = toFlat(targetGradient(tail, h0, options.target ?? 0))
    const hv = toFlat(h0)
    for (let i = 0; i < total.length; i++) total[i] = (total[i] ?? 0) + ((h1[i] ?? 0) - (hv[i] ?? 0)) * (g[i] ?? 0)
  }
  return fromStorage(total, first.shape, 'float64')
}

/**
 * Neuron conductance (Dhamdhere et al. 2018; captum NeuronConductance): how much each input feature's integrated
 * gradient flows through one hidden neuron, (x - x') times the path average of d f / d h_neuron times d h_neuron / d x
 * (by captum's default trapezoid rule). Summed over the inputs it is the neuron's layer conductance.
 *
 * @param head - Inputs to the layer's activations `[n, units]`.
 * @param tail - The layer to the outputs.
 * @param x - Inputs `[n, d]`.
 * @param neuron - The unit (a column of the layer).
 * @param options - target, baseline, nSteps and method (default `riemann_trapezoid`, captum's).
 * @returns Attributions `[n, d]`.
 * @example
 * ```ts
 * import * as explain from 'mdml-engine/explain'
 * import { matmul, toArray } from 'mdml-engine/tensor'
 * const head = (x) => matmul(x, [[1, 0], [1, 2]])
 * const tail = (h) => matmul(h, [[3], [1]])
 * // Through unit 0 (x1 + x2, weight 3): each input carries 3 times its own value.
 * toArray(explain.neuronConductance(head, tail, [[1, 2]], 0, { nSteps: 4, method: 'riemann_left' })).map((r) => r.map((v) => Math.round(v * 1e9) / 1e9)) // => [[3, 6]]
 * ```
 */
export function neuronConductance(
  head: DifferentiableModel,
  tail: DifferentiableModel,
  x: TensorLike,
  neuron: number,
  options: LayerPathOptions = {},
): Tensor {
  const xt = concrete(x, 'neuronConductance')
  const base = baselineLike(xt, options.baseline, 'neuronConductance')
  const diff = sub(xt, base) as Tensor
  const { alphas, weights } = pathRule(options.method ?? 'riemann_trapezoid', options.nSteps ?? 50)
  const n = xt.shape[0] ?? 1
  const total = new Float64Array(xt.data.length)
  const d = xt.data.length / n
  alphas.forEach((alpha, s) => {
    const point = add(base, mul(diff, alpha)) as Tensor
    const { out, pullback } = vjp(head, point as Value)
    const h = concrete(out, 'neuronConductance')
    const units = h.data.length / n
    if (!(Number.isInteger(neuron) && neuron >= 0 && neuron < units))
      throw new MdmlError('value', `neuronConductance: neuron ${neuron} outside [0, ${units})`)
    const mid = toFlat(targetGradient(tail, h, options.target ?? 0))
    const select = new Float64Array(h.data.length)
    for (let i = 0; i < n; i++) select[i * units + neuron] = 1
    const dh = toFlat(pullback(fromStorage(select, h.shape, 'float64')) as Tensor)
    for (let i = 0; i < n; i++)
      for (let j = 0; j < d; j++)
        total[i * d + j] =
          (total[i * d + j] ?? 0) + (weights[s] ?? 0) * (mid[i * units + neuron] ?? 0) * (dh[i * d + j] ?? 0)
  })
  return mul(diff, fromStorage(total, xt.shape, 'float64')) as Tensor
}

/**
 * Neuron integrated gradients (captum NeuronIntegratedGradients): integrated gradients of one hidden neuron's
 * activation with respect to the inputs, so the inputs' attributions sum to the neuron's change from the baseline.
 *
 * @param head - Inputs to the layer's activations `[n, units]`.
 * @param x - Inputs `[n, d]`.
 * @param neuron - The unit.
 * @param options - baseline, nSteps and method.
 * @returns Attributions `[n, d]`.
 * @example
 * ```ts
 * import * as explain from 'mdml-engine/explain'
 * import { matmul, toArray } from 'mdml-engine/tensor'
 * const r = explain.neuronIntegratedGradients((x) => matmul(x, [[1, 0], [1, 2]]), [[1, 2]], 1, { nSteps: 3 })
 * toArray(r).map((row) => row.map((v) => Math.round(v * 1e9) / 1e9)) // => [[0, 4]]
 * ```
 */
export function neuronIntegratedGradients(
  head: DifferentiableModel,
  x: TensorLike,
  neuron: number,
  options: Omit<LayerPathOptions, 'target'> = {},
): Tensor {
  return integratedGradients(head, x, {
    target: neuron,
    ...(options.baseline === undefined ? {} : { baseline: options.baseline }),
    ...(options.nSteps === undefined ? {} : { nSteps: options.nSteps }),
    ...(options.method === undefined ? {} : { method: options.method }),
  }).attributions
}

function withResidual(a: readonly (readonly number[])[], residual: boolean): number[][] {
  return a.map((row, i) => {
    const r = row.map((v, j) => (residual ? 0.5 * v + (i === j ? 0.5 : 0) : v))
    const s = r.reduce((x, y) => x + y, 0)
    return s === 0 ? r : r.map((v) => v / s)
  })
}

function checkAttentions(attentions: readonly (readonly (readonly number[])[])[], who: string): number {
  const t = attentions[0]?.length ?? 0
  if (attentions.length === 0 || t === 0) throw new MdmlError('value', `${who}: no attention matrices`)
  for (const a of attentions) {
    if (a.length !== t || a.some((r) => r.length !== t))
      throw new MdmlError('shape', `${who}: every layer's attention must be ${t} x ${t}`)
  }
  return t
}

/**
 * Attention rollout (Abnar and Zuidema 2020): how much each output token draws on each input token through a
 * transformer, the product of the layers' attention matrices (heads averaged), each first mixed half and half with
 * the identity for the residual connection and renormalised: rollout = A_L ... A_1, rows summing to 1.
 *
 * @param attentions - One `[T, T]` matrix per layer, bottom first, row i the attention of token i over the tokens.
 * @param options - residual (default true).
 * @returns The `[T, T]` rollout.
 * @example
 * ```ts
 * import * as explain from 'mdml-engine/explain'
 * // One layer where token 1 attends only to token 0: half of it stays itself through the residual.
 * explain.attentionRollout([[[1, 0], [1, 0]]]) // => [[1, 0], [0.5, 0.5]]
 * ```
 */
export function attentionRollout(
  attentions: readonly (readonly (readonly number[])[])[],
  options: { readonly residual?: boolean } = {},
): number[][] {
  const t = checkAttentions(attentions, 'attentionRollout')
  let r: number[][] = Array.from({ length: t }, (_, i) => Array.from({ length: t }, (_, j) => (i === j ? 1 : 0)))
  for (const a of attentions) {
    const m = withResidual(a, options.residual ?? true)
    r = m.map((row) => Array.from({ length: t }, (_, j) => row.reduce((s, v, k) => s + v * (r[k]?.[j] ?? 0), 0)))
  }
  return r
}

/**
 * Attention flow (Abnar and Zuidema 2020): the attention graph's maximum flow from each output token down to each
 * input token, with the (residual-mixed) attention weights as edge capacities between consecutive layers; unlike
 * rollout it bounds the mixing by the narrowest layer on the way.
 *
 * @param attentions - One `[T, T]` matrix per layer, bottom first.
 * @param options - residual (default true).
 * @returns The `[T, T]` flows, row i the output token.
 * @example
 * ```ts
 * import * as explain from 'mdml-engine/explain'
 * explain.attentionFlow([[[1, 0], [1, 0]]], { residual: false }) // => [[1, 0], [1, 0]]
 * ```
 */
export function attentionFlow(
  attentions: readonly (readonly (readonly number[])[])[],
  options: { readonly residual?: boolean } = {},
): number[][] {
  const t = checkAttentions(attentions, 'attentionFlow')
  const L = attentions.length
  const node = (layer: number, token: number): number => layer * t + token
  const edges: [number, number, number][] = []
  attentions.forEach((a, l) => {
    const m = withResidual(a, options.residual ?? true)
    m.forEach((row, i) => {
      row.forEach((v, j) => {
        if (v > 0) edges.push([node(l + 1, i), node(l, j), v])
      })
    })
  })
  const g = digraph(edges, { nodes: (L + 1) * t })
  return Array.from({ length: t }, (_, i) =>
    Array.from({ length: t }, (_, j) => maxFlow(g, node(L, i), node(0, j)).value),
  )
}
