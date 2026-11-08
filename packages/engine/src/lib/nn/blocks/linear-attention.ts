/**
 * Linear-time attention variants as sequence mixers, all with a matrix-valued state that the keys and values are summed
 * into: linear attention with a feature map (Katharopoulos et al.), RetNet's retention (a fixed exponential decay per
 * head, with parallel, recurrent and chunkwise forms), gated linear attention (a decay that depends on the input, per
 * key dimension) and DeltaNet (the delta rule: the state is corrected towards the new value rather than only added to).
 */

import { OmaiError } from '../../base/index.ts'
import { foldIn, type Key } from '../../random/index.ts'
import {
  add,
  broadcastTo,
  concat,
  div,
  exp,
  fromStorage,
  mul,
  ones,
  sigmoid,
  sliceAxis,
  sqrt,
  square,
  sum,
  swapaxes,
  type Value,
  zeros,
} from '../../tensor/index.ts'
import { elu, logsigmoid, relu, silu, softplus } from '../functional/activations.ts'
import { mergeHeads, splitHeads } from '../functional/attention.ts'
import { decayedLinearAttention, type LinearAttentionForm } from '../functional/linear-recurrent.ts'
import { layerNorm } from '../functional/normalisation.ts'
import { deltaRule } from '../functional/state-space.ts'
import { Linear } from '../module/layers.ts'
import { type ApplyContext, initKey, type Params } from '../module/module.ts'
import { type MixerOutput, type MixerState, SequenceMixer, type TransitionSpectrum } from './mixer.ts'

/** The positive feature maps of linear attention. */
export type FeatureMap = 'elu' | 'relu' | 'exp' | 'softplus'

const featureMaps: Readonly<Record<FeatureMap, (x: Value) => Value>> = {
  elu: (x) => add(elu(x), 1),
  relu: (x) => relu(x),
  exp: (x) => exp(x),
  softplus: (x) => softplus(x),
}

/** Options shared by the matrix-state mixers. */
export interface MatrixMixerOptions {
  /** The number of heads (default 2 when the width is even, else 1); the head width is `dim / heads`. */
  readonly heads?: number
  /** The algorithm: `parallel` (masked attention), `recurrent` or `chunked` (default `parallel`). */
  readonly form?: LinearAttentionForm
  /** The chunk length of the chunked form (default 8). */
  readonly chunk?: number
  readonly key?: Key
}

function headsOf(dim: number, heads: number | undefined, kind: string): number {
  const h = heads ?? (dim % 2 === 0 ? 2 : 1)
  if (dim % h !== 0) throw new OmaiError('value', `${kind}: ${h} heads do not divide the width ${dim}`)
  return h
}

const l2 = (x: Value): Value => div(x, sqrt(add(sum(square(x), -1, true), 1e-12)))
const headNorm = (x: Value): Value => layerNorm(x, x.shape[x.shape.length - 1] ?? 1, { eps: 1e-6 })

/** A `[B, L, H d]` projection split into heads `[B, H, L, d]`. */
function project(p: { run<I, O>(name: string, x: I): O }, name: string, x: Value, heads: number): Value {
  return splitHeads(p.run<Value, Value>(name, x), heads)
}

/**
 * Linear attention (Katharopoulos et al. 2020): softmax attention with the scores replaced by `phi(q) . phi(k)` for a
 * positive feature map, so that the keys and values are summed into a state `S = sum phi(k) v^T` and a normaliser
 * `z = sum phi(k)`, read by `phi(q)^T S / (phi(q) . z)`. The normaliser rides along as an extra value channel, so the
 * state `[B, H, dk, dv + 1]` holds both.
 *
 * @concept state-space-models
 * @reference Katharopoulos, Vyas, Pappas and Fleuret (2020), Transformers are RNNs.
 * @example
 * ```ts
 * import { LinearAttentionMixer } from 'omai-engine/nn/blocks'
 * import { zeros } from 'omai-engine/tensor'
 * new LinearAttentionMixer(8).forward(zeros([1, 5, 8])).output.shape // => [1, 5, 8]
 * ```
 */
export class LinearAttentionMixer extends SequenceMixer {
  readonly kind = 'LinearAttention'
  readonly dim: number
  readonly heads: number
  readonly form: LinearAttentionForm
  readonly chunk: number
  private readonly phi: (x: Value) => Value

  constructor(dim: number, options: MatrixMixerOptions & { readonly featureMap?: FeatureMap } = {}) {
    const k = initKey('LinearAttention', options.key)
    super({
      q_proj: new Linear(dim, dim, { bias: false, key: foldIn(k, 0) }),
      k_proj: new Linear(dim, dim, { bias: false, key: foldIn(k, 1) }),
      v_proj: new Linear(dim, dim, { bias: false, key: foldIn(k, 2) }),
      o_proj: new Linear(dim, dim, { bias: false, key: foldIn(k, 3) }),
    })
    this.dim = dim
    this.heads = headsOf(dim, options.heads, 'LinearAttention')
    this.form = options.form ?? 'parallel'
    this.chunk = options.chunk ?? 8
    this.phi = featureMaps[options.featureMap ?? 'elu']
  }

  initialState(batch: number): MixerState {
    const d = this.dim / this.heads
    return { s: zeros([batch, this.heads, d, d + 1]) }
  }

  protected mix(params: Params, x: Value, state: MixerState | undefined, ctx: ApplyContext): MixerOutput {
    const p = this.parts(params, ctx)
    const [bs = 1, l = 0] = x.shape
    const dv = this.dim / this.heads
    const q = this.phi(project(p, 'q_proj', x, this.heads))
    const k = this.phi(project(p, 'k_proj', x, this.heads))
    const v = project(p, 'v_proj', x, this.heads)
    const aug = concat([v, ones([bs, this.heads, l, 1])], 3)
    const r = decayedLinearAttention(
      { q, k, v: aug, ...(state === undefined ? {} : { state: state['s'] as Value }) },
      this.form,
      this.chunk,
    )
    const out = div(sliceAxis(r.output, 3, 0, dv), add(sliceAxis(r.output, 3, dv, dv + 1), 1e-6))
    const output = p.run<Value, Value>('o_proj', mergeHeads(out))
    p.done()
    return { output, state: { s: r.state } }
  }

  override extraRepr(): string {
    return `dim=${this.dim}, heads=${this.heads}, ${this.form}`
  }
}

/**
 * Multi-scale retention, the mixer of RetNet (Sun et al. 2023): `S_t = gamma_h S_{t-1} + k_t v_t^T`, `o_t = q_t S_t`
 * with a fixed decay `gamma_h = 1 - 2^(-5-h)` per head, so heads remember over different horizons. The same layer is
 * computed in parallel (masked attention with a decay mask), as a recurrence, or chunkwise (parallel inside chunks,
 * recurrent between them), all three equal. The per-head output is layer-normalised and gated by `silu(W_g x)`. The
 * xPos rotation of the paper is not included.
 *
 * @concept state-space-models
 * @reference Sun et al. (2023), Retentive network: a successor to transformer for large language models.
 * @example
 * ```ts
 * import { RetNetMixer } from 'omai-engine/nn/blocks'
 * import { zeros } from 'omai-engine/tensor'
 * new RetNetMixer(8, { heads: 2, form: 'chunked' }).forward(zeros([1, 5, 8])).output.shape // => [1, 5, 8]
 * ```
 */
export class RetNetMixer extends SequenceMixer {
  readonly kind = 'RetNet'
  readonly dim: number
  readonly heads: number
  readonly form: LinearAttentionForm
  readonly chunk: number
  /** The decay of each head, `1 - 2^(-5-h)`. */
  readonly gamma: readonly number[]

  constructor(dim: number, options: MatrixMixerOptions = {}) {
    const k = initKey('RetNet', options.key)
    super({
      q_proj: new Linear(dim, dim, { bias: false, key: foldIn(k, 0) }),
      k_proj: new Linear(dim, dim, { bias: false, key: foldIn(k, 1) }),
      v_proj: new Linear(dim, dim, { bias: false, key: foldIn(k, 2) }),
      g_proj: new Linear(dim, dim, { bias: false, key: foldIn(k, 3) }),
      o_proj: new Linear(dim, dim, { bias: false, key: foldIn(k, 4) }),
    })
    this.dim = dim
    this.heads = headsOf(dim, options.heads, 'RetNet')
    this.form = options.form ?? 'parallel'
    this.chunk = options.chunk ?? 8
    this.gamma = Array.from({ length: this.heads }, (_, h) => 1 - 2 ** (-5 - h))
  }

  override transitionSpectrum(): TransitionSpectrum {
    return { domain: 'discrete', re: this.gamma, im: this.gamma.map(() => 0) }
  }

  initialState(batch: number): MixerState {
    const d = this.dim / this.heads
    return { s: zeros([batch, this.heads, d, d]) }
  }

  protected mix(params: Params, x: Value, state: MixerState | undefined, ctx: ApplyContext): MixerOutput {
    const p = this.parts(params, ctx)
    const [bs = 1, l = 0] = x.shape
    const d = this.dim / this.heads
    const q = mul(project(p, 'q_proj', x, this.heads), d ** -0.5)
    const k = project(p, 'k_proj', x, this.heads)
    const v = project(p, 'v_proj', x, this.heads)
    const logDecay = broadcastTo(fromStorage(Float64Array.from(this.gamma, Math.log), [1, this.heads, 1], 'float64'), [
      bs,
      this.heads,
      l,
    ])
    const r = decayedLinearAttention(
      { q, k, v, logDecay, ...(state === undefined ? {} : { state: state['s'] as Value }) },
      this.form,
      this.chunk,
    )
    const gate = silu(p.run<Value, Value>('g_proj', x))
    const output = p.run<Value, Value>('o_proj', mul(mergeHeads(headNorm(r.output)), gate))
    p.done()
    return { output, state: { s: r.state } }
  }

  override extraRepr(): string {
    return `dim=${this.dim}, heads=${this.heads}, ${this.form}`
  }
}

/**
 * Gated linear attention (Yang et al. 2024): the state `S_t = diag(alpha_t) S_{t-1} + k_t v_t^T` is forgotten
 * through a gate `alpha_t = sigmoid(W_a x_t)^(1/16)` that depends on the input and has one entry per key dimension
 * (retention's decay is a fixed number per head). Forms as for retention; the per-head output is layer-normalised and
 * gated by `silu(W_g x)`.
 *
 * @concept state-space-models
 * @reference Yang, Wang, Shen, Panda and Kim (2024), Gated linear attention transformers with hardware-efficient
 *   training.
 * @example
 * ```ts
 * import { GatedLinearAttentionMixer } from 'omai-engine/nn/blocks'
 * import { zeros } from 'omai-engine/tensor'
 * new GatedLinearAttentionMixer(8).forward(zeros([1, 5, 8])).output.shape // => [1, 5, 8]
 * ```
 */
export class GatedLinearAttentionMixer extends SequenceMixer {
  readonly kind = 'GatedLinearAttention'
  readonly dim: number
  readonly heads: number
  readonly form: LinearAttentionForm
  readonly chunk: number

  constructor(dim: number, options: MatrixMixerOptions = {}) {
    const k = initKey('GatedLinearAttention', options.key)
    super({
      q_proj: new Linear(dim, dim, { bias: false, key: foldIn(k, 0) }),
      k_proj: new Linear(dim, dim, { bias: false, key: foldIn(k, 1) }),
      v_proj: new Linear(dim, dim, { bias: false, key: foldIn(k, 2) }),
      a_proj: new Linear(dim, dim, { key: foldIn(k, 3) }),
      g_proj: new Linear(dim, dim, { bias: false, key: foldIn(k, 4) }),
      o_proj: new Linear(dim, dim, { bias: false, key: foldIn(k, 5) }),
    })
    this.dim = dim
    this.heads = headsOf(dim, options.heads, 'GatedLinearAttention')
    this.form = options.form ?? 'parallel'
    this.chunk = options.chunk ?? 8
  }

  initialState(batch: number): MixerState {
    const d = this.dim / this.heads
    return { s: zeros([batch, this.heads, d, d]) }
  }

  protected mix(params: Params, x: Value, state: MixerState | undefined, ctx: ApplyContext): MixerOutput {
    const p = this.parts(params, ctx)
    const d = this.dim / this.heads
    const q = mul(project(p, 'q_proj', x, this.heads), d ** -0.5)
    const k = project(p, 'k_proj', x, this.heads)
    const v = project(p, 'v_proj', x, this.heads)
    const logDecay = splitHeads(mul(logsigmoid(p.run<Value, Value>('a_proj', x)), 1 / 16), this.heads)
    const r = decayedLinearAttention(
      { q, k, v, logDecay, ...(state === undefined ? {} : { state: state['s'] as Value }) },
      this.form,
      this.chunk,
    )
    const gate = silu(p.run<Value, Value>('g_proj', x))
    const output = p.run<Value, Value>('o_proj', mul(mergeHeads(headNorm(r.output)), gate))
    p.done()
    return { output, state: { s: r.state } }
  }

  override extraRepr(): string {
    return `dim=${this.dim}, heads=${this.heads}, ${this.form}`
  }
}

/**
 * DeltaNet (Schlag et al. 2021; Yang et al. 2024): linear attention with the delta rule. The state is a fast-weight
 * memory `S` that is corrected towards each new value, `S_t = S_{t-1} + beta_t (v_t - S_{t-1} k_t) k_t^T`, so a key
 * that is written again overwrites its old value instead of adding to it (associative recall). Queries and keys go
 * through SiLU and L2 normalisation, `beta = sigmoid(W_b x)` per head. Computed by the sequential recurrence only
 * (the chunkwise parallel algorithm of the paper is not implemented).
 *
 * @concept state-space-models
 * @reference Yang et al. (2024), Parallelizing linear transformers with the delta rule over sequence length.
 * @example
 * ```ts
 * import { DeltaNetMixer } from 'omai-engine/nn/blocks'
 * import { zeros } from 'omai-engine/tensor'
 * new DeltaNetMixer(8).forward(zeros([1, 5, 8])).output.shape // => [1, 5, 8]
 * ```
 */
export class DeltaNetMixer extends SequenceMixer {
  readonly kind = 'DeltaNet'
  readonly dim: number
  readonly heads: number

  constructor(dim: number, options: Pick<MatrixMixerOptions, 'heads' | 'key'> = {}) {
    const k = initKey('DeltaNet', options.key)
    const heads = headsOf(dim, options.heads, 'DeltaNet')
    super({
      q_proj: new Linear(dim, dim, { bias: false, key: foldIn(k, 0) }),
      k_proj: new Linear(dim, dim, { bias: false, key: foldIn(k, 1) }),
      v_proj: new Linear(dim, dim, { bias: false, key: foldIn(k, 2) }),
      b_proj: new Linear(dim, heads, { key: foldIn(k, 3) }),
      o_proj: new Linear(dim, dim, { bias: false, key: foldIn(k, 4) }),
    })
    this.dim = dim
    this.heads = heads
  }

  initialState(batch: number): MixerState {
    const d = this.dim / this.heads
    return { s: zeros([batch, this.heads, d, d]) }
  }

  protected mix(params: Params, x: Value, state: MixerState | undefined, ctx: ApplyContext): MixerOutput {
    const p = this.parts(params, ctx)
    const q = l2(silu(project(p, 'q_proj', x, this.heads)))
    const k = l2(silu(project(p, 'k_proj', x, this.heads)))
    const v = project(p, 'v_proj', x, this.heads)
    const beta = swapaxes(sigmoid(p.run<Value, Value>('b_proj', x)), 1, 2) // [B, H, L]
    const r = deltaRule(q, k, v, beta, state === undefined ? undefined : (state['s'] as Value))
    const output = p.run<Value, Value>('o_proj', mergeHeads(headNorm(r.output)))
    p.done()
    return { output, state: { s: r.state } }
  }

  override extraRepr(): string {
    return `dim=${this.dim}, heads=${this.heads}`
  }
}
