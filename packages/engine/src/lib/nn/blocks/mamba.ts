/**
 * Mamba, selective state-space models as sequence mixers: the Mamba block (Gu and Dao 2023: an expanding input
 * projection, a causal depthwise convolution, input-dependent step, input and output maps over a diagonal state, a
 * gated output projection) and the Mamba-2 block (Dao and Gu 2024: a scalar decay per head, so that the layer is
 * both a recurrence and masked attention, computed by the structured state-space duality `ssd`). Both carry the
 * convolution window and the SSM state between calls, so generation one token at a time gives the outputs of the
 * parallel pass.
 */

import { OmaiError } from '../../base/index.ts'
import { foldIn, type Key, uniform } from '../../random/index.ts'
import {
  add,
  asTensor,
  concat,
  exp,
  fromStorage,
  log,
  mul,
  neg,
  reshape,
  sliceAxis,
  swapaxes,
  type Tensor,
  type TensorLike,
  toFlat,
  type Value,
  zeros,
} from '../../tensor/index.ts'
import { silu, softplus } from '../functional/activations.ts'
import { conv1d } from '../functional/conv.ts'
import { selectiveScan, ssd } from '../functional/state-space.ts'
import { torchDefaultUniform } from '../init.ts'
import { Linear } from '../module/layers.ts'
import { type ApplyContext, initKey, type Params, tap } from '../module/module.ts'
import { RMSNorm } from '../module/normalisation.ts'
import { type MixerOutput, type MixerState, ownKey, SequenceMixer, type TransitionSpectrum } from './mixer.ts'
import { SequenceModel, type SequenceModelOptions } from './sequence-model.ts'

/**
 * A causal depthwise convolution over the time axis that continues from the last inputs of an earlier call.
 *
 * @param x - Inputs `[B, C, L]`.
 * @param weight - `[C, 1, K]`.
 * @param bias - `[C]` (optional).
 * @param window - The K - 1 inputs before the first (default zeros) `[B, C, K - 1]`.
 * @returns The outputs `[B, C, L]` and the window for the next call.
 * @concept state-space-models
 * @example
 * ```ts
 * import { causalDepthwiseConv } from 'omai-engine/nn/blocks'
 * import { toFlat } from 'omai-engine/tensor'
 * toFlat(causalDepthwiseConv([[[1, 2, 3]]], [[[1, 1]]]).y) // => [1, 3, 5]
 * ```
 */
export function causalDepthwiseConv(
  x: TensorLike,
  weight: TensorLike,
  bias?: Value,
  window?: Value,
): { readonly y: Value; readonly window: Value } {
  const xv = asTensor(x)
  const wv = asTensor(weight)
  const [bs = 1, c = 1, l = 0] = xv.shape
  const k = wv.shape[2] ?? 1
  if (k < 2) throw new OmaiError('value', 'causalDepthwiseConv: the kernel must have at least two taps')
  const padded = concat([window ?? zeros([bs, c, k - 1]), xv], 2)
  const y = conv1d(padded, wv, { groups: c, ...(bias === undefined ? {} : { bias }) })
  return { y, window: sliceAxis(padded, 2, l, l + k - 1) }
}

const f64 = (a: ArrayLike<number>, shape: number[]): Tensor => fromStorage(Float64Array.from(a), shape, 'float64')

/** The bias that makes softplus(bias) log-uniform in [dtMin, dtMax] (Mamba's step initialisation). */
function dtBias(key: Key, size: number, dtMin: number, dtMax: number, floor: number): Tensor {
  const u = uniform(key, [size]).data
  return f64(
    Array.from(u, (r) => {
      const dt = Math.max(Math.exp(r * (Math.log(dtMax) - Math.log(dtMin)) + Math.log(dtMin)), floor)
      return dt + Math.log(-Math.expm1(-dt))
    }),
    [size],
  )
}

/** Options of {@link MambaBlock}. */
export interface MambaOptions {
  /** The inner width as a multiple of the model width (default 2). */
  readonly expand?: number
  /** The SSM state size N per channel (default 16). */
  readonly stateSize?: number
  /** The taps of the causal convolution (default 4, at least 2). */
  readonly convKernel?: number
  /** The rank of the step's low-rank projection (default `ceil(dim / 16)`). */
  readonly dtRank?: number
  /** The range of the step at initialisation (default 1e-3 to 0.1) and its floor (1e-4). */
  readonly dtMin?: number
  readonly dtMax?: number
  readonly dtFloor?: number
  readonly bias?: boolean
  readonly key?: Key
}

interface MambaConfig {
  readonly inner: number
  readonly state: number
  readonly conv: number
  readonly rank: number
  readonly dtMin: number
  readonly dtMax: number
  readonly dtFloor: number
}

function mambaOwn(config: MambaConfig, key: Key): Params {
  const { inner, state, conv } = config
  return {
    aLog: f64(
      Array.from({ length: inner * state }, (_, i) => Math.log((i % state) + 1)),
      [inner, state],
    ),
    d: f64(new Array(inner).fill(1), [inner]),
    convWeight: torchDefaultUniform(foldIn(key, 0), [inner, 1, conv], conv),
    convBias: torchDefaultUniform(foldIn(key, 1), [inner], conv),
  }
}

function dtProj(config: MambaConfig, key: Key): Params {
  const std = config.rank ** -0.5
  return {
    weight: uniform(foldIn(key, 0), [config.inner, config.rank], { low: -std, high: std }),
    bias: dtBias(foldIn(key, 1), config.inner, config.dtMin, config.dtMax, config.dtFloor),
  }
}

/**
 * The Mamba block (Gu and Dao 2023). The input is projected to twice the inner width (a signal and a gate); the
 * signal goes through a causal depthwise convolution and SiLU; from it a low-rank projection gives the step
 * `Delta = softplus(.)` and linear maps give the input and output maps `B` and `C` of a diagonal state-space model
 * `h_t = exp(Delta_t A) h_{t-1} + Delta_t B_t x_t`, `y_t = C_t h_t + D x_t`, with `A = -exp(aLog)` fixed per channel
 * and state; the output is gated by SiLU of the gate and projected back. The state is the convolution window and the
 * SSM state, so the block runs on one token at a time with the outputs of the parallel pass.
 *
 * @concept state-space-models
 * @reference Gu and Dao (2023), Mamba: linear-time sequence modeling with selective state spaces.
 * @example
 * ```ts
 * import { MambaBlock } from 'omai-engine/nn/blocks'
 * import { zeros } from 'omai-engine/tensor'
 * new MambaBlock(8, { stateSize: 4 }).forward(zeros([1, 5, 8])).output.shape // => [1, 5, 8]
 * ```
 */
export class MambaBlock extends SequenceMixer {
  readonly kind = 'MambaBlock'
  readonly dim: number
  readonly inner: number
  readonly stateSize: number
  readonly convKernel: number
  private readonly config: MambaConfig

  constructor(dim: number, options: MambaOptions = {}) {
    const k = initKey('MambaBlock', options.key)
    const inner = (options.expand ?? 2) * dim
    const config: MambaConfig = {
      inner,
      state: options.stateSize ?? 16,
      conv: options.convKernel ?? 4,
      rank: options.dtRank ?? Math.ceil(dim / 16),
      dtMin: options.dtMin ?? 1e-3,
      dtMax: options.dtMax ?? 0.1,
      dtFloor: options.dtFloor ?? 1e-4,
    }
    if (config.conv < 2) throw new OmaiError('value', 'MambaBlock: convKernel must be at least 2')
    const bias = options.bias ?? false
    const dtLayer = new Linear(config.rank, inner, { key: foldIn(k, 2) })
    dtLayer.setParams(dtProj(config, foldIn(k, 2)))
    super(
      {
        in_proj: new Linear(dim, 2 * inner, { bias, key: foldIn(k, 0) }),
        x_proj: new Linear(inner, config.rank + 2 * config.state, { bias: false, key: foldIn(k, 1) }),
        dt_proj: dtLayer,
        out_proj: new Linear(inner, dim, { bias, key: foldIn(k, 3) }),
      },
      mambaOwn(config, ownKey(k)),
    )
    this.dim = dim
    this.inner = inner
    this.stateSize = config.state
    this.convKernel = config.conv
    this.config = config
  }

  protected override ownParams(key: Key): Params {
    return mambaOwn(this.config, key)
  }

  override initParams(key: Key): Params {
    const p = super.initParams(key)
    return { ...p, dt_proj: dtProj(this.config, foldIn(key, 2)) }
  }

  initialState(batch: number): MixerState {
    return {
      conv: zeros([batch, this.inner, this.convKernel - 1]),
      ssm: zeros([batch, this.inner, this.stateSize]),
    }
  }

  override transitionSpectrum(params: Params = this.params): TransitionSpectrum {
    return {
      domain: 'continuous',
      re: Array.from(toFlat(this.stateMatrix(params))),
      im: new Array<number>(this.inner * this.stateSize).fill(0),
    }
  }

  /** The state matrix `-exp(aLog)` `[inner, N]` (negative: the state decays). */
  stateMatrix(params: Params = this.params): Value {
    return neg(exp(params['aLog'] as Value))
  }

  protected mix(params: Params, x: Value, state: MixerState | undefined, ctx: ApplyContext): MixerOutput {
    const { inner, rank, state: n } = this.config
    const p = this.parts(params, ctx)
    const xz = p.run<Value, Value>('in_proj', x)
    const signal = sliceAxis(xz, 2, 0, inner)
    const gate = sliceAxis(xz, 2, inner, 2 * inner)
    const conv = causalDepthwiseConv(
      swapaxes(signal, 1, 2),
      params['convWeight'] as Value,
      params['convBias'] as Value,
      state?.['conv'] as Value | undefined,
    )
    const xc = silu(swapaxes(conv.y, 1, 2))
    const proj = p.run<Value, Value>('x_proj', xc)
    const dt = tap(ctx, 'dt', softplus(p.run<Value, Value>('dt_proj', sliceAxis(proj, 2, 0, rank))))
    const scan = selectiveScan({
      x: xc,
      dt,
      a: this.stateMatrix(params),
      b: sliceAxis(proj, 2, rank, rank + n),
      c: sliceAxis(proj, 2, rank + n, rank + 2 * n),
      d: params['d'] as Value,
      z: gate,
      ...(state === undefined ? {} : { h0: state['ssm'] as Value }),
    })
    const output = p.run<Value, Value>('out_proj', scan.y)
    p.done()
    return { output, state: { conv: conv.window, ssm: scan.state } }
  }

  override extraRepr(): string {
    return `dim=${this.dim}, inner=${this.inner}, state=${this.stateSize}, conv=${this.convKernel}`
  }
}

/** Options of {@link Mamba2Block}. */
export interface Mamba2Options {
  readonly expand?: number
  /** The SSM state size N, shared by the heads (default 16). */
  readonly stateSize?: number
  readonly convKernel?: number
  /** The width P of a head (default `min(inner, 16)`; must divide the inner width). */
  readonly headDim?: number
  readonly dtMin?: number
  readonly dtMax?: number
  readonly dtFloor?: number
  /** The range of the initial per-head decay rates `A` (default 1 to 16). */
  readonly aRange?: readonly [number, number]
  /** The algorithm of the duality (default `chunked`; all three agree). */
  readonly form?: 'recurrent' | 'quadratic' | 'chunked'
  readonly chunk?: number
  readonly bias?: boolean
  readonly key?: Key
}

interface Mamba2Config {
  readonly inner: number
  readonly heads: number
  readonly headDim: number
  readonly state: number
  readonly conv: number
  readonly dtMin: number
  readonly dtMax: number
  readonly dtFloor: number
  readonly aRange: readonly [number, number]
}

const mamba2Own = (c: Mamba2Config, key: Key): Params => ({
  dtBias: dtBias(foldIn(key, 2), c.heads, c.dtMin, c.dtMax, c.dtFloor),
  aLog: log(uniform(foldIn(key, 3), [c.heads], { low: c.aRange[0], high: c.aRange[1] })),
  d: f64(new Array(c.heads).fill(1), [c.heads]),
  convWeight: torchDefaultUniform(foldIn(key, 0), [c.inner + 2 * c.state, 1, c.conv], c.conv),
  convBias: torchDefaultUniform(foldIn(key, 1), [c.inner + 2 * c.state], c.conv),
})

/**
 * The Mamba-2 block (Dao and Gu 2024): one projection gives the gate, the convolved signal (with the input and output
 * maps `B`, `C` shared by all heads) and a step per head; the inner width splits into heads whose state matrix is a
 * scalar `-exp(aLog)` times the identity, so each head is `S_t = exp(Delta_t A) S_{t-1} + Delta_t x_t B_t^T`,
 * `y_t = S_t C_t + D x_t`. That is also masked attention with queries `C`, keys `B` and values `Delta x`; {@link ssd}
 * computes it as the recurrence, as the quadratic attention-like form or chunk by chunk, all equal. The output is
 * gated and RMS-normalised (`norm(y silu(z))`) before the output projection.
 *
 * @concept state-space-models
 * @reference Dao and Gu (2024), Transformers are SSMs: generalized models and efficient algorithms through structured
 *   state space duality.
 * @example
 * ```ts
 * import { Mamba2Block } from 'omai-engine/nn/blocks'
 * import { zeros } from 'omai-engine/tensor'
 * new Mamba2Block(8, { stateSize: 4, headDim: 8 }).forward(zeros([1, 5, 8])).output.shape // => [1, 5, 8]
 * ```
 */
export class Mamba2Block extends SequenceMixer {
  readonly kind = 'Mamba2Block'
  readonly dim: number
  readonly inner: number
  readonly heads: number
  readonly headDim: number
  readonly stateSize: number
  readonly convKernel: number
  readonly form: 'recurrent' | 'quadratic' | 'chunked'
  readonly chunk: number
  private readonly config: Mamba2Config

  constructor(dim: number, options: Mamba2Options = {}) {
    const k = initKey('Mamba2Block', options.key)
    const inner = (options.expand ?? 2) * dim
    const headDim = options.headDim ?? Math.min(inner, 16)
    if (inner % headDim !== 0) throw new OmaiError('value', `Mamba2Block: head width ${headDim} must divide ${inner}`)
    const config: Mamba2Config = {
      inner,
      heads: inner / headDim,
      headDim,
      state: options.stateSize ?? 16,
      conv: options.convKernel ?? 4,
      dtMin: options.dtMin ?? 1e-3,
      dtMax: options.dtMax ?? 0.1,
      dtFloor: options.dtFloor ?? 1e-4,
      aRange: options.aRange ?? [1, 16],
    }
    if (config.conv < 2) throw new OmaiError('value', 'Mamba2Block: convKernel must be at least 2')
    const bias = options.bias ?? false
    super(
      {
        in_proj: new Linear(dim, 2 * inner + 2 * config.state + config.heads, { bias, key: foldIn(k, 0) }),
        norm: new RMSNorm(inner, { eps: 1e-6 }),
        out_proj: new Linear(inner, dim, { bias, key: foldIn(k, 1) }),
      },
      mamba2Own(config, ownKey(k)),
    )
    this.dim = dim
    this.inner = inner
    this.heads = config.heads
    this.headDim = headDim
    this.stateSize = config.state
    this.convKernel = config.conv
    this.form = options.form ?? 'chunked'
    this.chunk = options.chunk ?? 8
    this.config = config
  }

  protected override ownParams(key: Key): Params {
    return mamba2Own(this.config, key)
  }

  initialState(batch: number): MixerState {
    return {
      conv: zeros([batch, this.inner + 2 * this.stateSize, this.convKernel - 1]),
      ssm: zeros([batch, this.heads, this.headDim, this.stateSize]),
    }
  }

  override transitionSpectrum(params: Params = this.params): TransitionSpectrum {
    return {
      domain: 'continuous',
      re: Array.from(toFlat(this.decayRates(params))),
      im: new Array<number>(this.heads).fill(0),
    }
  }

  /** The decay rates `-exp(aLog)` of the heads `[H]` (negative: the state decays). */
  decayRates(params: Params = this.params): Value {
    return neg(exp(params['aLog'] as Value))
  }

  protected mix(params: Params, x: Value, state: MixerState | undefined, ctx: ApplyContext): MixerOutput {
    const { inner, heads, headDim, state: n } = this.config
    const [bs = 1, l = 0] = x.shape
    const p = this.parts(params, ctx)
    const zxbcdt = p.run<Value, Value>('in_proj', x)
    const gate = sliceAxis(zxbcdt, 2, 0, inner)
    const xbc = sliceAxis(zxbcdt, 2, inner, 2 * inner + 2 * n)
    const dtRaw = sliceAxis(zxbcdt, 2, 2 * inner + 2 * n, 2 * inner + 2 * n + heads)
    const conv = causalDepthwiseConv(
      swapaxes(xbc, 1, 2),
      params['convWeight'] as Value,
      params['convBias'] as Value,
      state?.['conv'] as Value | undefined,
    )
    const act = silu(swapaxes(conv.y, 1, 2))
    const xs = sliceAxis(act, 2, 0, inner)
    const dt = tap(ctx, 'dt', softplus(add(dtRaw, params['dtBias'] as Value)))
    const heads4 = reshape(xs, [bs, l, heads, headDim])
    const r = ssd(
      {
        x: heads4,
        dt,
        a: this.decayRates(params),
        b: sliceAxis(act, 2, inner, inner + n),
        c: sliceAxis(act, 2, inner + n, inner + 2 * n),
        ...(state === undefined ? {} : { h0: state['ssm'] as Value }),
      },
      this.form,
      this.chunk,
    )
    const y = reshape(add(r.y, mul(heads4, reshape(params['d'] as Value, [heads, 1]))), [bs, l, inner])
    const normed = p.run<Value, Value>('norm', mul(y, silu(gate)))
    const output = p.run<Value, Value>('out_proj', normed)
    p.done()
    return { output, state: { conv: conv.window, ssm: r.state } }
  }

  override extraRepr(): string {
    return `dim=${this.dim}, inner=${this.inner}, heads=${this.heads}x${this.headDim}, state=${this.stateSize}, ${this.form}`
  }
}

/**
 * A Mamba language model or sequence classifier: a {@link SequenceModel} of Mamba blocks (the architecture of the
 * paper: token embedding, residual pre-norm Mamba blocks without a feed-forward, a final RMS norm, a tied output
 * projection).
 *
 * @param vocabSize - The vocabulary size.
 * @param dim - The model width.
 * @param numLayers - The number of blocks.
 * @param options - The Mamba options, and the model options (`outputSize`, `pooling`, `input`, ...).
 * @returns The model.
 * @concept sequence-models
 * @example
 * ```ts
 * import { mambaModel } from 'omai-engine/nn/blocks'
 * mambaModel(11, 8, 2, { stateSize: 4 }).forward([[1, 2, 3]]).shape // => [1, 3, 11]
 * ```
 */
export function mambaModel(
  vocabSize: number,
  dim: number,
  numLayers: number,
  options: MambaOptions & Omit<SequenceModelOptions, 'inputSize' | 'key'> & { readonly key?: Key } = {},
): SequenceModel {
  const { input, outputSize, pooling, tieEmbeddings, ffn, hidden, norm, key, ...mamba } = options
  return new SequenceModel(dim, numLayers, (d, k) => new MambaBlock(d, { ...mamba, key: k }), {
    inputSize: vocabSize,
    ...(input === undefined ? {} : { input }),
    ...(outputSize === undefined ? {} : { outputSize }),
    ...(pooling === undefined ? {} : { pooling }),
    ...(tieEmbeddings === undefined ? {} : { tieEmbeddings }),
    ...(ffn === undefined ? {} : { ffn }),
    ...(hidden === undefined ? {} : { hidden }),
    ...(norm === undefined ? {} : { norm }),
    ...(key === undefined ? {} : { key }),
  })
}
