/**
 * Building blocks of generative models. Timestep and noise-level embeddings with the time-conditioned residual, spatial
 * attention and down and up stages of a diffusion U-Net (laid out as diffusers' blocks), and the affine (RealNVP) and
 * additive (NICE) coupling layers of normalising flows, which return their output with the log-determinant of the
 * Jacobian.
 */

import { OmaiError } from '../../base/index.ts'
import { foldIn, type Key, normal } from '../../random/index.ts'
import {
  add,
  asTensor,
  concat,
  cos,
  exp,
  fromStorage,
  log,
  mul,
  neg,
  reshape,
  sin,
  sliceAxis,
  sub,
  sum,
  swapaxes,
  type Tensor,
  type TensorLike,
  tanh,
  type Value,
  zeros,
} from '../../tensor/index.ts'
import { silu } from '../functional/activations.ts'
import { attention } from '../functional/attention.ts'
import { interpolate } from '../functional/vision.ts'
import { Composite, ModuleList } from '../module/containers.ts'
import { Conv2d } from '../module/conv.ts'
import { Linear } from '../module/layers.ts'
import { type ApplyContext, initKey, type Module, type Params } from '../module/module.ts'
import { GroupNorm } from '../module/normalisation.ts'
import { Dropout } from '../module/regularisation.ts'

const keyAt =
  (k: Key) =>
  (i: number): Key =>
    foldIn(k, i)

/**
 * Sinusoidal timestep embeddings (DDPM's, from the Transformer's): for each t, [cos(t f_i), sin(t f_i)] with
 * frequencies f_i = maxPeriod^(-i / half) for i < half.
 *
 * @param t - Timesteps (or noise levels) `[B]`, any real values.
 * @param dim - Embedding width.
 * @param maxPeriod - The longest period (default 10000).
 * @returns `[B, dim]`, differentiable in t.
 * @concept diffusion-models
 * @example
 * ```ts
 * import { blocks } from 'omai-engine/nn'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(blocks.timestepEmbedding([0], 4)) // => [[1, 1, 0, 0]]
 * ```
 */
export function timestepEmbedding(t: TensorLike, dim: number, maxPeriod = 10000): Value {
  const tt = reshape(asTensor(t), [-1, 1])
  const half = Math.floor(dim / 2)
  const freqs = fromStorage(
    Float64Array.from({ length: half }, (_, i) => Math.exp((-Math.log(maxPeriod) * i) / half)),
    [1, half],
    'float64',
  )
  const args = mul(tt, freqs)
  const parts = [cos(args), sin(args)]
  if (dim % 2 === 1) parts.push(mul(sub(tt, tt), 0))
  return concat(parts, 1)
}

/**
 * Random Fourier features of the log noise level (EDM and NCSN++'s noise conditioning): [cos(2 pi w log sigma),
 * sin(2 pi w log sigma)] with fixed Gaussian frequencies w.
 *
 * @param sigma - Noise levels `[B]` (positive).
 * @param frequencies - Fixed frequencies `[dim / 2]` (draw them once with {@link fourierFrequencies}).
 * @returns `[B, dim]`.
 * @example
 * ```ts
 * import { blocks } from 'omai-engine/nn'
 * import { key } from 'omai-engine/random'
 * blocks.noiseLevelEmbedding([0.5, 2], blocks.fourierFrequencies(key(0), 8)).shape // => [2, 8]
 * ```
 */
export function noiseLevelEmbedding(sigma: TensorLike, frequencies: TensorLike): Value {
  const s = reshape(log(asTensor(sigma)), [-1, 1])
  const w = reshape(asTensor(frequencies), [1, -1])
  const args = mul(mul(s, w), 2 * Math.PI)
  return concat([cos(args), sin(args)], 1)
}

/**
 * Fixed Gaussian frequencies for {@link noiseLevelEmbedding}: N(0, scale^2), `dim / 2` of them.
 *
 * @param key - The key.
 * @param dim - The embedding width (even).
 * @param scale - Standard deviation (default 16, NCSN++'s).
 * @returns `[dim / 2]`.
 * @example
 * ```ts
 * import { blocks } from 'omai-engine/nn'
 * import { key } from 'omai-engine/random'
 * blocks.fourierFrequencies(key(0), 8).shape // => [4]
 * ```
 */
export function fourierFrequencies(key: Key, dim: number, scale = 16): Tensor {
  return normal(key, [Math.floor(dim / 2)], { scale })
}

/**
 * The time-embedding network of a diffusion U-Net: a sinusoidal embedding followed by Linear, SiLU, Linear (`linear_1`,
 * `linear_2`, as diffusers' TimestepEmbedding).
 *
 * @example
 * ```ts
 * import { blocks } from 'omai-engine/nn'
 * new blocks.TimeEmbedding(8, 32).forward([1, 500]).shape // => [2, 32]
 * ```
 */
export class TimeEmbedding extends Composite {
  readonly kind = 'TimeEmbedding'
  readonly dim: number

  constructor(dim: number, outDim: number, options: { readonly key?: Key } = {}) {
    const k = keyAt(initKey('TimeEmbedding', options.key))
    super({ linear_1: new Linear(dim, outDim, { key: k(0) }), linear_2: new Linear(outDim, outDim, { key: k(1) }) })
    this.dim = dim
  }

  apply(params: Params, t: Value, ctx: ApplyContext = {}): Value {
    const p = this.parts(params, ctx)
    const y = p.run<Value, Value>('linear_2', silu(p.run<Value, Value>('linear_1', timestepEmbedding(t, this.dim))))
    p.done()
    return y
  }

  outputShape(s: readonly number[]): number[] {
    return [s[0] ?? 1, this.part<Linear>('linear_2').outFeatures]
  }
}

/** The input of a diffusion block: feature maps and the time embedding `[B, T]`. */
export interface DiffusionInput {
  readonly x: TensorLike
  readonly t: TensorLike
}

/**
 * The residual block of diffusion U-Nets (DDPM, Stable Diffusion): GroupNorm, SiLU and a 3x3 convolution, plus a
 * projection of the time embedding added per channel, then GroupNorm, SiLU, dropout and a 3x3 convolution, with a
 * 1x1 shortcut when the width changes.
 *
 * @concept diffusion-models
 * @reference Ho, Jain and Abbeel (2020), Denoising diffusion probabilistic models.
 * @example
 * ```ts
 * import { blocks } from 'omai-engine/nn'
 * import { zeros } from 'omai-engine/tensor'
 * new blocks.DiffusionResBlock(4, 8, 16, { groups: 2 }).forward({ x: zeros([1, 4, 4, 4]), t: zeros([1, 16]) }).shape // => [1, 8, 4, 4]
 * ```
 */
export class DiffusionResBlock extends Composite<DiffusionInput, Value> {
  readonly kind = 'DiffusionResBlock'
  readonly outChannels: number

  constructor(
    inChannels: number,
    outChannels: number,
    timeDim: number,
    options: { readonly groups?: number; readonly dropout?: number; readonly key?: Key } = {},
  ) {
    const k = keyAt(initKey('DiffusionResBlock', options.key))
    const groups = options.groups ?? 32
    super({
      norm1: new GroupNorm(Math.min(groups, inChannels), inChannels),
      conv1: new Conv2d(inChannels, outChannels, 3, { padding: 1, key: k(1) }),
      time_emb_proj: new Linear(timeDim, outChannels, { key: k(2) }),
      norm2: new GroupNorm(Math.min(groups, outChannels), outChannels),
      dropout: new Dropout(options.dropout ?? 0),
      conv2: new Conv2d(outChannels, outChannels, 3, { padding: 1, key: k(5) }),
      ...(inChannels === outChannels ? {} : { conv_shortcut: new Conv2d(inChannels, outChannels, 1, { key: k(6) }) }),
    })
    this.outChannels = outChannels
  }

  protected override prepare(x: unknown): unknown {
    const r = x as DiffusionInput
    return { x: asTensor(r.x), t: asTensor(r.t) }
  }

  apply(params: Params, input: DiffusionInput, ctx: ApplyContext = {}): Value {
    const x = asTensor(input.x)
    const p = this.parts(params, ctx)
    const r = (name: string, v: Value): Value => p.run<Value, Value>(name, v)
    let h = r('conv1', silu(r('norm1', x)))
    const temb = reshape(r('time_emb_proj', silu(asTensor(input.t))), [-1, this.outChannels, 1, 1])
    h = add(h, temb)
    h = r('conv2', r('dropout', silu(r('norm2', h))))
    const y = add(this.has('conv_shortcut') ? r('conv_shortcut', x) : x, h)
    p.done()
    return y
  }

  outputShape(s: readonly number[]): number[] {
    return [s[0] ?? 1, this.outChannels, s[2] ?? 0, s[3] ?? 0]
  }
}

/**
 * Self-attention over the spatial positions of a feature map (the attention block of diffusion U-Nets): GroupNorm,
 * 1x1 query, key, value and output projections over H W tokens with several heads, plus the input.
 *
 * @example
 * ```ts
 * import { blocks } from 'omai-engine/nn'
 * import { zeros } from 'omai-engine/tensor'
 * new blocks.SpatialSelfAttention(8, { heads: 2, groups: 4 }).forward(zeros([1, 8, 3, 3])).shape // => [1, 8, 3, 3]
 * ```
 */
export class SpatialSelfAttention extends Composite {
  readonly kind = 'SpatialSelfAttention'
  readonly channels: number
  readonly heads: number

  constructor(
    channels: number,
    options: { readonly heads?: number; readonly groups?: number; readonly key?: Key } = {},
  ) {
    const k = keyAt(initKey('SpatialSelfAttention', options.key))
    super({
      norm: new GroupNorm(Math.min(options.groups ?? 32, channels), channels),
      to_q: new Linear(channels, channels, { key: k(1) }),
      to_k: new Linear(channels, channels, { key: k(2) }),
      to_v: new Linear(channels, channels, { key: k(3) }),
      to_out: new Linear(channels, channels, { key: k(4) }),
    })
    this.channels = channels
    this.heads = options.heads ?? 1
    if (channels % this.heads !== 0) throw new OmaiError('value', 'SpatialSelfAttention: heads must divide channels')
  }

  apply(params: Params, x: Value, ctx: ApplyContext = {}): Value {
    const [b = 1, c = 1, h = 1, w = 1] = x.shape
    const p = this.parts(params, ctx)
    const tokens = swapaxes(reshape(p.run<Value, Value>('norm', x), [b, c, h * w]), 1, 2)
    const heads = (v: Value): Value => swapaxes(reshape(v, [b, h * w, this.heads, c / this.heads]), 1, 2)
    const r = attention(
      heads(p.run<Value, Value>('to_q', tokens)),
      heads(p.run<Value, Value>('to_k', tokens)),
      heads(p.run<Value, Value>('to_v', tokens)),
    )
    const merged = reshape(swapaxes(r.output, 1, 2), [b, h * w, c])
    const out = reshape(swapaxes(p.run<Value, Value>('to_out', merged), 1, 2), [b, c, h, w])
    p.done()
    return add(x, out)
  }

  outputShape(s: readonly number[]): number[] {
    return [...s]
  }
}

/** Options of the diffusion U-Net stages. */
export interface DiffusionStageOptions {
  /** Residual blocks in the stage (default 2). */
  readonly layers?: number
  /** Self-attention after each residual block (default false). */
  readonly attention?: boolean
  readonly heads?: number
  readonly groups?: number
  readonly dropout?: number
  readonly key?: Key
}

/** The output of a {@link DiffusionDownStage}: the downsampled features and the skip features of each layer. */
export interface DownStageOutput {
  readonly output: Value
  readonly skips: readonly Value[]
}

/**
 * A diffusion U-Net encoder stage (diffusers' DownBlock2D and CrossAttnDownBlock2D without text conditioning):
 * residual blocks conditioned on the time embedding, optional self-attention, then a strided 3x3 convolution that
 * halves the resolution (unless `downsample` is false). Returns the skip features for the decoder.
 *
 * @concept diffusion-models
 * @example
 * ```ts
 * import { blocks } from 'omai-engine/nn'
 * import { zeros } from 'omai-engine/tensor'
 * const s = new blocks.DiffusionDownStage(4, 8, 16, { groups: 2, attention: true })
 * s.forward({ x: zeros([1, 4, 8, 8]), t: zeros([1, 16]) }).output.shape // => [1, 8, 4, 4]
 * ```
 */
export class DiffusionDownStage extends Composite<DiffusionInput, DownStageOutput> {
  readonly kind = 'DiffusionDownStage'
  readonly downsample: boolean

  constructor(
    inChannels: number,
    outChannels: number,
    timeDim: number,
    options: DiffusionStageOptions & { readonly downsample?: boolean } = {},
  ) {
    const k = keyAt(initKey('DiffusionDownStage', options.key))
    const n = options.layers ?? 2
    const block = {
      ...(options.groups === undefined ? {} : { groups: options.groups }),
      ...(options.dropout === undefined ? {} : { dropout: options.dropout }),
    }
    super({
      resnets: new ModuleList(
        Array.from(
          { length: n },
          (_, i) =>
            new DiffusionResBlock(i === 0 ? inChannels : outChannels, outChannels, timeDim, {
              ...block,
              key: foldIn(k(0), i),
            }),
        ),
      ),
      ...(options.attention === true
        ? {
            attentions: new ModuleList(
              Array.from(
                { length: n },
                (_, i) =>
                  new SpatialSelfAttention(outChannels, {
                    ...block,
                    ...(options.heads === undefined ? {} : { heads: options.heads }),
                    key: foldIn(k(1), i),
                  }),
              ),
            ),
          }
        : {}),
      ...((options.downsample ?? true)
        ? { downsampler: new Conv2d(outChannels, outChannels, 3, { stride: 2, padding: 1, key: k(2) }) }
        : {}),
    })
    this.downsample = options.downsample ?? true
  }

  protected override prepare(x: unknown): unknown {
    return x
  }

  apply(params: Params, input: DiffusionInput, ctx: ApplyContext = {}): DownStageOutput {
    const p = this.parts(params, ctx)
    const resnets = this.part<ModuleList<Module<DiffusionInput, Value>>>('resnets').modules
    const attns = this.has('attentions') ? this.part<ModuleList<Module>>('attentions').modules : []
    const rp = (params['resnets'] ?? {}) as Params
    const ap = (params['attentions'] ?? {}) as Params
    const t = asTensor(input.t)
    let h = asTensor(input.x)
    const skips: Value[] = []
    for (const [i, res] of resnets.entries()) {
      h = res.apply((rp[String(i)] ?? {}) as Params, { x: h, t }, p.context('resnets'))
      const a = attns[i]
      if (a !== undefined) h = a.apply((ap[String(i)] ?? {}) as Params, h)
      skips.push(h)
    }
    const output = this.downsample ? p.run<Value, Value>('downsampler', h) : h
    p.done()
    return { output, skips }
  }

  outputShape(s: readonly number[]): number[] {
    const f = this.downsample ? 2 : 1
    const c = this.part<ModuleList<DiffusionResBlock>>('resnets').at(0).outChannels
    return [s[0] ?? 1, c, Math.ceil((s[2] ?? 0) / f), Math.ceil((s[3] ?? 0) / f)]
  }
}

/** The input of a {@link DiffusionUpStage}: features, the time embedding and the encoder's skips (last first). */
export interface UpStageInput extends DiffusionInput {
  readonly skips: readonly TensorLike[]
}

/**
 * A diffusion U-Net decoder stage (diffusers' UpBlock2D): for each layer, concatenate the matching encoder skip, a
 * time-conditioned residual block and optional self-attention; then nearest upsampling and a 3x3 convolution (unless
 * `upsample` is false).
 *
 * @concept diffusion-models
 * @example
 * ```ts
 * import { blocks } from 'omai-engine/nn'
 * import { zeros } from 'omai-engine/tensor'
 * const s = new blocks.DiffusionUpStage(8, [4, 4], 4, 16, { groups: 2 })
 * s.forward({ x: zeros([1, 8, 4, 4]), t: zeros([1, 16]), skips: [zeros([1, 4, 4, 4]), zeros([1, 4, 4, 4])] }).shape // => [1, 4, 8, 8]
 * ```
 */
export class DiffusionUpStage extends Composite<UpStageInput, Value> {
  readonly kind = 'DiffusionUpStage'
  readonly upsample: boolean
  readonly outChannels: number

  constructor(
    inChannels: number,
    skipChannels: readonly number[],
    outChannels: number,
    timeDim: number,
    options: DiffusionStageOptions & { readonly upsample?: boolean } = {},
  ) {
    const k = keyAt(initKey('DiffusionUpStage', options.key))
    const block = {
      ...(options.groups === undefined ? {} : { groups: options.groups }),
      ...(options.dropout === undefined ? {} : { dropout: options.dropout }),
    }
    super({
      resnets: new ModuleList(
        skipChannels.map(
          (sc, i) =>
            new DiffusionResBlock((i === 0 ? inChannels : outChannels) + sc, outChannels, timeDim, {
              ...block,
              key: foldIn(k(0), i),
            }),
        ),
      ),
      ...(options.attention === true
        ? {
            attentions: new ModuleList(
              skipChannels.map(
                (_, i) =>
                  new SpatialSelfAttention(outChannels, {
                    ...block,
                    ...(options.heads === undefined ? {} : { heads: options.heads }),
                    key: foldIn(k(1), i),
                  }),
              ),
            ),
          }
        : {}),
      ...((options.upsample ?? true)
        ? { upsampler: new Conv2d(outChannels, outChannels, 3, { padding: 1, key: k(2) }) }
        : {}),
    })
    this.upsample = options.upsample ?? true
    this.outChannels = outChannels
  }

  protected override prepare(x: unknown): unknown {
    return x
  }

  apply(params: Params, input: UpStageInput, ctx: ApplyContext = {}): Value {
    const p = this.parts(params, ctx)
    const resnets = this.part<ModuleList<Module<DiffusionInput, Value>>>('resnets').modules
    const attns = this.has('attentions') ? this.part<ModuleList<Module>>('attentions').modules : []
    if (input.skips.length !== resnets.length) throw new OmaiError('value', 'DiffusionUpStage: one skip per layer')
    const rp = (params['resnets'] ?? {}) as Params
    const ap = (params['attentions'] ?? {}) as Params
    const t = asTensor(input.t)
    let h = asTensor(input.x)
    for (const [i, res] of resnets.entries()) {
      h = concat([h, asTensor(input.skips[i] as TensorLike)], 1)
      h = res.apply((rp[String(i)] ?? {}) as Params, { x: h, t }, p.context('resnets'))
      const a = attns[i]
      if (a !== undefined) h = a.apply((ap[String(i)] ?? {}) as Params, h)
    }
    if (this.upsample) h = p.run<Value, Value>('upsampler', interpolate(h, { scaleFactor: 2, mode: 'nearest' }))
    p.done()
    return h
  }

  outputShape(s: readonly number[]): number[] {
    const f = this.upsample ? 2 : 1
    return [s[0] ?? 1, this.outChannels, (s[2] ?? 0) * f, (s[3] ?? 0) * f]
  }
}

/** The output of a coupling layer: the transformed input and log |det J|, one per example. */
export interface CouplingOutput {
  readonly output: Value
  readonly logDet: Value
}

/** A 0/1 mask of the last axis: alternate features (`parity` 0 keeps the even ones fixed). */
const checkerMask = (dim: number, parity: number): Tensor =>
  fromStorage(
    Float64Array.from({ length: dim }, (_, i) => ((i + parity) % 2 === 0 ? 1 : 0)),
    [dim],
    'float64',
  )

/**
 * An affine coupling layer (RealNVP, Dinh et al. 2017): the masked features pass through unchanged and condition a
 * scale s and shift t for the others, y = x_a + (1 - m)(x_b exp(s) + t) with s bounded by tanh; log |det J| is the sum
 * of s over the transformed features, and the inverse is exact. The conditioner maps `[..., D]` to `[..., 2 D]`.
 *
 * @concept normalising-flows
 * @reference Dinh, Sohl-Dickstein and Bengio (2017), Density estimation using Real NVP.
 * @example
 * ```ts
 * import { blocks, Linear } from 'omai-engine/nn'
 * import { key } from 'omai-engine/random'
 * new blocks.AffineCoupling(new Linear(4, 8, { key: key(0) }), 4).forward([[1, 2, 3, 4]]).logDet.shape // => [1]
 * ```
 */
export class AffineCoupling extends Composite<Value, CouplingOutput> {
  readonly kind = 'AffineCoupling'
  readonly dim: number
  readonly parity: number

  constructor(conditioner: Module, dim: number, options: { readonly parity?: number } = {}) {
    super({ conditioner })
    this.dim = dim
    this.parity = options.parity ?? 0
  }

  private scaleShift(params: Params, fixed: Value, ctx: ApplyContext): { s: Value; t: Value } {
    const h = this.part('conditioner').apply((params['conditioner'] ?? {}) as Params, fixed, ctx)
    const m = checkerMask(this.dim, this.parity)
    const free = sub(1, m)
    const s = mul(tanh(sliceAxis(h, -1, 0, this.dim)), free)
    const t = mul(sliceAxis(h, -1, this.dim, 2 * this.dim), free)
    return { s, t }
  }

  apply(params: Params, x: Value, ctx: ApplyContext = {}): CouplingOutput {
    const m = checkerMask(this.dim, this.parity)
    const fixed = mul(x, m)
    const { s, t } = this.scaleShift(params, fixed, ctx)
    const y = add(fixed, mul(sub(1, m), add(mul(x, exp(s)), t)))
    return { output: y, logDet: sum(s, -1) }
  }

  /** The inverse map x = f^-1(y) and its log |det J| (the negative of the forward one). */
  inverse(params: Params, y: Value, ctx: ApplyContext = {}): CouplingOutput {
    const m = checkerMask(this.dim, this.parity)
    const fixed = mul(y, m)
    const { s, t } = this.scaleShift(params, fixed, ctx)
    const x = add(fixed, mul(sub(1, m), mul(sub(y, t), exp(neg(s)))))
    return { output: x, logDet: neg(sum(s, -1)) }
  }

  outputShape(s: readonly number[]): number[] {
    return [...s]
  }
}

/**
 * An additive coupling layer (NICE, Dinh et al. 2015): y = x_a + (1 - m)(x_b + t(x_a)); volume-preserving
 * (log |det J| = 0). The conditioner maps `[..., D]` to `[..., D]`.
 *
 * @reference Dinh, Krueger and Bengio (2015), NICE: non-linear independent components estimation.
 * @example
 * ```ts
 * import { blocks, Linear } from 'omai-engine/nn'
 * import { key } from 'omai-engine/random'
 * new blocks.AdditiveCoupling(new Linear(2, 2, { key: key(0) }), 2).forward([[1, 2]]).output.shape // => [1, 2]
 * ```
 */
export class AdditiveCoupling extends Composite<Value, CouplingOutput> {
  readonly kind = 'AdditiveCoupling'
  readonly dim: number
  readonly parity: number

  constructor(conditioner: Module, dim: number, options: { readonly parity?: number } = {}) {
    super({ conditioner })
    this.dim = dim
    this.parity = options.parity ?? 0
  }

  private shift(params: Params, fixed: Value, ctx: ApplyContext): Value {
    const t = this.part('conditioner').apply((params['conditioner'] ?? {}) as Params, fixed, ctx)
    return mul(t, sub(1, checkerMask(this.dim, this.parity)))
  }

  apply(params: Params, x: Value, ctx: ApplyContext = {}): CouplingOutput {
    const fixed = mul(x, checkerMask(this.dim, this.parity))
    const y = add(x, this.shift(params, fixed, ctx))
    return { output: y, logDet: zeros(x.shape.slice(0, -1)) }
  }

  /** The inverse map. */
  inverse(params: Params, y: Value, ctx: ApplyContext = {}): CouplingOutput {
    const fixed = mul(y, checkerMask(this.dim, this.parity))
    return { output: sub(y, this.shift(params, fixed, ctx)), logDet: zeros(y.shape.slice(0, -1)) }
  }

  outputShape(s: readonly number[]): number[] {
    return [...s]
  }
}
