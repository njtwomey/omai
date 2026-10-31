/**
 * Operator-learning blocks, maps between functions: the DeepONet (a branch network reads the input function at fixed
 * sensors, a trunk network reads the query point, and the output function's value is the dot product of their
 * features), and the Fourier neural operator (the spectral convolution that multiplies the lowest Fourier modes of a
 * field by learned complex weights, the Fourier layer that adds a pointwise path to it, and the lift-iterate-project
 * network in one and two dimensions).
 */

import { MdmlError } from '../../base/index.ts'
import { fft, ifft, irfft, rfft } from '../../numerics/fourier/index.ts'
import { foldIn, type Key, uniform } from '../../random/index.ts'
import {
  add,
  asTensor,
  concat,
  einsum,
  mul,
  sliceAxis,
  sub,
  type Tensor,
  type TensorLike,
  type Value,
  zeros,
} from '../../tensor/index.ts'
import { gelu } from '../functional/activations.ts'
import { Tanh } from '../module/activations.ts'
import { Composite, mlp, Sequential } from '../module/containers.ts'
import { Conv1d, Conv2d } from '../module/conv.ts'
import { type ApplyContext, initKey, Module, type Params } from '../module/module.ts'

/** The input of {@link DeepONet}: the input functions at the sensors `u` `[B, m]` and the query points `y`. */
export interface DeepONetInput {
  readonly u: TensorLike
  /** Query points `[Q, d]` shared by the batch, or `[B, Q, d]` per function. */
  readonly y: TensorLike
}

/** Options of {@link DeepONet}. */
export interface DeepONetOptions {
  /** The hidden widths of the branch network (default `[64, 64]`). */
  readonly branchHidden?: readonly number[]
  /** The hidden widths of the trunk network (default `[64, 64]`). */
  readonly trunkHidden?: readonly number[]
  /** Apply the activation after the trunk's last layer, as the original (default true). */
  readonly trunkOutputActivation?: boolean
  /** The activation of both networks (default tanh). */
  readonly activation?: () => Module
  readonly key?: Key
}

/**
 * The DeepONet (Lu et al. 2021) in its unstacked form: `G(u)(y) = sum_k branch_k(u) trunk_k(y) + b`, where the branch
 * network reads the input function sampled at `m` sensors and the trunk network reads the query location, both
 * producing `p` features. The universal approximation theorem of operators motivates exactly this split: a linear
 * operator is represented exactly by a linear branch.
 *
 * @reference Lu, Jin, Pang, Zhang and Karniadakis (2021), Learning nonlinear operators via DeepONet.
 * @example
 * ```ts
 * import { DeepONet } from 'mdml-engine/nn/blocks'
 * import { key } from 'mdml-engine/random'
 * import { zeros } from 'mdml-engine/tensor'
 * new DeepONet(10, 1, 16, { key: key(0) }).forward({ u: zeros([4, 10]), y: zeros([7, 1]) }).shape // => [4, 7]
 * ```
 */
export class DeepONet extends Composite<DeepONetInput, Value> {
  readonly kind = 'DeepONet'
  readonly sensors: number
  readonly coordinateDim: number
  readonly basis: number

  /**
   * @param sensors - The number of sensors `m` (the input function's samples).
   * @param coordinateDim - The dimension `d` of a query point.
   * @param basis - The number `p` of branch and trunk features.
   * @param options - Hidden widths, activation and key.
   */
  constructor(sensors: number, coordinateDim: number, basis: number, options: DeepONetOptions = {}) {
    const key = initKey('DeepONet', options.key)
    const act = options.activation ?? ((): Module => new Tanh())
    const branch = mlp([sensors, ...(options.branchHidden ?? [64, 64]), basis], {
      activation: act,
      key: foldIn(key, 0),
    })
    const trunkBase = mlp([coordinateDim, ...(options.trunkHidden ?? [64, 64]), basis], {
      activation: act,
      key: foldIn(key, 1),
    })
    const trunk = (options.trunkOutputActivation ?? true) ? new Sequential(...trunkBase.layers, act()) : trunkBase
    super({ branch, trunk }, { bias: zeros([]) })
    this.sensors = sensors
    this.coordinateDim = coordinateDim
    this.basis = basis
  }

  protected override prepare(x: unknown): unknown {
    return x
  }

  apply(params: Params, input: DeepONetInput, ctx: ApplyContext = {}): Value {
    const p = this.parts(params, ctx)
    const b = p.run<Value, Value>('branch', asTensor(input.u))
    const y = asTensor(input.y)
    const t = p.run<Value, Value>('trunk', y)
    p.done()
    const out = y.shape.length === 2 ? einsum('bp,qp->bq', b, t) : einsum('bp,bqp->bq', b, t)
    return add(out, params['bias'] as Value)
  }

  outputShape(s: readonly number[]): number[] {
    return [s[0] ?? 1]
  }

  override extraRepr(): string {
    return `sensors=${this.sensors}, coordinate_dim=${this.coordinateDim}, basis=${this.basis}`
  }
}

/** A complex multiplication of spectra: `(a + ib)(c + id)` summed over the input channel. */
function mixModes(re: Value, im: Value, wr: Value, wi: Value, spec: string): { re: Value; im: Value } {
  return {
    re: sub(einsum(spec, re, wr), einsum(spec, im, wi)),
    im: add(einsum(spec, re, wi), einsum(spec, im, wr)),
  }
}

/** Zeros appended along an axis so the spectrum has `total` entries there. */
function padSpectrum(x: Value, axis: number, total: number): Value {
  const have = x.shape[axis < 0 ? x.shape.length + axis : axis] ?? 0
  if (have >= total) return x
  const shape = [...x.shape]
  shape[axis < 0 ? x.shape.length + axis : axis] = total - have
  return concat([x, zeros(shape)], axis)
}

/** Options of the spectral convolutions. */
export interface SpectralConvOptions {
  readonly key?: Key
}

/**
 * The spectral convolution of the Fourier neural operator in one dimension: the real FFT of the field along its length,
 * the lowest `modes` frequencies multiplied by learned complex weights that mix the channels
 * (`out_k = sum_i W_{iok} x_{ik}`), all higher frequencies dropped, and the inverse FFT. With one mode it is a
 * circular convolution with a cosine; the resolution of the input is free, since the weights live on the modes.
 * Parameters `real` and `imag`, each `[in, out, modes]`.
 *
 * @reference Li, Kovachki, Azizzadenesheli, Liu, Bhattacharya, Stuart and Anandkumar (2021), Fourier neural operator for parametric partial differential equations.
 * @example
 * ```ts
 * import { SpectralConv1d } from 'mdml-engine/nn/blocks'
 * import { zeros } from 'mdml-engine/tensor'
 * new SpectralConv1d(2, 3, 4).forward(zeros([1, 2, 16])).shape // => [1, 3, 16]
 * ```
 */
export class SpectralConv1d extends Module {
  readonly kind = 'SpectralConv1d'
  readonly inChannels: number
  readonly outChannels: number
  readonly modes: number

  constructor(inChannels: number, outChannels: number, modes: number, options: SpectralConvOptions = {}) {
    super()
    this.inChannels = inChannels
    this.outChannels = outChannels
    this.modes = modes
    this.params = this.initParams(initKey('SpectralConv1d', options.key))
  }

  override initParams(key: Key): Params {
    const shape = [this.inChannels, this.outChannels, this.modes]
    const scale = 1 / (this.inChannels * this.outChannels)
    return {
      real: mul(uniform(foldIn(key, 0), shape), scale),
      imag: mul(uniform(foldIn(key, 1), shape), scale),
    }
  }

  apply(params: Params, x: Value): Value {
    const n = x.shape[x.shape.length - 1] ?? 0
    const half = Math.floor(n / 2) + 1
    if (this.modes > half)
      throw new MdmlError('value', `SpectralConv1d: ${this.modes} modes exceed the ${half} available`)
    const spectrum = rfft(x)
    const kept = {
      re: sliceAxis(spectrum.re, -1, 0, this.modes),
      im: sliceAxis(spectrum.im, -1, 0, this.modes),
    }
    const mixed = mixModes(kept.re, kept.im, params['real'] as Value, params['imag'] as Value, 'bik,iok->bok')
    return irfft({ re: padSpectrum(mixed.re, -1, half), im: padSpectrum(mixed.im, -1, half) }, n)
  }

  outputShape(s: readonly number[]): number[] {
    return [...s.slice(0, -2), this.outChannels, s[s.length - 1] ?? 0]
  }

  override extraRepr(): string {
    return `${this.inChannels}, ${this.outChannels}, modes=${this.modes}`
  }
}

/**
 * The spectral convolution in two dimensions: the 2-D real FFT, the lowest `modes1 x modes2` frequencies (both the
 * positive and the negative ones along the first axis) multiplied by learned complex weights, the rest dropped, and
 * the inverse FFT. Parameters `real1`, `imag1` (the non-negative frequencies of axis 0) and `real2`, `imag2` (the
 * negative ones), each `[in, out, modes1, modes2]`.
 *
 * @example
 * ```ts
 * import { SpectralConv2d } from 'mdml-engine/nn/blocks'
 * import { zeros } from 'mdml-engine/tensor'
 * new SpectralConv2d(1, 2, 3, 3).forward(zeros([1, 1, 8, 8])).shape // => [1, 2, 8, 8]
 * ```
 */
export class SpectralConv2d extends Module {
  readonly kind = 'SpectralConv2d'
  readonly inChannels: number
  readonly outChannels: number
  readonly modes1: number
  readonly modes2: number

  constructor(
    inChannels: number,
    outChannels: number,
    modes1: number,
    modes2: number,
    options: SpectralConvOptions = {},
  ) {
    super()
    this.inChannels = inChannels
    this.outChannels = outChannels
    this.modes1 = modes1
    this.modes2 = modes2
    this.params = this.initParams(initKey('SpectralConv2d', options.key))
  }

  override initParams(key: Key): Params {
    const shape = [this.inChannels, this.outChannels, this.modes1, this.modes2]
    const scale = 1 / (this.inChannels * this.outChannels)
    const draw = (i: number): Tensor => mul(uniform(foldIn(key, i), shape), scale) as Tensor
    return { real1: draw(0), imag1: draw(1), real2: draw(2), imag2: draw(3) }
  }

  apply(params: Params, x: Value): Value {
    const [h, w] = [x.shape[x.shape.length - 2] ?? 0, x.shape[x.shape.length - 1] ?? 0]
    const wHalf = Math.floor(w / 2) + 1
    if (this.modes2 > wHalf || 2 * this.modes1 > h)
      throw new MdmlError(
        'value',
        `SpectralConv2d: ${this.modes1} x ${this.modes2} modes do not fit a ${h} x ${w} field`,
      )
    // Real FFT over the last axis, complex FFT over the one before it.
    const along = rfft(x, undefined, -1)
    const spectrum = fft(along, undefined, -2)
    const block = (rows: readonly [number, number]): { re: Value; im: Value } => ({
      re: sliceAxis(sliceAxis(spectrum.re, -2, rows[0], rows[1]), -1, 0, this.modes2),
      im: sliceAxis(sliceAxis(spectrum.im, -2, rows[0], rows[1]), -1, 0, this.modes2),
    })
    const spec = 'bixy,ioxy->boxy'
    const low = block([0, this.modes1])
    const high = block([h - this.modes1, h])
    const lowOut = mixModes(low.re, low.im, params['real1'] as Value, params['imag1'] as Value, spec)
    const highOut = mixModes(high.re, high.im, params['real2'] as Value, params['imag2'] as Value, spec)
    const assemble = (a: Value, b: Value): Value => {
      const rows = padSpectrum(a, -1, wHalf)
      const rowsHigh = padSpectrum(b, -1, wHalf)
      const gap = h - 2 * this.modes1
      return gap > 0
        ? concat([rows, zeros([...rows.shape.slice(0, -2), gap, wHalf]), rowsHigh], -2)
        : concat([rows, rowsHigh], -2)
    }
    const full = { re: assemble(lowOut.re, highOut.re), im: assemble(lowOut.im, highOut.im) }
    const back = ifft(full, undefined, -2)
    return irfft(back, w, -1)
  }

  outputShape(s: readonly number[]): number[] {
    return [...s.slice(0, -3), this.outChannels, ...s.slice(-2)]
  }

  override extraRepr(): string {
    return `${this.inChannels}, ${this.outChannels}, modes=(${this.modes1}, ${this.modes2})`
  }
}

/** Options of {@link FourierLayer}. */
export interface FourierLayerOptions {
  /** Apply the GELU after the sum of the two paths (default true; the last layer of a network omits it). */
  readonly activation?: boolean
  readonly key?: Key
}

/**
 * A Fourier layer: `v -> sigma(SpectralConv(v) + W v)` where `W` is a pointwise (1x1) convolution carrying the
 * high-frequency and local content that the truncated spectral path drops.
 *
 * @example
 * ```ts
 * import { FourierLayer } from 'mdml-engine/nn/blocks'
 * import { zeros } from 'mdml-engine/tensor'
 * new FourierLayer(8, [4]).forward(zeros([2, 8, 32])).shape // => [2, 8, 32]
 * ```
 */
export class FourierLayer extends Composite {
  readonly kind = 'FourierLayer'
  readonly width: number
  readonly modes: readonly number[]
  readonly activation: boolean

  /**
   * @param width - Channels.
   * @param modes - Retained modes: one number for 1-D fields, two for 2-D.
   * @param options - Whether to apply the activation, and the key.
   */
  constructor(width: number, modes: readonly number[], options: FourierLayerOptions = {}) {
    const key = initKey('FourierLayer', options.key)
    const dims = modes.length
    if (dims !== 1 && dims !== 2) throw new MdmlError('value', 'FourierLayer: give one or two mode counts')
    super({
      spectral:
        dims === 1
          ? new SpectralConv1d(width, width, modes[0] as number, { key: foldIn(key, 0) })
          : new SpectralConv2d(width, width, modes[0] as number, modes[1] as number, { key: foldIn(key, 0) }),
      pointwise:
        dims === 1
          ? new Conv1d(width, width, 1, { key: foldIn(key, 1) })
          : new Conv2d(width, width, 1, { key: foldIn(key, 1) }),
    })
    this.width = width
    this.modes = modes
    this.activation = options.activation ?? true
  }

  apply(params: Params, x: Value, ctx: ApplyContext = {}): Value {
    const p = this.parts(params, ctx)
    const y = add(p.run<Value, Value>('spectral', x), p.run<Value, Value>('pointwise', x))
    p.done()
    return this.activation ? gelu(y) : y
  }

  outputShape(s: readonly number[]): number[] {
    return [...s]
  }
}

/** Options of {@link FNO}. */
export interface FnoOptions {
  /** Channels of the Fourier layers (default 32). */
  readonly width?: number
  /** Retained modes per spatial axis (default 12 in 1-D, 8 by 8 in 2-D). */
  readonly modes?: readonly number[]
  /** Number of Fourier layers (default 4). */
  readonly layers?: number
  /** The hidden width of the projection (default 4 times the width). */
  readonly projection?: number
  readonly key?: Key
}

/**
 * The Fourier neural operator: a pointwise lifting of the input channels to `width`, a stack of {@link FourierLayer}s,
 * and a two-layer pointwise projection to the output channels. The input is `[B, C, N]` (or `[B, C, H, W]` for two
 * dimensions) with the coordinates as extra channels if wanted; the same weights apply at any resolution.
 *
 * @example
 * ```ts
 * import { FNO } from 'mdml-engine/nn/blocks'
 * import { key } from 'mdml-engine/random'
 * import { zeros } from 'mdml-engine/tensor'
 * new FNO(1, 1, 1, { width: 8, modes: [4], layers: 2, key: key(0) }).forward(zeros([2, 1, 32])).shape // => [2, 1, 32]
 * ```
 */
export class FNO extends Composite {
  readonly kind = 'FNO'
  readonly dims: 1 | 2
  readonly layerCount: number
  readonly outChannels: number

  /**
   * @param dims - The number of spatial axes (1 or 2).
   * @param inChannels - Input channels.
   * @param outChannels - Output channels.
   * @param options - Width, modes, layers and projection width.
   */
  constructor(dims: 1 | 2, inChannels: number, outChannels: number, options: FnoOptions = {}) {
    const key = initKey('FNO', options.key)
    const width = options.width ?? 32
    const modes = options.modes ?? (dims === 1 ? [12] : [8, 8])
    const layers = options.layers ?? 4
    const hidden = options.projection ?? 4 * width
    const pointwise = (i: number, o: number, k: number): Module =>
      dims === 1 ? new Conv1d(i, o, 1, { key: foldIn(key, k) }) : new Conv2d(i, o, 1, { key: foldIn(key, k) })
    const parts: Record<string, Module<never, unknown>> = { lift: pointwise(inChannels, width, 0) }
    for (let l = 0; l < layers; l++)
      parts[`fourier${l}`] = new FourierLayer(width, modes, {
        activation: l < layers - 1,
        key: foldIn(key, 10 + l),
      }) as never
    parts['project1'] = pointwise(width, hidden, 1)
    parts['project2'] = pointwise(hidden, outChannels, 2)
    super(parts as never)
    this.dims = dims
    this.layerCount = layers
    this.outChannels = outChannels
  }

  apply(params: Params, x: Value, ctx: ApplyContext = {}): Value {
    const p = this.parts(params, ctx)
    let h = p.run<Value, Value>('lift', x)
    for (let l = 0; l < this.layerCount; l++) h = p.run<Value, Value>(`fourier${l}`, h)
    const y = p.run<Value, Value>('project2', gelu(p.run<Value, Value>('project1', h)))
    p.done()
    return y
  }

  outputShape(s: readonly number[]): number[] {
    return [...s.slice(0, -this.dims - 1), this.outChannels, ...s.slice(-this.dims)]
  }
}
