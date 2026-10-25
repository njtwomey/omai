/**
 * Weight reparametrisations that wrap a module, as torch.nn.utils does: `WeightNorm`, which splits a weight into a
 * direction and a magnitude, and `SpectralNorm`, which divides it by its largest singular value (tracked by power
 * iteration in buffers) to bound the layer's Lipschitz constant, as in GAN discriminators.
 */

import { MdmlError } from '../../base/index.ts'
import { foldIn, type Key, normal } from '../../random/index.ts'
import { item, norm, reshape, sizeOf, sqrt, square, sum, type Value } from '../../tensor/index.ts'
import { normalize, spectralNorm, weightNorm } from '../functional/normalisation.ts'
import { type ApplyContext, initKey, Module, type Params } from './module.ts'

const otherAxes = (rank: number, axis: number): number[] =>
  Array.from({ length: rank }, (_, i) => i).filter((i) => i !== axis)

/**
 * Weight normalisation of a module's weight (torch.nn.utils.weight_norm, Salimans and Kingma 2016): the weight is
 * reparametrised as `weight_g * weight_v / ||weight_v||` (norms over every axis but `dim`), and the wrapped module is
 * applied with the recomposed weight. Initialised so that the weight is unchanged.
 *
 * @concept weight-normalisation
 * @example
 * ```ts
 * import { WeightNorm, Linear } from 'mdml-engine/nn'
 * import { key } from 'mdml-engine/random'
 * Object.keys(new WeightNorm(new Linear(3, 2, { key: key(0) })).stateDict()) // => ['bias', 'weight_g', 'weight_v']
 * ```
 */
export class WeightNorm<I, O> extends Module<I, O> {
  readonly kind = 'WeightNorm'
  readonly module: Module<I, O>
  readonly name: string
  readonly dim: number | null

  constructor(module: Module<I, O>, options: { readonly name?: string; readonly dim?: number | null } = {}) {
    super()
    this.module = module
    this.name = options.name ?? 'weight'
    this.dim = options.dim === undefined ? 0 : options.dim
    this.params = this.decompose(module.params)
    this.buffers = module.buffers
  }

  private decompose(params: Params): Params {
    const w = params[this.name] as Value | undefined
    if (w === undefined) throw new MdmlError('value', `WeightNorm: the module has no parameter ${this.name}`)
    const { [this.name]: _, ...rest } = params
    const g =
      this.dim === null
        ? reshape(norm(reshape(w, [-1]), 2), [])
        : sqrt(sum(square(w), otherAxes(w.shape.length, this.dim < 0 ? w.shape.length + this.dim : this.dim), true))
    return { ...rest, [`${this.name}_g`]: g, [`${this.name}_v`]: w }
  }

  override initParams(key: Key): Params {
    return this.decompose(this.module.initParams(key))
  }

  override initBuffers(): Params {
    return this.module.initBuffers()
  }

  /** The recomposed weight for given parameters. */
  weight(params: Params = this.params): Value {
    return weightNorm(params[`${this.name}_v`] as Value, params[`${this.name}_g`] as Value, this.dim)
  }

  protected override prepare(x: unknown): unknown {
    return (this.module as unknown as { prepare(x: unknown): unknown }).prepare(x)
  }

  apply(params: Params, x: I, ctx: ApplyContext = {}): O {
    const { [`${this.name}_g`]: _g, [`${this.name}_v`]: _v, ...rest } = params
    return this.module.apply({ ...rest, [this.name]: this.weight(params) }, x, ctx)
  }

  outputShape(inputShape: readonly number[]): number[] {
    return this.module.outputShape(inputShape)
  }

  override setBuffers(buffers: Params): this {
    this.buffers = buffers
    this.module.setBuffers(buffers)
    return this
  }

  override train(mode = true): this {
    this.training = mode
    this.module.train(mode)
    return this
  }

  override extraRepr(): string {
    return `${this.module.kind}(${this.module.extraRepr()}), name=${this.name}, dim=${this.dim}`
  }
}

/** Options of {@link SpectralNorm}. */
export interface SpectralNormModuleOptions {
  readonly name?: string
  /** Power iterations per training forward pass (default 1). */
  readonly nPowerIterations?: number
  readonly eps?: number
  /** The output axis of the weight (default 0; 1 for transposed convolutions, as PyTorch chooses). */
  readonly dim?: number
  readonly key?: Key
}

/**
 * Spectral normalisation of a module's weight (torch.nn.utils.spectral_norm, Miyato et al. 2018): the parameter
 * becomes `weight_orig`, the power-iteration vectors `weight_u` and `weight_v` are buffers updated in training, and the
 * wrapped module is applied with `weight_orig / sigma`. Used to bound the Lipschitz constant of GAN discriminators.
 *
 * @concept spectral-normalisation
 * @example
 * ```ts
 * import { SpectralNorm, Linear } from 'mdml-engine/nn'
 * import { key } from 'mdml-engine/random'
 * Object.keys(new SpectralNorm(new Linear(3, 2, { key: key(0) }), { key: key(1) }).stateDict()) // => ['bias', 'weight_orig', 'weight_u', 'weight_v']
 * ```
 */
export class SpectralNorm<I, O> extends Module<I, O> {
  readonly kind = 'SpectralNorm'
  readonly module: Module<I, O>
  readonly name: string
  readonly nPowerIterations: number
  readonly eps: number
  readonly dim: number
  private readonly vectorKey: Key

  constructor(module: Module<I, O>, options: SpectralNormModuleOptions = {}) {
    super()
    this.module = module
    this.name = options.name ?? 'weight'
    this.nPowerIterations = options.nPowerIterations ?? 1
    this.eps = options.eps ?? 1e-12
    this.dim = options.dim ?? (module.kind.startsWith('ConvTranspose') ? 1 : 0)
    this.vectorKey = initKey('SpectralNorm', options.key)
    const w = module.params[this.name] as Value | undefined
    if (w === undefined) throw new MdmlError('value', `SpectralNorm: the module has no parameter ${this.name}`)
    const { [this.name]: _, ...rest } = module.params
    this.params = { ...rest, [`${this.name}_orig`]: w }
    this.buffers = this.initBuffers()
  }

  override initParams(key: Key): Params {
    const p = this.module.initParams(key)
    const { [this.name]: w, ...rest } = p
    return { ...rest, [`${this.name}_orig`]: w }
  }

  override initBuffers(): Params {
    const w = (this.params[`${this.name}_orig`] ?? this.module.params[this.name]) as Value
    const h = w.shape[this.dim] ?? 1
    const rest = sizeOf(w.shape) / h
    return {
      ...this.module.initBuffers(),
      [`${this.name}_u`]: normalize(normal(foldIn(this.vectorKey, 0), [h]), 2, 0, this.eps),
      [`${this.name}_v`]: normalize(normal(foldIn(this.vectorKey, 1), [rest]), 2, 0, this.eps),
    }
  }

  protected override prepare(x: unknown): unknown {
    return (this.module as unknown as { prepare(x: unknown): unknown }).prepare(x)
  }

  /** The normalised weight and the updated vectors for given parameters and buffers. */
  normalised(params: Params, buffers: Params, training: boolean): { weight: Value; u: Value; v: Value } {
    return spectralNorm(
      params[`${this.name}_orig`] as Value,
      buffers[`${this.name}_u`] as Value,
      buffers[`${this.name}_v`] as Value,
      { nPowerIterations: this.nPowerIterations, eps: this.eps, axis: this.dim, training },
    )
  }

  apply(params: Params, x: I, ctx: ApplyContext = {}): O {
    const buffers = ctx.buffers ?? this.buffers
    const training = ctx.train ?? true
    const r = this.normalised(params, buffers, training)
    const { [`${this.name}_orig`]: _, ...rest } = params
    const { [`${this.name}_u`]: _u, [`${this.name}_v`]: _v, ...inner } = buffers
    let innerUpdate: Params | undefined
    const y = this.module.apply({ ...rest, [this.name]: r.weight }, x, {
      ...ctx,
      buffers: inner,
      onBuffers: (b) => {
        innerUpdate = b
      },
    })
    if (ctx.onBuffers !== undefined && (training || innerUpdate !== undefined)) {
      ctx.onBuffers({ ...(innerUpdate ?? inner), [`${this.name}_u`]: r.u, [`${this.name}_v`]: r.v })
    }
    return y
  }

  outputShape(inputShape: readonly number[]): number[] {
    return this.module.outputShape(inputShape)
  }

  override train(mode = true): this {
    this.training = mode
    this.module.train(mode)
    return this
  }

  override extraRepr(): string {
    return `${this.module.kind}(${this.module.extraRepr()}), name=${this.name}, n_power_iterations=${this.nPowerIterations}`
  }
}

/**
 * The current spectral norm estimate of a spectrally normalised module (sigma = u^T W v from its buffers).
 *
 * @param module - The wrapped module.
 * @returns sigma.
 * @example
 * ```ts
 * import { SpectralNorm, Linear, spectralNormEstimate } from 'mdml-engine/nn'
 * import { key } from 'mdml-engine/random'
 * const sn = new SpectralNorm(new Linear(3, 2, { key: key(0) }), { key: key(1) })
 * sn.forward([[1, 2, 3]])
 * spectralNormEstimate(sn) > 0 // => true
 * ```
 */
export function spectralNormEstimate(module: SpectralNorm<never, unknown>): number {
  const r = spectralNorm(
    module.params[`${module.name}_orig`] as Value,
    module.buffers[`${module.name}_u`] as Value,
    module.buffers[`${module.name}_v`] as Value,
    { training: false, axis: module.dim },
  )
  return item(r.sigma)
}
