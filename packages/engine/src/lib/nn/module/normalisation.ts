import { WhiteboxError } from '../../base/index.ts'
import { full, item, tensor, type Value, zeros } from '../../tensor/index.ts'
import {
  batchNorm,
  groupNorm,
  instanceNorm,
  layerNorm,
  localResponseNorm,
  rmsNorm,
} from '../functional/normalisation.ts'
import { type ApplyContext, isTraining, Module, type Params } from './module.ts'

/** Options of the batch and instance normalisation modules. */
export interface BatchNormModuleOptions {
  /** Added to the variance (default 1e-5). */
  readonly eps?: number
  /** Weight of each new batch in the running statistics (default 0.1; `null` for a cumulative average). */
  readonly momentum?: number | null
  /** Learn a per-channel scale and shift. */
  readonly affine?: boolean
  /** Keep running statistics for evaluation. */
  readonly trackRunningStats?: boolean
}

const affineParams = (n: number): Params => ({ weight: full([n], 1), bias: zeros([n]) })

/** Batch or instance normalisation with optional running statistics (shared implementation). */
abstract class StatNorm extends Module {
  readonly kind: string
  readonly numFeatures: number
  readonly eps: number
  readonly momentum: number | null
  readonly affine: boolean
  readonly trackRunningStats: boolean
  protected readonly ranks: readonly number[]

  protected constructor(
    kind: string,
    numFeatures: number,
    ranks: readonly number[],
    options: BatchNormModuleOptions,
    defaults: { affine: boolean; track: boolean },
  ) {
    super()
    this.kind = kind
    this.numFeatures = numFeatures
    this.ranks = ranks
    this.eps = options.eps ?? 1e-5
    this.momentum = options.momentum === undefined ? 0.1 : options.momentum
    this.affine = options.affine ?? defaults.affine
    this.trackRunningStats = options.trackRunningStats ?? defaults.track
    this.params = this.initParams()
    this.buffers = this.initBuffers()
  }

  override initParams(): Params {
    return this.affine ? affineParams(this.numFeatures) : {}
  }

  override initBuffers(): Params {
    if (!this.trackRunningStats) return {}
    return {
      running_mean: zeros([this.numFeatures]),
      running_var: full([this.numFeatures], 1),
      num_batches_tracked: tensor(0),
    }
  }

  protected check(x: Value): void {
    if (!this.ranks.includes(x.shape.length) || x.shape[1] !== this.numFeatures)
      throw new WhiteboxError(
        'shape',
        `${this.kind}: expected ${this.numFeatures} channels on axis 1 of a ${this.ranks.join(' or ')}-D input, got [${x.shape.join(', ')}]`,
      )
  }

  outputShape(inputShape: readonly number[]): number[] {
    return [...inputShape]
  }

  override extraRepr(): string {
    return `${this.numFeatures}, eps=${this.eps}, momentum=${this.momentum}, affine=${this.affine}, track_running_stats=${this.trackRunningStats}`
  }
}

/** Batch normalisation over the given input ranks (shared by BatchNorm1d, 2d and 3d). */
export class BatchNormNd extends StatNorm {
  constructor(kind: string, numFeatures: number, ranks: readonly number[], options: BatchNormModuleOptions = {}) {
    super(kind, numFeatures, ranks, options, { affine: true, track: true })
  }

  apply(params: Params, x: Value, ctx: ApplyContext = {}): Value {
    this.check(x)
    const b = ctx.buffers ?? this.buffers
    const training = isTraining(ctx)
    const track = this.trackRunningStats && b['running_mean'] !== undefined
    const tracked = track ? item(b['num_batches_tracked'] as Value) : 0
    const r = batchNorm(x, {
      training: training || !track,
      eps: this.eps,
      momentum: this.momentum,
      numBatchesTracked: tracked,
      ...(track ? { runningMean: b['running_mean'] as Value, runningVar: b['running_var'] as Value } : {}),
      ...(params['weight'] === undefined ? {} : { weight: params['weight'] as Value }),
      ...(params['bias'] === undefined ? {} : { bias: params['bias'] as Value }),
    })
    if (training && track && ctx.onBuffers !== undefined && r.runningMean !== undefined && r.runningVar !== undefined) {
      ctx.onBuffers({
        running_mean: r.runningMean,
        running_var: r.runningVar,
        num_batches_tracked: tensor(tracked + 1),
      })
    }
    return r.output
  }
}

/**
 * Batch normalisation of `[B, C]` or `[B, C, L]` inputs (torch.nn.BatchNorm1d), with running statistics that
 * `forward` updates in training mode and uses in evaluation mode.
 *
 * @concept batch-normalisation
 * @example
 * ```ts
 * import { BatchNorm1d } from 'whitebox-engine/nn'
 * Object.keys(new BatchNorm1d(4).stateDict()) // => ['bias', 'weight', 'num_batches_tracked', 'running_mean', 'running_var']
 * ```
 */
export class BatchNorm1d extends BatchNormNd {
  constructor(numFeatures: number, options: BatchNormModuleOptions = {}) {
    super('BatchNorm1d', numFeatures, [2, 3], options)
  }
}

/**
 * Batch normalisation of `[B, C, H, W]` inputs (torch.nn.BatchNorm2d).
 *
 * @concept batch-normalisation
 * @example
 * ```ts
 * import { BatchNorm2d } from 'whitebox-engine/nn'
 * new BatchNorm2d(8).parameterCount() // => 16
 * ```
 */
export class BatchNorm2d extends BatchNormNd {
  constructor(numFeatures: number, options: BatchNormModuleOptions = {}) {
    super('BatchNorm2d', numFeatures, [4], options)
  }
}

/** Batch normalisation of `[B, C, D, H, W]` inputs (torch.nn.BatchNorm3d). @example new BatchNorm3d(2).kind // => 'BatchNorm3d' */
export class BatchNorm3d extends BatchNormNd {
  constructor(numFeatures: number, options: BatchNormModuleOptions = {}) {
    super('BatchNorm3d', numFeatures, [5], options)
  }
}

/** Instance normalisation over the given input ranks (shared by InstanceNorm1d, 2d and 3d). */
export class InstanceNormNd extends StatNorm {
  constructor(kind: string, numFeatures: number, ranks: readonly number[], options: BatchNormModuleOptions = {}) {
    super(kind, numFeatures, ranks, options, { affine: false, track: false })
  }

  apply(params: Params, x: Value, ctx: ApplyContext = {}): Value {
    this.check(x)
    const b = ctx.buffers ?? this.buffers
    const training = isTraining(ctx)
    const track = this.trackRunningStats && b['running_mean'] !== undefined
    const r = instanceNorm(x, {
      useInputStats: training || !track,
      eps: this.eps,
      momentum: this.momentum ?? 0.1,
      ...(track ? { runningMean: b['running_mean'] as Value, runningVar: b['running_var'] as Value } : {}),
      ...(params['weight'] === undefined ? {} : { weight: params['weight'] as Value }),
      ...(params['bias'] === undefined ? {} : { bias: params['bias'] as Value }),
    })
    if (training && track && ctx.onBuffers !== undefined && r.runningMean !== undefined && r.runningVar !== undefined) {
      // PyTorch's instance normalisation keeps num_batches_tracked at zero.
      ctx.onBuffers({ ...b, running_mean: r.runningMean, running_var: r.runningVar })
    }
    return r.output
  }
}

/** Instance normalisation of `[B, C, L]` inputs (torch.nn.InstanceNorm1d). @example new InstanceNorm1d(3).parameterCount() // => 0 */
export class InstanceNorm1d extends InstanceNormNd {
  constructor(numFeatures: number, options: BatchNormModuleOptions = {}) {
    super('InstanceNorm1d', numFeatures, [3], options)
  }
}

/**
 * Instance normalisation of `[B, C, H, W]` inputs (torch.nn.InstanceNorm2d; no affine parameters or running
 * statistics unless asked for).
 *
 * @example
 * ```ts
 * import { InstanceNorm2d } from 'whitebox-engine/nn'
 * new InstanceNorm2d(3, { affine: true }).parameterCount() // => 6
 * ```
 */
export class InstanceNorm2d extends InstanceNormNd {
  constructor(numFeatures: number, options: BatchNormModuleOptions = {}) {
    super('InstanceNorm2d', numFeatures, [4], options)
  }
}

/** Instance normalisation of `[B, C, D, H, W]` inputs (torch.nn.InstanceNorm3d). @example new InstanceNorm3d(2).kind // => 'InstanceNorm3d' */
export class InstanceNorm3d extends InstanceNormNd {
  constructor(numFeatures: number, options: BatchNormModuleOptions = {}) {
    super('InstanceNorm3d', numFeatures, [5], options)
  }
}

/** Options of {@link LayerNorm} and {@link RMSNorm}. */
export interface LayerNormOptions {
  readonly eps?: number
  /** Learn an elementwise scale (and shift, for layer normalisation); default true. */
  readonly elementwiseAffine?: boolean
  /** Learn the shift (layer normalisation only; default true). */
  readonly bias?: boolean
}

const shapeOf = (s: number | readonly number[]): number[] => (typeof s === 'number' ? [s] : [...s])

/**
 * Layer normalisation over the trailing axes (torch.nn.LayerNorm): elementwise weight (ones) and bias (zeros).
 *
 * @concept layer-normalisation
 * @example
 * ```ts
 * import { LayerNorm } from 'whitebox-engine/nn'
 * new LayerNorm(16).parameterCount() // => 32
 * ```
 */
export class LayerNorm extends Module {
  readonly kind = 'LayerNorm'
  readonly normalizedShape: readonly number[]
  readonly eps: number
  readonly elementwiseAffine: boolean
  readonly bias: boolean

  constructor(normalizedShape: number | readonly number[], options: LayerNormOptions = {}) {
    super()
    this.normalizedShape = shapeOf(normalizedShape)
    this.eps = options.eps ?? 1e-5
    this.elementwiseAffine = options.elementwiseAffine ?? true
    this.bias = this.elementwiseAffine && (options.bias ?? true)
    this.params = this.initParams()
  }

  override initParams(): Params {
    if (!this.elementwiseAffine) return {}
    return { weight: full(this.normalizedShape, 1), ...(this.bias ? { bias: zeros(this.normalizedShape) } : {}) }
  }

  apply(params: Params, x: Value): Value {
    return layerNorm(x, this.normalizedShape, {
      eps: this.eps,
      ...(params['weight'] === undefined ? {} : { weight: params['weight'] as Value }),
      ...(params['bias'] === undefined ? {} : { bias: params['bias'] as Value }),
    })
  }

  outputShape(inputShape: readonly number[]): number[] {
    return [...inputShape]
  }

  override extraRepr(): string {
    return `(${this.normalizedShape.join(', ')},), eps=${this.eps}, elementwise_affine=${this.elementwiseAffine}`
  }
}

/**
 * Root-mean-square normalisation over the trailing axes (torch.nn.RMSNorm): an elementwise weight (ones) and no
 * shift; eps defaults to the machine epsilon, as PyTorch.
 *
 * @example
 * ```ts
 * import { RMSNorm } from 'whitebox-engine/nn'
 * new RMSNorm(16).parameterCount() // => 16
 * ```
 */
export class RMSNorm extends Module {
  readonly kind = 'RMSNorm'
  readonly normalizedShape: readonly number[]
  readonly eps: number | undefined
  readonly elementwiseAffine: boolean

  constructor(normalizedShape: number | readonly number[], options: Omit<LayerNormOptions, 'bias'> = {}) {
    super()
    this.normalizedShape = shapeOf(normalizedShape)
    this.eps = options.eps
    this.elementwiseAffine = options.elementwiseAffine ?? true
    this.params = this.initParams()
  }

  override initParams(): Params {
    return this.elementwiseAffine ? { weight: full(this.normalizedShape, 1) } : {}
  }

  apply(params: Params, x: Value): Value {
    return rmsNorm(x, this.normalizedShape, {
      ...(this.eps === undefined ? {} : { eps: this.eps }),
      ...(params['weight'] === undefined ? {} : { weight: params['weight'] as Value }),
    })
  }

  outputShape(inputShape: readonly number[]): number[] {
    return [...inputShape]
  }

  override extraRepr(): string {
    return `(${this.normalizedShape.join(', ')},), eps=${this.eps ?? 'None'}, elementwise_affine=${this.elementwiseAffine}`
  }
}

/**
 * Group normalisation (torch.nn.GroupNorm): per-channel weight (ones) and bias (zeros) when affine.
 *
 * @example
 * ```ts
 * import { GroupNorm } from 'whitebox-engine/nn'
 * new GroupNorm(4, 16).parameterCount() // => 32
 * ```
 */
export class GroupNorm extends Module {
  readonly kind = 'GroupNorm'
  readonly numGroups: number
  readonly numChannels: number
  readonly eps: number
  readonly affine: boolean

  constructor(
    numGroups: number,
    numChannels: number,
    options: { readonly eps?: number; readonly affine?: boolean } = {},
  ) {
    super()
    if (numChannels % numGroups !== 0)
      throw new WhiteboxError('value', `GroupNorm: ${numChannels} channels do not divide into ${numGroups} groups`)
    this.numGroups = numGroups
    this.numChannels = numChannels
    this.eps = options.eps ?? 1e-5
    this.affine = options.affine ?? true
    this.params = this.initParams()
  }

  override initParams(): Params {
    return this.affine ? affineParams(this.numChannels) : {}
  }

  apply(params: Params, x: Value): Value {
    return groupNorm(x, this.numGroups, {
      eps: this.eps,
      ...(params['weight'] === undefined ? {} : { weight: params['weight'] as Value }),
      ...(params['bias'] === undefined ? {} : { bias: params['bias'] as Value }),
    })
  }

  outputShape(inputShape: readonly number[]): number[] {
    return [...inputShape]
  }

  override extraRepr(): string {
    return `${this.numGroups}, ${this.numChannels}, eps=${this.eps}, affine=${this.affine}`
  }
}

/**
 * Local response normalisation across channels (torch.nn.LocalResponseNorm).
 *
 * @example
 * ```ts
 * import { LocalResponseNorm } from 'whitebox-engine/nn'
 * new LocalResponseNorm(5).kind // => 'LocalResponseNorm'
 * ```
 */
export class LocalResponseNorm extends Module {
  readonly kind = 'LocalResponseNorm'
  readonly size: number
  readonly alpha: number
  readonly beta: number
  readonly k: number

  constructor(size: number, alpha = 1e-4, beta = 0.75, k = 1) {
    super()
    this.size = size
    this.alpha = alpha
    this.beta = beta
    this.k = k
  }

  apply(_params: Params, x: Value): Value {
    return localResponseNorm(x, this.size, this.alpha, this.beta, this.k)
  }

  outputShape(inputShape: readonly number[]): number[] {
    return [...inputShape]
  }

  override extraRepr(): string {
    return `${this.size}, alpha=${this.alpha}, beta=${this.beta}, k=${this.k}`
  }
}
