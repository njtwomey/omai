/**
 * Convolutional blocks for vision models: ResNet's basic and bottleneck blocks, squeeze-and-excitation channel gating,
 * the ConvNeXt block, MobileNet and EfficientNet's inverted bottleneck (`MBConv`), the encoder and decoder stages of a
 * U-Net and a feature pyramid network, plus `toTokens` for turning feature maps into token sequences.
 */

import { OmaiError } from '../../base/index.ts'
import { foldIn, type Key } from '../../random/index.ts'
import {
  add,
  asTensor,
  concat,
  full,
  moveaxis,
  mul,
  reshape,
  sigmoid,
  type TensorLike,
  type Value,
} from '../../tensor/index.ts'
import { gelu, relu, silu } from '../functional/activations.ts'
import { globalAvgPool } from '../functional/pooling.ts'
import { interpolate } from '../functional/vision.ts'
import { Composite, ModuleList, Sequential } from '../module/containers.ts'
import { Conv2d, ConvTranspose2d } from '../module/conv.ts'
import { Linear } from '../module/layers.ts'
import { type ApplyContext, initKey, type Module, type Params } from '../module/module.ts'
import { BatchNorm2d, LayerNorm } from '../module/normalisation.ts'
import { MaxPool2d } from '../module/pooling.ts'
import { DropPath } from '../module/regularisation.ts'

const keyAt =
  (k: Key) =>
  (i: number): Key =>
    foldIn(k, i)

/** Options shared by the convolutional blocks. */
export interface ConvBlockOptions {
  readonly stride?: number
  readonly key?: Key
}

/** The shortcut of a residual block whose shape changes: a strided 1x1 convolution and batch normalisation. */
const projection = (cin: number, cout: number, stride: number, k: Key): Sequential =>
  new Sequential(new Conv2d(cin, cout, 1, { stride, bias: false, key: k }), new BatchNorm2d(cout))

/**
 * ResNet's basic block (He et al. 2016; torchvision's BasicBlock): two 3x3 convolutions with batch normalisation and
 * ReLU, plus the input (projected by `downsample` when the stride or width changes).
 *
 * @concept residual-connections
 * @reference He, Zhang, Ren and Sun (2016), Deep residual learning for image recognition.
 * @example
 * ```ts
 * import { blocks } from 'omai-engine/nn'
 * new blocks.BasicBlock(4, 8, { stride: 2 }).outputShape([1, 4, 8, 8]) // => [1, 8, 4, 4]
 * ```
 */
export class BasicBlock extends Composite {
  readonly kind = 'BasicBlock'
  readonly outChannels: number
  readonly stride: number

  constructor(inChannels: number, outChannels: number, options: ConvBlockOptions = {}) {
    const k = keyAt(initKey('BasicBlock', options.key))
    const stride = options.stride ?? 1
    super({
      conv1: new Conv2d(inChannels, outChannels, 3, { stride, padding: 1, bias: false, key: k(0) }),
      bn1: new BatchNorm2d(outChannels),
      conv2: new Conv2d(outChannels, outChannels, 3, { padding: 1, bias: false, key: k(2) }),
      bn2: new BatchNorm2d(outChannels),
      ...(stride !== 1 || inChannels !== outChannels
        ? { downsample: projection(inChannels, outChannels, stride, k(4)) }
        : {}),
    })
    this.outChannels = outChannels
    this.stride = stride
  }

  apply(params: Params, x: Value, ctx: ApplyContext = {}): Value {
    const p = this.parts(params, ctx)
    const r = (name: string, v: Value): Value => p.run<Value, Value>(name, v)
    const h = r('bn2', r('conv2', relu(r('bn1', r('conv1', x)))))
    const y = relu(add(h, this.has('downsample') ? r('downsample', x) : x))
    p.done()
    return y
  }

  outputShape(s: readonly number[]): number[] {
    return [s[0] ?? 1, this.outChannels, Math.ceil((s[2] ?? 0) / this.stride), Math.ceil((s[3] ?? 0) / this.stride)]
  }
}

/**
 * ResNet's bottleneck block (torchvision's Bottleneck, stride on the 3x3 convolution as in ResNet v1.5): 1x1 reduce,
 * 3x3, 1x1 expand by `expansion` (4), with batch normalisation, ReLU and a projected shortcut when needed.
 *
 * @concept residual-connections
 * @example
 * ```ts
 * import { blocks } from 'omai-engine/nn'
 * new blocks.Bottleneck(16, 4).outputShape([1, 16, 8, 8]) // => [1, 16, 8, 8]
 * ```
 */
export class Bottleneck extends Composite {
  readonly kind = 'Bottleneck'
  readonly outChannels: number
  readonly stride: number

  constructor(
    inChannels: number,
    width: number,
    options: ConvBlockOptions & { readonly expansion?: number; readonly groups?: number } = {},
  ) {
    const k = keyAt(initKey('Bottleneck', options.key))
    const stride = options.stride ?? 1
    const out = width * (options.expansion ?? 4)
    super({
      conv1: new Conv2d(inChannels, width, 1, { bias: false, key: k(0) }),
      bn1: new BatchNorm2d(width),
      conv2: new Conv2d(width, width, 3, {
        stride,
        padding: 1,
        groups: options.groups ?? 1,
        bias: false,
        key: k(2),
      }),
      bn2: new BatchNorm2d(width),
      conv3: new Conv2d(width, out, 1, { bias: false, key: k(4) }),
      bn3: new BatchNorm2d(out),
      ...(stride !== 1 || inChannels !== out ? { downsample: projection(inChannels, out, stride, k(6)) } : {}),
    })
    this.outChannels = out
    this.stride = stride
  }

  apply(params: Params, x: Value, ctx: ApplyContext = {}): Value {
    const p = this.parts(params, ctx)
    const r = (name: string, v: Value): Value => p.run<Value, Value>(name, v)
    const h = r('bn3', r('conv3', relu(r('bn2', r('conv2', relu(r('bn1', r('conv1', x))))))))
    const y = relu(add(h, this.has('downsample') ? r('downsample', x) : x))
    p.done()
    return y
  }

  outputShape(s: readonly number[]): number[] {
    return [s[0] ?? 1, this.outChannels, Math.ceil((s[2] ?? 0) / this.stride), Math.ceil((s[3] ?? 0) / this.stride)]
  }
}

/**
 * Squeeze-and-excitation (Hu et al. 2018; torchvision's SqueezeExcitation): global average pooling, a 1x1 bottleneck
 * `fc1` with ReLU (or SiLU), `fc2` and a sigmoid give per-channel weights that rescale the input.
 *
 * @reference Hu, Shen and Sun (2018), Squeeze-and-excitation networks.
 * @example
 * ```ts
 * import { blocks } from 'omai-engine/nn'
 * import { zeros } from 'omai-engine/tensor'
 * new blocks.SqueezeExcitation(8, 2).forward(zeros([1, 8, 3, 3])).shape // => [1, 8, 3, 3]
 * ```
 */
export class SqueezeExcitation extends Composite {
  readonly kind = 'SqueezeExcitation'
  private readonly act: (x: Value) => Value

  constructor(
    channels: number,
    squeeze: number,
    options: { readonly activation?: 'relu' | 'silu'; readonly key?: Key } = {},
  ) {
    const k = keyAt(initKey('SqueezeExcitation', options.key))
    super({
      fc1: new Conv2d(channels, squeeze, 1, { key: k(0) }),
      fc2: new Conv2d(squeeze, channels, 1, { key: k(1) }),
    })
    this.act = options.activation === 'silu' ? (v) => silu(v) : (v) => relu(v)
  }

  apply(params: Params, x: Value, ctx: ApplyContext = {}): Value {
    const p = this.parts(params, ctx)
    const s = globalAvgPool(x, true)
    const w = sigmoid(p.run<Value, Value>('fc2', this.act(p.run<Value, Value>('fc1', s))))
    p.done()
    return mul(x, w)
  }

  outputShape(s: readonly number[]): number[] {
    return [...s]
  }
}

/**
 * The ConvNeXt block (Liu et al. 2022): a 7x7 depthwise convolution, layer normalisation over channels, an inverted
 * bottleneck of linear layers with GELU, a learnt per-channel layer scale and drop path, plus the input.
 *
 * @reference Liu, Mao, Wu, Feichtenhofer, Darrell and Xie (2022), A ConvNet for the 2020s.
 * @example
 * ```ts
 * import { blocks } from 'omai-engine/nn'
 * import { zeros } from 'omai-engine/tensor'
 * new blocks.ConvNeXtBlock(8).eval().forward(zeros([1, 8, 7, 7])).shape // => [1, 8, 7, 7]
 * ```
 */
export class ConvNeXtBlock extends Composite {
  readonly kind = 'ConvNeXtBlock'
  readonly dim: number

  constructor(
    dim: number,
    options: { readonly layerScale?: number; readonly dropPath?: number; readonly key?: Key } = {},
  ) {
    const k = keyAt(initKey('ConvNeXtBlock', options.key))
    super(
      {
        dwconv: new Conv2d(dim, dim, 7, { padding: 3, groups: dim, key: k(0) }),
        norm: new LayerNorm(dim, { eps: 1e-6 }),
        pwconv1: new Linear(dim, 4 * dim, { key: k(2) }),
        pwconv2: new Linear(4 * dim, dim, { key: k(3) }),
        drop_path: new DropPath(options.dropPath ?? 0),
      },
      { gamma: full([dim], options.layerScale ?? 1e-6) },
    )
    this.dim = dim
    this.layerScale = options.layerScale ?? 1e-6
  }

  readonly layerScale: number

  override initParams(key: Key): Params {
    return { ...super.initParams(key), gamma: full([this.dim], this.layerScale) }
  }

  apply(params: Params, x: Value, ctx: ApplyContext = {}): Value {
    const p = this.parts(params, ctx)
    // Channels last for the normalisation and the linear layers.
    let h = moveaxis(p.run<Value, Value>('dwconv', x), 1, -1)
    h = p.run<Value, Value>('pwconv2', gelu(p.run<Value, Value>('pwconv1', p.run<Value, Value>('norm', h))))
    h = mul(h, params['gamma'] as Value)
    const y = add(x, p.run<Value, Value>('drop_path', moveaxis(h, -1, 1)))
    p.done()
    return y
  }

  outputShape(s: readonly number[]): number[] {
    return [...s]
  }
}

/**
 * The mobile inverted bottleneck (MBConv; MobileNetV2, EfficientNet): a 1x1 expansion, a depthwise kxk convolution,
 * squeeze-excitation and a 1x1 projection, with batch normalisation and SiLU, and a residual connection with drop
 * path when the stride is 1 and the width is unchanged.
 *
 * @reference Tan and Le (2019), EfficientNet: rethinking model scaling for convolutional neural networks.
 * @example
 * ```ts
 * import { blocks } from 'omai-engine/nn'
 * new blocks.MBConv(8, 16, { stride: 2 }).outputShape([1, 8, 8, 8]) // => [1, 16, 4, 4]
 * ```
 */
export class MBConv extends Composite {
  readonly kind = 'MBConv'
  readonly outChannels: number
  readonly stride: number
  readonly residual: boolean

  constructor(
    inChannels: number,
    outChannels: number,
    options: ConvBlockOptions & {
      readonly expandRatio?: number
      readonly kernelSize?: number
      readonly seRatio?: number
      readonly dropPath?: number
    } = {},
  ) {
    const k = keyAt(initKey('MBConv', options.key))
    const stride = options.stride ?? 1
    const ratio = options.expandRatio ?? 4
    const hidden = inChannels * ratio
    const ks = options.kernelSize ?? 3
    const se = Math.max(1, Math.floor(inChannels * (options.seRatio ?? 0.25)))
    super({
      ...(ratio === 1
        ? {}
        : { expand: new Conv2d(inChannels, hidden, 1, { bias: false, key: k(0) }), bn0: new BatchNorm2d(hidden) }),
      depthwise: new Conv2d(hidden, hidden, ks, {
        stride,
        padding: Math.floor(ks / 2),
        groups: hidden,
        bias: false,
        key: k(2),
      }),
      bn1: new BatchNorm2d(hidden),
      se: new SqueezeExcitation(hidden, se, { activation: 'silu', key: k(4) }),
      project: new Conv2d(hidden, outChannels, 1, { bias: false, key: k(5) }),
      bn2: new BatchNorm2d(outChannels),
      drop_path: new DropPath(options.dropPath ?? 0),
    })
    this.outChannels = outChannels
    this.stride = stride
    this.residual = stride === 1 && inChannels === outChannels
  }

  apply(params: Params, x: Value, ctx: ApplyContext = {}): Value {
    const p = this.parts(params, ctx)
    const r = (name: string, v: Value): Value => p.run<Value, Value>(name, v)
    let h = this.has('expand') ? silu(r('bn0', r('expand', x))) : x
    h = silu(r('bn1', r('depthwise', h)))
    h = r('bn2', r('project', r('se', h)))
    const y = this.residual ? add(x, r('drop_path', h)) : h
    p.done()
    return y
  }

  outputShape(s: readonly number[]): number[] {
    return [s[0] ?? 1, this.outChannels, Math.ceil((s[2] ?? 0) / this.stride), Math.ceil((s[3] ?? 0) / this.stride)]
  }
}

/**
 * Two 3x3 convolutions with batch normalisation and ReLU (the unit of every U-Net stage).
 *
 * @example
 * ```ts
 * import { blocks } from 'omai-engine/nn'
 * new blocks.DoubleConv(3, 8).outputShape([1, 3, 6, 6]) // => [1, 8, 6, 6]
 * ```
 */
export class DoubleConv extends Composite {
  readonly kind = 'DoubleConv'
  readonly outChannels: number

  constructor(inChannels: number, outChannels: number, options: { readonly mid?: number; readonly key?: Key } = {}) {
    const k = keyAt(initKey('DoubleConv', options.key))
    const mid = options.mid ?? outChannels
    super({
      conv1: new Conv2d(inChannels, mid, 3, { padding: 1, bias: false, key: k(0) }),
      bn1: new BatchNorm2d(mid),
      conv2: new Conv2d(mid, outChannels, 3, { padding: 1, bias: false, key: k(2) }),
      bn2: new BatchNorm2d(outChannels),
    })
    this.outChannels = outChannels
  }

  apply(params: Params, x: Value, ctx: ApplyContext = {}): Value {
    const p = this.parts(params, ctx)
    const r = (name: string, v: Value): Value => p.run<Value, Value>(name, v)
    const y = relu(r('bn2', r('conv2', relu(r('bn1', r('conv1', x))))))
    p.done()
    return y
  }

  outputShape(s: readonly number[]): number[] {
    return [s[0] ?? 1, this.outChannels, s[2] ?? 0, s[3] ?? 0]
  }
}

/**
 * A U-Net encoder stage (Ronneberger et al. 2015): 2x2 max pooling, then a double convolution.
 *
 * @reference Ronneberger, Fischer and Brox (2015), U-Net: convolutional networks for biomedical image segmentation.
 * @example
 * ```ts
 * import { blocks } from 'omai-engine/nn'
 * new blocks.UNetDown(8, 16).outputShape([1, 8, 8, 8]) // => [1, 16, 4, 4]
 * ```
 */
export class UNetDown extends Composite {
  readonly kind = 'UNetDown'
  readonly outChannels: number

  constructor(inChannels: number, outChannels: number, options: { readonly key?: Key } = {}) {
    super({ pool: new MaxPool2d(2), conv: new DoubleConv(inChannels, outChannels, options) })
    this.outChannels = outChannels
  }

  apply(params: Params, x: Value, ctx: ApplyContext = {}): Value {
    const p = this.parts(params, ctx)
    const y = p.run<Value, Value>('conv', p.run<Value, Value>('pool', x))
    p.done()
    return y
  }

  outputShape(s: readonly number[]): number[] {
    return [s[0] ?? 1, this.outChannels, Math.floor((s[2] ?? 0) / 2), Math.floor((s[3] ?? 0) / 2)]
  }
}

/**
 * A U-Net decoder stage: upsample (a 2x2 transposed convolution halving the channels, or bilinear interpolation),
 * concatenate the encoder's skip connection, then a double convolution. Its input is `[x, skip]`.
 *
 * @example
 * ```ts
 * import { blocks } from 'omai-engine/nn'
 * import { zeros } from 'omai-engine/tensor'
 * new blocks.UNetUp(16, 8).forward([zeros([1, 16, 4, 4]), zeros([1, 8, 8, 8])]).shape // => [1, 8, 8, 8]
 * ```
 */
export class UNetUp extends Composite<readonly [Value, Value]> {
  readonly kind = 'UNetUp'
  readonly outChannels: number
  readonly bilinear: boolean

  constructor(
    inChannels: number,
    outChannels: number,
    options: { readonly bilinear?: boolean; readonly key?: Key } = {},
  ) {
    const k = keyAt(initKey('UNetUp', options.key))
    const bilinear = options.bilinear ?? false
    super({
      ...(bilinear ? {} : { up: new ConvTranspose2d(inChannels, inChannels / 2, 2, { stride: 2, key: k(0) }) }),
      conv: bilinear
        ? new DoubleConv(inChannels + outChannels, outChannels, { mid: inChannels / 2, key: k(1) })
        : new DoubleConv(inChannels / 2 + outChannels, outChannels, { key: k(1) }),
    })
    this.outChannels = outChannels
    this.bilinear = bilinear
  }

  protected override prepare(x: unknown): unknown {
    const [a, b] = x as readonly [TensorLike, TensorLike]
    return [asTensor(a), asTensor(b)]
  }

  apply(params: Params, input: readonly [Value, Value], ctx: ApplyContext = {}): Value {
    const [x, skip] = input
    const p = this.parts(params, ctx)
    const up = this.bilinear
      ? interpolate(x, { scaleFactor: 2, mode: 'bilinear', alignCorners: true })
      : p.run<Value, Value>('up', x)
    if (up.shape[2] !== skip.shape[2] || up.shape[3] !== skip.shape[3])
      throw new OmaiError('shape', 'UNetUp: the upsampled input and the skip connection differ in size')
    const y = p.run<Value, Value>('conv', concat([skip, up], 1))
    p.done()
    return y
  }

  outputShape(s: readonly number[]): number[] {
    return [s[0] ?? 1, this.outChannels, (s[2] ?? 0) * 2, (s[3] ?? 0) * 2]
  }
}

/**
 * A feature pyramid network (Lin et al. 2017; torchvision's FeaturePyramidNetwork): 1x1 lateral convolutions bring
 * every level to one width, a top-down pathway adds each coarser level upsampled (nearest) to the next finer one,
 * and 3x3 convolutions smooth each result. Input and output are lists of feature maps, finest first.
 *
 * @reference Lin, Dollar, Girshick, He, Hariharan and Belongie (2017), Feature pyramid networks for object detection.
 * @example
 * ```ts
 * import { blocks } from 'omai-engine/nn'
 * import { zeros } from 'omai-engine/tensor'
 * new blocks.FeaturePyramid([4, 8], 6).forward([zeros([1, 4, 8, 8]), zeros([1, 8, 4, 4])]).map((f) => f.shape) // => [[1, 6, 8, 8], [1, 6, 4, 4]]
 * ```
 */
export class FeaturePyramid extends Composite<readonly Value[], Value[]> {
  readonly kind = 'FeaturePyramid'
  readonly outChannels: number

  constructor(inChannels: readonly number[], outChannels: number, options: { readonly key?: Key } = {}) {
    const k = keyAt(initKey('FeaturePyramid', options.key))
    super({
      inner_blocks: new ModuleList(inChannels.map((c, i) => new Conv2d(c, outChannels, 1, { key: foldIn(k(0), i) }))),
      layer_blocks: new ModuleList(
        inChannels.map((_, i) => new Conv2d(outChannels, outChannels, 3, { padding: 1, key: foldIn(k(1), i) })),
      ),
    })
    this.outChannels = outChannels
  }

  protected override prepare(x: unknown): unknown {
    return (x as readonly TensorLike[]).map((v) => asTensor(v))
  }

  apply(params: Params, features: readonly Value[], _ctx: ApplyContext = {}): Value[] {
    const inner = this.part<ModuleList<Module>>('inner_blocks')
    const layer = this.part<ModuleList<Module>>('layer_blocks')
    const ip = (params['inner_blocks'] ?? {}) as Params
    const lp = (params['layer_blocks'] ?? {}) as Params
    const out: Value[] = new Array<Value>(features.length)
    let top: Value | undefined
    for (let i = features.length - 1; i >= 0; i--) {
      const f = features[i] as Value
      let lateral = inner.at(i).apply((ip[String(i)] ?? {}) as Params, f)
      if (top !== undefined) {
        const size = [lateral.shape[2] ?? 1, lateral.shape[3] ?? 1]
        lateral = add(lateral, interpolate(top, { size, mode: 'nearest' }))
      }
      top = lateral
      out[i] = layer.at(i).apply((lp[String(i)] ?? {}) as Params, lateral)
    }
    return out
  }

  outputShape(s: readonly number[]): number[] {
    return [s[0] ?? 1, this.outChannels, s[2] ?? 0, s[3] ?? 0]
  }
}

/**
 * Feature maps as a token sequence: `[B, C, H, W]` to `[B, H W, C]` (for attention over spatial positions).
 *
 * @param x - `[B, C, H, W]`.
 * @returns `[B, H W, C]`.
 * @example
 * ```ts
 * import { blocks } from 'omai-engine/nn'
 * import { zeros } from 'omai-engine/tensor'
 * blocks.toTokens(zeros([2, 3, 4, 5])).shape // => [2, 20, 3]
 * ```
 */
export function toTokens(x: TensorLike): Value {
  const v = asTensor(x)
  const [b = 1, c = 1] = v.shape
  return moveaxis(reshape(v, [b, c, -1]), 1, 2)
}
