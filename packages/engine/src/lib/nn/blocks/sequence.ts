import { foldIn, type Key } from '../../random/index.ts'
import {
  add,
  asTensor,
  isValue,
  mul,
  sigmoid,
  sliceAxis,
  type TensorLike,
  tanh,
  type Value,
} from '../../tensor/index.ts'
import { relu } from '../functional/activations.ts'
import { pad } from '../functional/vision.ts'
import { Composite, ModuleList } from '../module/containers.ts'
import { Conv1d } from '../module/conv.ts'
import { type ApplyContext, initKey, type Module, type Params } from '../module/module.ts'
import { LayerNorm } from '../module/normalisation.ts'
import { WeightNorm } from '../module/parametrisation.ts'
import { GRU, LSTM, type RecurrentOutput } from '../module/recurrent.ts'
import { Dropout } from '../module/regularisation.ts'

const keyAt =
  (k: Key) =>
  (i: number): Key =>
    foldIn(k, i)

/**
 * Pad the time axis on the left so a convolution of kernel k and dilation d sees only the past (causal convolution).
 *
 * @param x - `[B, C, T]`.
 * @param kernelSize - k.
 * @param dilation - d (default 1).
 * @returns `[B, C, T + d (k - 1)]`.
 * @example
 * ```ts
 * import { blocks } from 'whitebox-engine/nn'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(blocks.causalPad([[[1, 2]]], 2, 2)) // => [[[0, 0, 1, 2]]]
 * ```
 */
export function causalPad(x: TensorLike, kernelSize: number, dilation = 1): Value {
  return pad(x, [dilation * (kernelSize - 1), 0])
}

const conv = (cin: number, cout: number, k: number, dilation: number, norm: boolean, key: Key): Module => {
  const c = new Conv1d(cin, cout, k, { dilation, key })
  return norm ? (new WeightNorm(c) as unknown as Module) : c
}

/** Options of {@link TemporalBlock}. */
export interface TemporalBlockOptions {
  readonly kernelSize?: number
  readonly dilation?: number
  readonly dropout?: number
  /** Weight-normalise the convolutions (default true, as the TCN paper). */
  readonly weightNorm?: boolean
  readonly key?: Key
}

/**
 * The temporal convolutional network's residual block (Bai, Kolter and Koltun 2018): two dilated causal convolutions
 * (weight-normalised), ReLU and dropout, plus the input (through a 1x1 convolution when the width changes).
 *
 * @reference Bai, Kolter and Koltun (2018), An empirical evaluation of generic convolutional and recurrent networks for
 *   sequence modeling.
 * @example
 * ```ts
 * import { blocks } from 'whitebox-engine/nn'
 * import { zeros } from 'whitebox-engine/tensor'
 * new blocks.TemporalBlock(3, 5, { dilation: 2 }).eval().forward(zeros([2, 3, 10])).shape // => [2, 5, 10]
 * ```
 */
export class TemporalBlock extends Composite {
  readonly kind = 'TemporalBlock'
  readonly outChannels: number
  readonly kernelSize: number
  readonly dilation: number

  constructor(inChannels: number, outChannels: number, options: TemporalBlockOptions = {}) {
    const k = keyAt(initKey('TemporalBlock', options.key))
    const ks = options.kernelSize ?? 3
    const d = options.dilation ?? 1
    const norm = options.weightNorm ?? true
    super({
      conv1: conv(inChannels, outChannels, ks, d, norm, k(0)),
      dropout1: new Dropout(options.dropout ?? 0),
      conv2: conv(outChannels, outChannels, ks, d, norm, k(2)),
      dropout2: new Dropout(options.dropout ?? 0),
      ...(inChannels === outChannels ? {} : { downsample: new Conv1d(inChannels, outChannels, 1, { key: k(4) }) }),
    })
    this.outChannels = outChannels
    this.kernelSize = ks
    this.dilation = d
  }

  apply(params: Params, x: Value, ctx: ApplyContext = {}): Value {
    const p = this.parts(params, ctx)
    const r = (name: string, v: Value): Value => p.run<Value, Value>(name, v)
    const step = (name: string, v: Value): Value => relu(r(name, causalPad(v, this.kernelSize, this.dilation)))
    const h = r('dropout2', step('conv2', r('dropout1', step('conv1', x))))
    const y = relu(add(h, this.has('downsample') ? r('downsample', x) : x))
    p.done()
    return y
  }

  outputShape(s: readonly number[]): number[] {
    return [s[0] ?? 1, this.outChannels, s[2] ?? 0]
  }
}

/**
 * A temporal convolutional network: temporal blocks with dilations 1, 2, 4, ... so the receptive field grows
 * exponentially with depth; `receptiveField` gives 1 + 2 (k - 1) (2^L - 1).
 *
 * @example
 * ```ts
 * import { blocks } from 'whitebox-engine/nn'
 * new blocks.TCN(2, [4, 4, 4]).receptiveField // => 29
 * ```
 */
export class TCN extends Composite {
  readonly kind = 'TCN'
  readonly channels: readonly number[]
  readonly receptiveField: number

  constructor(inChannels: number, channels: readonly number[], options: Omit<TemporalBlockOptions, 'dilation'> = {}) {
    const k = keyAt(initKey('TCN', options.key))
    const ks = options.kernelSize ?? 3
    super({
      network: new ModuleList(
        channels.map(
          (c, i) =>
            new TemporalBlock(i === 0 ? inChannels : (channels[i - 1] ?? c), c, {
              ...options,
              dilation: 2 ** i,
              key: k(i),
            }),
        ),
      ),
    })
    this.channels = channels
    this.receptiveField = 1 + 2 * (ks - 1) * (2 ** channels.length - 1)
  }

  apply(params: Params, x: Value, ctx: ApplyContext = {}): Value {
    const list = this.part<ModuleList<TemporalBlock>>('network')
    const lp = (params['network'] ?? {}) as Params
    const p = this.parts(params, ctx)
    const listCtx = p.context('network')
    let h = x
    for (const [i, block] of list.modules.entries()) {
      h = block.apply((lp[String(i)] ?? {}) as Params, h, {
        ...listCtx,
        ...(listCtx.key === undefined ? {} : { key: foldIn(listCtx.key, i) }),
      })
    }
    p.done()
    return h
  }

  outputShape(s: readonly number[]): number[] {
    return [s[0] ?? 1, this.channels.at(-1) ?? s[1] ?? 1, s[2] ?? 0]
  }
}

/** The input of a WaveNet block: the residual stream, or with a conditioning signal `[B, Cc, T]`. */
export type WaveNetInput = Value | { readonly x: TensorLike; readonly condition?: TensorLike }

/** The output of a WaveNet block: the residual stream and the skip contribution. */
export interface WaveNetBlockOutput {
  readonly output: Value
  readonly skip: Value
}

/**
 * A WaveNet residual block (van den Oord et al. 2016): a dilated causal convolution to a filter and a gate,
 * tanh(filter) sigmoid(gate) (plus a 1x1-projected conditioning signal), then 1x1 convolutions to the residual and
 * the skip outputs.
 *
 * @reference van den Oord et al. (2016), WaveNet: a generative model for raw audio.
 * @example
 * ```ts
 * import { blocks } from 'whitebox-engine/nn'
 * import { zeros } from 'whitebox-engine/tensor'
 * new blocks.WaveNetBlock(4, 6, { dilation: 2 }).forward(zeros([1, 4, 8])).skip.shape // => [1, 6, 8]
 * ```
 */
export class WaveNetBlock extends Composite<WaveNetInput, WaveNetBlockOutput> {
  readonly kind = 'WaveNetBlock'
  readonly channels: number
  readonly dilation: number
  readonly kernelSize: number

  constructor(
    channels: number,
    skipChannels: number,
    options: {
      readonly dilation?: number
      readonly kernelSize?: number
      readonly conditionChannels?: number
      readonly key?: Key
    } = {},
  ) {
    const k = keyAt(initKey('WaveNetBlock', options.key))
    const ks = options.kernelSize ?? 2
    const d = options.dilation ?? 1
    super({
      dilated: new Conv1d(channels, 2 * channels, ks, { dilation: d, key: k(0) }),
      ...(options.conditionChannels === undefined
        ? {}
        : { condition: new Conv1d(options.conditionChannels, 2 * channels, 1, { key: k(1) }) }),
      residual: new Conv1d(channels, channels, 1, { key: k(2) }),
      skip: new Conv1d(channels, skipChannels, 1, { key: k(3) }),
    })
    this.channels = channels
    this.dilation = d
    this.kernelSize = ks
  }

  protected override prepare(x: unknown): unknown {
    return isValue(x) || Array.isArray(x) ? asTensor(x as TensorLike) : x
  }

  apply(params: Params, input: WaveNetInput, ctx: ApplyContext = {}): WaveNetBlockOutput {
    const record = isValue(input) ? { x: input } : input
    const x = asTensor(record.x)
    const p = this.parts(params, ctx)
    const r = (name: string, v: Value): Value => p.run<Value, Value>(name, v)
    let z = r('dilated', causalPad(x, this.kernelSize, this.dilation))
    if (record.condition !== undefined && this.has('condition')) z = add(z, r('condition', asTensor(record.condition)))
    const c = this.channels
    const h = mul(tanh(sliceAxis(z, 1, 0, c)), sigmoid(sliceAxis(z, 1, c, 2 * c)))
    const out = { output: add(x, r('residual', h)), skip: r('skip', h) }
    p.done()
    return out
  }

  outputShape(s: readonly number[]): number[] {
    return [...s]
  }
}

/** Options of {@link WaveNet}. */
export interface WaveNetOptions {
  /** Residual channels (default 16). */
  readonly residualChannels?: number
  /** Skip channels (default 32). */
  readonly skipChannels?: number
  /** Blocks per dilation cycle 1, 2, ..., 2^(n-1) (default 4). */
  readonly layersPerStack?: number
  /** Dilation cycles (default 2). */
  readonly stacks?: number
  readonly conditionChannels?: number
  readonly key?: Key
}

/**
 * A WaveNet: a causal 1x1 input projection, stacks of residual blocks with dilations cycling through powers of two,
 * the sum of their skip outputs through ReLU and two 1x1 convolutions to the output channels (logits of a
 * quantised signal, for example).
 *
 * @example
 * ```ts
 * import { blocks } from 'whitebox-engine/nn'
 * import { zeros } from 'whitebox-engine/tensor'
 * new blocks.WaveNet(1, 8, { layersPerStack: 2, stacks: 1 }).forward(zeros([1, 1, 6])).shape // => [1, 8, 6]
 * ```
 */
export class WaveNet extends Composite<WaveNetInput, Value> {
  readonly kind = 'WaveNet'
  readonly outChannels: number

  constructor(inChannels: number, outChannels: number, options: WaveNetOptions = {}) {
    const k = keyAt(initKey('WaveNet', options.key))
    const rc = options.residualChannels ?? 16
    const sc = options.skipChannels ?? 32
    const per = options.layersPerStack ?? 4
    const stacks = options.stacks ?? 2
    super({
      input: new Conv1d(inChannels, rc, 1, { key: k(0) }),
      blocks: new ModuleList(
        Array.from(
          { length: per * stacks },
          (_, i) =>
            new WaveNetBlock(rc, sc, {
              dilation: 2 ** (i % per),
              ...(options.conditionChannels === undefined ? {} : { conditionChannels: options.conditionChannels }),
              key: foldIn(k(1), i),
            }),
        ),
      ),
      out1: new Conv1d(sc, sc, 1, { key: k(2) }),
      out2: new Conv1d(sc, outChannels, 1, { key: k(3) }),
    })
    this.outChannels = outChannels
  }

  protected override prepare(x: unknown): unknown {
    return isValue(x) || Array.isArray(x) ? asTensor(x as TensorLike) : x
  }

  apply(params: Params, input: WaveNetInput, ctx: ApplyContext = {}): Value {
    const record = isValue(input) ? { x: input } : input
    const p = this.parts(params, ctx)
    let h = p.run<Value, Value>('input', asTensor(record.x))
    const list = this.part<ModuleList<WaveNetBlock>>('blocks')
    const lp = (params['blocks'] ?? {}) as Params
    let skips: Value | undefined
    for (const [i, block] of list.modules.entries()) {
      const r = block.apply((lp[String(i)] ?? {}) as Params, {
        x: h,
        ...(record.condition === undefined ? {} : { condition: record.condition }),
      })
      h = r.output
      skips = skips === undefined ? r.skip : add(skips, r.skip)
    }
    const y = p.run<Value, Value>('out2', relu(p.run<Value, Value>('out1', relu(skips as Value))))
    p.done()
    return y
  }

  outputShape(s: readonly number[]): number[] {
    return [s[0] ?? 1, this.outChannels, s[2] ?? 0]
  }
}

/** Options of {@link StackedRecurrent}. */
export interface StackedRecurrentOptions {
  readonly cell?: 'lstm' | 'gru'
  /** Add each layer's input to its output when the widths match (GNMT's residual stacking; default true). */
  readonly residual?: boolean
  /** Layer-normalise each layer's output (default true). */
  readonly layerNorm?: boolean
  readonly dropout?: number
  readonly bidirectional?: boolean
  readonly key?: Key
}

/**
 * Stacked recurrent layers with residual connections, layer normalisation and dropout between them (GNMT-style deep
 * LSTMs), batch-first `[B, T, F]` in and out.
 *
 * @example
 * ```ts
 * import { blocks } from 'whitebox-engine/nn'
 * import { zeros } from 'whitebox-engine/tensor'
 * new blocks.StackedRecurrent(3, 4, 3).eval().forward(zeros([2, 5, 3])).shape // => [2, 5, 4]
 * ```
 */
export class StackedRecurrent extends Composite {
  readonly kind = 'StackedRecurrent'
  readonly hiddenSize: number
  readonly residual: boolean
  readonly width: number

  constructor(inputSize: number, hiddenSize: number, numLayers: number, options: StackedRecurrentOptions = {}) {
    const k = keyAt(initKey('StackedRecurrent', options.key))
    const dirs = options.bidirectional === true ? 2 : 1
    const make = (i: number): Module<never, unknown> => {
      const width = i === 0 ? inputSize : hiddenSize * dirs
      const o = { batchFirst: true, bidirectional: dirs === 2, key: foldIn(k(0), i) }
      return options.cell === 'gru' ? new GRU(width, hiddenSize, o) : new LSTM(width, hiddenSize, o)
    }
    const layers = Array.from({ length: numLayers }, (_, i) => make(i))
    super({
      layers: new ModuleList(layers),
      ...(options.layerNorm === false
        ? {}
        : { norms: new ModuleList(layers.map(() => new LayerNorm(hiddenSize * dirs))) }),
      dropout: new Dropout(options.dropout ?? 0),
    })
    this.hiddenSize = hiddenSize
    this.residual = options.residual ?? true
    this.width = hiddenSize * dirs
  }

  apply(params: Params, x: Value, ctx: ApplyContext = {}): Value {
    const p = this.parts(params, ctx)
    const layers = this.part<ModuleList<Module<Value, RecurrentOutput>>>('layers').modules
    const lp = (params['layers'] ?? {}) as Params
    const np = (params['norms'] ?? {}) as Params
    const norms = this.has('norms') ? this.part<ModuleList<Module>>('norms').modules : []
    const listCtx = p.context('layers')
    let h = x
    for (const [i, layer] of layers.entries()) {
      const out = layer.apply((lp[String(i)] ?? {}) as Params, h, {
        ...listCtx,
        ...(listCtx.key === undefined ? {} : { key: foldIn(listCtx.key, i) }),
      }).output as Value
      let y = this.residual && out.shape[2] === h.shape[2] ? add(out, h) : out
      const norm = norms[i]
      if (norm !== undefined) y = norm.apply((np[String(i)] ?? {}) as Params, y)
      // Each gap between layers draws its own dropout mask.
      const dctx = p.context('dropout')
      h =
        i + 1 < layers.length
          ? this.part<Dropout>('dropout').apply({}, y, {
              ...dctx,
              ...(dctx.key === undefined ? {} : { key: foldIn(dctx.key, i) }),
            })
          : y
    }
    p.done()
    return h
  }

  outputShape(s: readonly number[]): number[] {
    return [s[0] ?? 1, s[1] ?? 0, this.width]
  }
}
