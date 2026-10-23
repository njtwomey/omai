import { WhiteboxError } from '../../base/index.ts'
import { foldIn, type Key } from '../../random/index.ts'
import { asTensor, expandDims, isValue, squeeze, type TensorLike, type Value, zeros } from '../../tensor/index.ts'
import {
  type CellParams,
  gruCell,
  lstmCell,
  type PackedSequence,
  packPaddedSequence,
  padPackedSequence,
  type RecurrentMode,
  recurrent,
  rnnCell,
} from '../functional/recurrent.ts'
import { torchDefaultUniform } from '../init.ts'
import { type ApplyContext, initKey, isTraining, Module, type Params } from './module.ts'

/** Options of {@link RNN}, {@link LSTM} and {@link GRU}. */
export interface RecurrentModuleOptions {
  readonly numLayers?: number
  readonly bias?: boolean
  /** `[B, T, F]` inputs and outputs (default false: `[T, B, F]`). */
  readonly batchFirst?: boolean
  /** Dropout on the outputs of every layer but the last, in training. */
  readonly dropout?: number
  readonly bidirectional?: boolean
  readonly key?: Key
}

/** The input of a recurrent module: a sequence, a packed sequence, or a record with initial states and lengths. */
export type RecurrentInput =
  | Value
  | PackedSequence
  | {
      readonly input: TensorLike | PackedSequence
      readonly h0?: TensorLike
      readonly c0?: TensorLike
      /** Valid lengths of a padded batch. */
      readonly lengths?: readonly number[]
    }

/** The output of a recurrent module. */
export interface RecurrentOutput {
  /** The top layer's outputs at every step (packed when the input was packed). */
  readonly output: Value | PackedSequence
  /** Final hidden states `[D L, B, H]`. */
  readonly h: Value
  /** Final cell states (LSTM). */
  readonly c?: Value
}

const isPacked = (x: unknown): x is PackedSequence =>
  typeof x === 'object' && x !== null && 'batchSizes' in x && 'data' in x

/** The shared implementation of RNN, LSTM and GRU (PyTorch's RNNBase): weights `weight_ih_l{k}[_reverse]` etc. */
export class RNNBase extends Module<RecurrentInput, RecurrentOutput> {
  readonly kind: string
  readonly mode: RecurrentMode
  readonly inputSize: number
  readonly hiddenSize: number
  readonly numLayers: number
  readonly bias: boolean
  readonly batchFirst: boolean
  readonly dropout: number
  readonly bidirectional: boolean
  readonly projSize: number

  constructor(
    mode: RecurrentMode,
    inputSize: number,
    hiddenSize: number,
    options: RecurrentModuleOptions & { readonly projSize?: number } = {},
  ) {
    super()
    this.mode = mode
    this.kind = mode === 'LSTM' ? 'LSTM' : mode === 'GRU' ? 'GRU' : 'RNN'
    this.inputSize = inputSize
    this.hiddenSize = hiddenSize
    this.numLayers = options.numLayers ?? 1
    this.bias = options.bias ?? true
    this.batchFirst = options.batchFirst ?? false
    this.dropout = options.dropout ?? 0
    this.bidirectional = options.bidirectional ?? false
    this.projSize = options.projSize ?? 0
    if (this.projSize > 0 && mode !== 'LSTM') throw new WhiteboxError('value', 'projSize is only for LSTM')
    if (this.projSize >= hiddenSize && this.projSize > 0)
      throw new WhiteboxError('value', 'projSize must be smaller than hiddenSize')
    this.params = this.initParams(initKey(this.kind, options.key))
  }

  private get gates(): number {
    return this.mode === 'LSTM' ? 4 : this.mode === 'GRU' ? 3 : 1
  }

  private get directions(): number {
    return this.bidirectional ? 2 : 1
  }

  /** The width of each direction's output (the projection size, if any). */
  get outputSize(): number {
    return this.projSize > 0 ? this.projSize : this.hiddenSize
  }

  override initParams(key: Key): Params {
    const h = this.hiddenSize
    const out: Record<string, unknown> = {}
    let i = 0
    const u = (shape: number[]): Value => torchDefaultUniform(foldIn(key, i++), shape, h)
    for (let l = 0; l < this.numLayers; l++) {
      for (let d = 0; d < this.directions; d++) {
        const sfx = `_l${l}${d === 1 ? '_reverse' : ''}`
        const inputWidth = l === 0 ? this.inputSize : this.outputSize * this.directions
        out[`weight_ih${sfx}`] = u([this.gates * h, inputWidth])
        out[`weight_hh${sfx}`] = u([this.gates * h, this.outputSize])
        if (this.bias) {
          out[`bias_ih${sfx}`] = u([this.gates * h])
          out[`bias_hh${sfx}`] = u([this.gates * h])
        }
        if (this.projSize > 0) out[`weight_hr${sfx}`] = u([this.projSize, h])
      }
    }
    return out
  }

  /** The cell weights of each layer and direction, in the functional form's layout. */
  layerParams(params: Params): CellParams[][] {
    return Array.from({ length: this.numLayers }, (_, l) =>
      Array.from({ length: this.directions }, (_, d) => {
        const sfx = `_l${l}${d === 1 ? '_reverse' : ''}`
        const get = (name: string): Value | undefined => params[`${name}${sfx}`] as Value | undefined
        const biasIh = get('bias_ih')
        const biasHh = get('bias_hh')
        const weightHr = get('weight_hr')
        return {
          weightIh: get('weight_ih') as Value,
          weightHh: get('weight_hh') as Value,
          ...(biasIh === undefined ? {} : { biasIh }),
          ...(biasHh === undefined ? {} : { biasHh }),
          ...(weightHr === undefined ? {} : { weightHr }),
        }
      }),
    )
  }

  protected override prepare(x: unknown): unknown {
    if (isPacked(x)) return x
    return isValue(x) || Array.isArray(x) ? asTensor(x as TensorLike) : x
  }

  apply(params: Params, x: RecurrentInput, ctx: ApplyContext = {}): RecurrentOutput {
    const record = isValue(x) || isPacked(x) ? { input: x } : x
    const packed = isPacked(record.input) ? record.input : undefined
    let input: Value
    let lengths = 'lengths' in record ? record.lengths : undefined
    let batchFirst = this.batchFirst
    if (packed !== undefined) {
      const unpacked = padPackedSequence(packed)
      input = unpacked.padded
      lengths = unpacked.lengths
      batchFirst = false
    } else {
      input = asTensor(record.input as TensorLike)
    }
    const unbatched = input.shape.length === 2
    if (unbatched) input = expandDims(input, batchFirst ? 0 : 1)
    const h0 = 'h0' in record && record.h0 !== undefined ? asTensor(record.h0) : undefined
    const c0 = 'c0' in record && record.c0 !== undefined ? asTensor(record.c0) : undefined
    const training = isTraining(ctx)
    const r = recurrent(
      input,
      this.layerParams(params),
      {
        mode: this.mode,
        numLayers: this.numLayers,
        bidirectional: this.bidirectional,
        batchFirst,
        ...(lengths === undefined ? {} : { lengths }),
        ...(this.dropout > 0 && training ? { dropout: this.dropout, training } : {}),
        ...(ctx.key === undefined ? {} : { key: ctx.key }),
      },
      h0 === undefined ? undefined : unbatched ? expandDims(h0, 1) : h0,
      c0 === undefined ? undefined : unbatched ? expandDims(c0, 1) : c0,
    )
    const output: Value | PackedSequence =
      packed !== undefined
        ? packPaddedSequence(r.output, lengths ?? [])
        : unbatched
          ? squeeze(r.output, batchFirst ? 0 : 1)
          : r.output
    const h = unbatched ? squeeze(r.h, 1) : r.h
    const c = r.c === undefined ? undefined : unbatched ? squeeze(r.c, 1) : r.c
    return { output, h, ...(c === undefined ? {} : { c }) }
  }

  outputShape(inputShape: readonly number[]): number[] {
    return [...inputShape.slice(0, -1), this.outputSize * this.directions]
  }

  override extraRepr(): string {
    const parts = [`${this.inputSize}, ${this.hiddenSize}`]
    if (this.numLayers !== 1) parts.push(`num_layers=${this.numLayers}`)
    if (!this.bias) parts.push('bias=False')
    if (this.batchFirst) parts.push('batch_first=True')
    if (this.dropout > 0) parts.push(`dropout=${this.dropout}`)
    if (this.bidirectional) parts.push('bidirectional=True')
    if (this.projSize > 0) parts.push(`proj_size=${this.projSize}`)
    return parts.join(', ')
  }
}

/**
 * An Elman RNN (torch.nn.RNN): stacked, optionally bidirectional, tanh or ReLU, with PyTorch's weight names and
 * initialisation; accepts padded batches with lengths or packed sequences.
 *
 * @concept recurrent-networks
 * @example
 * ```ts
 * import { RNN } from 'whitebox-engine/nn'
 * import { zeros } from 'whitebox-engine/tensor'
 * new RNN(3, 4, { numLayers: 2, bidirectional: true }).forward(zeros([5, 2, 3])).h.shape // => [4, 2, 4]
 * ```
 */
export class RNN extends RNNBase {
  constructor(
    inputSize: number,
    hiddenSize: number,
    options: RecurrentModuleOptions & { readonly nonlinearity?: 'tanh' | 'relu' } = {},
  ) {
    super(options.nonlinearity === 'relu' ? 'RNN_RELU' : 'RNN_TANH', inputSize, hiddenSize, options)
  }
}

/**
 * A long short-term memory network (torch.nn.LSTM): stacked, optionally bidirectional, with an optional output
 * projection (`projSize`); returns the outputs and the final hidden and cell states.
 *
 * @concept recurrent-networks
 * @example
 * ```ts
 * import { LSTM } from 'whitebox-engine/nn'
 * import { zeros } from 'whitebox-engine/tensor'
 * new LSTM(3, 4, { batchFirst: true }).forward(zeros([2, 5, 3])).c?.shape // => [1, 2, 4]
 * ```
 */
export class LSTM extends RNNBase {
  constructor(
    inputSize: number,
    hiddenSize: number,
    options: RecurrentModuleOptions & { readonly projSize?: number } = {},
  ) {
    super('LSTM', inputSize, hiddenSize, options)
  }
}

/**
 * A gated recurrent unit network (torch.nn.GRU): stacked and optionally bidirectional.
 *
 * @concept recurrent-networks
 * @example
 * ```ts
 * import { GRU } from 'whitebox-engine/nn'
 * import { zeros } from 'whitebox-engine/tensor'
 * new GRU(3, 4).forward(zeros([5, 2, 3])).output.shape // => [5, 2, 4]
 * ```
 */
export class GRU extends RNNBase {
  constructor(inputSize: number, hiddenSize: number, options: RecurrentModuleOptions = {}) {
    super('GRU', inputSize, hiddenSize, options)
  }
}

/** The input of a recurrent cell: an input, or a record with the previous state. */
export type CellInput = Value | { readonly input: TensorLike; readonly h?: TensorLike; readonly c?: TensorLike }

/** A recurrent cell (RNNCell, LSTMCell, GRUCell): weights `weight_ih`, `weight_hh`, `bias_ih`, `bias_hh`. */
abstract class CellBase<O> extends Module<CellInput, O> {
  readonly inputSize: number
  readonly hiddenSize: number
  readonly bias: boolean
  protected abstract readonly gates: number

  protected constructor(inputSize: number, hiddenSize: number, bias: boolean) {
    super()
    this.inputSize = inputSize
    this.hiddenSize = hiddenSize
    this.bias = bias
  }

  override initParams(key: Key): Params {
    const h = this.hiddenSize
    return {
      weight_ih: torchDefaultUniform(foldIn(key, 0), [this.gates * h, this.inputSize], h),
      weight_hh: torchDefaultUniform(foldIn(key, 1), [this.gates * h, h], h),
      ...(this.bias
        ? {
            bias_ih: torchDefaultUniform(foldIn(key, 2), [this.gates * h], h),
            bias_hh: torchDefaultUniform(foldIn(key, 3), [this.gates * h], h),
          }
        : {}),
    }
  }

  protected weights(params: Params): CellParams {
    return {
      weightIh: params['weight_ih'] as Value,
      weightHh: params['weight_hh'] as Value,
      ...(params['bias_ih'] === undefined ? {} : { biasIh: params['bias_ih'] as Value }),
      ...(params['bias_hh'] === undefined ? {} : { biasHh: params['bias_hh'] as Value }),
    }
  }

  protected override prepare(x: unknown): unknown {
    return isValue(x) || Array.isArray(x) ? asTensor(x as TensorLike) : x
  }

  /** The input `[B, I]` and previous state (zeros by default), and whether the input was unbatched. */
  protected unpack(x: CellInput): { x: Value; h: Value; c: Value; unbatched: boolean } {
    const record = isValue(x) ? { input: x } : x
    let input = asTensor(record.input)
    const unbatched = input.shape.length === 1
    if (unbatched) input = expandDims(input, 0)
    const b = input.shape[0] ?? 1
    const state = (s: TensorLike | undefined): Value =>
      s === undefined ? zeros([b, this.hiddenSize]) : unbatched ? expandDims(asTensor(s), 0) : asTensor(s)
    return { x: input, h: state(record.h), c: state('c' in record ? record.c : undefined), unbatched }
  }

  outputShape(inputShape: readonly number[]): number[] {
    return [...inputShape.slice(0, -1), this.hiddenSize]
  }

  override extraRepr(): string {
    return `${this.inputSize}, ${this.hiddenSize}${this.bias ? '' : ', bias=False'}`
  }
}

/**
 * An Elman RNN cell (torch.nn.RNNCell).
 *
 * @example
 * ```ts
 * import { RNNCell } from 'whitebox-engine/nn'
 * import { zeros } from 'whitebox-engine/tensor'
 * new RNNCell(3, 4).forward(zeros([2, 3])).shape // => [2, 4]
 * ```
 */
export class RNNCell extends CellBase<Value> {
  readonly kind = 'RNNCell'
  protected readonly gates = 1
  readonly nonlinearity: 'tanh' | 'relu'

  constructor(
    inputSize: number,
    hiddenSize: number,
    options: { readonly bias?: boolean; readonly nonlinearity?: 'tanh' | 'relu'; readonly key?: Key } = {},
  ) {
    super(inputSize, hiddenSize, options.bias ?? true)
    this.nonlinearity = options.nonlinearity ?? 'tanh'
    this.params = this.initParams(initKey('RNNCell', options.key))
  }

  apply(params: Params, x: CellInput): Value {
    const s = this.unpack(x)
    const h = rnnCell(s.x, s.h, this.weights(params), this.nonlinearity)
    return s.unbatched ? squeeze(h, 0) : h
  }
}

/**
 * An LSTM cell (torch.nn.LSTMCell): returns the next `[h, c]`.
 *
 * @example
 * ```ts
 * import { LSTMCell } from 'whitebox-engine/nn'
 * import { zeros } from 'whitebox-engine/tensor'
 * new LSTMCell(3, 4).forward(zeros([2, 3]))[1].shape // => [2, 4]
 * ```
 */
export class LSTMCell extends CellBase<readonly [Value, Value]> {
  readonly kind = 'LSTMCell'
  protected readonly gates = 4

  constructor(inputSize: number, hiddenSize: number, options: { readonly bias?: boolean; readonly key?: Key } = {}) {
    super(inputSize, hiddenSize, options.bias ?? true)
    this.params = this.initParams(initKey('LSTMCell', options.key))
  }

  apply(params: Params, x: CellInput): readonly [Value, Value] {
    const s = this.unpack(x)
    const [h, c] = lstmCell(s.x, [s.h, s.c], this.weights(params))
    return s.unbatched ? [squeeze(h, 0), squeeze(c, 0)] : [h, c]
  }
}

/**
 * A GRU cell (torch.nn.GRUCell).
 *
 * @example
 * ```ts
 * import { GRUCell } from 'whitebox-engine/nn'
 * import { zeros } from 'whitebox-engine/tensor'
 * new GRUCell(3, 4).forward({ input: zeros([3]), h: zeros([4]) }).shape // => [4]
 * ```
 */
export class GRUCell extends CellBase<Value> {
  readonly kind = 'GRUCell'
  protected readonly gates = 3

  constructor(inputSize: number, hiddenSize: number, options: { readonly bias?: boolean; readonly key?: Key } = {}) {
    super(inputSize, hiddenSize, options.bias ?? true)
    this.params = this.initParams(initKey('GRUCell', options.key))
  }

  apply(params: Params, x: CellInput): Value {
    const s = this.unpack(x)
    const h = gruCell(s.x, s.h, this.weights(params))
    return s.unbatched ? squeeze(h, 0) : h
  }
}
