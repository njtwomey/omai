/**
 * Sequence models built from interchangeable mixers: a `MixerBlock` (a pre-norm residual block around any
 * `SequenceMixer`, with an optional gated feed-forward), and `SequenceModel`, a stack of such blocks between an input
 * layer (token embedding or feature projection) and a head. With a vocabulary as output it is a language model, with a
 * pooled output a sequence classifier or regressor. `run` threads the per-layer states, so the model generates one
 * token at a time with the outputs of its parallel pass; `forward` is the parallel pass a trainer calls.
 */

import { foldIn, type Key } from '../../random/index.ts'
import { add, asTensor, isValue, mean, reshape, sliceAxis, type TensorLike, type Value } from '../../tensor/index.ts'
import { linear } from '../functional/linear.ts'
import { Composite, ModuleList } from '../module/containers.ts'
import { Embedding, Linear } from '../module/layers.ts'
import { type ApplyContext, childScope, initKey, type Params } from '../module/module.ts'
import { LayerNorm, RMSNorm } from '../module/normalisation.ts'
import type { MixerInput, MixerOutput, MixerState, SequenceMixer } from './mixer.ts'
import { GatedFeedForward } from './transformer.ts'

/** Options of {@link MixerBlock}. */
export interface MixerBlockOptions {
  /** A gated feed-forward sub-block after the mixer (default `none`: Mamba's blocks have none). */
  readonly ffn?: 'none' | 'gated'
  /** The feed-forward hidden width (default `round(8 dim / 3)`). */
  readonly hidden?: number
  readonly norm?: 'rms' | 'layer'
  readonly key?: Key
}

const normOf = (kind: 'rms' | 'layer', dim: number): RMSNorm | LayerNorm =>
  kind === 'layer' ? new LayerNorm(dim) : new RMSNorm(dim, { eps: 1e-6 })

/**
 * A pre-norm residual block around a sequence mixer: `x + mixer(norm(x))`, then optionally `h + ffn(norm(h))`. It
 * threads the mixer's state, so it is itself a mixer-shaped module.
 *
 * @concept sequence-models
 * @example
 * ```ts
 * import { MixerBlock, MambaBlock } from 'mdml-engine/nn/blocks'
 * import { zeros } from 'mdml-engine/tensor'
 * new MixerBlock(new MambaBlock(8, { stateSize: 4 })).forward(zeros([1, 3, 8])).output.shape // => [1, 3, 8]
 * ```
 */
export class MixerBlock extends Composite<MixerInput | Value, MixerOutput> {
  readonly kind = 'MixerBlock'
  readonly dim: number
  readonly mixer: SequenceMixer
  readonly ffn: 'none' | 'gated'

  constructor(mixer: SequenceMixer, options: MixerBlockOptions = {}) {
    const k = initKey('MixerBlock', options.key)
    const dim = mixer.dim
    const ffn = options.ffn ?? 'none'
    const norm = options.norm ?? 'rms'
    super({
      norm: normOf(norm, dim),
      mixer,
      ...(ffn === 'none'
        ? {}
        : {
            ffn_norm: normOf(norm, dim),
            ffn: new GatedFeedForward(dim, options.hidden ?? Math.round((8 * dim) / 3), { key: foldIn(k, 1) }),
          }),
    })
    this.dim = dim
    this.mixer = mixer
    this.ffn = ffn
  }

  protected override prepare(x: unknown): unknown {
    return isValue(x) || Array.isArray(x) ? { x: asTensor(x as TensorLike) } : x
  }

  apply(params: Params, input: MixerInput | Value, ctx: ApplyContext = {}): MixerOutput {
    const inp: MixerInput = isValue(input) ? { x: input } : input
    const p = this.parts(params, ctx)
    const x = asTensor(inp.x)
    const r = p.run<MixerInput, MixerOutput>('mixer', {
      x: p.run<Value, Value>('norm', x),
      ...(inp.state === undefined ? {} : { state: inp.state }),
    })
    let h = add(x, r.output)
    if (this.ffn !== 'none') h = add(h, p.run<Value, Value>('ffn', p.run<Value, Value>('ffn_norm', h)))
    p.done()
    return { output: h, state: r.state }
  }

  outputShape(inputShape: readonly number[]): number[] {
    return [...inputShape]
  }

  override extraRepr(): string {
    return `dim=${this.dim}, mixer=${this.mixer.kind}${this.ffn === 'none' ? '' : ', ffn'}`
  }
}

/** Options of {@link SequenceModel}. */
export interface SequenceModelOptions extends MixerBlockOptions {
  /** What the input is: token ids `[B, L]` (default) or features `[B, L, F]`. */
  readonly input?: 'tokens' | 'features'
  /** The vocabulary size (tokens) or the feature count. */
  readonly inputSize: number
  /** The output width per position: the vocabulary of a language model (default `inputSize` for tokens). */
  readonly outputSize?: number
  /** `none`: an output per position (a language model); `last` or `mean`: one output per sequence (a classifier). */
  readonly pooling?: 'none' | 'last' | 'mean'
  /** Share the embedding with the output projection (default true for a language model over the same vocabulary). */
  readonly tieEmbeddings?: boolean
}

/** The per-layer states of a {@link SequenceModel}, keyed by layer index. */
export type SequenceModelState = Readonly<Record<string, MixerState>>

/**
 * A sequence model: an input layer, a stack of {@link MixerBlock}s (each around a mixer the caller builds: Mamba, a
 * state-space layer, linear attention, ...), a final normalisation and a head. Tokens give logits `[B, L, V]` (a
 * language model), or `[B, C]` with pooling (a classifier); features give a regression output. `forward` is the
 * parallel pass; `run` is the same function with the states threaded, which is how generation proceeds token by token.
 *
 * @concept sequence-models
 * @example
 * ```ts
 * import { MambaBlock, SequenceModel } from 'mdml-engine/nn/blocks'
 * const m = new SequenceModel(8, 2, (dim) => new MambaBlock(dim, { stateSize: 4 }), { inputSize: 11 })
 * m.forward([[1, 2, 3]]).shape // => [1, 3, 11]
 * ```
 */
export class SequenceModel extends Composite<TensorLike, Value> {
  readonly kind = 'SequenceModel'
  readonly dim: number
  readonly numLayers: number
  readonly inputKind: 'tokens' | 'features'
  readonly inputSize: number
  readonly outputSize: number
  readonly pooling: 'none' | 'last' | 'mean'
  readonly tieEmbeddings: boolean

  constructor(
    dim: number,
    numLayers: number,
    mixer: (dim: number, key: Key, layer: number) => SequenceMixer,
    options: SequenceModelOptions,
  ) {
    const k = initKey('SequenceModel', options.key)
    const inputKind = options.input ?? 'tokens'
    const pooling = options.pooling ?? 'none'
    const outputSize = options.outputSize ?? options.inputSize
    const tie = (options.tieEmbeddings ?? true) && inputKind === 'tokens' && outputSize === options.inputSize
    super({
      embed:
        inputKind === 'tokens'
          ? new Embedding(options.inputSize, dim, { key: foldIn(k, 0) })
          : new Linear(options.inputSize, dim, { key: foldIn(k, 0) }),
      layers: new ModuleList(
        Array.from(
          { length: numLayers },
          (_, i) =>
            new MixerBlock(mixer(dim, foldIn(foldIn(k, 1), i), i), {
              ...(options.ffn === undefined ? {} : { ffn: options.ffn }),
              ...(options.hidden === undefined ? {} : { hidden: options.hidden }),
              ...(options.norm === undefined ? {} : { norm: options.norm }),
              key: foldIn(foldIn(k, 2), i),
            }),
        ),
      ),
      norm: normOf(options.norm ?? 'rms', dim),
      ...(tie ? {} : { head: new Linear(dim, outputSize, { bias: pooling !== 'none', key: foldIn(k, 3) }) }),
    })
    this.dim = dim
    this.numLayers = numLayers
    this.inputKind = inputKind
    this.inputSize = options.inputSize
    this.outputSize = outputSize
    this.pooling = pooling
    this.tieEmbeddings = tie
  }

  protected override prepare(x: unknown): unknown {
    return this.inputKind === 'tokens' ? asTensor(x as TensorLike, 'int32') : asTensor(x as TensorLike)
  }

  /** The blocks, in order. */
  get blockList(): readonly MixerBlock[] {
    return this.part<ModuleList<MixerBlock>>('layers').modules
  }

  /** The states before the first position of a batch: one per layer, keyed by index. */
  initialState(batch: number): SequenceModelState {
    return Object.fromEntries(this.blockList.map((b, i) => [String(i), b.mixer.initialState(batch)]))
  }

  /**
   * The model on a sequence from the states of earlier calls: the outputs and the states after the last position.
   * Without states it is the parallel pass; one token at a time with the states handed on it is generation, with the
   * same outputs.
   */
  run(
    params: Params,
    input: TensorLike,
    state?: SequenceModelState,
    ctx: ApplyContext = {},
  ): { readonly output: Value; readonly state: SequenceModelState } {
    const p = this.parts(params, ctx)
    const list = this.part<ModuleList<MixerBlock>>('layers')
    const inner = childScope(p.context('layers'), list.buffers)
    const listParams = (params['layers'] ?? {}) as Params
    let h = p.run<Value, Value>('embed', asTensor(input, this.inputKind === 'tokens' ? 'int32' : undefined))
    const next: Record<string, MixerState> = {}
    for (const [i, block] of this.blockList.entries()) {
      const r = block.apply(
        (listParams[String(i)] ?? {}) as Params,
        { x: h, ...(state?.[String(i)] === undefined ? {} : { state: state[String(i)] as MixerState }) },
        inner.child(String(i), i),
      )
      h = r.output
      next[String(i)] = r.state
    }
    inner.done()
    h = p.run<Value, Value>('norm', h)
    if (this.pooling === 'last')
      h = reshape(sliceAxis(h, 1, (h.shape[1] ?? 1) - 1, h.shape[1] ?? 1), [h.shape[0] ?? 1, -1])
    else if (this.pooling === 'mean') h = mean(h, 1)
    const output = this.tieEmbeddings
      ? linear(h, (params['embed'] as Params)['weight'] as Value)
      : p.run<Value, Value>('head', h)
    p.done()
    return { output, state: next }
  }

  apply(params: Params, input: TensorLike, ctx: ApplyContext = {}): Value {
    return this.run(params, input, undefined, ctx).output
  }

  outputShape(inputShape: readonly number[]): number[] {
    const lead = this.pooling === 'none' ? [...inputShape] : [inputShape[0] ?? 1]
    return this.inputKind === 'tokens' || this.pooling !== 'none'
      ? [...lead.slice(0, this.pooling === 'none' ? undefined : 1), this.outputSize]
      : [...lead.slice(0, 2), this.outputSize]
  }

  override extraRepr(): string {
    return `dim=${this.dim}, layers=${this.numLayers}, ${this.inputKind}=${this.inputSize}, out=${this.outputSize}, pooling=${this.pooling}`
  }
}
