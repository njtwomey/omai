/**
 * The mixer slot: every sequence mixer of the library in one option table, so that a sequence model chooses its layer
 * by name and knobs (`mixer: { kind: 'retnet', heads: 4 }`) and the controls, grids and tuners see every mixer's knobs
 * through one space. Beside the state-space layers, Mamba, the linear-attention family, RWKV, the minimal RNNs and
 * Hyena it holds softmax attention (with a key-value cache for a state) as the baseline every other mixer is
 * measured against.
 */

import { foldIn, type Key } from '../../random/index.ts'
import {
  bool,
  choice,
  dict,
  int,
  type ModeConfig,
  type Option,
  type OptionTable,
  optionTable,
} from '../../space/index.ts'
import type { Value } from '../../tensor/index.ts'
import { emptyKVCache, GroupedQueryAttention, type GroupedQueryOutput, type KVCache } from '../module/attention.ts'
import { type ApplyContext, initKey, type Params } from '../module/module.ts'
import { DeltaNetMixer, GatedLinearAttentionMixer, LinearAttentionMixer, RetNetMixer } from './linear-attention.ts'
import { Mamba2Block, MambaBlock } from './mamba.ts'
import { type MixerOutput, type MixerState, SequenceMixer } from './mixer.ts'
import { Hyena, MinGRU, MinLSTM, RWKVMixer } from './recurrent-mixers.ts'
import { LRUMixer, S4, S4D, S5 } from './state-space.ts'

/**
 * Causal multi-head softmax self-attention with rotary positions as a mixer, the baseline of the table. Its state is
 * the key-value cache, which grows with the sequence (every other mixer has a state of fixed size, except Hyena).
 *
 * @concept sequence-models
 * @example
 * ```ts
 * import { AttentionMixer } from 'omai-engine/nn/blocks'
 * import { zeros } from 'omai-engine/tensor'
 * new AttentionMixer(8, { heads: 2 }).forward(zeros([1, 5, 8])).output.shape // => [1, 5, 8]
 * ```
 */
export class AttentionMixer extends SequenceMixer {
  readonly kind = 'Attention'
  readonly dim: number
  readonly heads: number

  constructor(dim: number, options: { readonly heads?: number; readonly window?: number; readonly key?: Key } = {}) {
    const k = initKey('Attention', options.key)
    const heads = options.heads ?? (dim % 2 === 0 ? 2 : 1)
    super({
      attn: new GroupedQueryAttention(dim, heads, {
        causal: true,
        bias: false,
        ...(options.window === undefined ? {} : { window: options.window }),
        key: foldIn(k, 0),
      }),
    })
    this.dim = dim
    this.heads = heads
  }

  initialState(batch: number): MixerState {
    const c = emptyKVCache(batch, this.heads, this.dim / this.heads)
    return { keys: c.keys, values: c.values }
  }

  protected mix(params: Params, x: Value, state: MixerState | undefined, ctx: ApplyContext): MixerOutput {
    const p = this.parts(params, ctx)
    const cache: KVCache | undefined =
      state === undefined ? undefined : { keys: state['keys'] as Value, values: state['values'] as Value }
    const r = p.run<object, GroupedQueryOutput>('attn', { x, ...(cache === undefined ? {} : { cache }) })
    p.done()
    return { output: r.output, state: { keys: r.cache.keys, values: r.cache.values } }
  }
}

/** Builds a mixer of a given width from a key (the layer index is folded into the key by the caller). */
export type MixerFactory = (dim: number, key: Key) => SequenceMixer

/** What a mixer is, for selectors and validity rules. */
export interface MixerProperties {
  /** The state has a fixed size whatever the sequence length (false for attention's cache and Hyena's history). */
  readonly constantState: boolean
  /** The family: the layer's mathematical kind. */
  readonly family: 'state-space' | 'selective' | 'linear-attention' | 'recurrent' | 'convolution' | 'attention'
}

const mixer = (
  description: string,
  properties: MixerProperties,
  space: Option<MixerFactory, MixerProperties>['space'],
  make: (knobs: ModeConfig) => MixerFactory,
): Option<MixerFactory, MixerProperties> => ({ description, properties, space, make })

const num = (knobs: ModeConfig, name: string): number | undefined =>
  typeof knobs[name] === 'number' ? (knobs[name] as number) : undefined
const str = <T extends string>(knobs: ModeConfig, name: string): T | undefined =>
  typeof knobs[name] === 'string' ? (knobs[name] as T) : undefined
const opt = <K extends string, V>(name: K, value: V | undefined): { readonly [P in K]?: V } =>
  (value === undefined ? {} : { [name]: value }) as { readonly [P in K]?: V }

const heads = int(1, 32, { default: 2, label: 'heads' })
const stateSize = int(2, 256, { log: true, default: 16, label: 'state size' })
const form3 = choice(['parallel', 'recurrent', 'chunked'] as const, { default: 'parallel', label: 'algorithm' })

/**
 * The sequence mixers as one slot. Pass a configuration to {@link sequenceMixers}`.make` to get a factory
 * `(dim, key) => SequenceMixer` for {@link SequenceModel}.
 *
 * @concept sequence-models
 * @index table Every sequence mixer (Mamba, RetNet, RWKV, attention) as one slot.
 * @example
 * ```ts
 * import { sequenceMixers } from 'omai-engine/nn/blocks'
 * sequenceMixers.names.slice(0, 3) // => ['mamba', 'mamba2', 's4d']
 * ```
 */
export const sequenceMixers: OptionTable<MixerFactory, MixerProperties> = optionTable(
  {
    mamba: mixer(
      'Mamba: input-dependent step, input and output maps over a diagonal state (selective scan)',
      { constantState: true, family: 'selective' },
      dict({
        stateSize,
        expand: int(1, 8, { default: 2, label: 'expansion' }),
        convKernel: int(2, 16, { default: 4, label: 'convolution taps' }),
      }),
      (c) => (dim, key) =>
        new MambaBlock(dim, {
          ...opt('stateSize', num(c, 'stateSize')),
          ...opt('expand', num(c, 'expand')),
          ...opt('convKernel', num(c, 'convKernel')),
          key,
        }),
    ),
    mamba2: mixer(
      'Mamba-2: scalar decay per head, the SSD duality of recurrence and masked attention',
      { constantState: true, family: 'selective' },
      dict({
        stateSize,
        expand: int(1, 8, { default: 2, label: 'expansion' }),
        headDim: int(1, 128, { default: 16, label: 'head width' }),
        form: choice(['chunked', 'quadratic', 'recurrent'] as const, { default: 'chunked', label: 'algorithm' }),
      }),
      (c) => (dim, key) =>
        new Mamba2Block(dim, {
          ...opt('stateSize', num(c, 'stateSize')),
          ...opt('expand', num(c, 'expand')),
          ...opt('headDim', num(c, 'headDim')),
          ...opt('form', str<'chunked' | 'quadratic' | 'recurrent'>(c, 'form')),
          key,
        }),
    ),
    s4d: mixer(
      'S4D: diagonal complex state per channel, convolution or recurrence',
      { constantState: true, family: 'state-space' },
      dict({
        stateSize,
        init: choice(['lin', 'inv', 'real'] as const, { default: 'lin', label: 'eigenvalues' }),
        mode: choice(['convolution', 'recurrence'] as const, { default: 'convolution', label: 'mode' }),
      }),
      (c) => (dim, key) =>
        new S4D(dim, {
          ...opt('stateSize', num(c, 'stateSize')),
          ...opt('init', str<'lin' | 'inv' | 'real'>(c, 'init')),
          ...opt('mode', str<'convolution' | 'recurrence'>(c, 'mode')),
          key,
        }),
    ),
    s4: mixer(
      'S4: HiPPO normal-plus-low-rank matrix in its eigenbasis, bilinear discretisation',
      { constantState: true, family: 'state-space' },
      dict({
        stateSize,
        measure: choice(['legs', 'legt', 'lagt'] as const, { default: 'legs', label: 'HiPPO measure' }),
        mode: choice(['convolution', 'recurrence'] as const, { default: 'convolution', label: 'mode' }),
      }),
      (c) => (dim, key) =>
        new S4(dim, {
          ...opt('stateSize', num(c, 'stateSize')),
          ...opt('measure', str<'legs' | 'legt' | 'lagt'>(c, 'measure')),
          ...opt('mode', str<'convolution' | 'recurrence'>(c, 'mode')),
          key,
        }),
    ),
    s5: mixer(
      'S5: one MIMO complex diagonal state shared by the channels, parallel scan',
      { constantState: true, family: 'state-space' },
      dict({ stateSize, blocks: int(1, 16, { default: 1, label: 'HiPPO blocks' }) }),
      (c) => (dim, key) =>
        new S5(dim, { ...opt('stateSize', num(c, 'stateSize')), ...opt('blocks', num(c, 'blocks')), key }),
    ),
    lru: mixer(
      'LRU: stable complex diagonal recurrence (modulus below one by construction)',
      { constantState: true, family: 'state-space' },
      dict({ stateSize }),
      (c) => (dim, key) => new LRUMixer(dim, { ...opt('stateSize', num(c, 'stateSize')), key }),
    ),
    'linear-attention': mixer(
      'linear attention with a positive feature map',
      { constantState: true, family: 'linear-attention' },
      dict({
        heads,
        featureMap: choice(['elu', 'relu', 'exp', 'softplus'] as const, { default: 'elu', label: 'feature map' }),
        form: form3,
      }),
      (c) => (dim, key) =>
        new LinearAttentionMixer(dim, {
          ...opt('heads', num(c, 'heads')),
          ...opt('featureMap', str<'elu' | 'relu' | 'exp' | 'softplus'>(c, 'featureMap')),
          ...opt('form', str<'parallel' | 'recurrent' | 'chunked'>(c, 'form')),
          key,
        }),
    ),
    retnet: mixer(
      'RetNet: retention with a fixed decay per head (parallel, recurrent, chunkwise)',
      { constantState: true, family: 'linear-attention' },
      dict({ heads, form: form3 }),
      (c) => (dim, key) =>
        new RetNetMixer(dim, {
          ...opt('heads', num(c, 'heads')),
          ...opt('form', str<'parallel' | 'recurrent' | 'chunked'>(c, 'form')),
          key,
        }),
    ),
    gla: mixer(
      'gated linear attention: an input-dependent decay per key dimension',
      { constantState: true, family: 'linear-attention' },
      dict({ heads, form: form3 }),
      (c) => (dim, key) =>
        new GatedLinearAttentionMixer(dim, {
          ...opt('heads', num(c, 'heads')),
          ...opt('form', str<'parallel' | 'recurrent' | 'chunked'>(c, 'form')),
          key,
        }),
    ),
    deltanet: mixer(
      'DeltaNet: linear attention with the delta rule (overwrite, not only add)',
      { constantState: true, family: 'linear-attention' },
      dict({ heads }),
      (c) => (dim, key) => new DeltaNetMixer(dim, { ...opt('heads', num(c, 'heads')), key }),
    ),
    rwkv: mixer(
      'RWKV: token-shifted time mixing with the WKV recurrence, then channel mixing',
      { constantState: true, family: 'recurrent' },
      dict({}),
      () => (dim, key) => new RWKVMixer(dim, { key }),
    ),
    mingru: mixer(
      'minGRU: input-only gates, a linear recurrence run as a parallel scan',
      { constantState: true, family: 'recurrent' },
      dict({
        expand: int(1, 8, { default: 2, label: 'expansion' }),
        scan: bool({ default: true, label: 'parallel scan' }),
      }),
      (c) => (dim, key) =>
        new MinGRU(dim, { ...opt('expand', num(c, 'expand')), form: c['scan'] === false ? 'recurrent' : 'scan', key }),
    ),
    minlstm: mixer(
      'minLSTM: normalised input-only forget and input gates, a linear recurrence',
      { constantState: true, family: 'recurrent' },
      dict({
        expand: int(1, 8, { default: 2, label: 'expansion' }),
        scan: bool({ default: true, label: 'parallel scan' }),
      }),
      (c) => (dim, key) =>
        new MinLSTM(dim, { ...opt('expand', num(c, 'expand')), form: c['scan'] === false ? 'recurrent' : 'scan', key }),
    ),
    hyena: mixer(
      'Hyena: long implicit convolutions from a filter network, gated by projections',
      { constantState: false, family: 'convolution' },
      dict({ filterWidth: int(2, 256, { default: 16, label: 'filter network width' }) }),
      (c) => (dim, key) => new Hyena(dim, { ...opt('filterWidth', num(c, 'filterWidth')), key }),
    ),
    attention: mixer(
      'softmax self-attention with rotary positions (the baseline)',
      { constantState: false, family: 'attention' },
      dict({ heads }),
      (c) => (dim, key) => new AttentionMixer(dim, { ...opt('heads', num(c, 'heads')), key }),
    ),
  },
  {
    tag: 'kind',
    default: 'mamba',
    label: 'sequence mixer',
    description: 'the layer that mixes information along the sequence',
  },
)
