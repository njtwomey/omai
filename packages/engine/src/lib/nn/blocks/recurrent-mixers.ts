/**
 * Gated linear recurrences and long convolutions as sequence mixers: RWKV (token-shifted time mixing with the WKV
 * recurrence, and channel mixing), the minimal GRU and LSTM of "Were RNNs all we needed?" (gates that depend only on
 * the input, so the recurrence is linear and runs as a parallel scan) and Hyena (long implicit convolutions generated
 * by a small network, gated by projections of the input).
 */

import { foldIn, type Key, normal } from '../../random/index.ts'
import {
  add,
  concat,
  div,
  exp,
  fromStorage,
  mul,
  reshape,
  sigmoid,
  sin,
  sliceAxis,
  square,
  stack,
  sub,
  swapaxes,
  type Tensor,
  type Value,
  zeros,
} from '../../tensor/index.ts'
import { relu } from '../functional/activations.ts'
import { wkv, wkvInitialState } from '../functional/linear-recurrent.ts'
import { channelCausalConvolution, parallelScan } from '../functional/state-space.ts'
import { Linear } from '../module/layers.ts'
import { type ApplyContext, initKey, type Params } from '../module/module.ts'
import { RMSNorm } from '../module/normalisation.ts'
import { causalDepthwiseConv } from './mamba.ts'
import { type MixerInput, type MixerOutput, type MixerState, ownKey, SequenceMixer } from './mixer.ts'

const f64 = (a: ArrayLike<number>, shape: number[]): Tensor => fromStorage(Float64Array.from(a), shape, 'float64')
const full = (n: number, v: number): Tensor => f64(new Array(n).fill(v), [n])

/** The previous input of every position: the carried one first, then the inputs shifted by one. */
function shifted(x: Value, previous: Value | undefined): Value {
  const [bs = 1, l = 0, d = 1] = x.shape
  const first = previous === undefined ? zeros([bs, 1, d]) : reshape(previous, [bs, 1, d])
  return l <= 1 ? first : concat([first, sliceAxis(x, 1, 0, l - 1)], 1)
}

const lastOf = (x: Value): Value =>
  reshape(sliceAxis(x, 1, (x.shape[1] ?? 1) - 1, x.shape[1] ?? 1), [x.shape[0] ?? 1, -1])
const lerp = (x: Value, prev: Value, mix: Value): Value => add(mul(x, mix), mul(prev, sub(1, mix)))

const timeMixOwn = (dim: number): Params => ({
  mixK: full(dim, 0.5),
  mixV: full(dim, 0.5),
  mixR: full(dim, 0.5),
  // Decay rates w = exp(decay) between 0.01 (long memory) and 1 (about three steps).
  decay: f64(
    Array.from({ length: dim }, (_, c) => Math.log(0.01 * 100 ** (dim === 1 ? 0 : c / (dim - 1)))),
    [dim],
  ),
  first: full(dim, 0),
})

/**
 * RWKV time mixing (Peng et al. 2023): each channel is a decaying average of the past values, the WKV recurrence
 * (see {@link wkv}), over keys and values taken from a blend of the current and the previous input (the token shift),
 * gated by a receptance: `out = W_o (sigmoid(W_r x_r) * wkv(W_k x_k, W_v x_v))`. State: the previous input and the
 * WKV state.
 *
 * @concept state-space-models
 * @reference Peng et al. (2023), RWKV: reinventing RNNs for the transformer era.
 * @example
 * ```ts
 * import { RWKVTimeMix } from 'omai-engine/nn/blocks'
 * import { zeros } from 'omai-engine/tensor'
 * new RWKVTimeMix(6).forward(zeros([1, 4, 6])).output.shape // => [1, 4, 6]
 * ```
 */
export class RWKVTimeMix extends SequenceMixer {
  readonly kind = 'RWKVTimeMix'
  readonly dim: number

  constructor(dim: number, options: { readonly key?: Key } = {}) {
    const k = initKey('RWKVTimeMix', options.key)
    super(
      {
        key_proj: new Linear(dim, dim, { bias: false, key: foldIn(k, 0) }),
        value_proj: new Linear(dim, dim, { bias: false, key: foldIn(k, 1) }),
        receptance_proj: new Linear(dim, dim, { bias: false, key: foldIn(k, 2) }),
        output_proj: new Linear(dim, dim, { bias: false, key: foldIn(k, 3) }),
      },
      timeMixOwn(dim),
    )
    this.dim = dim
  }

  protected override ownParams(): Params {
    return timeMixOwn(this.dim)
  }

  initialState(batch: number): MixerState {
    const s = wkvInitialState(batch, this.dim)
    return { shift: zeros([batch, this.dim]), a: s.a, b: s.b, p: s.p }
  }

  protected mix(params: Params, x: Value, state: MixerState | undefined, ctx: ApplyContext): MixerOutput {
    const p = this.parts(params, ctx)
    const prev = shifted(x, state?.['shift'] as Value | undefined)
    const xk = lerp(x, prev, params['mixK'] as Value)
    const xv = lerp(x, prev, params['mixV'] as Value)
    const xr = lerp(x, prev, params['mixR'] as Value)
    const r = sigmoid(p.run<Value, Value>('receptance_proj', xr))
    const s =
      state === undefined ? undefined : { a: state['a'] as Value, b: state['b'] as Value, p: state['p'] as Value }
    const w = wkv(
      p.run<Value, Value>('key_proj', xk),
      p.run<Value, Value>('value_proj', xv),
      exp(params['decay'] as Value),
      params['first'] as Value,
      s,
    )
    const output = p.run<Value, Value>('output_proj', mul(r, w.output))
    p.done()
    return { output, state: { shift: lastOf(x), a: w.state.a, b: w.state.b, p: w.state.p } }
  }
}

/**
 * RWKV channel mixing: a feed-forward block over the token-shifted input, `sigmoid(W_r x_r) * W_v relu(W_k x_k)^2`
 * (squared ReLU), with the previous input as its only state.
 *
 * @concept state-space-models
 * @reference Peng et al. (2023), RWKV.
 * @example
 * ```ts
 * import { RWKVChannelMix } from 'omai-engine/nn/blocks'
 * import { zeros } from 'omai-engine/tensor'
 * new RWKVChannelMix(6).forward(zeros([1, 4, 6])).output.shape // => [1, 4, 6]
 * ```
 */
export class RWKVChannelMix extends SequenceMixer {
  readonly kind = 'RWKVChannelMix'
  readonly dim: number

  constructor(dim: number, options: { readonly hidden?: number; readonly key?: Key } = {}) {
    const k = initKey('RWKVChannelMix', options.key)
    const hidden = options.hidden ?? 4 * dim
    super(
      {
        key_proj: new Linear(dim, hidden, { bias: false, key: foldIn(k, 0) }),
        value_proj: new Linear(hidden, dim, { bias: false, key: foldIn(k, 1) }),
        receptance_proj: new Linear(dim, dim, { bias: false, key: foldIn(k, 2) }),
      },
      { mixK: full(dim, 0.5), mixR: full(dim, 0.5) },
    )
    this.dim = dim
  }

  protected override ownParams(): Params {
    return { mixK: full(this.dim, 0.5), mixR: full(this.dim, 0.5) }
  }

  initialState(batch: number): MixerState {
    return { shift: zeros([batch, this.dim]) }
  }

  protected mix(params: Params, x: Value, state: MixerState | undefined, ctx: ApplyContext): MixerOutput {
    const p = this.parts(params, ctx)
    const prev = shifted(x, state?.['shift'] as Value | undefined)
    const k = relu(p.run<Value, Value>('key_proj', lerp(x, prev, params['mixK'] as Value)))
    const r = sigmoid(p.run<Value, Value>('receptance_proj', lerp(x, prev, params['mixR'] as Value)))
    const output = mul(r, p.run<Value, Value>('value_proj', square(k)))
    p.done()
    return { output, state: { shift: lastOf(x) } }
  }
}

/**
 * An RWKV layer: time mixing, then channel mixing on a normalised residual stream, `d1 = TM(x)`, `d2 = CM(norm(x +
 * d1))`, output `d1 + d2` (which the surrounding residual block adds to its input). In the paper's block the
 * channel mixing reads the un-normalised residual; here it reads the normalised input of the mixer plus `d1`.
 *
 * @concept state-space-models
 * @reference Peng et al. (2023), RWKV.
 * @example
 * ```ts
 * import { RWKVMixer } from 'omai-engine/nn/blocks'
 * import { zeros } from 'omai-engine/tensor'
 * new RWKVMixer(6).forward(zeros([1, 4, 6])).output.shape // => [1, 4, 6]
 * ```
 */
export class RWKVMixer extends SequenceMixer {
  readonly kind = 'RWKV'
  readonly dim: number
  private readonly tm: RWKVTimeMix
  private readonly cm: RWKVChannelMix

  constructor(dim: number, options: { readonly hidden?: number; readonly key?: Key } = {}) {
    const k = initKey('RWKV', options.key)
    const tm = new RWKVTimeMix(dim, { key: foldIn(k, 0) })
    const cm = new RWKVChannelMix(dim, {
      ...(options.hidden === undefined ? {} : { hidden: options.hidden }),
      key: foldIn(k, 1),
    })
    super({ time_mix: tm, norm: new RMSNorm(dim, { eps: 1e-6 }), channel_mix: cm })
    this.dim = dim
    this.tm = tm
    this.cm = cm
  }

  initialState(batch: number): MixerState {
    return { time: this.tm.initialState(batch), channel: this.cm.initialState(batch) }
  }

  protected mix(params: Params, x: Value, state: MixerState | undefined, ctx: ApplyContext): MixerOutput {
    const p = this.parts(params, ctx)
    const t = p.run<MixerInput, MixerOutput>('time_mix', {
      x,
      ...(state === undefined ? {} : { state: state['time'] as MixerState }),
    })
    const h = p.run<Value, Value>('norm', add(x, t.output))
    const c = p.run<MixerInput, MixerOutput>('channel_mix', {
      x: h,
      ...(state === undefined ? {} : { state: state['channel'] as MixerState }),
    })
    p.done()
    return { output: add(t.output, c.output), state: { time: t.state, channel: c.state } }
  }
}

/** Options of {@link MinGRU} and {@link MinLSTM}. */
export interface MinRecurrentOptions {
  /** The hidden width as a multiple of the model width (default 2). */
  readonly expand?: number
  /** `scan` (default): a parallel scan over the sequence; `recurrent`: step by step. */
  readonly form?: 'scan' | 'recurrent'
  readonly key?: Key
}

/** h_t = a_t h_{t-1} + b_t over the time axis 1 by the chosen form. */
function recur(a: Value, b: Value, h0: Value, form: 'scan' | 'recurrent'): Value {
  if (form === 'scan') return parallelScan(a, b, 1, h0)
  const [bs = 1, l = 0, d = 1] = a.shape
  let h = h0
  const hs: Value[] = []
  for (let t = 0; t < l; t++) {
    const at = reshape(sliceAxis(a, 1, t, t + 1), [bs, d])
    const bt = reshape(sliceAxis(b, 1, t, t + 1), [bs, d])
    h = add(mul(at, h), bt)
    hs.push(h)
  }
  return stack(hs, 1)
}

/**
 * The minimal GRU (Feng et al. 2024): the update gate and the candidate depend on the input only,
 * `z_t = sigmoid(W_z x_t)`, `h~_t = W_h x_t`, `h_t = (1 - z_t) h_{t-1} + z_t h~_t`, so the recurrence is linear in the
 * state and runs as a parallel scan. State: the hidden state `[B, inner]`.
 *
 * @concept state-space-models
 * @reference Feng et al. (2024), Were RNNs all we needed?
 * @example
 * ```ts
 * import { MinGRU } from 'omai-engine/nn/blocks'
 * import { zeros } from 'omai-engine/tensor'
 * new MinGRU(6).forward(zeros([1, 4, 6])).output.shape // => [1, 4, 6]
 * ```
 */
export class MinGRU extends SequenceMixer {
  readonly kind = 'MinGRU'
  readonly dim: number
  readonly inner: number
  readonly form: 'scan' | 'recurrent'

  constructor(dim: number, options: MinRecurrentOptions = {}) {
    const k = initKey('MinGRU', options.key)
    const inner = (options.expand ?? 2) * dim
    super({
      in_proj: new Linear(dim, 2 * inner, { key: foldIn(k, 0) }),
      out_proj: new Linear(inner, dim, { key: foldIn(k, 1) }),
    })
    this.dim = dim
    this.inner = inner
    this.form = options.form ?? 'scan'
  }

  initialState(batch: number): MixerState {
    return { h: zeros([batch, this.inner]) }
  }

  protected mix(params: Params, x: Value, state: MixerState | undefined, ctx: ApplyContext): MixerOutput {
    const p = this.parts(params, ctx)
    const [bs = 1, l = 0] = x.shape
    const proj = p.run<Value, Value>('in_proj', x)
    const z = sigmoid(sliceAxis(proj, 2, 0, this.inner))
    const cand = sliceAxis(proj, 2, this.inner, 2 * this.inner)
    const h0 = state === undefined ? zeros([bs, this.inner]) : (state['h'] as Value)
    const h = recur(sub(1, z), mul(z, cand), h0, this.form)
    const output = p.run<Value, Value>('out_proj', h)
    p.done()
    return { output, state: { h: reshape(sliceAxis(h, 1, l - 1, l), [bs, this.inner]) } }
  }
}

/**
 * The minimal LSTM (Feng et al. 2024): forget and input gates that depend on the input only, normalised to sum to
 * one, `f'_t = f_t / (f_t + i_t)`, `i'_t = i_t / (f_t + i_t)`, `h_t = f'_t h_{t-1} + i'_t h~_t`; a linear recurrence
 * run as a parallel scan. State: the hidden state `[B, inner]`.
 *
 * @concept state-space-models
 * @reference Feng et al. (2024), Were RNNs all we needed?
 * @example
 * ```ts
 * import { MinLSTM } from 'omai-engine/nn/blocks'
 * import { zeros } from 'omai-engine/tensor'
 * new MinLSTM(6).forward(zeros([1, 4, 6])).output.shape // => [1, 4, 6]
 * ```
 */
export class MinLSTM extends SequenceMixer {
  readonly kind = 'MinLSTM'
  readonly dim: number
  readonly inner: number
  readonly form: 'scan' | 'recurrent'

  constructor(dim: number, options: MinRecurrentOptions = {}) {
    const k = initKey('MinLSTM', options.key)
    const inner = (options.expand ?? 2) * dim
    super({
      in_proj: new Linear(dim, 3 * inner, { key: foldIn(k, 0) }),
      out_proj: new Linear(inner, dim, { key: foldIn(k, 1) }),
    })
    this.dim = dim
    this.inner = inner
    this.form = options.form ?? 'scan'
  }

  initialState(batch: number): MixerState {
    return { h: zeros([batch, this.inner]) }
  }

  protected mix(params: Params, x: Value, state: MixerState | undefined, ctx: ApplyContext): MixerOutput {
    const p = this.parts(params, ctx)
    const [bs = 1, l = 0] = x.shape
    const proj = p.run<Value, Value>('in_proj', x)
    const f = sigmoid(sliceAxis(proj, 2, 0, this.inner))
    const i = sigmoid(sliceAxis(proj, 2, this.inner, 2 * this.inner))
    const cand = sliceAxis(proj, 2, 2 * this.inner, 3 * this.inner)
    const total = add(f, i)
    const h0 = state === undefined ? zeros([bs, this.inner]) : (state['h'] as Value)
    const h = recur(mul(f, reciprocal(total)), mul(mul(i, reciprocal(total)), cand), h0, this.form)
    const output = p.run<Value, Value>('out_proj', h)
    p.done()
    return { output, state: { h: reshape(sliceAxis(h, 1, l - 1, l), [bs, this.inner]) } }
  }
}

const reciprocal = (x: Value): Value => div(1, x)

/** Options of {@link Hyena}. */
export interface HyenaOptions {
  /** The taps of the short depthwise convolutions (default 3). */
  readonly shortKernel?: number
  /** The width of the filter network (default 16). */
  readonly filterWidth?: number
  /** The number of sine/cosine frequency pairs of the position features (default 3). */
  readonly bands?: number
  readonly key?: Key
}

const POSITION_PERIOD = 64

/** Position features `[T, 1 + 2 bands]` of the positions 0 .. T - 1: t / period and sin/cos of its harmonics. */
function positionFeatures(length: number, bands: number): Tensor {
  const width = 1 + 2 * bands
  const out = new Float64Array(length * width)
  for (let t = 0; t < length; t++) {
    out[t * width] = t / POSITION_PERIOD
    for (let j = 1; j <= bands; j++) {
      out[t * width + 2 * j - 1] = Math.sin((2 * Math.PI * j * t) / POSITION_PERIOD)
      out[t * width + 2 * j] = Math.cos((2 * Math.PI * j * t) / POSITION_PERIOD)
    }
  }
  return fromStorage(out, [length, width], 'float64')
}

function hyenaOwn(dim: number, shortKernel: number, key: Key): Params {
  const channels = 2 * dim
  return {
    // Window decays exp(-alpha t), alpha log-spaced from 0.02 (long filters) to 0.5.
    alpha: f64(
      Array.from({ length: channels }, (_, c) => 0.02 * 25 ** (c / Math.max(1, channels - 1))),
      [channels],
    ),
    skip: normal(foldIn(key, 0), [channels]),
    shortWeight: mul(normal(foldIn(key, 1), [3 * dim, 1, shortKernel]), 1 / Math.sqrt(shortKernel)) as Tensor,
    shortBias: zeros([3 * dim]) as Tensor,
  }
}

/**
 * The Hyena operator of order two (Poli et al. 2023): the input is projected to a value and two gates, each passed
 * through a short depthwise convolution; then `z = v`, `z <- x_1 * (h_1 conv z + b_1 z)`, `z <- x_2 * (h_2 conv z +
 * b_2 z)`, where the long filters `h_n` (as long as the sequence) are not parameters but the output of a small
 * network of the position (sine-activated, times an exponentially decaying window), so their number of parameters is
 * independent of the length. The convolution is causal; its state is the history of its inputs (the memory grows with
 * the sequence, as the operator has no finite recurrence), which is what lets generation continue with the outputs of
 * the parallel pass.
 *
 * @concept state-space-models
 * @reference Poli et al. (2023), Hyena hierarchy: towards larger convolutional language models.
 * @example
 * ```ts
 * import { Hyena } from 'omai-engine/nn/blocks'
 * import { zeros } from 'omai-engine/tensor'
 * new Hyena(6).forward(zeros([1, 5, 6])).output.shape // => [1, 5, 6]
 * ```
 */
export class Hyena extends SequenceMixer {
  readonly kind = 'Hyena'
  readonly dim: number
  readonly shortKernel: number
  readonly bands: number

  constructor(dim: number, options: HyenaOptions = {}) {
    const k = initKey('Hyena', options.key)
    const width = options.filterWidth ?? 16
    const bands = options.bands ?? 3
    const shortKernel = options.shortKernel ?? 3
    super(
      {
        in_proj: new Linear(dim, 3 * dim, { key: foldIn(k, 0) }),
        filter_in: new Linear(1 + 2 * bands, width, { key: foldIn(k, 1) }),
        filter_hidden: new Linear(width, width, { key: foldIn(k, 2) }),
        filter_out: new Linear(width, 2 * dim, { bias: false, key: foldIn(k, 3) }),
        out_proj: new Linear(dim, dim, { key: foldIn(k, 4) }),
      },
      hyenaOwn(dim, shortKernel, ownKey(k)),
    )
    this.dim = dim
    this.shortKernel = shortKernel
    this.bands = bands
  }

  protected override ownParams(key: Key): Params {
    return hyenaOwn(this.dim, this.shortKernel, key)
  }

  initialState(batch: number): MixerState {
    return {
      conv: zeros([batch, 3 * this.dim, this.shortKernel - 1]),
      history1: zeros([batch, 0, this.dim]),
      history2: zeros([batch, 0, this.dim]),
    }
  }

  /** The long filters `[2 D, T]` for the positions 0 .. T - 1 (a function of the position only). */
  filters(params: Params, length: number, ctx: ApplyContext = {}): Value {
    const p = this.parts(params, ctx)
    const pe = positionFeatures(length, this.bands)
    let h = sin(p.run<Value, Value>('filter_in', pe))
    h = sin(p.run<Value, Value>('filter_hidden', h))
    const raw = swapaxes(p.run<Value, Value>('filter_out', h), 0, 1) // [2D, T]
    const t = f64(
      Array.from({ length }, (_, i) => i),
      [1, length],
    )
    return mul(raw, exp(mul(mul(reshape(params['alpha'] as Value, [-1, 1]), t), -1)))
  }

  protected mix(params: Params, x: Value, state: MixerState | undefined, ctx: ApplyContext): MixerOutput {
    const p = this.parts(params, ctx)
    const [bs = 1, l = 0] = x.shape
    const d = this.dim
    const u = swapaxes(p.run<Value, Value>('in_proj', x), 1, 2) // [B, 3D, L]
    const short = causalDepthwiseConv(
      u,
      params['shortWeight'] as Value,
      params['shortBias'] as Value,
      state?.['conv'] as Value | undefined,
    )
    const parts = swapaxes(short.y, 1, 2) // [B, L, 3D]
    const v = sliceAxis(parts, 2, 0, d)
    const gates = [sliceAxis(parts, 2, d, 2 * d), sliceAxis(parts, 2, 2 * d, 3 * d)]
    const past1 = (state?.['history1'] as Value | undefined) ?? zeros([bs, 0, d])
    const total = (past1.shape[1] ?? 0) + l
    const filters = this.filters(params, total, ctx)
    const skip = params['skip'] as Value
    const histories: Value[] = []
    let z = v
    for (const n of [0, 1]) {
      const past = n === 0 ? past1 : ((state?.[`history${n + 1}`] as Value | undefined) ?? zeros([bs, 0, d]))
      const all = (past.shape[1] ?? 0) === 0 ? z : concat([past, z], 1)
      histories.push(all)
      const h = sliceAxis(filters, 0, n * d, (n + 1) * d)
      const conv = sliceAxis(channelCausalConvolution(all, h), 1, total - l, total)
      z = mul(gates[n] as Value, add(conv, mul(z, sliceAxis(skip, 0, n * d, (n + 1) * d))))
    }
    const output = p.run<Value, Value>('out_proj', z)
    p.done()
    return {
      output,
      state: { conv: short.window, history1: histories[0] as Value, history2: histories[1] as Value },
    }
  }
}
