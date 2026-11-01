/**
 * Linear recurrences with matrix or per-channel state, the cores of linear attention and its descendants: attention
 * whose keys and values are summed into a state that may decay (retention, gated linear attention), computed as a
 * recurrence, as masked quadratic attention or chunk by chunk, and the WKV recurrence of RWKV.
 */

import {
  add,
  concat,
  cumsum,
  div,
  einsum,
  exp,
  expandDims,
  fromStorage,
  maximum,
  mul,
  promoteFloat,
  reshape,
  sliceAxis,
  stack,
  stopGradient,
  sub,
  sum,
  type TensorLike,
  type Value,
  where,
  zeros,
} from '../../tensor/index.ts'

const fl = (x: TensorLike): Value => promoteFloat(x)[0] as Value

/** The algorithms for attention with a decaying state. */
export type LinearAttentionForm = 'recurrent' | 'parallel' | 'chunked'

/** Inputs of {@link decayedLinearAttention}. */
export interface DecayedLinearAttentionInput {
  /** Queries `[B, H, L, dk]`. */
  readonly q: TensorLike
  /** Keys `[B, H, L, dk]`. */
  readonly k: TensorLike
  /** Values `[B, H, L, dv]`. */
  readonly v: TensorLike
  /** Log-decays (<= 0): `[B, H, L]` for one decay per head and step, `[B, H, L, dk]` for one per key dimension. Default none. */
  readonly logDecay?: TensorLike
  /** The state before the first step `[B, H, dk, dv]` (default zeros). */
  readonly state?: TensorLike
}

/**
 * Attention with a decaying state: `S_t = a_t * S_{t-1} + k_t v_t^T` and `o_t = S_t^T q_t`, where the decay `a_t =
 * exp(logDecay_t)` is one number per head and step (retention, Mamba-2) or one per key dimension (gated linear
 * attention), or absent (plain linear attention). The three forms give the same outputs and final state: `recurrent`
 * steps through time; `parallel` is masked attention with scores `(q_t . k_s)` weighted by the accumulated decay
 * between s and t; `chunked` applies the parallel form inside chunks and the recurrence between them.
 *
 * @param input - q, k, v, the log-decays and the initial state.
 * @param form - The algorithm (default `parallel`).
 * @param chunk - The chunk length of the chunked form (default 8).
 * @returns The outputs `[B, H, L, dv]` and the final state `[B, H, dk, dv]`.
 * @concept state-space-models
 * @reference Sun et al. (2023), Retentive network; Yang et al. (2024), Gated linear attention transformers with
 *   hardware-efficient training.
 * @example
 * ```ts
 * import { decayedLinearAttention } from 'mdml-engine/nn'
 * import { toFlat } from 'mdml-engine/tensor'
 * const q = [[[[1], [1]]]]
 * toFlat(decayedLinearAttention({ q, k: q, v: [[[[1], [1]]]], logDecay: [[[Math.log(0.5), Math.log(0.5)]]] }).output) // => [1, 1.5]
 * ```
 */
export function decayedLinearAttention(
  input: DecayedLinearAttentionInput,
  form: LinearAttentionForm = 'parallel',
  chunk = 8,
): { readonly output: Value; readonly state: Value } {
  const q = fl(input.q)
  const k = fl(input.k)
  const v = fl(input.v)
  const [bs = 1, heads = 1, l = 0, dk = 1] = q.shape
  const dv = v.shape[3] ?? 1
  const g = input.logDecay === undefined ? undefined : fl(input.logDecay)
  const vector = g !== undefined && g.shape.length === 4
  const s0 = input.state === undefined ? zeros([bs, heads, dk, dv]) : fl(input.state)
  if (form === 'recurrent') {
    let s = s0
    const outs: Value[] = []
    const step = (x: Value, t: number, shape: number[]): Value => reshape(sliceAxis(x, 2, t, t + 1), shape)
    for (let t = 0; t < l; t++) {
      const write = mul(expandDims(step(k, t, [bs, heads, dk]), 3), expandDims(step(v, t, [bs, heads, dv]), 2))
      const decayed =
        g === undefined
          ? s
          : mul(exp(vector ? expandDims(step(g, t, [bs, heads, dk]), 3) : step(g, t, [bs, heads, 1, 1])), s)
      s = add(decayed, write)
      outs.push(sum(mul(expandDims(step(q, t, [bs, heads, dk]), 3), s), 2))
    }
    return { output: stack(outs, 2), state: s }
  }
  if (form === 'chunked' && l > chunk) {
    let s = s0
    const outs: Value[] = []
    for (let lo = 0; lo < l; lo += chunk) {
      const hi = Math.min(l, lo + chunk)
      const cut = (x: Value): Value => sliceAxis(x, 2, lo, hi)
      const r = decayedLinearAttention(
        { q: cut(q), k: cut(k), v: cut(v), ...(g === undefined ? {} : { logDecay: cut(g) }), state: s },
        'parallel',
      )
      outs.push(r.output)
      s = r.state
    }
    return { output: concat(outs, 2), state: s }
  }
  const lower = new Uint8Array(l * l)
  for (let t = 0; t < l; t++) for (let s = 0; s <= t; s++) lower[t * l + s] = 1
  const mask = fromStorage(lower, [l, l], 'bool')
  if (g === undefined) {
    const scores = where(mask, einsum('bhtd,bhsd->bhts', q, k), 0)
    return {
      output: add(einsum('bhts,bhsv->bhtv', scores, v), einsum('bhtd,bhdv->bhtv', q, s0)),
      state: add(s0, einsum('bhsd,bhsv->bhdv', k, v)),
    }
  }
  const cum = cumsum(g, 2) // [B, H, L(, dk)]
  const last = sliceAxis(cum, 2, l - 1, l)
  if (!vector) {
    const diff = sub(expandDims(cum, 3), expandDims(cum, 2)) // [B, H, T, S]
    const decay = where(mask, exp(where(mask, diff, 0)), 0)
    const scores = mul(einsum('bhtd,bhsd->bhts', q, k), decay)
    const fromState = mul(einsum('bhtd,bhdv->bhtv', q, s0), expandDims(exp(cum), 3))
    const weight = exp(sub(last, cum)) // [B, H, L]
    return {
      output: add(einsum('bhts,bhsv->bhtv', scores, v), fromState),
      state: add(mul(reshape(exp(last), [bs, heads, 1, 1]), s0), einsum('bhs,bhsd,bhsv->bhdv', weight, k, v)),
    }
  }
  const m5 = reshape(mask, [1, 1, l, l, 1])
  const diff = sub(expandDims(cum, 3), expandDims(cum, 2)) // [B, H, T, S, dk]
  const decay = where(m5, exp(where(m5, diff, 0)), 0)
  const scores = einsum('bhtd,bhsd,bhtsd->bhts', q, k, decay)
  const fromState = einsum('bhtd,bhdv->bhtv', mul(q, exp(cum)), s0)
  const weight = exp(sub(last, cum)) // [B, H, L, dk]
  return {
    output: add(einsum('bhts,bhsv->bhtv', scores, v), fromState),
    state: add(mul(reshape(exp(last), [bs, heads, dk, 1]), s0), einsum('bhsd,bhsd,bhsv->bhdv', weight, k, v)),
  }
}

/** The state of {@link wkv}: the running numerator `a`, denominator `b` and exponent shift `p`, each `[B, D]`. */
export interface WkvState {
  readonly a: Value
  readonly b: Value
  readonly p: Value
}

/** The state before anything is read. */
export function wkvInitialState(batch: number, width: number): WkvState {
  return {
    a: zeros([batch, width]),
    b: zeros([batch, width]),
    p: fromStorage(new Float64Array(batch * width).fill(-1e30), [batch, width], 'float64'),
  }
}

/**
 * The WKV recurrence of RWKV (Peng et al. 2023): per channel, a softmax-like average of the past values with weights
 * `exp(k_i - (t - 1 - i) w)` that decay exponentially with age at rate `w >= 0`, plus a bonus `u` for the current
 * token: `wkv_t = (sum_{i<t} e^{-(t-1-i)w + k_i} v_i + e^{u + k_t} v_t) / (sum_{i<t} e^{-(t-1-i)w + k_i} + e^{u + k_t})`.
 * Computed with a running exponent shift so that it does not overflow.
 *
 * @param k - Keys `[B, L, D]`.
 * @param v - Values `[B, L, D]`.
 * @param w - Decay rates `[D]` (non-negative).
 * @param u - The bonus for the current token `[D]`.
 * @param state - The state before the first step (default {@link wkvInitialState}).
 * @returns The outputs `[B, L, D]` and the final state.
 * @concept state-space-models
 * @reference Peng et al. (2023), RWKV: reinventing RNNs for the transformer era.
 * @example
 * ```ts
 * import { wkv } from 'mdml-engine/nn'
 * import { toFlat } from 'mdml-engine/tensor'
 * toFlat(wkv([[[0], [0]]], [[[1], [3]]], [0], [0]).output) // => [1, 2]
 * ```
 */
export function wkv(
  k: TensorLike,
  v: TensorLike,
  w: TensorLike,
  u: TensorLike,
  state?: WkvState,
): { readonly output: Value; readonly state: WkvState } {
  const kv = fl(k)
  const vv = fl(v)
  const wv = fl(w)
  const uv = fl(u)
  const [bs = 1, l = 0, d = 1] = kv.shape
  let { a, b, p } = state ?? wkvInitialState(bs, d)
  const outs: Value[] = []
  const at = (x: Value, t: number): Value => reshape(sliceAxis(x, 1, t, t + 1), [bs, d])
  for (let t = 0; t < l; t++) {
    const kt = at(kv, t)
    const vt = at(vv, t)
    const bonus = add(kt, uv)
    // Shifts only rescale numerator and denominator together, so they carry no gradient.
    const q1 = stopGradient(maximum(p, bonus))
    const e1 = exp(sub(p, q1))
    const e2 = exp(sub(bonus, q1))
    outs.push(div(add(mul(e1, a), mul(e2, vt)), add(mul(e1, b), e2)))
    const pd = sub(p, wv)
    const q2 = stopGradient(maximum(pd, kt))
    const f1 = exp(sub(pd, q2))
    const f2 = exp(sub(kt, q2))
    a = add(mul(f1, a), mul(f2, vt))
    b = add(mul(f1, b), f2)
    p = q2
  }
  return { output: stack(outs, 1), state: { a, b, p } }
}
