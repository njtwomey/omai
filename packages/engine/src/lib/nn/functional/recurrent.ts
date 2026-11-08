/**
 * Recurrent networks as functions: single steps of the Elman RNN, LSTM and GRU cells with PyTorch's weight names, a
 * JAX-style `scan` over time, `recurrent` for stacked and bidirectional networks over whole sequences (the computation
 * of torch.nn.RNN, LSTM and GRU), and packing and unpacking of variable-length batches.
 */

import { OmaiError } from '../../base/index.ts'
import { foldIn, type Key } from '../../random/index.ts'
import {
  add,
  asTensor,
  concat,
  fromStorage,
  matmul,
  mul,
  promoteFloat,
  relu,
  reshape,
  sigmoid,
  sliceAxis,
  stack,
  sub,
  swapaxes,
  type Tensor,
  type TensorLike,
  take,
  takeAlongAxis,
  tanh,
  toFlat,
  type Value,
  where,
  zeros,
} from '../../tensor/index.ts'
import { dropout } from './regularisation.ts'

const fl = (x: TensorLike): Value => promoteFloat(x)[0] as Value

/** The weights of one recurrent cell (PyTorch's names without the layer suffix). */
export interface CellParams {
  /** `[gates H, I]`. */
  readonly weightIh: Value
  /** `[gates H, H]` (or `[gates H, proj]` with a projection). */
  readonly weightHh: Value
  readonly biasIh?: Value
  readonly biasHh?: Value
  /** LSTM output projection `[proj, H]` (proj_size). */
  readonly weightHr?: Value
}

const affine = (x: Value, w: Value, b: Value | undefined): Value => {
  const y = matmul(x, swapaxes(w, 0, 1))
  return b === undefined ? y : add(y, b)
}

/** The gate pre-activations from the input part (precomputed for a whole sequence) and the hidden part. */
const hiddenPart = (h: Value, p: CellParams): Value => affine(h, p.weightHh, p.biasHh)

/**
 * One step of an Elman RNN (torch.nn.RNNCell): h' = act(x W_ih^T + b_ih + h W_hh^T + b_hh).
 *
 * @param x - Input `[B, I]`.
 * @param h - Hidden state `[B, H]`.
 * @param params - The cell's weights.
 * @param nonlinearity - `tanh` (default) or `relu`.
 * @returns The next hidden state.
 * @concept recurrent-networks
 * @example
 * ```ts
 * import { rnnCell } from 'omai-engine/nn'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(rnnCell([[1]], [[0]], { weightIh: [[2]], weightHh: [[1]] }, 'relu')) // => [[2]]
 * ```
 */
export function rnnCell(
  x: TensorLike,
  h: TensorLike,
  params: {
    readonly weightIh: TensorLike
    readonly weightHh: TensorLike
    readonly biasIh?: TensorLike
    readonly biasHh?: TensorLike
  },
  nonlinearity: 'tanh' | 'relu' = 'tanh',
): Value {
  const p = cellParams(params)
  const pre = add(affine(fl(x), p.weightIh, p.biasIh), hiddenPart(fl(h), p))
  return nonlinearity === 'relu' ? relu(pre) : tanh(pre)
}

const cellParams = (params: {
  readonly weightIh: TensorLike
  readonly weightHh: TensorLike
  readonly biasIh?: TensorLike
  readonly biasHh?: TensorLike
  readonly weightHr?: TensorLike
}): CellParams => ({
  weightIh: fl(params.weightIh),
  weightHh: fl(params.weightHh),
  ...(params.biasIh === undefined ? {} : { biasIh: fl(params.biasIh) }),
  ...(params.biasHh === undefined ? {} : { biasHh: fl(params.biasHh) }),
  ...(params.weightHr === undefined ? {} : { weightHr: fl(params.weightHr) }),
})

const gate = (z: Value, i: number, h: number): Value => sliceAxis(z, -1, i * h, (i + 1) * h)

/** An LSTM step from the precomputed input part `xi` (`[B, 4H]`). */
function lstmStep(xi: Value, h: Value, c: Value, p: CellParams): [Value, Value] {
  const hidden = p.weightIh.shape[0] === undefined ? 0 : (p.weightIh.shape[0] as number) / 4
  const z = add(xi, hiddenPart(h, p))
  const i = sigmoid(gate(z, 0, hidden))
  const f = sigmoid(gate(z, 1, hidden))
  const g = tanh(gate(z, 2, hidden))
  const o = sigmoid(gate(z, 3, hidden))
  const c2 = add(mul(f, c), mul(i, g))
  let h2 = mul(o, tanh(c2))
  if (p.weightHr !== undefined) h2 = matmul(h2, swapaxes(p.weightHr, 0, 1))
  return [h2, c2]
}

/** A GRU step from the precomputed input part `xi` (`[B, 3H]`, including b_ih). */
function gruStep(xi: Value, h: Value, p: CellParams): Value {
  const hidden = (p.weightIh.shape[0] ?? 0) / 3
  const hh = hiddenPart(h, p)
  const r = sigmoid(add(gate(xi, 0, hidden), gate(hh, 0, hidden)))
  const z = sigmoid(add(gate(xi, 1, hidden), gate(hh, 1, hidden)))
  const n = tanh(add(gate(xi, 2, hidden), mul(r, gate(hh, 2, hidden))))
  // h' = (1 - z) n + z h
  return add(n, mul(z, sub(h, n)))
}

/**
 * One step of an LSTM (torch.nn.LSTMCell, Hochreiter and Schmidhuber 1997): input, forget, cell and output gates in
 * PyTorch's order; c' = f c + i g, h' = o tanh(c'), projected by W_hr when given (proj_size).
 *
 * @param x - Input `[B, I]`.
 * @param state - `[h, c]`, each `[B, H]` (h `[B, proj]` with a projection).
 * @param params - The cell's weights (`weightIh` `[4H, I]`, `weightHh` `[4H, H]`, biases, optional `weightHr`).
 * @returns The next `[h, c]`.
 * @concept recurrent-networks
 * @example
 * ```ts
 * import { lstmCell } from 'omai-engine/nn'
 * import { zeros } from 'omai-engine/tensor'
 * lstmCell(zeros([2, 3]), [zeros([2, 4]), zeros([2, 4])], { weightIh: zeros([16, 3]), weightHh: zeros([16, 4]) })[1].shape // => [2, 4]
 * ```
 */
export function lstmCell(
  x: TensorLike,
  state: readonly [TensorLike, TensorLike],
  params: {
    readonly weightIh: TensorLike
    readonly weightHh: TensorLike
    readonly biasIh?: TensorLike
    readonly biasHh?: TensorLike
    readonly weightHr?: TensorLike
  },
): [Value, Value] {
  const p = cellParams(params)
  return lstmStep(affine(fl(x), p.weightIh, p.biasIh), fl(state[0]), fl(state[1]), p)
}

/**
 * One step of a GRU (torch.nn.GRUCell, Cho et al. 2014): reset and update gates, n = tanh(W_in x + b_in + r (W_hn h +
 * b_hn)), h' = (1 - z) n + z h, in PyTorch's gate order (r, z, n).
 *
 * @param x - Input `[B, I]`.
 * @param h - Hidden state `[B, H]`.
 * @param params - The cell's weights (`weightIh` `[3H, I]`, `weightHh` `[3H, H]`, biases).
 * @returns The next hidden state.
 * @concept recurrent-networks
 * @example
 * ```ts
 * import { gruCell } from 'omai-engine/nn'
 * import { toArray, zeros } from 'omai-engine/tensor'
 * toArray(gruCell(zeros([1, 2]), [[1]], { weightIh: zeros([3, 2]), weightHh: zeros([3, 1]) })) // => [[0.5]]
 * ```
 */
export function gruCell(
  x: TensorLike,
  h: TensorLike,
  params: {
    readonly weightIh: TensorLike
    readonly weightHh: TensorLike
    readonly biasIh?: TensorLike
    readonly biasHh?: TensorLike
  },
): Value {
  const p = cellParams(params)
  return gruStep(affine(fl(x), p.weightIh, p.biasIh), fl(h), p)
}

/** Options of {@link scan}. */
export interface ScanOptions {
  /** Run from the last step to the first. */
  readonly reverse?: boolean
}

/**
 * Scan a step function over the leading (time) axis (jax.lax.scan): carry' , y_t = step(carry, x_t), collecting the
 * outputs. The recurrent layers are scans; so are any custom recurrences.
 *
 * @param step - `(carry, x_t, t) => [carry', y_t]`.
 * @param init - The initial carry.
 * @param xs - Inputs `[T, ...]`.
 * @param options - `reverse`.
 * @returns The final carry and the stacked outputs `[T, ...]` (in time order also when reversed).
 * @concept recurrent-networks
 * @example
 * ```ts
 * import { scan } from 'omai-engine/nn'
 * import { add, toArray } from 'omai-engine/tensor'
 * toArray(scan((c, x) => { const s = add(c, x); return [s, s] }, [0], [[1], [2], [3]]).ys) // => [[1], [3], [6]]
 * ```
 */
export function scan<C>(
  step: (carry: C, x: Value, t: number) => readonly [C, Value],
  init: C,
  xs: TensorLike,
  options: ScanOptions = {},
): { readonly carry: C; readonly ys: Value } {
  const x = fl(xs)
  const t = x.shape[0] ?? 0
  let carry = init
  const ys: Value[] = new Array<Value>(t)
  for (let s = 0; s < t; s++) {
    const i = options.reverse === true ? t - 1 - s : s
    const [next, y] = step(carry, reshape(sliceAxis(x, 0, i, i + 1), x.shape.slice(1)), i)
    carry = next
    ys[i] = y
  }
  return { carry, ys: stack(ys, 0) }
}

/** The recurrent cell kinds. */
export type RecurrentMode = 'RNN_TANH' | 'RNN_RELU' | 'LSTM' | 'GRU'

/** One direction of one layer: its cell weights. */
export type LayerParams = CellParams

/** Options of {@link recurrent}. */
export interface RecurrentOptions {
  readonly mode: RecurrentMode
  readonly numLayers?: number
  readonly bidirectional?: boolean
  /** `[B, T, F]` input and output (default false: `[T, B, F]`). */
  readonly batchFirst?: boolean
  /** Valid lengths per sequence (the packed-sequence semantics): later steps neither update the state nor output. */
  readonly lengths?: readonly number[]
  /** Dropout on the outputs of every layer but the last (needs `key` when training). */
  readonly dropout?: number
  readonly key?: Key
  readonly training?: boolean
}

/** Result of {@link recurrent}. */
export interface RecurrentResult {
  /** `[T, B, D H]` (or batch first), zero after each sequence's length. */
  readonly output: Value
  /** Final hidden states `[D L, B, H]`. */
  readonly h: Value
  /** Final cell states `[D L, B, H]` (LSTM only). */
  readonly c?: Value
}

/** Per batch element, the time index visited at step t when running a sequence of length len backwards. */
function reverseIndex(t: number, b: number, f: number, lengths: readonly number[]): Tensor {
  const idx = new Int32Array(t * b * f)
  for (let s = 0; s < t; s++) {
    for (let n = 0; n < b; n++) {
      const len = lengths[n] ?? t
      const src = s < len ? len - 1 - s : s
      idx.fill(src, (s * b + n) * f, (s * b + n + 1) * f)
    }
  }
  return fromStorage(idx, [t, b, f], 'int32')
}

const reverseTime = (x: Value, lengths: readonly number[]): Value => {
  const [t = 0, b = 0, f = 0] = x.shape
  return takeAlongAxis(x, reverseIndex(t, b, f, lengths), 0)
}

/** Run one direction of one layer over `[T, B, F]`. */
function runDirection(
  mode: RecurrentMode,
  x: Value,
  p: CellParams,
  h0: Value,
  c0: Value | undefined,
  lengths: readonly number[] | undefined,
): { ys: Value; h: Value; c?: Value } {
  const [t = 0, b = 0] = x.shape
  const xi = affine(x, p.weightIh, p.biasIh)
  const masks = lengths
  let h = h0
  let c = c0
  const ys: Value[] = []
  for (let s = 0; s < t; s++) {
    const xt = reshape(sliceAxis(xi, 0, s, s + 1), [b, -1])
    let hn: Value
    let cn: Value | undefined
    if (mode === 'LSTM') [hn, cn] = lstmStep(xt, h, c as Value, p)
    else if (mode === 'GRU') hn = gruStep(xt, h, p)
    else {
      const pre = add(xt, hiddenPart(h, p))
      hn = mode === 'RNN_RELU' ? relu(pre) : tanh(pre)
    }
    if (masks === undefined) {
      h = hn
      if (cn !== undefined) c = cn
      ys.push(hn)
    } else {
      const active = fromStorage(
        Uint8Array.from(masks, (len) => (s < len ? 1 : 0)),
        [b, 1],
        'bool',
      )
      h = where(active, hn, h)
      if (cn !== undefined) c = where(active, cn, c as Value)
      ys.push(where(active, hn, 0))
    }
  }
  return { ys: stack(ys, 0), h, ...(c === undefined ? {} : { c }) }
}

/**
 * A stacked, optionally bidirectional recurrent network over a whole sequence (the computation of torch.nn.RNN, LSTM
 * and GRU): each layer runs its cell forward (and backward) in time, the next layer reads the concatenated outputs,
 * with dropout between layers and per-sequence lengths for padded batches (packed-sequence semantics: the backward
 * direction starts at each sequence's own last step).
 *
 * @param input - `[T, B, I]` (or `[B, T, I]` with batchFirst).
 * @param layers - Weights per layer and direction: `layers[l][d]` (d = 0 forward, 1 backward).
 * @param options - Mode, layers, bidirectional, batchFirst, lengths, dropout.
 * @param h0 - Initial hidden states `[D L, B, H]` (default zeros).
 * @param c0 - Initial cell states for an LSTM.
 * @returns The output and final states.
 * @concept recurrent-networks
 * @example
 * ```ts
 * import { recurrent } from 'omai-engine/nn'
 * import { zeros } from 'omai-engine/tensor'
 * const cell = { weightIh: zeros([2, 3]), weightHh: zeros([2, 2]) }
 * recurrent(zeros([5, 4, 3]), [[cell, cell]], { mode: 'RNN_TANH', bidirectional: true }).output.shape // => [5, 4, 4]
 * ```
 */
export function recurrent(
  input: TensorLike,
  layers: readonly (readonly CellParams[])[],
  options: RecurrentOptions,
  h0?: TensorLike,
  c0?: TensorLike,
): RecurrentResult {
  let x = fl(input)
  if (options.batchFirst === true) x = swapaxes(x, 0, 1)
  const [t = 0, b = 0] = x.shape
  const dirs = options.bidirectional === true ? 2 : 1
  const numLayers = options.numLayers ?? layers.length
  if (layers.length !== numLayers || layers.some((l) => l.length !== dirs))
    throw new OmaiError('shape', 'recurrent: need weights for every layer and direction')
  if (
    options.lengths !== undefined &&
    (options.lengths.length !== b || options.lengths.some((len) => len < 0 || len > t))
  )
    throw new OmaiError('value', 'recurrent: one length in [0, T] per sequence')
  const lstm = options.mode === 'LSTM'
  const hiddenOf = (p: CellParams): number => (p.weightHh.shape[0] ?? 0) / (lstm ? 4 : options.mode === 'GRU' ? 3 : 1)
  const outOf = (p: CellParams): number => (p.weightHr === undefined ? hiddenOf(p) : (p.weightHr.shape[0] ?? 0))
  const h0v = h0 === undefined ? undefined : fl(h0)
  const c0v = c0 === undefined ? undefined : fl(c0)
  const hs: Value[] = []
  const cs: Value[] = []
  for (let l = 0; l < numLayers; l++) {
    const outs: Value[] = []
    for (let d = 0; d < dirs; d++) {
      const p = (layers[l] as readonly CellParams[])[d] as CellParams
      const k = l * dirs + d
      const hInit = h0v === undefined ? zeros([b, outOf(p)]) : reshape(sliceAxis(h0v, 0, k, k + 1), [b, -1])
      const cInit = !lstm
        ? undefined
        : c0v === undefined
          ? zeros([b, hiddenOf(p)])
          : reshape(sliceAxis(c0v, 0, k, k + 1), [b, -1])
      const lengths = options.lengths ?? new Array<number>(b).fill(t)
      const xin = d === 1 ? reverseTime(x, lengths) : x
      const r = runDirection(options.mode, xin, p, hInit, cInit, options.lengths)
      outs.push(d === 1 ? reverseTime(r.ys, lengths) : r.ys)
      hs.push(r.h)
      if (r.c !== undefined) cs.push(r.c)
    }
    x = dirs === 1 ? (outs[0] as Value) : concat(outs, 2)
    if (l + 1 < numLayers && (options.dropout ?? 0) > 0 && options.training !== false) {
      x = dropout(x, options.dropout, {
        training: true,
        ...(options.key === undefined ? {} : { key: foldIn(options.key, l) }),
      })
    }
  }
  const output = options.batchFirst === true ? swapaxes(x, 0, 1) : x
  return { output, h: stack(hs, 0), ...(lstm ? { c: stack(cs, 0) } : {}) }
}

/** A packed batch of variable-length sequences (torch.nn.utils.rnn.PackedSequence). */
export interface PackedSequence {
  /** `[sum of lengths, ...]`: time step 0 of every sequence, then step 1 of those still running, and so on. */
  readonly data: Value
  /** The number of sequences still running at each step. */
  readonly batchSizes: readonly number[]
  /** The original batch index of each sorted position (longest first). */
  readonly sortedIndices: readonly number[]
  /** The sorted position of each original batch index. */
  readonly unsortedIndices: readonly number[]
}

/** Options of {@link packPaddedSequence}. */
export interface PackOptions {
  readonly batchFirst?: boolean
  /** Require lengths already in decreasing order (default false: sort them). */
  readonly enforceSorted?: boolean
}

/**
 * Pack a padded batch (torch.nn.utils.rnn.pack_padded_sequence): sort by length, longest first, and keep only the
 * valid steps, time-major. A gather, so differentiable in the data.
 *
 * @param padded - `[T, B, ...]` (or `[B, T, ...]` with batchFirst).
 * @param lengths - The valid length of each sequence.
 * @param options - `batchFirst` and `enforceSorted`.
 * @returns The packed sequence.
 * @example
 * ```ts
 * import { packPaddedSequence } from 'omai-engine/nn'
 * packPaddedSequence([[[1], [4]], [[2], [5]], [[3], [0]]], [3, 2]).batchSizes // => [2, 2, 1]
 * ```
 */
export function packPaddedSequence(
  padded: TensorLike,
  lengths: readonly number[],
  options: PackOptions = {},
): PackedSequence {
  let x = fl(padded)
  if (options.batchFirst === true) x = swapaxes(x, 0, 1)
  const [t = 0, b = 0, ...rest] = x.shape
  if (lengths.length !== b) throw new OmaiError('shape', 'packPaddedSequence: one length per sequence')
  const order = Array.from({ length: b }, (_, i) => i).sort((i, j) => (lengths[j] ?? 0) - (lengths[i] ?? 0) || i - j)
  if (options.enforceSorted === true && order.some((v, i) => v !== i))
    throw new OmaiError('value', 'packPaddedSequence: lengths are not sorted in decreasing order')
  const maxLen = lengths[order[0] ?? 0] ?? 0
  if (maxLen > t) throw new OmaiError('value', 'packPaddedSequence: a length exceeds the time axis')
  const rows: number[] = []
  const batchSizes: number[] = []
  for (let s = 0; s < maxLen; s++) {
    let count = 0
    for (const i of order) {
      if ((lengths[i] ?? 0) > s) {
        rows.push(s * b + i)
        count += 1
      }
    }
    batchSizes.push(count)
  }
  const flat = reshape(x, [t * b, ...rest])
  const unsorted = new Array<number>(b)
  for (const [pos, i] of order.entries()) unsorted[i] = pos
  return {
    data: take(flat, fromStorage(Int32Array.from(rows), [rows.length], 'int32'), 0),
    batchSizes,
    sortedIndices: order,
    unsortedIndices: unsorted,
  }
}

/**
 * The lengths of a packed batch, in the original order.
 *
 * @param packed - A packed sequence.
 * @returns One length per sequence.
 * @example
 * ```ts
 * import { packPaddedSequence, packedLengths } from 'omai-engine/nn'
 * packedLengths(packPaddedSequence([[[1], [4]], [[2], [5]]], [1, 2])) // => [1, 2]
 * ```
 */
export function packedLengths(packed: PackedSequence): number[] {
  const sorted = packed.sortedIndices.map((_, pos) => packed.batchSizes.filter((n) => n > pos).length)
  return packed.unsortedIndices.map((pos) => sorted[pos] ?? 0)
}

/**
 * Unpack to a padded batch (torch.nn.utils.rnn.pad_packed_sequence), in the original order.
 *
 * @param packed - A packed sequence.
 * @param options - `batchFirst`, the padding value (default 0) and a total length (default the longest).
 * @returns The padded batch and the lengths.
 * @example
 * ```ts
 * import { packPaddedSequence, padPackedSequence } from 'omai-engine/nn'
 * padPackedSequence(packPaddedSequence([[[1], [4]], [[2], [5]], [[3], [0]]], [3, 2])).lengths // => [3, 2]
 * ```
 */
export function padPackedSequence(
  packed: PackedSequence,
  options: { readonly batchFirst?: boolean; readonly paddingValue?: number; readonly totalLength?: number } = {},
): { readonly padded: Value; readonly lengths: number[] } {
  const lengths = packedLengths(packed)
  const b = lengths.length
  const t = options.totalLength ?? packed.batchSizes.length
  const rest = packed.data.shape.slice(1)
  // Gather from the packed rows plus one padding row at the end.
  const pad = add(zeros([1, ...rest]), options.paddingValue ?? 0)
  const source = concat([packed.data, pad], 0)
  const padRow = packed.data.shape[0] ?? 0
  const offsets: number[] = []
  let at = 0
  for (const n of packed.batchSizes) {
    offsets.push(at)
    at += n
  }
  const rows = new Int32Array(t * b)
  for (let s = 0; s < t; s++) {
    for (let i = 0; i < b; i++) {
      const pos = packed.unsortedIndices[i] ?? 0
      rows[s * b + i] = s < (lengths[i] ?? 0) ? (offsets[s] ?? 0) + pos : padRow
    }
  }
  let out = reshape(take(source, fromStorage(rows, [rows.length], 'int32'), 0), [t, b, ...rest])
  if (options.batchFirst === true) out = swapaxes(out, 0, 1)
  return { padded: out, lengths }
}

/**
 * The valid lengths of a padded batch from a boolean or 0/1 mask `[B, T]` (true = valid): one past the last valid
 * step of each row.
 *
 * @param mask - `[B, T]`.
 * @returns One length per row.
 * @example
 * ```ts
 * import { lengthsFromMask } from 'omai-engine/nn'
 * lengthsFromMask([[1, 1, 0], [1, 0, 0]]) // => [2, 1]
 * ```
 */
export function lengthsFromMask(mask: TensorLike): number[] {
  const m = asTensor(mask)
  const [b = 0, t = 0] = m.shape
  const v = toFlat(m)
  return Array.from({ length: b }, (_, i) => {
    let n = 0
    for (let s = 0; s < t; s++) if ((v[i * t + s] ?? 0) !== 0) n = s + 1
    return n
  })
}
