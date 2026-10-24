import { WhiteboxError } from '../../base/index.ts'
import {
  add,
  concat,
  fromStorage,
  matmul,
  mul,
  reshape,
  scatterAdd,
  sliceAxis,
  type TensorLike,
  take,
  type Value,
} from '../../tensor/index.ts'
import { type ExtensionMode, extendRows } from '../_extend.ts'
import { constant, float, indexTensor, rowsOf } from '../_shared.ts'
import { type DiscreteWavelet, dwtMaxLevel, type WaveletSpec, wavelet } from './families.ts'

// The discrete wavelet transform as PyWavelets: single-level dwt and idwt with pywt's signal extension modes,
// multilevel wavedec and waverec, and their two-dimensional (separable) counterparts. Coefficients are gathers of
// the extended signal times the filters, so every transform is differentiable in the data.

/** Signal extension of the DWT (pywt.Modes). */
export type WaveletMode =
  | 'zero'
  | 'constant'
  | 'symmetric'
  | 'reflect'
  | 'periodic'
  | 'smooth'
  | 'antisymmetric'
  | 'antireflect'
  | 'periodization'

const EXTENSION: Readonly<Record<Exclude<WaveletMode, 'periodization'>, ExtensionMode>> = {
  zero: 'zero',
  constant: 'edge',
  symmetric: 'symmetric',
  reflect: 'reflect',
  periodic: 'wrap',
  smooth: 'smooth',
  antisymmetric: 'antisymmetric',
  antireflect: 'antireflect',
}

/** Options shared by the transforms. */
export interface DwtOptions {
  /** Extension mode (default 'symmetric'). */
  readonly mode?: WaveletMode
  readonly axis?: number
}

/** Approximation and detail coefficients of one level. */
export interface DwtCoefficients {
  readonly cA: Value
  readonly cD: Value
}

/** cA and cD of rows `[batch, n]`. */
function dwtRows(rows: Value, w: DiscreteWavelet, mode: WaveletMode): [Value, Value] {
  const n = rows.shape[1] ?? 0
  const f = w.decLo.length
  const filters = constant(
    Array.from({ length: f * 2 }, (_, i) =>
      i % 2 === 0 ? (w.decLo[Math.floor(i / 2)] ?? 0) : (w.decHi[Math.floor(i / 2)] ?? 0),
    ),
    [f, 2],
  )
  let source: Value
  const idx: number[] = []
  let len: number
  if (mode === 'periodization') {
    source = n % 2 === 1 ? concat([rows, sliceAxis(rows, 1, n - 1, n)], 1) : rows
    const m = source.shape[1] ?? 0
    len = m / 2
    for (let i = 0; i < len; i++) for (let j = 0; j < f; j++) idx.push((((2 * i + f / 2 - j) % m) + m) % m)
  } else {
    source = extendRows(rows, f - 1, EXTENSION[mode])
    len = Math.floor((n + f - 1) / 2)
    for (let i = 0; i < len; i++) for (let j = 0; j < f; j++) idx.push(2 * i + f - j)
  }
  const batch = rows.shape[0] ?? 1
  const windows = reshape(take(source, indexTensor(idx), 1), [batch, len, f])
  const out = matmul(windows, filters)
  return [reshape(sliceAxis(out, 2, 0, 1), [batch, len]), reshape(sliceAxis(out, 2, 1, 2), [batch, len])]
}

/** The inverse of one level for rows: cA and cD `[batch, k]`. */
function idwtRows(ca: Value, cd: Value, w: DiscreteWavelet, mode: WaveletMode): Value {
  const [batch = 1, k = 0] = ca.shape
  const f = w.recLo.length
  const out = mode === 'periodization' ? 2 * k : 2 * k - f + 2
  if (out < 1) throw new WhiteboxError('value', 'idwt: too few coefficients for this wavelet')
  const idx = new Int32Array(batch * k * f)
  for (let b = 0; b < batch; b++) {
    for (let i = 0; i < k; i++) {
      for (let j = 0; j < f; j++) {
        const p = mode === 'periodization' ? (((2 * i + j + 1 - f / 2) % out) + out) % out : 2 * i + j - (f - 2)
        idx[(b * k + i) * f + j] = p >= 0 && p < out ? p : out
      }
    }
  }
  const lo = mul(reshape(ca, [batch, k, 1]), constant(w.recLo, [1, 1, f]))
  const hi = mul(reshape(cd, [batch, k, 1]), constant(w.recHi, [1, 1, f]))
  const updates = reshape(add(lo, hi), [batch, k * f])
  const indices = fromStorage(idx, [batch, k * f], 'int32')
  const summed = scatterAdd([batch, out + 1], indices, updates, 1)
  return sliceAxis(summed, 1, 0, out)
}

/**
 * The single-level discrete wavelet transform (pywt.dwt): convolve the extended signal with the decomposition
 * filters and keep every second sample, giving floor((n + F - 1) / 2) coefficients (ceil(n / 2) for
 * 'periodization'). Differentiable.
 *
 * @param x - Data.
 * @param w - The wavelet.
 * @param options - Extension mode (default 'symmetric') and axis (default -1).
 * @returns The approximation and detail coefficients.
 * @concept discrete-wavelet-transform
 * @example
 * ```ts
 * import { dwt } from 'whitebox-engine/signal'
 * import { toFlat } from 'whitebox-engine/tensor'
 * toFlat(dwt([1, 1, 2, 2], 'haar').cA).map((v) => Math.round(v * 1e12) / 1e12) // => [1.414213562373, 2.828427124746]
 * ```
 */
export function dwt(x: TensorLike, w: WaveletSpec, options: DwtOptions = {}): DwtCoefficients {
  const r = rowsOf(float(x), options.axis ?? -1, 'dwt')
  const [ca, cd] = dwtRows(r.rows, wavelet(w), options.mode ?? 'symmetric')
  return { cA: r.restore(ca), cD: r.restore(cd) }
}

/**
 * The single-level inverse DWT (pywt.idwt): upsample, convolve with the reconstruction filters and keep the central
 * 2 k - F + 2 samples (2 k for 'periodization').
 *
 * @param ca - Approximation coefficients.
 * @param cd - Detail coefficients (same shape).
 * @param w - The wavelet.
 * @param options - Extension mode and axis.
 * @returns The reconstruction.
 * @concept discrete-wavelet-transform
 * @example
 * ```ts
 * import { dwt, idwt } from 'whitebox-engine/signal'
 * import { toFlat } from 'whitebox-engine/tensor'
 * const { cA, cD } = dwt([3, 1, 4, 1, 5, 9], 'db2')
 * toFlat(idwt(cA, cD, 'db2')).map((v) => Math.round(v * 1e12) / 1e12) // => [3, 1, 4, 1, 5, 9]
 * ```
 */
export function idwt(ca: TensorLike, cd: TensorLike, w: WaveletSpec, options: DwtOptions = {}): Value {
  const a = float(ca)
  const d = float(cd)
  if (a.shape.length !== d.shape.length || a.shape.some((s, i) => s !== d.shape[i]))
    throw new WhiteboxError('shape', 'idwt: cA and cD must have the same shape')
  const ra = rowsOf(a, options.axis ?? -1, 'idwt')
  const rd = rowsOf(d, options.axis ?? -1, 'idwt')
  return ra.restore(idwtRows(ra.rows, rd.rows, wavelet(w), options.mode ?? 'symmetric'))
}

/**
 * The multilevel DWT (pywt.wavedec): repeated dwt of the approximation, returning [cA_n, cD_n, ..., cD_1].
 *
 * @param x - Data.
 * @param w - The wavelet.
 * @param options - Level (default dwtMaxLevel), extension mode and axis.
 * @returns The coefficients, coarsest first.
 * @concept discrete-wavelet-transform
 * @example
 * ```ts
 * import { wavedec } from 'whitebox-engine/signal'
 * wavedec(new Array(16).fill(1), 'haar', { level: 2 }).map((c) => c.shape[0]) // => [4, 4, 8]
 * ```
 */
export function wavedec(
  x: TensorLike,
  w: WaveletSpec,
  options: DwtOptions & { readonly level?: number } = {},
): Value[] {
  const wv = wavelet(w)
  const v = float(x)
  const axis = options.axis ?? -1
  const a = axis < 0 ? axis + v.shape.length : axis
  const level = options.level ?? dwtMaxLevel(v.shape[a] ?? 0, wv)
  if (level < 0) throw new WhiteboxError('value', 'wavedec: the level must be non-negative')
  const details: Value[] = []
  let approx = v
  for (let l = 0; l < level; l++) {
    const c = dwt(approx, wv, options)
    details.push(c.cD)
    approx = c.cA
  }
  return [approx, ...details.reverse()]
}

/**
 * The multilevel inverse DWT (pywt.waverec), trimming an approximation one sample longer than the next detail.
 *
 * @param coeffs - [cA_n, cD_n, ..., cD_1].
 * @param w - The wavelet.
 * @param options - Extension mode and axis.
 * @returns The reconstruction.
 * @concept discrete-wavelet-transform
 * @example
 * ```ts
 * import { wavedec, waverec } from 'whitebox-engine/signal'
 * import { toFlat } from 'whitebox-engine/tensor'
 * const x = [3, 1, 4, 1, 5, 9, 2, 6]
 * toFlat(waverec(wavedec(x, 'db1', { level: 3 }), 'db1')).map((v) => Math.round(v * 1e12) / 1e12) // => [3, 1, 4, 1, 5, 9, 2, 6]
 * ```
 */
export function waverec(coeffs: readonly TensorLike[], w: WaveletSpec, options: DwtOptions = {}): Value {
  const first = coeffs[0]
  if (first === undefined) throw new WhiteboxError('value', 'waverec: no coefficients')
  let a = float(first)
  const axis = options.axis ?? -1
  for (const dl of coeffs.slice(1)) {
    const d = float(dl)
    const ax = axis < 0 ? axis + a.shape.length : axis
    const la = a.shape[ax] ?? 0
    const ld = d.shape[ax] ?? 0
    if (la === ld + 1) a = sliceAxis(a, ax, 0, ld)
    else if (la !== ld) throw new WhiteboxError('shape', 'waverec: coefficient lengths do not match')
    a = idwt(a, d, w, options)
  }
  return a
}

/** The four subbands of a 2-D DWT: approximation and horizontal, vertical and diagonal details. */
export interface Dwt2Coefficients {
  readonly cA: Value
  readonly cH: Value
  readonly cV: Value
  readonly cD: Value
}

/**
 * The single-level 2-D DWT (pywt.dwt2): the 1-D transform along the last two axes in turn, giving the approximation
 * and the horizontal (cH, detail along rows), vertical (cV) and diagonal (cD) details.
 *
 * @param x - Data with at least two axes (the last two are transformed).
 * @param w - The wavelet.
 * @param mode - Extension mode (default 'symmetric').
 * @returns The four subbands.
 * @concept discrete-wavelet-transform
 * @example
 * ```ts
 * import { dwt2 } from 'whitebox-engine/signal'
 * dwt2([[1, 2], [3, 4]], 'haar').cA.shape // => [1, 1]
 * ```
 */
export function dwt2(x: TensorLike, w: WaveletSpec, mode: WaveletMode = 'symmetric'): Dwt2Coefficients {
  const v = float(x)
  if (v.shape.length < 2) throw new WhiteboxError('shape', 'dwt2: expected at least two axes')
  const rows = dwt(v, w, { mode, axis: -2 })
  const a = dwt(rows.cA, w, { mode, axis: -1 })
  const d = dwt(rows.cD, w, { mode, axis: -1 })
  return { cA: a.cA, cH: d.cA, cV: a.cD, cD: d.cD }
}

/**
 * The single-level inverse 2-D DWT (pywt.idwt2).
 *
 * @param c - The four subbands.
 * @param w - The wavelet.
 * @param mode - Extension mode.
 * @returns The reconstruction.
 * @concept discrete-wavelet-transform
 * @example
 * ```ts
 * import { dwt2, idwt2 } from 'whitebox-engine/signal'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(idwt2(dwt2([[1, 2], [3, 4]], 'haar'), 'haar')).flat().map((v) => Math.round(Number(v) * 1e12) / 1e12) // => [1, 2, 3, 4]
 * ```
 */
export function idwt2(c: Dwt2Coefficients, w: WaveletSpec, mode: WaveletMode = 'symmetric'): Value {
  const a = idwt(c.cA, c.cV, w, { mode, axis: -1 })
  const d = idwt(c.cH, c.cD, w, { mode, axis: -1 })
  return idwt(a, d, w, { mode, axis: -2 })
}

/** Coefficients of a multilevel 2-D DWT: the coarsest approximation and the detail triples, coarsest first. */
export interface Wavedec2Coefficients {
  readonly cA: Value
  readonly details: readonly { readonly cH: Value; readonly cV: Value; readonly cD: Value }[]
}

/**
 * The multilevel 2-D DWT (pywt.wavedec2).
 *
 * @param x - Data (the last two axes are transformed).
 * @param w - The wavelet.
 * @param options - Level (default the smaller of the axes' maximum levels) and mode.
 * @returns The approximation and the details, coarsest first.
 * @concept discrete-wavelet-transform
 * @example
 * ```ts
 * import { wavedec2 } from 'whitebox-engine/signal'
 * wavedec2(new Array(8).fill(new Array(8).fill(1)), 'haar', { level: 2 }).details.length // => 2
 * ```
 */
export function wavedec2(
  x: TensorLike,
  w: WaveletSpec,
  options: { readonly level?: number; readonly mode?: WaveletMode } = {},
): Wavedec2Coefficients {
  const v = float(x)
  const wv = wavelet(w)
  const rank = v.shape.length
  const level =
    options.level ?? Math.min(dwtMaxLevel(v.shape[rank - 2] ?? 0, wv), dwtMaxLevel(v.shape[rank - 1] ?? 0, wv))
  const details: { cH: Value; cV: Value; cD: Value }[] = []
  let a = v
  for (let l = 0; l < level; l++) {
    const c = dwt2(a, wv, options.mode ?? 'symmetric')
    details.push({ cH: c.cH, cV: c.cV, cD: c.cD })
    a = c.cA
  }
  return { cA: a, details: details.reverse() }
}

/**
 * The multilevel inverse 2-D DWT (pywt.waverec2), trimming approximations one sample longer than the details.
 *
 * @param c - The approximation and details from {@link wavedec2}.
 * @param w - The wavelet.
 * @param mode - Extension mode.
 * @returns The reconstruction.
 * @concept discrete-wavelet-transform
 * @example
 * ```ts
 * import { wavedec2, waverec2 } from 'whitebox-engine/signal'
 * waverec2(wavedec2([[1, 2, 3, 4], [5, 6, 7, 8], [9, 1, 2, 3], [4, 5, 6, 7]], 'haar'), 'haar').shape // => [4, 4]
 * ```
 */
export function waverec2(c: Wavedec2Coefficients, w: WaveletSpec, mode: WaveletMode = 'symmetric'): Value {
  let a = c.cA
  for (const d of c.details) {
    const rank = a.shape.length
    for (const ax of [rank - 2, rank - 1]) {
      const target = d.cH.shape[ax] ?? 0
      if ((a.shape[ax] ?? 0) === target + 1) a = sliceAxis(a, ax, 0, target)
    }
    a = idwt2({ cA: a, cH: d.cH, cV: d.cV, cD: d.cD }, w, mode)
  }
  return a
}
