import { WhiteboxError } from '../base/index.ts'
import { fftConvolve } from '../numerics/convolution/index.ts'
import { fftshift, irfft, rfft } from '../numerics/fourier/index.ts'
import type { Lti } from '../systems/index.ts'
import {
  add,
  concat,
  cScale,
  flip,
  max,
  mean,
  min,
  mul,
  pad,
  primalOf,
  reshape,
  sliceAxis,
  sort,
  stack,
  sub,
  type TensorLike,
  take,
  toFlat,
  type Value,
  zeros,
} from '../tensor/index.ts'
import { constant, float, gcd, indexTensor, positiveInt, rowsOf } from './_shared.ts'
import { filtfilt, lfilter, sosfilt, sosfiltfilt } from './filters/apply.ts'
import { firwin } from './filters/fir.ts'
import { cheby1 } from './filters/iir.ts'
import { getWindow, type WindowSpec } from './windows/index.ts'

// Multirate processing as scipy.signal: upsampling, filtering and downsampling in one step (upfirdn), polyphase
// rational resampling (resample_poly), Fourier resampling (resample), decimation, interpolation, and the polyphase
// decomposition behind them. All are compositions of differentiable operations.

/** How upfirdn extends the signal beyond its ends. */
export type UpfirdnMode =
  | 'constant'
  | 'symmetric'
  | 'reflect'
  | 'edge'
  | 'wrap'
  | 'antisymmetric'
  | 'antireflect'
  | 'smooth'
  | 'line'

/** Insert up - 1 zeros after every sample of rows `[batch, n]`, giving `[batch, (n - 1) up + 1]`. */
function stuff(rows: Value, up: number): Value {
  if (up === 1) return rows
  const [batch = 1, n = 0] = rows.shape
  const z = zeros([batch, n, up - 1])
  const inter = reshape(concat([reshape(rows, [batch, n, 1]), z], 2), [batch, n * up])
  return sliceAxis(inter, 1, 0, (n - 1) * up + 1)
}

/** Extend rows by p samples at both ends under an upfirdn mode. */
function extendRows(rows: Value, p: number, mode: UpfirdnMode): Value {
  const n = rows.shape[1] ?? 0
  const idx = (f: (i: number) => number): Value => take(rows, indexTensor(Array.from({ length: p }, (_, i) => f(i))), 1)
  const first = sliceAxis(rows, 1, 0, 1)
  const last = sliceAxis(rows, 1, n - 1, n)
  const mirror = (k: number, period: number): number => {
    const j = ((k % period) + period) % period
    return j
  }
  switch (mode) {
    case 'edge':
      return concat([idx(() => 0), rows, idx(() => n - 1)], 1)
    case 'wrap':
      return concat([idx((i) => mirror(n - p + i, n)), rows, idx((i) => mirror(i, n))], 1)
    case 'symmetric':
    case 'antisymmetric': {
      const sym = (k: number): number => {
        const j = mirror(k, 2 * n)
        return j < n ? j : 2 * n - 1 - j
      }
      const left = idx((i) => sym(-p + i))
      const right = idx((i) => sym(n + i))
      if (mode === 'symmetric') return concat([left, rows, right], 1)
      return concat([mul(left, -1), rows, mul(right, -1)], 1)
    }
    case 'reflect':
    case 'antireflect': {
      const ref = (k: number): number => {
        if (n === 1) return 0
        const j = mirror(k, 2 * (n - 1))
        return j < n ? j : 2 * (n - 1) - j
      }
      const left = idx((i) => ref(-p + i))
      const right = idx((i) => ref(n + i))
      if (mode === 'reflect') return concat([left, rows, right], 1)
      return concat([sub(mul(first, 2), left), rows, sub(mul(last, 2), right)], 1)
    }
    case 'smooth':
    case 'line': {
      const second = sliceAxis(rows, 1, Math.min(1, n - 1), Math.min(2, n))
      const penult = sliceAxis(rows, 1, Math.max(n - 2, 0), Math.max(n - 1, 1))
      const slopeL = mode === 'line' ? mul(sub(last, first), 1 / Math.max(n - 1, 1)) : sub(second, first)
      const slopeR = mode === 'line' ? slopeL : sub(last, penult)
      const steps = (k: number[]): Value => constant(k, [1, k.length])
      const left = add(first, mul(slopeL, steps(Array.from({ length: p }, (_, i) => i - p))))
      const right = add(last, mul(slopeR, steps(Array.from({ length: p }, (_, i) => i + 1))))
      return concat([left, rows, right], 1)
    }
    default:
      return rows
  }
}

function upfirdnRows(h: Value, rows: Value, up: number, down: number, mode: UpfirdnMode, cval: number): Value {
  const nh = h.shape[0] ?? 0
  const n = rows.shape[1] ?? 0
  const outLen = Math.floor(((n - 1) * up + nh - 1) / down) + 1
  if (mode === 'constant' && cval === 0) {
    const full = fftConvolve(stuff(rows, up), reshape(h, [1, nh]), 'full', [1])
    const idx = Array.from({ length: outLen }, (_, k) => k * down)
    return take(full, indexTensor(idx), 1)
  }
  if (mode === 'constant') {
    // A constant extension: filter the deviation from cval, then add cval times the sum of the taps in each phase.
    const base = upfirdnRows(h, sub(rows, cval), up, down, 'constant', 0)
    const hv = toFlat(primalOf(h))
    const phase = Array.from({ length: outLen }, (_, k) => {
      let s = 0
      for (let j = (k * down) % up; j < nh; j += up) s += hv[j] ?? 0
      return cval * s
    })
    return add(base, constant(phase, [1, outLen]))
  }
  const p = Math.ceil(nh / up) + 1
  const ext = extendRows(rows, p, mode)
  const full = fftConvolve(stuff(ext, up), reshape(h, [1, nh]), 'full', [1])
  const idx = Array.from({ length: outLen }, (_, k) => p * up + k * down)
  return take(full, indexTensor(idx), 1)
}

/** Options for {@link upfirdn}. */
export interface UpfirdnOptions {
  readonly up?: number
  readonly down?: number
  readonly axis?: number
  /** Extension beyond the ends (default 'constant'). */
  readonly mode?: UpfirdnMode
  /** The constant for mode 'constant' (default 0). */
  readonly cval?: number
}

/**
 * Upsample, FIR filter and downsample in one step (scipy.signal.upfirdn): insert up - 1 zeros between samples,
 * convolve with h, keep every down-th sample. The output has floor(((n - 1) up + len(h) - 1) / down) + 1 samples.
 * Differentiable in h and x.
 *
 * @param h - FIR filter taps.
 * @param x - Signal.
 * @param options - Up and down factors (default 1), axis (default -1), extension mode and constant.
 * @returns The resampled, filtered signal.
 * @concept multirate
 * @example
 * ```ts
 * import { upfirdn } from 'whitebox-engine/signal'
 * import { toFlat } from 'whitebox-engine/tensor'
 * toFlat(upfirdn([1, 1], [1, 2, 3], { up: 2 })).map((v) => Math.round(v * 1e12) / 1e12) // => [1, 1, 2, 2, 3, 3]
 * ```
 */
export function upfirdn(h: TensorLike, x: TensorLike, options: UpfirdnOptions = {}): Value {
  const up = positiveInt(options.up ?? 1, 'upfirdn: up')
  const down = positiveInt(options.down ?? 1, 'upfirdn: down')
  const hv = float(h)
  if (hv.shape.length !== 1 || (hv.shape[0] ?? 0) === 0) throw new WhiteboxError('shape', 'upfirdn: h must be 1-D')
  const r = rowsOf(float(x), options.axis ?? -1, 'upfirdn')
  return r.restore(upfirdnRows(hv, r.rows, up, down, options.mode ?? 'constant', options.cval ?? 0))
}

/** How resamplePoly extends the signal: an upfirdn mode, or the statistic of the signal it is extended with. */
export type ResamplePadType = UpfirdnMode | 'mean' | 'median' | 'minimum' | 'maximum'

/** The median of each row, `[batch, 1]` (differentiable through the sort). */
function rowMedian(rows: Value): Value {
  const n = rows.shape[1] ?? 0
  const s = sort(rows, 1)
  const mid = Math.floor(n / 2)
  if (n % 2 === 1) return sliceAxis(s, 1, mid, mid + 1)
  return mul(add(sliceAxis(s, 1, mid - 1, mid), sliceAxis(s, 1, mid, mid + 1)), 0.5)
}

/** Options for {@link resamplePoly}. */
export interface ResamplePolyOptions {
  readonly axis?: number
  /** The FIR design window (default ['kaiser', 5]), or the filter taps themselves. */
  readonly window?: WindowSpec | TensorLike
  /** Extension (default 'constant'). */
  readonly padtype?: ResamplePadType
  readonly cval?: number
}

/**
 * Resample by the rational factor up / down with a polyphase FIR filter (scipy.signal.resample_poly): a windowed-sinc
 * lowpass at the lower Nyquist rate (or the given taps), applied by upfirdn and aligned so that output sample k sits
 * at input time k down / up. Differentiable in x.
 *
 * @param x - Signal.
 * @param up - Upsampling factor.
 * @param down - Downsampling factor.
 * @param options - Axis (default -1), window or taps, padding type and constant.
 * @returns ceil(n up / down) samples.
 * @concept multirate
 * @example
 * ```ts
 * import { resamplePoly } from 'whitebox-engine/signal'
 * resamplePoly(new Array(30).fill(1), 2, 3).shape // => [20]
 * ```
 */
export function resamplePoly(x: TensorLike, up: number, down: number, options: ResamplePolyOptions = {}): Value {
  positiveInt(up, 'resamplePoly: up')
  positiveInt(down, 'resamplePoly: down')
  const g = gcd(up, down)
  const u = up / g
  const d = down / g
  const v = float(x)
  if (u === 1 && d === 1) return v
  const r = rowsOf(v, options.axis ?? -1, 'resamplePoly')
  const nIn = r.n
  const nOut = Math.floor((nIn * u) / d) + ((nIn * u) % d === 0 ? 0 : 1)
  const w = options.window ?? ['kaiser', 5]
  let h: Value
  let halfLen: number
  const isTaps = typeof w !== 'string' && typeof w !== 'number' && !(Array.isArray(w) && typeof w[0] === 'string')
  if (isTaps) {
    h = reshape(float(w as TensorLike), [-1])
    halfLen = Math.floor(((h.shape[0] ?? 1) - 1) / 2)
  } else {
    const maxRate = Math.max(u, d)
    halfLen = 10 * maxRate
    h = firwin(2 * halfLen + 1, 1 / maxRate, { window: w as WindowSpec })
  }
  h = mul(h, u)
  const nh = h.shape[0] ?? 0
  const prePad = d - (halfLen % d)
  const preRemove = Math.floor((halfLen + prePad) / d)
  const outputLength = (lenH: number): number => Math.floor(((nIn - 1) * u + lenH - 1) / d) + 1
  let postPad = 0
  while (outputLength(nh + prePad + postPad) < nOut + preRemove) postPad++
  const hp = pad(h, [[prePad, postPad]])
  const padtype = options.padtype ?? 'constant'
  let rows = r.rows
  let background: Value | null = null
  if (padtype === 'mean' || padtype === 'median' || padtype === 'minimum' || padtype === 'maximum') {
    background =
      padtype === 'mean'
        ? mean(rows, 1, true)
        : padtype === 'median'
          ? rowMedian(rows)
          : padtype === 'minimum'
            ? min(rows, 1, true)
            : max(rows, 1, true)
    rows = sub(rows, background)
  }
  const mode: UpfirdnMode = background === null ? (padtype as UpfirdnMode) : 'constant'
  const y = upfirdnRows(hp, rows, u, d, mode, mode === 'constant' && background === null ? (options.cval ?? 0) : 0)
  let keep = sliceAxis(y, 1, preRemove, preRemove + nOut)
  if (background !== null) keep = add(keep, background)
  return r.restore(keep)
}

/**
 * Fourier resampling to num samples (scipy.signal.resample): truncate or zero-pad the spectrum (splitting or merging
 * the Nyquist bin of even lengths) and invert; exact for periodic band-limited signals. Differentiable.
 *
 * @param x - Real signal.
 * @param num - Number of output samples.
 * @param options - Axis (default 0, as scipy) and a spectral window.
 * @returns The resampled signal.
 * @concept multirate
 * @example
 * ```ts
 * import { resample } from 'whitebox-engine/signal'
 * import { toFlat } from 'whitebox-engine/tensor'
 * toFlat(resample([1, 0, -1, 0], 8)).map((v) => Math.round(v * 1e12) / 1e12) // => [1, 0.707106781187, 0, -0.707106781187, -1, -0.707106781187, 0, 0.707106781187]
 * ```
 */
export function resample(
  x: TensorLike,
  num: number,
  options: { readonly axis?: number; readonly window?: WindowSpec } = {},
): Value {
  positiveInt(num, 'resample: num')
  const r = rowsOf(float(x), options.axis ?? 0, 'resample')
  const nx = r.n
  const sFac = nx / num
  const m = Math.min(num, nx)
  const m2 = Math.floor(m / 2) + 1
  let spec = rfft(r.rows)
  const nX = spec.re.shape[1] ?? 0
  if (options.window !== undefined) {
    const wv = toFlat(primalOf(fftshift(getWindow(options.window, nx))))
    const w = Float64Array.from(wv)
    for (let k = 1; k < nX; k++) w[k] = ((w[k] ?? 0) + (wv[nx - k] ?? 0)) / 2
    spec = cScale(spec, constant(w.slice(0, nX), [1, nX]))
  }
  spec = { re: sliceAxis(spec.re, 1, 0, m2), im: sliceAxis(spec.im, 1, 0, m2) }
  if (m % 2 === 0 && num !== nx) {
    const f = Array.from({ length: m2 }, (_, k) => (k === m / 2 ? (num < nx ? 2 : 0.5) : 1))
    spec = cScale(spec, constant(f, [1, m2]))
  }
  return r.restore(irfft(cScale(spec, 1 / sFac), num))
}

/** Options for {@link decimate}. */
export interface DecimateOptions {
  /** Filter order (default 8 for 'iir', 20 q for 'fir'). */
  readonly n?: number
  /** 'iir' (Chebyshev I, default), 'fir' (Hamming-windowed sinc) or an LTI system to use. */
  readonly ftype?: 'iir' | 'fir' | Lti
  readonly axis?: number
  /** Filter forwards and backwards (default true), so the result has no delay. */
  readonly zeroPhase?: boolean
}

/**
 * Downsample by an integer factor after an anti-aliasing lowpass (scipy.signal.decimate): by default an order 8
 * Chebyshev I filter at 0.8 / q of Nyquist, applied forwards and backwards. Differentiable in x.
 *
 * @param x - Signal.
 * @param q - Downsampling factor.
 * @param options - Order, filter type, axis and zero phase.
 * @returns The decimated signal.
 * @concept multirate
 * @example
 * ```ts
 * import { decimate } from 'whitebox-engine/signal'
 * decimate(new Array(64).fill(1), 4).shape // => [16]
 * ```
 */
export function decimate(x: TensorLike, q: number, options: DecimateOptions = {}): Value {
  positiveInt(q, 'decimate: q')
  const axis = options.axis ?? -1
  const zeroPhase = options.zeroPhase ?? true
  const v = float(x)
  const ftype = options.ftype ?? 'iir'
  const r = rowsOf(v, axis, 'decimate')
  const every = (y: Value): Value =>
    take(y, indexTensor(Array.from({ length: Math.ceil((y.shape[1] ?? 0) / q) }, (_, i) => i * q)), 1)
  if (ftype === 'fir' || (typeof ftype === 'object' && ftype.kind === 'tf' && (ftype.den.shape[0] ?? 1) === 1)) {
    const b =
      ftype === 'fir'
        ? firwin((options.n ?? 20 * q) + 1, 1 / q, { window: 'hamming' })
        : mul(ftype.num, 1 / (toFlat(primalOf(ftype.den))[0] ?? 1))
    if (zeroPhase) return r.restore(resamplePoly(r.rows, 1, q, { axis: 1, window: b }))
    const nOut = Math.floor(r.n / q) + (r.n % q === 0 ? 0 : 1)
    return r.restore(sliceAxis(upfirdn(b, r.rows, { down: q, axis: 1 }), 1, 0, nOut))
  }
  if (typeof ftype === 'object') {
    if (ftype.kind === 'tf') {
      const y = zeroPhase
        ? filtfilt(ftype.num, ftype.den, r.rows, { axis: 1 })
        : lfilter(ftype.num, ftype.den, r.rows, { axis: 1 })
      return r.restore(every(y))
    }
    const y = zeroPhase ? sosfiltfilt(ftype, r.rows, { axis: 1 }) : sosfilt(ftype, r.rows, { axis: 1 })
    return r.restore(every(y))
  }
  const sos = cheby1(options.n ?? 8, 0.05, 0.8 / q, { output: 'sos' })
  const y = zeroPhase ? sosfiltfilt(sos, r.rows, { axis: 1 }) : sosfilt(sos, r.rows, { axis: 1 })
  return r.restore(every(y))
}

/**
 * Interpolate by an integer factor: zero-stuffing followed by the anti-imaging lowpass of resamplePoly (up = factor,
 * down = 1), so the original samples are kept up to the filter's accuracy.
 *
 * @param x - Signal.
 * @param factor - Upsampling factor.
 * @param options - Axis and design window.
 * @returns n factor samples.
 * @concept multirate
 * @example
 * ```ts
 * import { interpolate } from 'whitebox-engine/signal'
 * interpolate([0, 1, 0, -1], 3).shape // => [12]
 * ```
 */
export function interpolate(
  x: TensorLike,
  factor: number,
  options: { readonly axis?: number; readonly window?: WindowSpec } = {},
): Value {
  return resamplePoly(x, factor, 1, options)
}

/**
 * Keep every m-th sample starting at a phase (downsampling without filtering).
 *
 * @param x - Signal (along the last axis).
 * @param m - Factor.
 * @param phase - First sample kept (default 0).
 * @returns The downsampled signal.
 * @concept multirate
 * @example
 * ```ts
 * import { downsample } from 'whitebox-engine/signal'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(downsample([0, 1, 2, 3, 4, 5, 6], 3, 1)) // => [1, 4]
 * ```
 */
export function downsample(x: TensorLike, m: number, phase = 0): Value {
  positiveInt(m, 'downsample: m')
  const r = rowsOf(float(x), -1, 'downsample')
  const idx: number[] = []
  for (let i = phase; i < r.n; i += m) idx.push(i)
  return r.restore(take(r.rows, indexTensor(idx), 1))
}

/**
 * Insert l - 1 zeros after every sample (upsampling without filtering), giving n l samples.
 *
 * @param x - Signal (along the last axis).
 * @param l - Factor.
 * @returns The zero-stuffed signal.
 * @concept multirate
 * @example
 * ```ts
 * import { upsample } from 'whitebox-engine/signal'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(upsample([1, 2], 3)) // => [1, 0, 0, 2, 0, 0]
 * ```
 */
export function upsample(x: TensorLike, l: number): Value {
  positiveInt(l, 'upsample: l')
  const r = rowsOf(float(x), -1, 'upsample')
  return r.restore(
    pad(stuff(r.rows, l), [
      [0, 0],
      [0, l - 1],
    ]),
  )
}

/**
 * The type 1 polyphase components of a filter: E_k[n] = h[n m + k] for k = 0 .. m - 1, so that H(z) = sum_k z^-k
 * E_k(z^m).
 *
 * @param h - Filter taps.
 * @param m - Number of phases.
 * @returns The components `[m, ceil(len(h) / m)]` (zero-padded).
 * @concept polyphase
 * @example
 * ```ts
 * import { polyphase } from 'whitebox-engine/signal'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(polyphase([1, 2, 3, 4, 5], 2)) // => [[1, 3, 5], [2, 4, 0]]
 * ```
 */
export function polyphase(h: TensorLike, m: number): Value {
  positiveInt(m, 'polyphase: m')
  const v = reshape(float(h), [-1])
  const n = v.shape[0] ?? 0
  const cols = Math.ceil(n / m)
  const padded = pad(v, [[0, cols * m - n]])
  return transposeCols(reshape(padded, [cols, m]))
}

function transposeCols(v: Value): Value {
  const [r = 0, c = 0] = v.shape
  const parts = Array.from({ length: c }, (_, k) => reshape(sliceAxis(v, 1, k, k + 1), [r]))
  return stack(parts, 0)
}

/**
 * Reassemble a filter from its type 1 polyphase components (the inverse of {@link polyphase}).
 *
 * @param components - `[m, k]` components.
 * @param length - Length of the filter (default m k).
 * @returns The taps.
 * @concept polyphase
 * @example
 * ```ts
 * import { polyphaseRecombine } from 'whitebox-engine/signal'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(polyphaseRecombine([[1, 3, 5], [2, 4, 0]], 5)) // => [1, 2, 3, 4, 5]
 * ```
 */
export function polyphaseRecombine(components: TensorLike, length?: number): Value {
  const c = float(components)
  if (c.shape.length !== 2) throw new WhiteboxError('shape', 'polyphaseRecombine: components are [m, k]')
  const [m = 0, k = 0] = c.shape
  const flat = reshape(transposeCols(c), [m * k])
  return sliceAxis(flat, 0, 0, length ?? m * k)
}

/**
 * The filter H(z^m): m - 1 zeros between taps. By the noble identities, filtering with H(z^m) then downsampling by m
 * equals downsampling then filtering with H(z), and upsampling then filtering with H(z^m) equals filtering with H(z)
 * then upsampling.
 *
 * @param h - Taps.
 * @param m - Factor.
 * @returns The expanded taps `[(len(h) - 1) m + 1]`.
 * @concept polyphase
 * @example
 * ```ts
 * import { expandFilter } from 'whitebox-engine/signal'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(expandFilter([1, 2, 3], 2)) // => [1, 0, 2, 0, 3]
 * ```
 */
export function expandFilter(h: TensorLike, m: number): Value {
  positiveInt(m, 'expandFilter: m')
  return reshape(stuff(reshape(float(h), [1, -1]), m), [-1])
}
