/**
 * Signal helpers of the audio metrics: windows, polyphase resampling and framed spectra. TODO: move them to
 * `signal/windows.ts`, `signal/multirate.ts` and `signal/time-frequency.ts` once the signal domain exists.
 */
import { WhiteboxError } from '../base/index.ts'
import { rfft } from '../numerics/fourier/index.ts'
import { i0 } from '../numerics/special/index.ts'
import { type TensorLike, toFlat } from '../tensor/index.ts'
import { matrix } from './metric.ts'

/** A mono signal as a Float64Array. */
export function signal(name: string, x: TensorLike): Float64Array {
  const m = matrix(x)
  if (m.shape.length !== 1) throw new WhiteboxError('shape', `${name}: signals are 1-D`)
  return m.data
}

/** Signals `[T]` or `[S, T]` as rows. */
export function signals(name: string, x: TensorLike): Float64Array[] {
  const m = matrix(x)
  if (m.shape.length === 1) return [m.data]
  if (m.shape.length !== 2) throw new WhiteboxError('shape', `${name}: signals are [T] or [S, T]`)
  const t = m.shape[1] ?? 0
  return Array.from({ length: m.shape[0] ?? 0 }, (_, i) => m.data.slice(i * t, (i + 1) * t))
}

/** numpy.hanning(m): the symmetric Hann window with zero end points. */
export function hanning(m: number): Float64Array {
  if (m === 1) return Float64Array.of(1)
  return Float64Array.from({ length: m }, (_, n) => 0.5 - 0.5 * Math.cos((2 * Math.PI * n) / (m - 1)))
}

/** scipy.signal.get_window('hann', m): the periodic Hann window. */
export function periodicHann(m: number): Float64Array {
  return Float64Array.from({ length: m }, (_, n) => 0.5 - 0.5 * Math.cos((2 * Math.PI * n) / m))
}

/** numpy.kaiser(m, beta). */
export function kaiser(m: number, beta: number): Float64Array {
  if (m === 1) return Float64Array.of(1)
  const n = Float64Array.from({ length: m }, (_, k) => beta * Math.sqrt(1 - ((2 * k) / (m - 1) - 1) ** 2))
  const num = toFlat(i0(n))
  const den = toFlat(i0([beta]))[0] ?? 1
  return Float64Array.from(num, (v) => v / den)
}

/** numpy.sinc: sin(pi x) / (pi x). */
export const sinc = (x: number): number => (x === 0 ? 1 : Math.sin(Math.PI * x) / (Math.PI * x))

const gcd = (a: number, b: number): number => (b === 0 ? a : gcd(b, a % b))

/** Upsample by `up`, filter with `h`, downsample by `down` (scipy.signal.upfirdn, zero padding). */
export function upfirdn(h: Float64Array, x: Float64Array, up: number, down: number): Float64Array {
  const length = Math.floor(((x.length - 1) * up + h.length - 1) / down) + 1
  const out = new Float64Array(length)
  for (let k = 0; k < length; k++) {
    const n = k * down
    let s = 0
    // Upsampled index i * up must satisfy 0 <= n - i * up < h.length.
    const lo = Math.max(0, Math.ceil((n - h.length + 1) / up))
    const hi = Math.min(x.length - 1, Math.floor(n / up))
    for (let i = lo; i <= hi; i++) s += (h[n - i * up] ?? 0) * (x[i] ?? 0)
    out[k] = s
  }
  return out
}

/** scipy.signal.resample_poly(x, up, down, window=h) with an explicit FIR window and zero padding. */
export function resamplePoly(x: Float64Array, upIn: number, downIn: number, window: Float64Array): Float64Array {
  const g = gcd(upIn, downIn)
  const up = upIn / g
  const down = downIn / g
  if (up === 1 && down === 1) return x.slice()
  const nIn = x.length
  const nOut = Math.floor((nIn * up) / down) + ((nIn * up) % down === 0 ? 0 : 1)
  const halfLen = Math.floor((window.length - 1) / 2)
  const prePad = down - (halfLen % down)
  const preRemove = Math.floor((halfLen + prePad) / down)
  const outputLength = (lenH: number): number => Math.floor(((nIn - 1) * up + lenH - 1) / down) + 1
  let postPad = 0
  while (outputLength(window.length + prePad + postPad) < nOut + preRemove) postPad++
  const h = new Float64Array(prePad + window.length + postPad)
  for (const [i, v] of window.entries()) h[prePad + i] = v * up
  return upfirdn(h, x, up, down).slice(preRemove, preRemove + nOut)
}

/** Power spectra |rfft(w * frame, nfft)|^2 of frames starting at the given offsets. */
export function framePowers(
  x: Float64Array,
  window: Float64Array,
  nfft: number,
  starts: readonly number[],
  scale = 1,
): Float64Array[] {
  if (starts.length === 0) return []
  const m = window.length
  const frames = Array.from({ length: starts.length }, (_, f) =>
    Array.from({ length: m }, (_, i) => (window[i] ?? 0) * (x[(starts[f] ?? 0) + i] ?? 0)),
  )
  const spec = rfft(frames, nfft)
  const re = toFlat(spec.re)
  const im = toFlat(spec.im)
  const bins = Math.floor(nfft / 2) + 1
  return starts.map((_, f) =>
    Float64Array.from(
      { length: bins },
      (_, k) => ((re[f * bins + k] ?? 0) ** 2 + (im[f * bins + k] ?? 0) ** 2) * scale,
    ),
  )
}

/** Frame offsets 0, hop, 2 hop, ... with each frame fully inside the signal (range(0, n - m + 1, hop)). */
export const frameStarts = (n: number, m: number, hop: number, strict = false): number[] => {
  const out: number[] = []
  for (let s = 0; strict ? s < n - m : s <= n - m; s += hop) out.push(s)
  return out
}
