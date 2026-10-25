/**
 * The constant-Q transform, a time-frequency analysis whose bins are geometrically spaced (a fixed number per octave)
 * with a bandwidth proportional to frequency, which suits musical pitch: the centre frequencies, the bank of windowed
 * complex exponentials and the transform itself, matching librosa.cqt without early downsampling.
 */

import { WhiteboxError } from '../base/index.ts'
import { fft } from '../numerics/fourier/index.ts'
import {
  add,
  type ComplexValue,
  matmul,
  mul,
  sliceAxis,
  sub,
  type Tensor,
  type TensorLike,
  toFlat,
} from '../tensor/index.ts'
import { constant, float } from './_shared.ts'
import type { TimeFrequency } from './signal.ts'
import { stft } from './time-frequency.ts'
import { enbw, getWindow, type WindowSpec } from './windows/index.ts'

// The constant-Q transform with librosa's wavelet filters (librosa.cqt): logarithmically spaced centre frequencies,
// each analysed by a windowed complex exponential whose length gives a constant ratio of frequency to bandwidth.
// The transform is computed at the full sampling rate for every bin (librosa resamples the lower octaves, which is
// equivalent up to the resampling filter); for a single octave it is librosa.cqt exactly.

/**
 * Centre frequencies of a constant-Q transform (librosa.cqt_frequencies): fmin 2^(k / binsPerOctave).
 *
 * @param nBins - Number of bins.
 * @param fmin - Lowest frequency.
 * @param binsPerOctave - Bins per octave (default 12).
 * @returns The frequencies.
 * @concept constant-q
 * @example
 * ```ts
 * import { cqtFrequencies } from 'whitebox-engine/signal'
 * cqtFrequencies(3, 100, 1) // => [100, 200, 400]
 * ```
 */
export function cqtFrequencies(nBins: number, fmin: number, binsPerOctave = 12): number[] {
  return Array.from({ length: nBins }, (_, k) => fmin * 2 ** (k / binsPerOctave))
}

/** Options for {@link cqtFilters} and {@link cqt}. */
export interface CqtOptions {
  /** Sampling rate (default 22050). */
  readonly sr?: number
  /** Lowest frequency (default C1, 32.70 Hz). */
  readonly fmin?: number
  /** Number of bins (default 84). */
  readonly nBins?: number
  readonly binsPerOctave?: number
  /** Scale of the filter lengths (default 1). */
  readonly filterScale?: number
  /** Window of the filters (default 'hann'). */
  readonly window?: WindowSpec
  /** Normalise each filter's l1 norm (default true). */
  readonly norm?: boolean
}

/** The relative bandwidths alpha of librosa's wavelet filters from the log-spacing of the centre frequencies. */
function relativeBandwidth(freqs: readonly number[]): number[] {
  const n = freqs.length
  if (n < 2) throw new WhiteboxError('value', 'cqt: at least two frequencies are needed')
  const logf = freqs.map(Math.log2)
  return freqs.map((_, k) => {
    const bpo =
      k === 0
        ? 1 / ((logf[1] ?? 0) - (logf[0] ?? 0))
        : k === n - 1
          ? 1 / ((logf[n - 1] ?? 0) - (logf[n - 2] ?? 0))
          : 2 / ((logf[k + 1] ?? 0) - (logf[k - 1] ?? 0))
    const r = 2 ** (2 / bpo)
    return (r - 1) / (r + 1)
  })
}

/** The time-domain filters of a constant-Q transform. */
export interface CqtFilters {
  /** Complex filters `[nBins, nFft]`, each centred in its row. */
  readonly filters: ComplexValue
  /** The (fractional) filter lengths. */
  readonly lengths: number[]
  /** The frequencies. */
  readonly frequencies: number[]
  /** Highest frequency the filters reach. */
  readonly cutoff: number
}

/**
 * The constant-Q filter bank (librosa.filters.wavelet): for each centre frequency f, the windowed complex exponential
 * of length Q sr / f with Q = filterScale / alpha, l1-normalised and centred in a row of power-of-two length.
 *
 * @param options - Sampling rate, frequencies, filter scale, window and normalisation.
 * @returns The filters, their lengths, the frequencies and the cutoff.
 * @throws WhiteboxError with code `value` when the filters would exceed the Nyquist frequency.
 * @concept constant-q
 * @example
 * ```ts
 * import { cqtFilters } from 'whitebox-engine/signal'
 * cqtFilters({ sr: 8000, fmin: 2000, nBins: 12 }).filters.re.shape // => [12, 128]
 * ```
 */
export function cqtFilters(options: CqtOptions = {}): CqtFilters {
  const sr = options.sr ?? 22050
  const fmin = options.fmin ?? 32.70319566257483
  const nBins = options.nBins ?? 84
  const bpo = options.binsPerOctave ?? 12
  const freqs = cqtFrequencies(nBins, fmin, bpo)
  const r = 2 ** (2 / bpo)
  const alpha = nBins === 1 ? [(r - 1) / (r + 1)] : relativeBandwidth(freqs)
  const window = options.window ?? 'hann'
  const scale = options.filterScale ?? 1
  const bw = enbw(getWindow(window, 1000))
  let cutoff = 0
  const lengths = freqs.map((f, k) => {
    const q = scale / (alpha[k] ?? 1)
    cutoff = Math.max(cutoff, f * (1 + (0.5 * bw) / q))
    return (q * sr) / f
  })
  if (cutoff > sr / 2) throw new WhiteboxError('value', `cqt: the filters reach ${cutoff} Hz, beyond Nyquist ${sr / 2}`)
  let maxLen = 0
  for (const l of lengths) maxLen = Math.max(maxLen, l)
  const nFft = 2 ** Math.ceil(Math.log2(maxLen))
  const re = new Float64Array(nBins * nFft)
  const im = new Float64Array(nBins * nFft)
  for (const [k, l] of lengths.entries()) {
    const start = Math.floor(-l / 2)
    const count = Math.ceil(Math.floor(l / 2) - start)
    const w = toFlat(getWindow(window, count))
    const f = freqs[k] ?? 0
    let norm = 0
    const sr2 = new Float64Array(count)
    const si = new Float64Array(count)
    for (let i = 0; i < count; i++) {
      const ph = ((start + i) * 2 * Math.PI * f) / sr
      sr2[i] = Math.cos(ph) * (w[i] ?? 0)
      si[i] = Math.sin(ph) * (w[i] ?? 0)
      norm += Math.hypot(sr2[i] ?? 0, si[i] ?? 0)
    }
    if (!(options.norm ?? true) || norm === 0) norm = 1
    const off = Math.floor((nFft - count) / 2)
    for (let i = 0; i < count; i++) {
      re[k * nFft + off + i] = (sr2[i] ?? 0) / norm
      im[k * nFft + off + i] = (si[i] ?? 0) / norm
    }
  }
  return {
    filters: { re: constant(re, [nBins, nFft]), im: constant(im, [nBins, nFft]) },
    lengths,
    frequencies: freqs,
    cutoff,
  }
}

/**
 * The constant-Q transform (librosa.cqt with no early downsampling): the one-sided spectra of rectangular frames of
 * the centred signal times the filters' spectra, scaled by length / nFft and then by 1 / sqrt(length). Differentiable
 * in y.
 *
 * @param y - Real signal.
 * @param options - Filter-bank options, the hop (default 512) and the padding mode ('constant' or 'reflect').
 * @returns Frequencies (the centre frequencies), frame times and the complex transform `[nBins, frames]`.
 * @concept constant-q
 * @example
 * ```ts
 * import { cqt } from 'whitebox-engine/signal'
 * const y = Array.from({ length: 2048 }, (_, n) => Math.cos((2 * Math.PI * 2500 * n) / 8000))
 * cqt(y, { sr: 8000, fmin: 2000, nBins: 12, hop: 256 }).values.re.shape // => [12, 9]
 * ```
 */
export function cqt(
  y: TensorLike,
  options: CqtOptions & { readonly hop?: number; readonly padMode?: 'constant' | 'reflect' } = {},
): TimeFrequency<ComplexValue> {
  const sr = options.sr ?? 22050
  const hop = options.hop ?? 512
  const bank = cqtFilters(options)
  const nFft = bank.filters.re.shape[1] ?? 0
  const nBins = bank.lengths.length
  const spec = fft(bank.filters)
  const half = Math.floor(nFft / 2) + 1
  const scale = constant(
    bank.lengths.map((l) => l / nFft),
    [nBins, 1],
  )
  const br = mul(sliceAxis(spec.re, 1, 0, half), scale)
  const bi = mul(sliceAxis(spec.im, 1, 0, half), scale)
  const d = stft(float(y), {
    window: new Float64Array(nFft).fill(1),
    nperseg: nFft,
    noverlap: nFft - hop,
    boundary: options.padMode === 'reflect' ? 'reflect' : 'zeros',
    padded: false,
    scaling: 'none',
  }).values
  const cr = sub(matmul(br, d.re), matmul(bi, d.im))
  const ci = add(matmul(br, d.im), matmul(bi, d.re))
  const norm = constant(
    bank.lengths.map((l) => 1 / Math.sqrt(l)),
    [nBins, 1],
  )
  const frames = cr.shape[1] ?? 0
  return {
    frequencies: constant(bank.frequencies) as Tensor,
    times: constant(Array.from({ length: frames }, (_, j) => (j * hop) / sr)),
    values: { re: mul(cr, norm), im: mul(ci, norm) },
    fs: sr,
  }
}
