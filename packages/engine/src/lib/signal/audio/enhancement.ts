import { WhiteboxError } from '../../base/index.ts'
import { fromStorage, primalOf, type Tensor, type TensorLike } from '../../tensor/index.ts'
import { samples } from '../_shared.ts'
import { librosaIstft, librosaStft } from './_stft.ts'

// Speech enhancement in the short-time Fourier domain: power spectral subtraction with over-subtraction and a
// spectral floor (Boll 1979, Berouti et al. 1979), Wiener filtering with the decision-directed a priori SNR (Ephraim
// and Malah 1984), and an energy-based voice activity detector with hangover.

/** Options shared by the enhancement methods. */
export interface EnhancementOptions {
  /** FFT length (default 512). */
  readonly nFft?: number
  /** Hop (default nFft / 4). */
  readonly hopLength?: number
  /** Frames at the start assumed to be noise only, for the noise estimate (default 6). */
  readonly noiseFrames?: number
  /** A noise-only recording to estimate the noise spectrum from instead. */
  readonly noise?: TensorLike
}

interface Spectra {
  readonly re: Float64Array
  readonly im: Float64Array
  readonly bins: number
  readonly frames: number
}

function spectra(x: Float64Array, nFft: number, hop: number): Spectra {
  const z = librosaStft(x, nFft, hop)
  const re = primalOf(z.re)
  return {
    re: Float64Array.from(re.data, Number),
    im: Float64Array.from(primalOf(z.im).data, Number),
    bins: re.shape[0] ?? 0,
    frames: re.shape[1] ?? 0,
  }
}

/** The mean noise power per bin, from a noise recording or the first frames. */
function noisePower(s: Spectra, options: EnhancementOptions, nFft: number, hop: number): Float64Array {
  const src = options.noise === undefined ? s : spectra(samples(options.noise, 'noise'), nFft, hop)
  const frames = options.noise === undefined ? Math.min(src.frames, options.noiseFrames ?? 6) : src.frames
  if (frames < 1) throw new WhiteboxError('value', 'enhancement: no noise frames')
  const p = new Float64Array(s.bins)
  for (let b = 0; b < s.bins; b++) {
    let acc = 0
    for (let t = 0; t < frames; t++)
      acc += (src.re[b * src.frames + t] ?? 0) ** 2 + (src.im[b * src.frames + t] ?? 0) ** 2
    p[b] = acc / frames
  }
  return p
}

function resynthesise(s: Spectra, gain: Float64Array, nFft: number, hop: number, length: number): Float64Array {
  const re = Float64Array.from(s.re, (v, i) => v * (gain[i] ?? 0))
  const im = Float64Array.from(s.im, (v, i) => v * (gain[i] ?? 0))
  return librosaIstft(
    { re: fromStorage(re, [s.bins, s.frames], 'float64'), im: fromStorage(im, [s.bins, s.frames], 'float64') },
    nFft,
    hop,
    length,
  )
}

/**
 * Power spectral subtraction: |X|^2 = max(|Y|^2 - alpha |N|^2, beta |N|^2) with the noisy phase, resynthesised by the
 * inverse STFT.
 *
 * @param y - Noisy signal.
 * @param options - Framing, noise estimate, over-subtraction alpha (default 2) and floor beta (default 0.01).
 * @returns The enhanced signal, the length of y.
 * @concept speech-enhancement
 * @example
 * ```ts
 * import { audio } from 'whitebox-engine/signal'
 * const y = Array.from({ length: 4000 }, (_, i) => (i > 1500 ? Math.sin(i / 5) : 0) + 0.1 * Math.sin(i * 2.1))
 * audio.spectralSubtraction(y, { nFft: 256 }).length // => 4000
 * ```
 */
export function spectralSubtraction(
  y: TensorLike,
  options: EnhancementOptions & { readonly alpha?: number; readonly beta?: number } = {},
): Float64Array {
  const x = samples(y, 'spectralSubtraction')
  const nFft = options.nFft ?? 512
  const hop = options.hopLength ?? Math.floor(nFft / 4)
  const s = spectra(x, nFft, hop)
  const np = noisePower(s, options, nFft, hop)
  const alpha = options.alpha ?? 2
  const beta = options.beta ?? 0.01
  const gain = new Float64Array(s.bins * s.frames)
  for (let b = 0; b < s.bins; b++)
    for (let t = 0; t < s.frames; t++) {
      const i = b * s.frames + t
      const p = (s.re[i] ?? 0) ** 2 + (s.im[i] ?? 0) ** 2
      const clean = Math.max(p - alpha * (np[b] ?? 0), beta * (np[b] ?? 0))
      gain[i] = p > 0 ? Math.sqrt(clean / p) : 0
    }
  return resynthesise(s, gain, nFft, hop, x.length)
}

/**
 * Wiener-filter denoising with the decision-directed a priori SNR: xi_t = a G_{t-1}^2 gamma_{t-1} + (1 - a)
 * max(gamma_t - 1, 0) with gamma the a posteriori SNR, and gain xi / (1 + xi) (floored).
 *
 * @param y - Noisy signal.
 * @param options - Framing, noise estimate, smoothing a (default 0.98) and gain floor (default 0.05).
 * @returns The enhanced signal.
 * @concept speech-enhancement
 * @example
 * ```ts
 * import { audio } from 'whitebox-engine/signal'
 * const y = Array.from({ length: 4000 }, (_, i) => (i > 1500 ? Math.sin(i / 5) : 0) + 0.1 * Math.sin(i * 2.1))
 * audio.wienerDenoise(y, { nFft: 256 }).length // => 4000
 * ```
 */
export function wienerDenoise(
  y: TensorLike,
  options: EnhancementOptions & { readonly smoothing?: number; readonly floor?: number } = {},
): Float64Array {
  const x = samples(y, 'wienerDenoise')
  const nFft = options.nFft ?? 512
  const hop = options.hopLength ?? Math.floor(nFft / 4)
  const s = spectra(x, nFft, hop)
  const np = noisePower(s, options, nFft, hop)
  const a = options.smoothing ?? 0.98
  const floor = options.floor ?? 0.05
  const gain = new Float64Array(s.bins * s.frames)
  for (let b = 0; b < s.bins; b++) {
    let prev = 1
    let prevGamma = 1
    const nb = Math.max(np[b] ?? 0, Number.MIN_VALUE * 2 ** 52)
    for (let t = 0; t < s.frames; t++) {
      const i = b * s.frames + t
      const gamma = ((s.re[i] ?? 0) ** 2 + (s.im[i] ?? 0) ** 2) / nb
      const xi = t === 0 ? Math.max(gamma - 1, 0) : a * prev * prev * prevGamma + (1 - a) * Math.max(gamma - 1, 0)
      const g = Math.max(floor, xi / (1 + xi))
      gain[i] = g
      prev = g
      prevGamma = gamma
    }
  }
  return resynthesise(s, gain, nFft, hop, x.length)
}

/**
 * Energy-based voice activity detection: a frame is active when its energy is within `rangeDb` of the loudest
 * frame's and above the noise floor by `thresholdDb`; activity is held for `hangover` frames after it ends.
 *
 * @param y - Signal.
 * @param options - Frame length (default 400), hop (default 160), threshold over the floor (default 6 dB), range
 *   below the peak (default 40 dB) and hangover (default 3 frames).
 * @returns 0/1 activity per frame.
 * @concept voice-activity-detection
 * @example
 * ```ts
 * import { audio } from 'whitebox-engine/signal'
 * import { toFlat } from 'whitebox-engine/tensor'
 * const y = Array.from({ length: 8000 }, (_, i) => (i > 4000 ? Math.sin(i / 5) : 0.001 * Math.sin(i * 2.1)))
 * const v = toFlat(audio.voiceActivity(y))
 * [v[0], v[v.length - 1]] // => [0, 1]
 * ```
 */
export function voiceActivity(
  y: TensorLike,
  options: {
    readonly frameLength?: number
    readonly hopLength?: number
    readonly thresholdDb?: number
    readonly rangeDb?: number
    readonly hangover?: number
  } = {},
): Tensor {
  const x = samples(y, 'voiceActivity')
  const fl = options.frameLength ?? 400
  const hop = options.hopLength ?? 160
  if (x.length < fl) throw new WhiteboxError('value', 'voiceActivity: the signal is shorter than a frame')
  const frames = 1 + Math.floor((x.length - fl) / hop)
  const energy = Float64Array.from({ length: frames }, (_, t) => {
    let e = 0
    for (let i = 0; i < fl; i++) e += (x[t * hop + i] ?? 0) ** 2
    return 10 * Math.log10(e / fl + 1e-12)
  })
  const sorted = Float64Array.from(energy).sort()
  const floor = sorted[Math.floor(0.1 * (frames - 1))] ?? 0
  let peak = Number.NEGATIVE_INFINITY
  for (const e of energy) peak = Math.max(peak, e)
  const thr = Math.max(floor + (options.thresholdDb ?? 6), peak - (options.rangeDb ?? 40))
  const out = new Float64Array(frames)
  let hold = 0
  for (let t = 0; t < frames; t++) {
    if ((energy[t] ?? 0) > thr) {
      out[t] = 1
      hold = options.hangover ?? 3
    } else if (hold > 0) {
      out[t] = 1
      hold--
    }
  }
  return fromStorage(out, [frames], 'float64')
}
