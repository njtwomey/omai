import { irfft } from '../../numerics/fourier/index.ts'
import { type ComplexValue, primalOf, type TensorLike, toFlat } from '../../tensor/index.ts'
import { stft } from '../time-frequency.ts'
import { getWindow } from '../windows/index.ts'

// librosa's STFT conventions shared by the audio modules: centred frames padded with zeros, a periodic Hann window,
// no scaling; and its inverse, which keeps the tail of the last frame and divides by the summed squared window only
// where it is not tiny.

/** librosa.stft(y, n_fft, hop_length) as a complex value `[bins, frames]`. */
export function librosaStft(y: TensorLike, nFft: number, hop: number): ComplexValue {
  return stft(y, { nperseg: nFft, noverlap: nFft - hop, boundary: 'zeros', padded: false, scaling: 'none' }).values
}

/** librosa.istft: overlap-add of windowed frames over the summed squared window, trimmed and fixed to `length`. */
export function librosaIstft(z: ComplexValue, nFft: number, hop: number, length: number): Float64Array {
  const frames = z.re.shape[1] ?? 0
  const keep = Math.min(frames, Math.ceil((length + 2 * Math.floor(nFft / 2)) / hop))
  const re = primalOf(z.re)
  const im = primalOf(z.im)
  const bins = re.shape[0] ?? 0
  const win = toFlat(getWindow('hann', nFft))
  const total = nFft + hop * (keep - 1)
  const ola = new Float64Array(total)
  const wss = new Float64Array(total)
  for (let t = 0; t < keep; t++) {
    const fr = new Float64Array(bins)
    const fi = new Float64Array(bins)
    for (let b = 0; b < bins; b++) {
      fr[b] = Number(re.data[b * frames + t] ?? 0)
      fi[b] = Number(im.data[b * frames + t] ?? 0)
    }
    const frame = toFlat(irfft({ re: fr, im: fi }, nFft))
    for (let j = 0; j < nFft; j++) {
      ola[t * hop + j] = (ola[t * hop + j] ?? 0) + (win[j] ?? 0) * (frame[j] ?? 0)
      wss[t * hop + j] = (wss[t * hop + j] ?? 0) + (win[j] ?? 0) ** 2
    }
  }
  // Trim half a frame at the start (centred frames), divide where the summed squared window is not tiny, fix length.
  const start = Math.floor(nFft / 2)
  const out = new Float64Array(length)
  for (let i = 0; i < length && start + i < total; i++) {
    const w = wss[start + i] ?? 0
    const v = ola[start + i] ?? 0
    out[i] = w > Number.MIN_VALUE * 2 ** 52 ? v / w : v
  }
  return out
}
