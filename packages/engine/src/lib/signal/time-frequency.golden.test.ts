import { goldenSuite, type Params } from '../../testing/suite.ts'
import { type ComplexValue, reshape, sliceAxis, stack, type Value } from '../tensor/index.ts'
import * as S from './index.ts'

const pair = (z: ComplexValue): Value => stack([z.re, z.im])
const unpair = (p: Value): ComplexValue => {
  const shape = p.shape.slice(1)
  return { re: reshape(sliceAxis(p, 0, 0, 1), shape), im: reshape(sliceAxis(p, 0, 1, 2), shape) }
}

goldenSuite('short-time Fourier analysis against scipy.signal and librosa', 'signal/time_frequency', {
  stft: ([x], o) => pair(S.stft(x, o as S.StftOptions).values),
  stftTimes: ([x], o) => S.stft(x, o as S.StftOptions).times,
  istft: ([z], o) => S.istft(unpair(z), o as S.IstftOptions).x,
  spectrogram: ([x], o) => S.spectrogram(x, o as S.SpectrogramOptions, o['mode'] as 'psd').values,
  librosaStft: ([x], o: Params) =>
    pair(
      S.stft(x, {
        nperseg: o['nFft'] as number,
        noverlap: (o['nFft'] as number) - (o['hop'] as number),
        boundary: (o['boundary'] as S.StftBoundary | undefined) ?? 'zeros',
        padded: false,
        scaling: 'none',
      }).values,
    ),
  instantaneousFrequency: ([x], o) => S.instantaneousFrequency(x, o['fs'] as number),
})
