import { goldenSuite, type Params } from '../../../testing/suite.ts'
import { type ComplexValue, concat, reshape, stack, tensor, type Value } from '../../tensor/index.ts'
import * as S from '../index.ts'

const pair = (z: ComplexValue): Value => stack([z.re, z.im])
const name = (o: Params): string => o['wavelet'] as string
const mode = (o: Params): S.WaveletMode => (o['mode'] as S.WaveletMode | undefined) ?? 'symmetric'
const flat = (v: Value): Value => reshape(v, [-1])

goldenSuite('discrete wavelet transforms against PyWavelets', 'signal/wavelets/dwt', {
  filters: (_i, o) => {
    const w = S.wavelet(name(o))
    return tensor([Array.from(w.decLo), Array.from(w.decHi), Array.from(w.recLo), Array.from(w.recHi)])
  },
  dwt: ([x], o) => {
    const c = S.dwt(x, name(o), { mode: mode(o), axis: (o['axis'] as number | undefined) ?? -1 })
    return stack([c.cA, c.cD])
  },
  idwt: ([a, d], o) => S.idwt(a, d, name(o), { mode: mode(o) }),
  wavedec: ([x], o) =>
    concat(
      S.wavedec(x, name(o), { mode: mode(o), ...(o['level'] === undefined ? {} : { level: o['level'] as number }) }),
    ),
  waverec: ([x], o) =>
    S.waverec(
      S.wavedec(x, name(o), { mode: mode(o), ...(o['level'] === undefined ? {} : { level: o['level'] as number }) }),
      name(o),
      { mode: mode(o) },
    ),
  dwt2: ([x], o) => {
    const c = S.dwt2(x, name(o), mode(o))
    return stack([c.cA, c.cH, c.cV, c.cD])
  },
  idwt2: ([x], o) => S.idwt2(S.dwt2(x, name(o), mode(o)), name(o), mode(o)),
  wavedec2: ([x], o) => {
    const c = S.wavedec2(x, name(o), { level: o['level'] as number })
    return concat([flat(c.cA), ...c.details.flatMap((d) => [flat(d.cH), flat(d.cV), flat(d.cD)])])
  },
  waverec2: ([x], o) => S.waverec2(S.wavedec2(x, name(o), { level: o['level'] as number }), name(o)),
  dwtMaxLevel: (_i, o) => S.dwtMaxLevel(o['n'] as number, name(o)),
})

goldenSuite('stationary wavelet transforms against PyWavelets', 'signal/wavelets/swt', {
  swt: ([x], o) =>
    concat(
      S.swt(x, name(o), {
        level: o['level'] as number,
        ...(o['startLevel'] === undefined ? {} : { startLevel: o['startLevel'] as number }),
      }).flatMap((c) => [c.cA, c.cD]),
    ),
  iswt: ([x], o) => S.iswt(S.swt(x, name(o), { level: o['level'] as number }), name(o)),
})

goldenSuite('continuous wavelet transforms against PyWavelets', 'signal/wavelets/cwt', {
  wavefun: (_i, o) => {
    const w = S.continuousWavelet(name(o), o['precision'] as number)
    return stack([w.re, w.im])
  },
  wavefunX: (_i, o) => S.continuousWavelet(name(o), o['precision'] as number).x,
  centralFrequency: (_i, o) => S.centralFrequency(name(o)),
  cwt: ([x, s], o) => pair(S.cwt(x, Array.from(s.data), name(o), o['samplingPeriod'] as number).coefficients),
  cwtFreqs: ([x, s], o) => tensor(S.cwt(x, Array.from(s.data), name(o), o['samplingPeriod'] as number).frequencies),
})

goldenSuite('wavelet thresholding against PyWavelets and scikit-image', 'signal/wavelets/denoise', {
  threshold: ([x], o) =>
    S.threshold(x, o['value'] as number, o['mode'] as S.ThresholdMode, (o['substitute'] as number | undefined) ?? 0),
  waveletDenoise: ([x], o) => S.waveletDenoise(x, o as S.WaveletDenoiseOptions),
})
