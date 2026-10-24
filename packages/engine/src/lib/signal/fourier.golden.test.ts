import { goldenSuite } from '../../testing/suite.ts'
import { type ComplexValue, stack, type Value } from '../tensor/index.ts'
import * as S from './index.ts'

const pair = (z: ComplexValue): Value => stack([z.re, z.im])

goldenSuite('DTFT and analytic signals against scipy.signal', 'signal/fourier', {
  dtft: ([x, f], o) => pair(S.dtft(x, f, o['fs'] as number)),
  hilbert: ([x], o) => pair(S.hilbert(x, o['n'] as number | undefined, (o['axis'] as number | undefined) ?? -1)),
})

goldenSuite('mu-law companding against librosa and sinc interpolation', 'signal/sampling', {
  muLawEncode: ([x], o) => S.muLawEncode(x, o['mu'] as number, o['quantize'] as boolean),
  muLawDecode: ([x], o) => S.muLawDecode(x, o['mu'] as number, o['quantize'] as boolean),
  sincInterpolate: ([x, t], o) => S.sincInterpolate(x, o['fs'] as number, t),
})
