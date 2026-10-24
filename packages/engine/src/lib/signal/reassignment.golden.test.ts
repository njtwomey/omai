import { goldenSuite } from '../../testing/suite.ts'
import { type ComplexValue, stack, tensor, type Value } from '../tensor/index.ts'
import * as S from './index.ts'

const pair = (z: ComplexValue): Value => stack([z.re, z.im])

goldenSuite('reassigned spectrograms against librosa', 'signal/reassignment', {
  reassignedFrequencies: ([y], o) => S.reassignedSpectrogram(y, o as S.ReassignmentOptions).frequencies,
  reassignedTimes: ([y], o) => S.reassignedSpectrogram(y, o as S.ReassignmentOptions).times,
  reassignedMagnitudes: ([y], o) => S.reassignedSpectrogram(y, o as S.ReassignmentOptions).magnitudes,
})

goldenSuite('constant-Q transforms against librosa', 'signal/constant_q', {
  cqt: ([y], o) => pair(S.cqt(y, o as S.CqtOptions).values),
  cqtFilters: (_i, o) => pair(S.cqtFilters(o as S.CqtOptions).filters),
  cqtLengths: (_i, o) => tensor(S.cqtFilters(o as S.CqtOptions).lengths),
})
