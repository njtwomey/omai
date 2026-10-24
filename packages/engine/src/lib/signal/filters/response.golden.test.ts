import { goldenSuite } from '../../../testing/suite.ts'
import { type ComplexValue, stack, toFlat, type Value } from '../../tensor/index.ts'
import * as S from '../index.ts'

const pair = (z: ComplexValue): Value => stack([z.re, z.im])

goldenSuite('frequency responses against scipy.signal', 'signal/filters/response', {
  sosfreqz: ([sos], o) => pair(S.sosfreqz(sos, o['worN'] as number).h),
  sosfreqzAt: ([sos, f], o) => pair(S.sosfreqz(sos, f, { fs: o['fs'] as number }).h),
  freqzZpk: (_i, o) =>
    pair(
      S.freqzZpk(
        {
          zeros: { re: toFlat(o['zr'] as Value), im: toFlat(o['zi'] as Value) },
          poles: { re: toFlat(o['pr'] as Value), im: toFlat(o['pi'] as Value) },
          gain: o['k'] as number,
        },
        o['worN'] as number,
        { whole: o['whole'] as boolean },
      ).h,
    ),
  groupDelay: ([b, a], o) => S.groupDelay(b, a, o['worN'] as number).delay,
  groupDelayAt: ([b, a, f], o) => S.groupDelay(b, a, f, { fs: o['fs'] as number }).delay,
})
