import { describe, it } from 'vitest'
import { complexOf, type GoldenFn, runFixture, sortedParts } from '../../testing/systems.ts'
import { tensor } from '../tensor/index.ts'
import * as S from './index.ts'

const FNS: Readonly<Record<string, GoldenFn>> = {
  zpk2sos: ([z, p], q) =>
    S.zpk2sos(complexOf(z as never), complexOf(p as never), q['gain'] as number, {
      pairing: q['pairing'] as S.SosPairing,
      analog: q['analog'] as boolean,
    }),
  sos2tf: ([s], q) => (q['part'] === 'num' ? S.sos2tf(s as never).num : S.sos2tf(s as never).den),
  sos2zpk: ([s], q) => {
    const r = S.sos2zpk(s as never)
    return q['part'] === 'zeros' ? sortedParts(r.zeros) : q['part'] === 'poles' ? sortedParts(r.poles) : tensor(r.gain)
  },
  tf2sos: ([b, a]) => S.tf2sos(b as never, a as never),
}

describe('systems/sections against scipy.signal', () => {
  runFixture('systems/sections', FNS, it)
})
