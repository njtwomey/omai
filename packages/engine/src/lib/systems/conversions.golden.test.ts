import { describe, it } from 'vitest'
import { complexOf, type GoldenFn, runFixture, sortedParts } from '../../testing/systems.ts'
import { tensor } from '../tensor/index.ts'
import * as S from './index.ts'

type Part = 'a' | 'b' | 'c' | 'd'

const zpkPart = (z: S.ZerosPolesGainData, part: unknown) =>
  part === 'zeros' ? sortedParts(z.zeros) : part === 'poles' ? sortedParts(z.poles) : tensor(z.gain)

const FNS: Readonly<Record<string, GoldenFn>> = {
  'normalize-num': ([n, d]) => S.normalize(n as never, d as never).num,
  'normalize-den': ([n, d]) => S.normalize(n as never, d as never).den,
  tf2ss: ([n, d], p) => S.tf2ss(n as never, d as never)[p['part'] as Part],
  tf2zpk: ([n, d], p) => zpkPart(S.tf2zpk(n as never, d as never), p['part']),
  ss2tf: ([a, b, c, d], p) => {
    const r = S.ss2tf(a as never, b as never, c as never, d, p['input'] as number)
    return p['part'] === 'num' ? r.num : r.den
  },
  ss2zpk: ([a, b, c, d], p) => zpkPart(S.ss2zpk(a as never, b as never, c as never, d), p['part']),
  zpk2tf: ([z, q], p) => {
    const r = S.zpk2tf(complexOf(z as never), complexOf(q as never), p['gain'] as number)
    return p['part'] === 'num' ? r.num : r.den
  },
  zpk2ss: ([z, q], p) => S.zpk2ss(complexOf(z as never), complexOf(q as never), p['gain'] as number)[p['part'] as Part],
}

describe('systems/conversions against scipy.signal', () => {
  runFixture('systems/conversions', FNS, it)
})
