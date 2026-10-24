import { describe, it } from 'vitest'
import { type GoldenFn, runFixture } from '../../testing/systems.ts'
import { reshape, stack, type Tensor, tensor, toFlat } from '../tensor/index.ts'
import * as S from './index.ts'

const FNS: Readonly<Record<string, GoldenFn>> = {
  inverseLaplace: ([n, d, t]) => {
    const inv = S.inverseLaplace(n as Tensor, d as Tensor)
    return tensor(toFlat(t as Tensor).map(inv.at))
  },
  inverseZ: ([b, a], p) => {
    const inv = S.inverseZ(b as Tensor, a as Tensor)
    return tensor(Array.from({ length: p['n'] as number }, (_, k) => inv.at(k)))
  },
  differenceToTf: ([b, a, w]) => {
    const h = S.freqresp(S.differenceToTf(b as Tensor, a as Tensor), w).h
    return stack([reshape(h.re, [-1]), reshape(h.im, [-1])], 0)
  },
}

describe('systems/transforms against scipy.signal', () => {
  runFixture('systems/transforms', FNS, it)
})
