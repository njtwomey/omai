import { describe, it } from 'vitest'
import { type GoldenFn, runFixture, sortedParts } from '../../testing/systems.ts'
import { type Tensor, tensor, toFlat } from '../tensor/index.ts'
import * as S from './index.ts'

type Part = 'a' | 'b' | 'c' | 'd'

const FNS: Readonly<Record<string, GoldenFn>> = {
  'c2d-ss': ([a, b, c, d], p) =>
    S.cont2discrete(S.ss(a as Tensor, b as Tensor, c as Tensor, d), p['dt'] as number, {
      method: p['method'] as S.DiscretisationMethod,
      ...(p['alpha'] === undefined ? {} : { alpha: p['alpha'] as number }),
    })[p['part'] as Part],
  'c2d-tf': ([n, d], p) => {
    const r = S.cont2discrete(S.tf(n as Tensor, d as Tensor), p['dt'] as number, {
      method: p['method'] as S.DiscretisationMethod,
    })
    return p['part'] === 'num' ? r.num : r.den
  },
  'c2d-zpk': ([z, q], p) => {
    const r = S.cont2discrete(S.zpk(toFlat(z as Tensor), toFlat(q as Tensor), p['gain'] as number), p['dt'] as number)
    return p['part'] === 'zeros' ? sortedParts(r.zeros) : p['part'] === 'poles' ? sortedParts(r.poles) : tensor(r.gain)
  },
  'c2d-prewarp': ([a, b, c, d], p) =>
    S.cont2discrete(S.ss(a as Tensor, b as Tensor, c as Tensor, d), p['dt'] as number, {
      method: 'bilinear',
      prewarpFrequency: p['w'] as number,
    })[p['part'] as Part],
  'c2d-matched': ([n, d], p) => {
    const r = S.cont2discrete(S.tf(n as Tensor, d as Tensor), p['dt'] as number, { method: 'matched' })
    return p['part'] === 'num' ? r.num : r.den
  },
}

describe('systems/discretise against scipy.signal and python-control', () => {
  runFixture('systems/discretise', FNS, it)
})
