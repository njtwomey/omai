import { describe, it } from 'vitest'
import { type GoldenFn, runFixture } from '../../testing/systems.ts'
import { div, reshape, sliceAxis, stack, tensor } from '../tensor/index.ts'
import * as S from './index.ts'

const FNS: Readonly<Record<string, GoldenFn>> = {
  'minreal-ss-order': ([a, b, c, d]) =>
    tensor(S.dimensions(S.minreal(S.ss(a as never, b as never, c as never, d))).states),
  'minreal-ss-tf': ([a, b, c, d], p) => {
    const reduced = S.minreal(S.ss(a as never, b as never, c as never, d)) as S.StateSpace
    const g = S.toTf(reduced, { input: p['input'] as number })
    const o = p['output_index'] as number
    const num = g.num.shape.length === 2 ? reshape(sliceAxis(g.num, 0, o, o + 1), [-1]) : g.num
    return p['part'] === 'num' ? div(num, sliceAxis(g.den, 0, 0, 1)) : div(g.den, sliceAxis(g.den, 0, 0, 1))
  },
  'minreal-ss-freq': ([a, b, c, d, w]) => {
    const h = S.freqresp(S.minreal(S.ss(a as never, b as never, c as never, d)), w).h
    return stack([reshape(h.re, [-1]), reshape(h.im, [-1])], 0)
  },
  'minreal-tf': ([n, d], p) => {
    const r = S.minreal(S.tf(n as never, d as never)) as S.TransferFunction
    return p['part'] === 'num' ? r.num : r.den
  },
}

describe('systems/representations against python-control', () => {
  runFixture('systems/representations', FNS, it)
})
