import { describe, it } from 'vitest'
import { type GoldenFn, runFixture } from '../../testing/systems.ts'
import { type ComplexValue, div, reshape, sliceAxis, stack, type Tensor, type Value } from '../tensor/index.ts'
import * as S from './index.ts'

const W = [0.1, 0.31622776601683794, 1, 3.1622776601683795, 10]
const parts = (h: ComplexValue): Value => stack([reshape(h.re, [-1]), reshape(h.im, [-1])], 0)

const FNS: Readonly<Record<string, GoldenFn>> = {
  'tf-op': ([n1, d1, n2, d2], p) => {
    const g1 = S.tf(n1 as Tensor, d1 as Tensor)
    const g2 = S.tf(n2 as Tensor, d2 as Tensor)
    const op = p['op']
    const r =
      op === 'series'
        ? S.series(g1, g2)
        : op === 'parallel'
          ? S.parallel(g1, g2)
          : op === 'unity'
            ? S.feedback(g1)
            : S.feedback(g1, g2, p['sign'] as number)
    const lead = sliceAxis(r.den, 0, 0, 1)
    return p['part'] === 'num' ? div(r.num, lead) : div(r.den, lead)
  },
  'ss-op': ([a1, b1, c1, d1, a2, b2, c2, d2], p) => {
    const s1 = S.ss(a1 as Tensor, b1 as Tensor, c1 as Tensor, d1)
    const s2 = S.ss(a2 as Tensor, b2 as Tensor, c2 as Tensor, d2)
    const op = p['op']
    const r =
      op === 'series'
        ? S.series(s1, s2)
        : op === 'parallel'
          ? S.parallel(s1, s2)
          : S.feedback(s1, s2, p['sign'] as number)
    return parts(S.freqresp(r, W).h)
  },
  connect: ([n1, d1, n2, d2]) => {
    const blocks = S.append(S.tf(n1 as Tensor, d1 as Tensor), S.tf(n2 as Tensor, d2 as Tensor))
    return parts(
      S.freqresp(
        S.connect(
          blocks,
          [
            [2, 1],
            [1, -2],
          ],
          [1],
          [1, 2],
        ),
        W,
      ).h,
    )
  },
}

describe('systems/interconnect against python-control', () => {
  runFixture('systems/interconnect', FNS, it)
})
